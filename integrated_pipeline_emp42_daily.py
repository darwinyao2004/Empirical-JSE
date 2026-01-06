# -*- coding: utf-8 -*-
"""
Integrated empirical pipeline (42-day fit):
- Read each yyyymm_full.csv (permno + ~63 daily returns)
- Drop rows with any NaN across the full 3-month window (keeps stock universe consistent for OOS)
- Use first ~42 trading days to estimate covariance (PCA / JSE)
- Construct GMV portfolio weights from that covariance
- Use the remaining days (~21 trading days) as the OOS (third-month) daily return window,
  compute the within-month variance of the portfolio's DAILY returns, and treat that as the monthly OOS metric

Outputs (all suffixed with 42):
- covariance_outputs_emp42/{PCA,JSE,raw,meta,logs}/...
- portfolio_outputs_emp42/{PCA,JSE}/PortfolioA_GMV/yyyymm_weights.csv
- results_emp42_daily/{PCA,JSE}/PortfolioA_GMV_daily_variance.csv
- results_emp42_daily/{PCA,JSE}/PortfolioA_GMV_daily_variance_timeseries.png
- results_emp42_daily/summary.csv
- results_emp42_daily/PortfolioA_GMV_daily_variance_timeseries.png

Notes:
- This script intentionally preserves the core estimator implementations and GMV solver logic
  from the original scripts, and only changes the rolling logic to "first two months fit,
  third month test" inside the same 3-month file to avoid missing-stock issues.

Author: (integrated from Darwin Yao scripts)
"""
from __future__ import annotations

import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Configuration (defaults)
# -----------------------------
METHODS = ["PCA", "JSE"]
PORTFOLIO_TYPES = ["PortfolioA_GMV"]

FILE_SUFFIX_FULL = "_full.csv"         # input daily returns
FILE_SUFFIX_COV = "_full_cov.csv"      # output covariance file suffix (upstream convention)

DEFAULT_IN_DIR = "500_ret_exlude_zero_var"   # keep original default from covariance script
DEFAULT_OUT_COV_ROOT = "covariance_outputs_emp42_daily"
DEFAULT_OUT_PORT_ROOT = "portfolio_outputs_emp42_daily"
DEFAULT_OUT_RESULTS_ROOT = "results_emp42_daily"

DEFAULT_NUM_FACTORS = 1
DEFAULT_EPS = 1e-12

# "42 days around 2 months" as requested
DEFAULT_EST_DAYS = 42


# -----------------------------
# Utilities (from covariance script)
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def demean_over_time(X: np.ndarray) -> np.ndarray:
    """Demean by row (each stock minus its time mean)."""
    return X - X.mean(axis=1, keepdims=True)


def sample_cov(X: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    X: (p, n) demeaned return matrix (rows=assets, columns=time)
    Returns: S = (1/n) X X^T (not n-1), n
    """
    p, n = X.shape
    if n <= 1:
        raise ValueError("Need at least 2 sample periods.")
    S = (X @ X.T) / n
    S = (S + S.T) / 2.0
    return S, n


def top_k_eigenpairs(S: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Eigendecomposition of symmetric matrix, return top k (sorted by eigenvalue descending)."""
    vals, vecs = np.linalg.eigh(S)     # ascending
    idx = np.argsort(vals)[::-1]       # descending
    vals = vals[idx]
    vecs = vecs[:, idx]
    k = min(k, vecs.shape[1])
    return vecs[:, :k], vals[:k]


def top1_eigvec_mean(Sigma: np.ndarray) -> Tuple[float, float]:
    """
    Return (mean of top-eigenvalue eigenvector, top eigenvalue),
    with sign aligned so eigenvector mean is non-negative.
    """
    A = (Sigma + Sigma.T) / 2.0
    vals, vecs = np.linalg.eigh(A)  # ascending
    j = int(np.argmax(vals))
    v = vecs[:, j]
    if float(v.mean()) < 0:
        v = -v
    return float(v.mean()), float(vals[j])


def pca_factor_cov(S: np.ndarray, k: int, eps: float = 1e-12) -> np.ndarray:
    """
    PCA factor covariance (paper-aligned baseline for JSE comparisons):

        Σ = U_k diag(max(λ_j - ℓ^2, 0)) U_k^T + (n/p) ℓ^2 I,

    where ℓ^2 is the average of the remaining nonzero eigenvalues of S, and n is
    inferred as rank(S) (in the HL setting with p>n, rank(S)≈n).
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))

    scale = float(np.linalg.norm(S, ord=2)) if p > 0 else 0.0
    tol = max(eps, 1e-12 * scale)
    n_eff = int(np.linalg.matrix_rank(S, tol=tol))
    n_eff = max(1, min(n_eff, p))

    trS = float(np.trace(S))

    if k_eff == 0:
        delta2 = max(trS / float(p), eps)
        Sigma = delta2 * np.eye(p)
        Sigma = (Sigma + Sigma.T) / 2.0
        return Sigma

    U, lam = top_k_eigenpairs(S, k_eff)

    denom = n_eff - k_eff
    if denom <= 0:
        ell2 = 0.0
    else:
        ell2 = (trS - float(np.sum(lam))) / float(denom)
        ell2 = max(ell2, 0.0)

    spike = np.maximum(lam - ell2, 0.0)
    Sig_k = U @ (spike[:, None] * U.T)

    delta2 = (float(n_eff) / float(p)) * ell2
    delta2 = max(delta2, 0.0)

    Sigma = Sig_k + delta2 * np.eye(p)
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def js_eigvec_factor_cov(S: np.ndarray, k: int, n: int, eps: float = 1e-12) -> np.ndarray:
    """
    JS-eigvec factor covariance (aligned to Goldberg–Kercheval JSE paper):
      1) Top k eigenpairs of S: (U_k, Λ_k)
      2) Apply JSE shrinkage to each column of U_k:
         h^JSE = m(h)·1 + c^JSE (h - m(h)·1),
         c^JSE = 1 - ϖ^2 / s^2(h),
         s^2(h) = (λ_j/p) * Σ (h_i - m(h))^2,
         ϖ^2 = ℓ^2 / p,  where ℓ^2 is the average of the remaining nonzero eigenvalues of S
      3) Covariance estimator consistent with Eq. (37)-style construction:
         Σ = Q diag(max(λ_j - ℓ^2, 0)) Q^T + (n/p) ℓ^2 I
         where Q is orthonormalized from the JSE-shrunk vectors.
    """
    p = S.shape[0]
    k_eff = max(0, min(k, p - 1))

    scale = float(np.linalg.norm(S, ord=2)) if p > 0 else 0.0
    tol = max(eps, 1e-12 * scale)
    n_eff = int(np.linalg.matrix_rank(S, tol=tol))
    n_eff = max(1, min(n_eff, p, int(n)))

    trS = float(np.trace(S))
    one = np.ones((p, 1))

    if k_eff == 0:
        delta2 = max(trS / float(p), eps)
        Sigma = delta2 * np.eye(p)
        Sigma = (Sigma + Sigma.T) / 2.0
        return Sigma

    U, lam = top_k_eigenpairs(S, k_eff)

    denom = n_eff - k_eff
    if denom <= 0:
        ell2 = 0.0
    else:
        ell2 = (trS - float(np.sum(lam))) / float(denom)
        ell2 = max(ell2, 0.0)

    varpi2 = ell2 / float(p)

    U_js = U.copy()
    for j in range(k_eff):
        h = U[:, j:j+1]
        if float(h.mean()) < 0:
            h = -h

        m = float(h.mean())
        h_c = h - m * one

        lamj = float(lam[j])
        s2 = lamj * float(np.sum(h_c ** 2)) / float(p)

        c = 1.0 - (varpi2 / (s2 + 1e-18))
        c = max(0.0, c)  # positive-part truncation
        U_js[:, j:j+1] = m * one + c * h_c

    Q, _ = np.linalg.qr(U_js)
    for j in range(k_eff):
        if float(np.dot(Q[:, j], U[:, j])) < 0:
            Q[:, j] *= -1.0

    spike = np.maximum(lam - ell2, 0.0)
    Sig_k = Q @ (spike[:, None] * Q.T)

    delta2 = (float(n_eff) / float(p)) * ell2
    Sigma = Sig_k + delta2 * np.eye(p)

    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


# -----------------------------
# Utilities (from portfolio construction script)
# -----------------------------
def save_weights(permnos: np.ndarray, w: np.ndarray, out_path: Path) -> None:
    df = pd.DataFrame({"permno": permnos, "weight": w})
    df.to_csv(out_path, index=False)


def solve_gmv_closed(Sigma: np.ndarray) -> np.ndarray:
    """
    Simplest GMV (unconstrained):
    - Only depends on Sigma
    - Weights sum to 1
    """
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square 2D array.")

    p = Sigma.shape[0]
    ones = np.ones(p)

    Sigma_inv_ones = np.linalg.solve(Sigma, ones)
    weights = Sigma_inv_ones / Sigma_inv_ones.sum()
    return weights


# -----------------------------
# New OOS logic: within-file 2-month fit, 3rd-month test
# -----------------------------
def compound_return(daily_ret: np.ndarray) -> np.ndarray:
    """
    daily_ret: (p, n_days) daily simple returns in decimal form.
    returns: (p,) compounded return over the window.
    """
    # stable product; assumes no NaN
    return np.prod(1.0 + daily_ret, axis=1) - 1.0


def parse_yyyymm_from_key(key: str) -> Optional[str]:
    """
    key is typically like '201501_full' (from yyyymm_full.csv stem).
    Returns '201501' if parseable, else None.
    """
    token = key.split("_")[0]
    if len(token) == 6 and token.isdigit():
        return token
    return None


def month_end_timestamp(yyyymm: str) -> pd.Timestamp:
    """
    Convert 'YYYYMM' to a month-end Timestamp.
    """
    per = pd.Period(f"{yyyymm[:4]}-{yyyymm[4:]}", freq="M")
    return per.to_timestamp(how="end").normalize()


@dataclass
class BacktestPoint:
    date: pd.Timestamp
    realized_variance: float
    n_oos_days: int


def integrated_run(
    in_dir: Path,
    cov_out_root: Path,
    port_out_root: Path,
    results_root: Path,
    est_days: int = DEFAULT_EST_DAYS,
    eps: float = DEFAULT_EPS,
    num_factors: int = DEFAULT_NUM_FACTORS,
) -> None:
    # Output dirs
    out_js = cov_out_root / "JSE"
    out_pca = cov_out_root / "PCA"
    out_meta = cov_out_root / "meta"
    out_logs = cov_out_root / "logs"
    out_raw = cov_out_root / "raw"

    for d in (out_js, out_pca, out_meta, out_logs, out_raw):
        ensure_dir(d)

    # Portfolio outputs
    for method in METHODS:
        ensure_dir(port_out_root / method / "PortfolioA_GMV")

    ensure_dir(results_root)

    # Input files
    files = sorted([p for p in in_dir.glob("*.csv") if p.name.endswith(FILE_SUFFIX_FULL) and not p.name.startswith("._")])
    if not files:
        print(f"[WARN] No *{FILE_SUFFIX_FULL} files found in {in_dir}.")
        return

    cov_log_rows: List[dict] = []
    backtests: Dict[str, List[BacktestPoint]] = {m: [] for m in METHODS}

    def save_with_permno(mat: np.ndarray, out_path: Path, permno_vec: np.ndarray) -> None:
        df_out = pd.DataFrame(mat)
        df_out.insert(0, "permno", permno_vec)
        df_out.to_csv(out_path, header=False, index=False)

    for fpath in files:
        key = fpath.stem  # yyyymm_full
        yyyymm = parse_yyyymm_from_key(key)
        if yyyymm is None:
            print(f"[SKIP] Cannot parse YYYYMM from {fpath.name}; expected leading 6 digits.")
            continue

        print(f"[Processing] {fpath.name} (fit≈{est_days} days; OOS=remaining days)...")

        # Read returns
        try:
            df_all = pd.read_csv(fpath, header=None)
        except UnicodeDecodeError:
            df_all = pd.read_csv(fpath, header=None, encoding="latin-1")

        permno_col = df_all.iloc[:, 0].values
        ret_df = df_all.iloc[:, 1:]

        X_raw_all = ret_df.values.astype(float)  # expected: (500, ~63)
        # Drop rows with any NaN across the full window so fit and test share the same universe
        mask = ~np.any(np.isnan(X_raw_all), axis=1)
        kept_idx = np.where(mask)[0]
        kept_permno = permno_col[mask]
        X_all = X_raw_all[mask, :]

        p, n_total = X_all.shape
        if n_total <= est_days:
            print(f"[SKIP] {key}: total days={n_total} <= est_days={est_days}. Need at least 1 OOS day.")
            continue

        X_est_raw = X_all[:, :est_days]
        X_oos_raw = X_all[:, est_days:]
        n_est = X_est_raw.shape[1]
        n_oos = X_oos_raw.shape[1]

        # --- Covariance estimation on the first ~42 days (demeaned) ---
        X_est = demean_over_time(X_est_raw)
        S, nobs = sample_cov(X_est)

        # k logic is preserved: fixed k, but still capped by p-1 and n-1
        k = int(num_factors)
        k_eff = max(0, min(k, p - 1, nobs - 1))

        Sigma_PCA = pca_factor_cov(S, k_eff, eps=eps)
        Sigma_JS = js_eigvec_factor_cov(S, k_eff, nobs, eps=eps)

        raw_vmean, _ = top1_eigvec_mean(S)
        pca_vmean, _ = top1_eigvec_mean(Sigma_PCA)
        jse_vmean, _ = top1_eigvec_mean(Sigma_JS)
        print(f"    Top-eigvec mean | Raw: {raw_vmean:.6e} | PCA: {pca_vmean:.6e} | JSE: {jse_vmean:.6e}")

        # --- Save covariances (same format as upstream scripts) ---
        save_with_permno(Sigma_PCA, out_pca / f"{key}_cov.csv", kept_permno)
        save_with_permno(Sigma_JS, out_js / f"{key}_cov.csv", kept_permno)
        save_with_permno(S,        out_raw / f"{key}_cov.csv", kept_permno)

        # Metadata (preserve existing keys; add fit/test lengths)
        meta = {
            "file": fpath.name,
            "p_after_drop": int(p),
            "n_obs": int(nobs),                 # fit-window observations
            "n_total_days": int(n_total),       # full 3-month window
            "est_days": int(n_est),
            "oos_days": int(n_oos),
            "k_requested": int(k),
            "k_used": int(k_eff),
            "num_dropped_rows": int(X_raw_all.shape[0] - p),
            "kept_row_indices": kept_idx.tolist(),
            "lw_delta": 0.0,  # placeholder (LW disabled upstream)
        }
        with open(out_meta / f"{key}.json", "w", encoding="utf-8") as jf:
            json.dump(meta, jf, ensure_ascii=False, indent=2)

        cov_log_rows.append(
            dict(
                month=key,
                p_after_drop=p,
                n_obs=nobs,
                n_total_days=n_total,
                est_days=n_est,
                oos_days=n_oos,
                k_req=k,
                k_used=k_eff,
                dropped=int(X_raw_all.shape[0] - p),
                lw_delta=float(0.0),
            )
        )

        # --- Portfolio construction (GMV) ---
        # Keep solver logic exactly: use solve_gmv_closed
        w_pca = solve_gmv_closed(Sigma_PCA)
        w_jse = solve_gmv_closed(Sigma_JS)

        out_w_pca = port_out_root / "PCA" / "PortfolioA_GMV" / f"{yyyymm}_weights.csv"
        out_w_jse = port_out_root / "JSE" / "PortfolioA_GMV" / f"{yyyymm}_weights.csv"
        save_weights(kept_permno, w_pca, out_w_pca)
        save_weights(kept_permno, w_jse, out_w_jse)

        # --- OOS evaluation within the same file (third month; DAILY returns) ---
        # Compute portfolio daily returns in the OOS window, then within-month variance.
        # No renormalization: keep weights exactly as produced by solver.
        port_daily_pca = np.dot(w_pca, X_oos_raw)  # (n_oos,)
        port_daily_jse = np.dot(w_jse, X_oos_raw)  # (n_oos,)

        var_pca = float(np.var(port_daily_pca, ddof=1)) if n_oos > 1 else float("nan")
        var_jse = float(np.var(port_daily_jse, ddof=1)) if n_oos > 1 else float("nan")

        dt = month_end_timestamp(yyyymm)
        backtests["PCA"].append(BacktestPoint(date=dt, realized_variance=var_pca, n_oos_days=int(n_oos)))
        backtests["JSE"].append(BacktestPoint(date=dt, realized_variance=var_jse, n_oos_days=int(n_oos)))

        print(f"[OK] {key}: p={p}, fit_days={n_est}, oos_days={n_oos} -> cov+weights+oos done")

    # Save covariance summary log
    if cov_log_rows:
        df_covlog = pd.DataFrame(cov_log_rows)
        df_covlog.to_csv(out_logs / "summary.csv", index=False)
        print(f"[OK] Wrote covariance log: {out_logs / 'summary.csv'}")

    # Save per-method return curves + summary
    summary_rows = []
    for method in METHODS:
        pts = backtests.get(method, [])
        if not pts:
            continue
        pts_df = pd.DataFrame(
            {
                "date": [p.date for p in pts],
                "realized_variance": [p.realized_variance for p in pts],
                "n_oos_days": [p.n_oos_days for p in pts],
            }
        ).sort_values("date").reset_index(drop=True)

        mu = float(pts_df["realized_variance"].mean())
        var = float(pts_df["realized_variance"].var(ddof=1)) if len(pts_df) > 1 else float("nan")

        out_dir = results_root / method
        ensure_dir(out_dir)
        out_curve = out_dir / "PortfolioA_GMV_daily_variance.csv"
        pts_df.to_csv(out_curve, index=False)

        # Plot per-method monthly time series of within-month DAILY variance
        fig = plt.figure()
        plt.plot(pts_df["date"], pts_df["realized_variance"], label=f"{method}")
        plt.xlabel("date")
        plt.ylabel("within-month daily variance")
        plt.title(f"{method} PortfolioA_GMV: within-month daily variance (monthly series)")
        plt.tight_layout()
        out_png = out_dir / "PortfolioA_GMV_daily_variance_timeseries.png"
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        summary_rows.append(
            dict(method=method, portfolio_type="PortfolioA_GMV", n_points=len(pts_df), mean_daily_variance=mu, var_daily_variance=var)
        )
        print(f"[OK] Saved daily-variance curve: {out_curve} (N={len(pts_df)}, mean={mu:.6e}, var={var:.6e})")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = results_root / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Wrote summary: {summary_path}")
        print(summary_df.to_string(index=False))

        # Combined plot across methods
        fig = plt.figure()
        for m in METHODS:
            curve_path = results_root / m / "PortfolioA_GMV_daily_variance.csv"
            if curve_path.exists():
                df_m = pd.read_csv(curve_path, parse_dates=["date"])
                plt.plot(df_m["date"], df_m["realized_variance"], label=m)
        plt.xlabel("date")
        plt.ylabel("within-month daily variance")
        plt.title("PortfolioA_GMV: within-month daily variance (monthly series)")
        plt.legend()
        plt.tight_layout()
        out_png = results_root / "PortfolioA_GMV_daily_variance_timeseries.png"
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        # Print mean of the monthly within-month daily variance series (requested "10-year mean")
        for m in METHODS:
            curve_path = results_root / m / "PortfolioA_GMV_daily_variance.csv"
            if curve_path.exists():
                df_m = pd.read_csv(curve_path, parse_dates=["date"])
                mean_var = float(df_m["realized_variance"].mean())
                n_months = int(df_m.shape[0])
                approx_years = n_months / 12.0
                print(f"[MEAN] {m} PortfolioA_GMV within-month daily variance: {mean_var:.6e} (N={n_months} months, ~{approx_years:.2f} years)")
    else:
        print("[WARN] No results produced; check your inputs.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Integrated cov->portfolio->OOS pipeline (42-day fit within each file).")
    p.add_argument("--in_dir", type=str, default=DEFAULT_IN_DIR,
                   help=f"Directory containing *{FILE_SUFFIX_FULL} files (default: {DEFAULT_IN_DIR})")
    p.add_argument("--cov_out_root", type=str, default=DEFAULT_OUT_COV_ROOT,
                   help=f"Covariance output root (default: {DEFAULT_OUT_COV_ROOT})")
    p.add_argument("--port_out_root", type=str, default=DEFAULT_OUT_PORT_ROOT,
                   help=f"Portfolio output root (default: {DEFAULT_OUT_PORT_ROOT})")
    p.add_argument("--results_root", type=str, default=DEFAULT_OUT_RESULTS_ROOT,
                   help=f"Results output root (default: {DEFAULT_OUT_RESULTS_ROOT})")
    p.add_argument("--est_days", type=int, default=DEFAULT_EST_DAYS,
                   help=f"Number of trading days used for estimation (default: {DEFAULT_EST_DAYS})")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS,
                   help=f"Minimum truncation parameter (default: {DEFAULT_EPS})")
    p.add_argument("--num_factors", type=int, default=DEFAULT_NUM_FACTORS,
                   help=f"Fixed number of factors k (default: {DEFAULT_NUM_FACTORS})")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser().resolve()
    cov_out_root = Path(args.cov_out_root).expanduser().resolve()
    port_out_root = Path(args.port_out_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()

    print("Input directory:      ", in_dir)
    print("Cov output root:      ", cov_out_root)
    print("Portfolio output root:", port_out_root)
    print("Results output root:  ", results_root)
    print("Fit days (≈42):       ", args.est_days)
    print("Fixed k:              ", args.num_factors)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    ensure_dir(cov_out_root)
    ensure_dir(port_out_root)
    ensure_dir(results_root)

    integrated_run(
        in_dir=in_dir,
        cov_out_root=cov_out_root,
        port_out_root=port_out_root,
        results_root=results_root,
        est_days=int(args.est_days),
        eps=float(args.eps),
        num_factors=int(args.num_factors),
    )


if __name__ == "__main__":
    main()
