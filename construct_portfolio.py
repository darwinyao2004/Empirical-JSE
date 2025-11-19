"""
Portfolio construction from monthly full-rank covariance matrices.

Inputs
------
Folder structure:
covariance_outputs/
  ├─ raw/
  │   ├─ 201501_full_cov.csv
  │   ├─ 201502_full_cov.csv
  │   └─ ...
  ├─ LW/
  ├─ PCA/
  └─ JSE/
Each CSV: first column = permno (length p), remaining columns = p×p covariance (row-major).
Row/列顺序一致（上游已对齐并清洗缺失），我们将保持该顺序输出权重。

Outputs
-------
portfolio_outputs/
  ├─ raw/
  │   ├─ PortfolioA_GMV/
  │   │   ├─ 201501_weights.csv
  │   │   └─ ...
  │   └─ PortfolioB_TVMV/
  │       ├─ 201501_weights.csv
  │       └─ ...
  ├─ LW/
  ├─ PCA/
  └─ JSE/
Each weights.csv has two columns: permno, weight

Author: Darwin Yao
"""

import os
import sys
import math
import glob
import warnings
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd

# ==== Optional solvers ====
_HAS_CVXPY = False
try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False
    warnings.warn("cvxpy not available; will fallback to scipy.optimize for constrained QP.")
try:
    from scipy.optimize import minimize, LinearConstraint, Bounds
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    warnings.warn("scipy not available; constrained QP may fail if cvxpy is also missing.")


# ---------------- Config knobs (adjust here) ----------------
# INPUT_ROOT = "c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/covariance_outputs_sim"
# OUTPUT_ROOT = "c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/portfolio_outputs_sim"
INPUT_ROOT = "covariance_outputs_sim"
OUTPUT_ROOT = "portfolio_outputs_sim"

# Common knobs
RIDGE_EPS = 0          # ε for Σ + εI to stabilize inverses
L2_LAMBDA = 0          # λ for λ||w||_2^2
# LONG_ONLY = True          # default: long-only
LONG_ONLY = False          # default: long-only
WEIGHT_CAP = None         # u upper bound for long-only; set to None to remove cap
# ALLOW_SHORT_IF_SET = False  # if True and no bounds => allow shorting (uses closed form for GMV)
ALLOW_SHORT_IF_SET = True  # if True and no bounds => allow shorting (uses closed form for GMV)
LEVERAGE_L1_BUDGET = None   # e.g., 1.5 for ||w||_1 ≤ L; only with cvxpy in this template

# Portfolio B (Target-Volatility MV with zero-mean prior)
TARGET_VOL_ANNUAL = 0.10  # σ*  e.g., 10% annualized
TRADING_DAYS = 252        # to convert monthly Σ (already monthly?) => we treat Σ as covariance at analysis horizon.
# If Σ is monthly covariance, TARGET_VOL should be monthly. If Σ is daily covariance, adjust accordingly.
# Here we assume Σ is at the analysis horizon used for optimization (consistent across dates).
# If your Σ is monthly, TARGET_VOL_EFFECTIVE should be monthly too:
TARGET_VOL_EFFECTIVE = TARGET_VOL_ANNUAL  # adjust if needed

# γ search
GAMMA_MIN = 1e-6
GAMMA_MAX = 1e6
GAMMA_TOL = 1e-4
VOL_TOL = 1e-5
MAX_BISECT_ITERS = 50

# File pattern
METHOD_FOLDERS = ["PCA", "JSE"]  # LW disabled - may have issues
FILE_SUFFIX = "_full_cov.csv"

# Maximum number of files to process (set to None to process all files)
MAX_FILES = None  # Process only first 30 files for faster testing; set to None for all files

# -----------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_cov_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSV that has first column = permno, remaining p columns = p x p covariance.

    We try to robustly parse regardless of header.
    """
    df = pd.read_csv(path, header=None)
    # Heuristic: first column must be integers-like permno (or strings). We won't enforce dtype.
    permno_col = df.iloc[:, 0].to_numpy()
    # Remaining should be p columns that together form a p x p matrix:
    rest = df.iloc[:, 1:].to_numpy()
    p = len(permno_col)

    if rest.shape != (p, p):
        # Try with header=0 if first row was header
        df2 = pd.read_csv(path)  # header=0
        permno_col = df2.iloc[:, 0].to_numpy()
        rest = df2.iloc[:, 1:].to_numpy()
        p = len(permno_col)
        if rest.shape != (p, p):
            raise ValueError(f"File {path}: cannot reshape remaining columns into ({p},{p}). Got {rest.shape}")

    Sigma = rest.astype(float, copy=False)
    return permno_col, Sigma


def stabilize_sigma(Sigma: np.ndarray, eps: float = RIDGE_EPS) -> np.ndarray:
    p = Sigma.shape[0]
    # Symmetrize just in case of small numeric asymmetry:
    Sigma = 0.5 * (Sigma + Sigma.T)
    # Add ridge
    return Sigma + eps * np.eye(p)


def gmv_unconstrained(Sigma: np.ndarray) -> np.ndarray:
    """
    GMV closed form: w ~ Σ^{-1} 1, normalized to 1' w = 1
    """
    p = Sigma.shape[0]
    ones = np.ones(p)
    try:
        inv_S = np.linalg.solve(Sigma, np.eye(p))
    except np.linalg.LinAlgError:
        inv_S = np.linalg.pinv(Sigma)
    w = inv_S @ ones
    denom = ones @ w
    if abs(denom) < 1e-12:
        raise ValueError("GMV closed form: denominator nearly zero.")
    return w / denom


def _solve_qp_cvxpy_gmv(Sigma: np.ndarray,
                        long_only: bool = True,
                        weight_cap: Optional[float] = WEIGHT_CAP,
                        l2_lambda: float = L2_LAMBDA,
                        l1_budget: Optional[float] = LEVERAGE_L1_BUDGET) -> np.ndarray:
    p = Sigma.shape[0]
    w = cp.Variable(p)

    obj = cp.quad_form(w, Sigma) + l2_lambda * cp.sum_squares(w)
    cons = [cp.sum(w) == 1]

    if l1_budget is not None:
        cons.append(cp.norm1(w) <= l1_budget)

    if long_only:
        cons.append(w >= 0)
        if weight_cap is not None:
            cons.append(w <= weight_cap)
    else:
        # Optional symmetric box if you want: uncomment to enforce -cap <= w <= cap
        # if weight_cap is not None:
        #     cons.append(w <= weight_cap)
        #     cons.append(w >= -weight_cap)
        pass

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-7, eps_rel=1e-7)  # OSQP often robust for box QPs
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        # Try a different solver as fallback
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-8, reltol=1e-8, feastol=1e-8)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed to solve GMV QP: status={prob.status}")
    return np.array(w.value).reshape(-1)


def _solve_qp_scipy_gmv(Sigma: np.ndarray,
                        long_only: bool = True,
                        weight_cap: Optional[float] = WEIGHT_CAP,
                        l2_lambda: float = L2_LAMBDA) -> np.ndarray:
    p = Sigma.shape[0]
    ones = np.ones(p)

    def obj(w):
        return float(w @ Sigma @ w + l2_lambda * (w @ w))

    def grad(w):
        return 2 * (Sigma @ w) + 2 * l2_lambda * w

    # equality constraint: 1' w = 1
    lc = LinearConstraint(ones, lb=1.0, ub=1.0)

    if long_only:
        lb = np.zeros(p)
        ub = np.full(p, +np.inf if weight_cap is None else float(weight_cap))
    else:
        # unconstrained box (very loose)
        lb = np.full(p, -np.inf)
        ub = np.full(p, +np.inf)
        # If you want symmetric caps when shorting, set them here.

    bounds = Bounds(lb, ub)

    x0 = np.full(p, 1.0 / p)  # feasible start for long-only
    res = minimize(obj, x0, jac=grad, method="trust-constr",
                   constraints=[lc],
                   bounds=bounds,
                   options={"maxiter": 10_000, "gtol": 1e-10, "xtol": 1e-12, "verbose": 0})
    if not res.success:
        raise RuntimeError(f"Scipy trust-constr failed: {res.message}")
    return res.x


def solve_gmv(Sigma: np.ndarray,
              long_only: bool = LONG_ONLY,
              weight_cap: Optional[float] = WEIGHT_CAP,
              l2_lambda: float = L2_LAMBDA,
              allow_short_if_set: bool = ALLOW_SHORT_IF_SET) -> np.ndarray:
    """
    Portfolio A: GMV
    - if allow_short and no (useful) bounds -> closed form
    - else -> constrained QP
    """
    p = Sigma.shape[0]
    Sigma = stabilize_sigma(Sigma, RIDGE_EPS)

    if (not long_only) and (weight_cap is None) and allow_short_if_set:
        return gmv_unconstrained(Sigma)

    if _HAS_CVXPY:
        return _solve_qp_cvxpy_gmv(Sigma, long_only=long_only, weight_cap=weight_cap, l2_lambda=l2_lambda)
    elif _HAS_SCIPY:
        return _solve_qp_scipy_gmv(Sigma, long_only=long_only, weight_cap=weight_cap, l2_lambda=l2_lambda)
    else:
        raise RuntimeError("No QP solver available (cvxpy/scipy not found).")


def _solve_qp_cvxpy_gamma(Sigma: np.ndarray,
                          gamma: float,
                          long_only: bool = True,
                          weight_cap: Optional[float] = WEIGHT_CAP,
                          l2_lambda: float = L2_LAMBDA,
                          l1_budget: Optional[float] = LEVERAGE_L1_BUDGET) -> np.ndarray:
    """
    Solve: min (γ/2) w' Σ w + λ‖w‖²  s.t. 1'w=1, box/l1 constraints.
    """
    p = Sigma.shape[0]
    w = cp.Variable(p)

    obj = (gamma / 2.0) * cp.quad_form(w, Sigma) + l2_lambda * cp.sum_squares(w)
    cons = [cp.sum(w) == 1]

    if l1_budget is not None:
        cons.append(cp.norm1(w) <= l1_budget)

    if long_only:
        cons.append(w >= 0)
        if weight_cap is not None:
            cons.append(w <= weight_cap)

    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1e-7, eps_rel=1e-7)
    if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        prob.solve(solver=cp.ECOS, verbose=False, abstol=1e-8, reltol=1e-8, feastol=1e-8)
        if w.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"CVXPY failed for gamma={gamma}: status={prob.status}")
    return np.array(w.value).reshape(-1)


def _solve_qp_scipy_gamma(Sigma: np.ndarray,
                          gamma: float,
                          long_only: bool = True,
                          weight_cap: Optional[float] = WEIGHT_CAP,
                          l2_lambda: float = L2_LAMBDA) -> np.ndarray:
    p = Sigma.shape[0]
    ones = np.ones(p)

    def obj(w):
        return float(0.5 * gamma * (w @ Sigma @ w) + l2_lambda * (w @ w))

    def grad(w):
        return gamma * (Sigma @ w) + 2 * l2_lambda * w

    lc = LinearConstraint(ones, lb=1.0, ub=1.0)

    if long_only:
        lb = np.zeros(p)
        ub = np.full(p, +np.inf if weight_cap is None else float(weight_cap))
    else:
        lb = np.full(p, -np.inf)
        ub = np.full(p, +np.inf)

    bounds = Bounds(lb, ub)
    x0 = np.full(p, 1.0 / p)

    res = minimize(obj, x0, jac=grad, method="trust-constr",
                   constraints=[lc],
                   bounds=bounds,
                   options={"maxiter": 10_000, "gtol": 1e-10, "xtol": 1e-12, "verbose": 0})
    if not res.success:
        raise RuntimeError(f"Solve gamma={gamma} failed: {res.message}")
    return res.x


def solve_target_vol(Sigma: np.ndarray,
                     target_vol: float = TARGET_VOL_EFFECTIVE,
                     long_only: bool = LONG_ONLY,
                     weight_cap: Optional[float] = WEIGHT_CAP,
                     l2_lambda: float = L2_LAMBDA,
                     gamma_bounds: Tuple[float, float] = (GAMMA_MIN, GAMMA_MAX),
                     max_iters: int = MAX_BISECT_ITERS,
                     vol_tol: float = VOL_TOL) -> Tuple[np.ndarray, float, float]:
    """
    Portfolio B: Target-Volatility mean-variance with zero-mean prior
    min (γ/2) w' Σ w + λ||w||²  s.t. 1'w=1, bounds
    Find γ by bisection so that sqrt(w' Σ w) ≈ target_vol.

    Returns: (weights, achieved_vol, gamma)
    """
    Sigma = stabilize_sigma(Sigma, RIDGE_EPS)
    lo, hi = gamma_bounds

    def solve_given_gamma(gamma):
        if _HAS_CVXPY:
            w = _solve_qp_cvxpy_gamma(Sigma, gamma, long_only, weight_cap, l2_lambda)
        elif _HAS_SCIPY:
            w = _solve_qp_scipy_gamma(Sigma, gamma, long_only, weight_cap, l2_lambda)
        else:
            raise RuntimeError("No solver available for target-vol optimization.")
        var = float(w @ Sigma @ w)
        vol = math.sqrt(max(var, 0.0))
        return w, vol

    # First evaluate ends
    w_lo, vol_lo = solve_given_gamma(lo)
    w_hi, vol_hi = solve_given_gamma(hi)

    # γ increases => stronger risk penalty => *lower* vol typically
    # Ensure monotonic bracket; if not, still bisect safely using vol comparison
    if not (vol_hi <= vol_lo):
        # swap to enforce monotonicity assumption
        lo, hi = hi, lo
        w_lo, vol_lo, w_hi, vol_hi = w_hi, vol_hi, w_lo, vol_lo

    # If already within tolerance at ends, return
    if abs(vol_lo - target_vol) <= vol_tol:
        return w_lo, vol_lo, lo
    if abs(vol_hi - target_vol) <= vol_tol:
        return w_hi, vol_hi, hi

    w_best, vol_best, g_best = (w_lo, vol_lo, lo) if abs(vol_lo - target_vol) < abs(vol_hi - target_vol) else (w_hi, vol_hi, hi)

    for _ in range(max_iters):
        mid = math.sqrt(lo * hi)  # geometric mean for wide γ range
        w_mid, vol_mid = solve_given_gamma(mid)
        # Track best
        if abs(vol_mid - target_vol) < abs(vol_best - target_vol):
            w_best, vol_best, g_best = w_mid, vol_mid, mid

        if abs(vol_mid - target_vol) <= vol_tol:
            return w_mid, vol_mid, mid

        # Decide which side to keep: recall higher γ => lower vol
        if vol_mid > target_vol:
            # need more penalty => increase γ
            lo = mid
        else:
            hi = mid

        if (hi / lo) < (1.0 + GAMMA_TOL):
            break

    return w_best, vol_best, g_best


def save_weights(permnos: np.ndarray, w: np.ndarray, out_path: str):
    df = pd.DataFrame({"permno": permnos, "weight": w})
    df.to_csv(out_path, index=False)


def process_method_folder(method_name: str,
                          in_root: str = INPUT_ROOT,
                          out_root: str = OUTPUT_ROOT):
    method_in = os.path.join(in_root, method_name)
    files = sorted(glob.glob(os.path.join(method_in, f"*{FILE_SUFFIX}")))
    if not files:
        print(f"[{method_name}] No files matched in {method_in}")
        return
    
    # Limit to first MAX_FILES if specified
    total_files = len(files)
    if MAX_FILES is not None and MAX_FILES > 0:
        files = files[:MAX_FILES]
        print(f"[{method_name}] Processing {len(files)} of {total_files} files (limited by MAX_FILES={MAX_FILES})")
    else:
        print(f"[{method_name}] Processing all {total_files} files")

    out_gmv = os.path.join(out_root, method_name, "PortfolioA_GMV")
    out_tvmv = os.path.join(out_root, method_name, "PortfolioB_TVMV")
    ensure_dir(out_gmv)
    ensure_dir(out_tvmv)

    for fp in files:
        try:
            fname = os.path.basename(fp)
            yyyymm = fname.replace(FILE_SUFFIX, "")
            permno, Sigma = load_cov_csv(fp)

            # A) GMV
            w_gmv = solve_gmv(Sigma,
                              long_only=LONG_ONLY,
                              weight_cap=WEIGHT_CAP,
                              l2_lambda=L2_LAMBDA,
                              allow_short_if_set=ALLOW_SHORT_IF_SET)
            save_weights(permno, w_gmv, os.path.join(out_gmv, f"{yyyymm}_weights.csv"))

            # B) Target-Vol MV (zero-mean prior)
            '''w_tvmv, vol_tvmv, gamma_used = solve_target_vol(
                Sigma,
                target_vol=TARGET_VOL_EFFECTIVE,
                long_only=LONG_ONLY,
                weight_cap=WEIGHT_CAP,
                l2_lambda=L2_LAMBDA
            )
            save_weights(permno, w_tvmv, os.path.join(out_tvmv, f"{yyyymm}_weights.csv"))'''

            print(f"[{method_name}] {yyyymm}: GMV done")# | TVMV done (vol≈{vol_tvmv:.6f}, γ≈{gamma_used:.4g})")

        except Exception as e:
            print(f"[{method_name}] ERROR on file {fp}: {e}", file=sys.stderr)


def main():
    for m in METHOD_FOLDERS:
        process_method_folder(m)


if __name__ == "__main__":
    main()
