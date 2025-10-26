"""
Visualization with combined GMV plots.

Adds:
- results/figures/GMV_monthly_returns_all_methods.png
- results/figures/GMV_cumulative_returns_all_methods.png

Author: Darwin Yao
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
RESULTS_ROOT = Path("results")
FIG_ROOT = RESULTS_ROOT / "figures"
METHODS = ["raw", "LW", "PCA", "JSE"]
PORTFOLIO_TYPES = ["PortfolioA_GMV", "PortfolioB_TVMV"]

# Keep the earlier toggle
PLOT_DRAWDOWNS = True

# -----------------------------
# Helpers
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def wealth_index(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0).astype(float)
    return (1.0 + r).cumprod() * start_value

def compute_drawdown(wealth: pd.Series) -> pd.Series:
    return wealth / wealth.cummax() - 1.0

def load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    df = pd.read_csv(path)
    needed = {"method", "portfolio_type", "n_points", "mean_return", "var_return"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in summary: {missing}")
    return df

def load_curve(method: str, ptype: str) -> pd.DataFrame | None:
    curve_path = RESULTS_ROOT / method / f"{ptype}_returns.csv"
    if not curve_path.exists():
        print(f"[WARN] Missing curve file: {curve_path}")
        return None
    df = pd.read_csv(curve_path, parse_dates=["date"])
    if "date" not in df.columns or "realized_return" not in df.columns:
        print(f"[WARN] Unexpected columns in {curve_path}; expected ['date','realized_return']")
        return None
    return df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Existing plots (summary + per-portfolio)
# -----------------------------
def plot_summary_bar_mean_with_std(summary: pd.DataFrame, out_path: Path) -> None:
    df = summary.copy()
    df["std_return"] = np.sqrt(df["var_return"].clip(lower=0.0)).fillna(0.0)
    df["label"] = df["method"] + " – " + df["portfolio_type"].str.replace("Portfolio", "", regex=False)
    df = df.sort_values("mean_return", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["label"], df["mean_return"], yerr=df["std_return"], capsize=4)
    ax.set_ylabel("Mean monthly return")
    ax.set_title("Mean monthly return (error bars = std. dev.)")
    ax.set_xticklabels(df["label"], rotation=30, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_mean_variance_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    markers = {"PortfolioA_GMV": "o", "PortfolioB_TVMV": "s"}
    for ptype in PORTFOLIO_TYPES:
        sub = summary[summary["portfolio_type"] == ptype]
        if sub.empty:
            continue
        ax.scatter(sub["var_return"], sub["mean_return"], marker=markers.get(ptype, "o"), label=ptype)
        for _, row in sub.iterrows():
            ax.annotate(row["method"], (row["var_return"], row["mean_return"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Variance of monthly return")
    ax.set_ylabel("Mean monthly return")
    ax.set_title("Mean–Variance Overview")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_per_portfolio_time_series(curve: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["date"], curve["realized_return"])
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly realized return")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_per_portfolio_cumulative(curve: pd.DataFrame, title: str, out_path: Path) -> None:
    w = wealth_index(curve["realized_return"], start_value=1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["date"], w)
    ax.set_title(title + " — Cumulative return (wealth index)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index (start = 1.0)")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_per_portfolio_drawdown(curve: pd.DataFrame, title: str, out_path: Path) -> None:
    w = wealth_index(curve["realized_return"], start_value=1.0)
    dd = compute_drawdown(w)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(curve["date"], dd)
    ax.set_title(title + " — Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.axhline(0.0, linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# NEW: Combined GMV plots
# -----------------------------
def plot_combined_gmv_monthly(curves: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Plot the 4 GMV monthly-return lines (raw/LW/PCA/JSE) on one figure.
    'curves' maps method -> DataFrame with columns ['date','realized_return'].
    Handles different date ranges gracefully (lines may start/end at different times).
    """
    if not curves:
        print("[WARN] No GMV curves found for combined monthly plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for method, df in curves.items():
        if df is None or df.empty:
            continue
        ax.plot(df["date"], df["realized_return"], label=method)

    ax.set_title("GMV — Monthly realized returns (all methods)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Monthly realized return")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_combined_gmv_cumulative(curves: Dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Plot the 4 GMV cumulative wealth-index lines on one figure.
    Each method’s wealth index is computed independently from its own monthly returns.
    """
    if not curves:
        print("[WARN] No GMV curves found for combined cumulative plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for method, df in curves.items():
        if df is None or df.empty:
            continue
        w = wealth_index(df["realized_return"], start_value=1.0)
        ax.plot(df["date"], w, label=method)

    ax.set_title("GMV — Cumulative returns (wealth index), all methods")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth index (start = 1.0)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    ensure_dir(FIG_ROOT)

    # 1) Summary charts
    summary = load_summary(RESULTS_ROOT / "summary.csv")
    plot_summary_bar_mean_with_std(summary, FIG_ROOT / "summary_bar_mean_with_std.png")
    plot_mean_variance_scatter(summary, FIG_ROOT / "mean_vs_variance_scatter.png")

    # 2) Per-portfolio charts
    for method in METHODS:
        for ptype in PORTFOLIO_TYPES:
            curve = load_curve(method, ptype)
            if curve is None or curve.empty:
                continue
            base_title = f"{method} – {ptype}"
            plot_per_portfolio_time_series(curve, base_title, FIG_ROOT / f"{method}_{ptype}_monthly_returns.png")
            plot_per_portfolio_cumulative(curve, base_title, FIG_ROOT / f"{method}_{ptype}_cumulative_returns.png")
            if PLOT_DRAWDOWNS:
                plot_per_portfolio_drawdown(curve, base_title, FIG_ROOT / f"{method}_{ptype}_drawdowns.png")

    # 3) NEW: Combined GMV plots
    gmv_curves: Dict[str, pd.DataFrame] = {}
    for method in METHODS:
        df = load_curve(method, "PortfolioA_GMV")
        if df is not None and not df.empty:
            gmv_curves[method] = df

    plot_combined_gmv_monthly(
        gmv_curves,
        FIG_ROOT / "GMV_monthly_returns_all_methods.png"
    )

    plot_combined_gmv_cumulative(
        gmv_curves,
        FIG_ROOT / "GMV_cumulative_returns_all_methods.png"
    )

    print(f"[OK] Figures written to: {FIG_ROOT.resolve()}")

if __name__ == "__main__":
    main()
