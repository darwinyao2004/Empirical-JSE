"""
Portfolio out-of-sample (next-month) performance backtest

What this script does
---------------------
1) Loads monthly returns from 'monthly_ret.csv'
   - First column: 'permno'
   - Subsequent columns: dates with header format 'YYYY/MM/DD'
   - Values are monthly returns (e.g., in decimal form like 0.0123 for +1.23%)

2) Iterates through the directory tree:
   portfolio_outputs/
       raw|LW|PCA|JSE/
           PortfolioA_GMV|PortfolioB_TVMV/
               yyyymm_weights.csv  (columns: permno, weight)

3) For each weights file (constructed at YYYYMM), the script finds the *next month*
   in monthly_ret and computes the realized portfolio return as the weighted sum
   of stock returns for that next month. If some stocks in the weights file do not have
   returns in the next month, those rows are dropped and the remaining weights are
   renormalized to sum to 1 before computing the realized return.

4) Outputs:
   - results/{method}/{portfolio_type}_returns.csv :
       two columns: date (YYYY-MM-DD) and realized_return
   - results/summary.csv :
       method, portfolio_type, n_points, mean_return, var_return

Assumptions / Notes
-------------------
- 'monthly_ret.csv' contains exactly one observation per month per stock (date columns).
- The next-month lookup is done by matching on (year, month) of the *date column names*.
- If multiple columns exist for the same (year, month), the last one (chronologically)
  will be used; if none exists, that backtest point is skipped.
- We treat missing return values as genuinely missing. We drop those rows and renormalize
  weights among the remaining stocks. If no returns survive, that backtest point is skipped.

Directory Layout Example
------------------------
portfolio_outputs/
    raw/
        PortfolioA_GMV/
            201801_weights.csv
            201802_weights.csv
            ...
        PortfolioB_TVMV/
            ...
    LW/
        ...
    PCA/
        ...
    JSE/
        ...

How to run
----------
Ensure this script sits alongside:
- monthly_ret.csv
- the portfolio_outputs/ directory

Then run:  python backtest_next_month_returns.py
Author: Darwin Yao
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Configuration
# -----------------------------
# PORTFOLIO_ROOT = Path("c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/portfolio_outputs_sim")
# MONTHLY_RET_CSV = Path("c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/500_ret_sim/monthly_ret.csv")
PORTFOLIO_ROOT = Path("portfolio_outputs_sim")
MONTHLY_RET_CSV = Path("500_ret_sim/monthly_ret.csv")
METHODS = ["LW", "PCA", "JSE"]
PORTFOLIO_TYPES = ["PortfolioA_GMV"]
# RESULTS_ROOT = Path("c:/Users/remote/Desktop/temp/tmp1030/code by Darwin/results_sim")
RESULTS_ROOT = Path("results_sim")


# -----------------------------
# Utilities
# -----------------------------
@dataclass
class BacktestPoint:
    date: pd.Timestamp    # the *next month's* date (from monthly_ret columns)
    realized_return: float


def parse_year_month_from_filename(filename: str) -> Tuple[int, int]:
    """
    Given a filename like '201902_weights.csv', return (2019, 2).
    """
    stem = Path(filename).stem
    # Expect pattern yyyymm_weights
    # Split at underscore, take first token
    token = stem.split("_")[0]
    if len(token) != 6 or not token.isdigit():
        raise ValueError(f"Cannot parse YYYYMM from filename: {filename}")
    year = int(token[:4])
    month = int(token[4:])
    if not (1 <= month <= 12):
        raise ValueError(f"Invalid month parsed from filename: {filename}")
    return year, month


def next_year_month(year: int, month: int) -> Tuple[int, int]:
    """
    Return (year, month) for the next calendar month.
    """
    if month == 12:
        return year + 1, 1
    return year, month + 1


def build_month_index(columns: List[str]) -> Dict[Tuple[int, int], List[pd.Timestamp]]:
    """
    From date-like column names such as 'YYYY/MM/DD', build a mapping:
        (year, month) -> [sorted list of pd.Timestamp within that month]
    We will later pick the last available date in a given (year, month).
    """
    out: Dict[Tuple[int, int], List[pd.Timestamp]] = {}
    for c in columns:
        # Robust parse: allow either 'YYYY/MM/DD' or 'YYYY-MM-DD'
        try:
            # Try with '/' first
            dt = pd.to_datetime(c, format="%Y/%m/%d", errors="raise")
        except Exception:
            try:
                dt = pd.to_datetime(c, format="%Y-%m-%d", errors="raise")
            except Exception:
                # Skip non-date columns just in case; main code already filters 'permno'
                continue
        key = (dt.year, dt.month)
        out.setdefault(key, []).append(dt)
    # sort within each (year,month)
    for k in out:
        out[k].sort()
    return out


def pick_month_column(
    month_index: Dict[Tuple[int, int], List[pd.Timestamp]],
    year: int,
    month: int,
) -> Optional[pd.Timestamp]:
    """
    Pick the last available date in the target (year, month).
    Returns None if no such month is present.
    """
    days = month_index.get((year, month))
    if not days:
        return None
    return days[-1]  # last (latest) date of that month present in the data


def safe_read_weights_csv(path: Path) -> pd.DataFrame:
    """
    Read a weights CSV with columns: permno, weight.
    Coerce column names to lowercase for robustness.
    """
    df = pd.read_csv(path)
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    # Standardize expected columns
    if "permno" not in df.columns or "weight" not in df.columns:
        raise ValueError(f"Expected columns 'permno' and 'weight' in {path}")
    return df[["permno", "weight"]].copy()


def compute_realized_return_for_next_month(
    weights_df: pd.DataFrame,
    monthly_ret_df: pd.DataFrame,
    next_month_col: str,
) -> Optional[float]:
    """
    Merge weights with the next-month return column, drop missing, re-normalize weights,
    and return the realized portfolio return. Returns None if nothing to compute.
    """
    # Join on permno
    merged = weights_df.merge(
        monthly_ret_df[["permno", next_month_col]], on="permno", how="inner"
    )
    # Drop missing returns
    merged = merged.dropna(subset=[next_month_col])
    if merged.empty:
        return None

    # Renormalize weights to sum to 1 over the surviving names
    w = merged["weight"].astype(float).values
    r = merged[next_month_col].astype(float).values

    wsum = np.sum(w)
    if wsum == 0.0:
        # If all weights zero, nothing to compute
        return None
    w = w / wsum

    # Realized return is weighted sum
    realized = float(np.dot(w, r))
    return realized


# -----------------------------
# Main
# -----------------------------
def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Load returns
    ret = pd.read_csv(MONTHLY_RET_CSV)
    if "permno" not in ret.columns:
        raise ValueError("monthly_ret.csv must have 'permno' as the first column.")
    # Build list of date columns and a (year,month)->dates mapping
    date_cols = [c for c in ret.columns if c != "permno"]
    if not date_cols:
        raise ValueError("monthly_ret.csv has no date columns beyond 'permno'.")

    # Build a mapping of (year, month) -> list of timestamps in that month
    month_index = build_month_index(date_cols)

    # We will also create a mapping from Timestamp->column name string for final selection
    colname_by_ts: Dict[pd.Timestamp, str] = {}
    for c in date_cols:
        # need to figure out the parse that succeeded in build_month_index
        # We'll try both formats; if both fail, skip (shouldn't happen for valid columns)
        dt: Optional[pd.Timestamp] = None
        for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
            try:
                dt = pd.to_datetime(c, format=fmt, errors="raise")
                break
            except Exception:
                continue
        if dt is not None:
            colname_by_ts[dt] = c

    # Storage for a combined summary
    summary_rows = []

    for method in METHODS:
        method_dir = PORTFOLIO_ROOT / method
        if not method_dir.is_dir():
            print(f"[WARN] Method directory not found: {method_dir} (skipping)")
            continue

        for ptype in PORTFOLIO_TYPES:
            ptype_dir = method_dir / ptype
            if not ptype_dir.is_dir():
                print(f"[WARN] Portfolio type directory not found: {ptype_dir} (skipping)")
                continue

            # Gather all *_weights.csv files and sort by YYYYMM
            weight_files = sorted(
                [f for f in ptype_dir.glob("*_weights.csv") if f.is_file()],
                key=lambda p: p.name
            )

            backtest_points: List[BacktestPoint] = []

            for wf in weight_files:
                try:
                    y, m = parse_year_month_from_filename(wf.name)
                except Exception as e:
                    print(f"[WARN] {e} (skipping file {wf})")
                    continue

                ny, nm = next_year_month(y, m)
                # Pick the last available date within that next month
                ts = pick_month_column(month_index, ny, nm)
                if ts is None:
                    # No returns for that next month; skip
                    print(f"[INFO] No return column found for {ny}-{nm:02d} (next of {y}-{m:02d}); skipping {wf.name}")
                    continue

                next_col = colname_by_ts[ts]

                # Load weights, compute realized return
                try:
                    wdf = safe_read_weights_csv(wf)
                except Exception as e:
                    print(f"[WARN] Failed to read weights from {wf}: {e}")
                    continue

                realized = compute_realized_return_for_next_month(wdf, ret, next_col)
                if realized is None:
                    print(f"[INFO] No realized return computed for {wf.name} (likely empty after merge)")
                    continue

                backtest_points.append(BacktestPoint(date=pd.Timestamp(ts), realized_return=realized))

            # Build a DataFrame for this (method, ptype)
            if not backtest_points:
                print(f"[INFO] No backtest points for {method}/{ptype}")
                continue

            pts_df = pd.DataFrame(
                {
                    "date": [bp.date.normalize() for bp in backtest_points],
                    "realized_return": [bp.realized_return for bp in backtest_points],
                }
            ).sort_values("date").reset_index(drop=True)

            # Compute summary statistics
            mu = float(pts_df["realized_return"].mean())
            var = float(pts_df["realized_return"].var(ddof=1)) if len(pts_df) > 1 else float("nan")

            summary_rows.append(
                {
                    "method": method,
                    "portfolio_type": ptype,
                    "n_points": len(pts_df),
                    "mean_return": mu,
                    "var_return": var,
                }
            )

            # Save per-portfolio return curve
            out_dir = RESULTS_ROOT / method
            out_dir.mkdir(parents=True, exist_ok=True)
            out_curve = out_dir / f"{ptype}_returns.csv"
            pts_df.to_csv(out_curve, index=False)
            print(f"[OK] Saved return curve: {out_curve}  (N={len(pts_df)}, mean={mu:.6f}, var={var:.6f})")

    # Save combined summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = RESULTS_ROOT / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[OK] Wrote summary to {summary_path}")
        print(summary_df.to_string(index=False))
    else:
        print("[WARN] No results produced; check your inputs and directory structure.")


if __name__ == "__main__":
    main()
