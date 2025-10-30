# single_factor_sim.py
"""
Generates:
- 120 CSVs under 500_ret_sim: yyyymm_full.csv, each with 500 rows x 64 columns
  (first column = integer id 1..500, next 63 columns = daily returns for the last 3 months)
- monthly_ret.csv: first column = id; first row = month-end dates (yyyy/mm/dd);
  each cell is the geometric product of (1 + daily return) over the last 21 days
  of that month's file.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from calendar import monthrange

# ----------------------------
# Configuration
# ----------------------------
np.random.seed(42)

N_ASSETS = 500
DAYS_PER_MONTH = 21               # approximate trading days per month
WINDOW_DAYS = DAYS_PER_MONTH * 3  # 3 months = 63 days

# Single-factor model parameters
FACTOR_MEAN = 0.0003          # daily factor drift
FACTOR_STD  = 0.01            # daily factor volatility
ALPHA_MEAN  = 0.00005         # cross-sectional alpha mean (daily)
ALPHA_STD   = 0.0002          # cross-sectional alpha std  (daily)
BETA_MEAN   = 1.0             # cross-sectional beta mean
BETA_STD    = 0.3             # cross-sectional beta std
IDIO_STD_MEAN = 0.012         # idiosyncratic daily vol mean
IDIO_STD_STD  = 0.004         # idiosyncratic daily vol dispersion

OUT_DIR = "500_ret_sim"
os.makedirs(OUT_DIR, exist_ok=True)

asset_ids = np.arange(1, N_ASSETS + 1, dtype=int)

# Fixed cross-sectional parameters for the entire experiment
alphas = np.random.normal(ALPHA_MEAN, ALPHA_STD, size=N_ASSETS)
betas  = np.random.normal(BETA_MEAN, BETA_STD, size=N_ASSETS)
idio_stds = np.clip(np.random.normal(IDIO_STD_MEAN, IDIO_STD_STD, size=N_ASSETS), 0.003, None)

def month_iter(start_ym="201501", end_ym="202412"):
    ys, ms = int(start_ym[:4]), int(start_ym[4:])
    ye, me = int(end_ym[:4]), int(end_ym[4:])
    y, m = ys, ms
    while (y < ye) or (y == ye and m <= me):
        yield y, m
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1

monthly_dates = []
monthly_rets  = []

for (year, month) in month_iter("201501", "202412"):
    yyyymm = f"{year}{month:02d}"

    # Simulate the common factor for 63 days (iid here for simplicity)
    F = np.random.normal(FACTOR_MEAN, FACTOR_STD, size=WINDOW_DAYS)

    # Idiosyncratic shocks (N_ASSETS x 63)
    eps = np.random.normal(0.0, idio_stds[:, None], size=(N_ASSETS, WINDOW_DAYS))

    # Single-factor model: r_it = alpha_i + beta_i * F_t + eps_it
    R = alphas[:, None] + betas[:, None] * F[None, :] + eps  # shape (500, 63)

    # Write yyyymm_full.csv: first col = id, then 63 daily returns (no header)
    df = pd.DataFrame(np.column_stack([asset_ids, R]))
    df.to_csv(os.path.join(OUT_DIR, f"{yyyymm}_full.csv"), header=False, index=False, float_format="%.10f")

    # Monthly gross return = product of (1 + daily return) over the last 21 columns
    last21 = R[:, -DAYS_PER_MONTH:]
    gross = np.prod(1.0 + last21, axis=1)  # store the product (not minus 1), as requested
    monthly_rets.append(gross - 1)

    # Month-end date yyyy/mm/dd
    last_day = monthrange(year, month)[1]
    monthly_dates.append(f"{year}/{month}/{last_day}")

# monthly_ret.csv: columns = ['id'] + monthly_dates; first row carries the dates as header
monthly_mat = np.column_stack([asset_ids] + [np.array(col) for col in monthly_rets])
monthly_df = pd.DataFrame(monthly_mat, columns=["id"] + monthly_dates)
monthly_df.to_csv(os.path.join(OUT_DIR, "monthly_ret.csv"), index=False, float_format="%.10f")
