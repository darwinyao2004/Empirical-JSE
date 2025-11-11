# single_factor_sim.py
"""
Single-factor model simulation (without alpha, for covariance research).

Model: r_it = beta_i * F_t + eps_it
where:
  - F_t ~ N(FACTOR_MEAN, FACTOR_STD): common factor
  - beta_i ~ N(BETA_MEAN, BETA_STD): factor loading (cross-sectional)
  - eps_it ~ N(0, sigma_i): idiosyncratic shock
  
Risk types (configurable via HOMOSKEDASTIC flag):
  - Homoskedastic: sigma_i = constant for all i (equal idiosyncratic risks)
  - Heteroskedastic: sigma_i ~ N(IDIO_STD_MEAN, IDIO_STD_STD) (different risks)

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

# Risk type switch
HOMOSKEDASTIC = True  # Set to True for homoskedastic (equal) idiosyncratic risks
                       # Set to False for heteroskedastic (different) idiosyncratic risks

# Single-factor model parameters
# FACTOR_MEAN = 0.0003          # daily factor drift
FACTOR_MEAN = 0.0          # daily factor drift
# FACTOR_STD  = 0.01            # daily factor volatility
FACTOR_STD  = 0.16/np.sqrt(252)            # daily factor volatility
# ALPHA_MEAN  = 0.00005         # cross-sectional alpha mean (daily) - COMMENTED OUT: not used for covariance research
# ALPHA_STD   = 0.0002          # cross-sectional alpha std  (daily) - COMMENTED OUT: not used for covariance research
BETA_MEAN   = 1.0             # cross-sectional beta mean
# BETA_STD    = 0.3             # cross-sectional beta std
BETA_STD    = 0.5             # cross-sectional beta std
# IDIO_STD_MEAN = 0.012         # idiosyncratic daily vol mean (used for both homo- and heteroskedastic)
IDIO_STD_MEAN = 0.6/np.sqrt(252)         # idiosyncratic daily vol mean (used for both homo- and heteroskedastic)
# IDIO_STD_STD  = 0.004         # idiosyncratic daily vol dispersion (only used for heteroskedastic)
IDIO_STD_STD  = 0.1/np.sqrt(252)         # idiosyncratic daily vol dispersion (only used for heteroskedastic)

OUT_DIR = "500_ret_sim"
os.makedirs(OUT_DIR, exist_ok=True)

asset_ids = np.arange(1, N_ASSETS + 1, dtype=int)

# Fixed cross-sectional parameters for the entire experiment
# alphas = np.random.normal(ALPHA_MEAN, ALPHA_STD, size=N_ASSETS)  # COMMENTED OUT: not used for covariance research
betas  = np.random.normal(BETA_MEAN, BETA_STD, size=N_ASSETS)

# Save true betas for validation tests
np.save(os.path.join(OUT_DIR, "true_betas.npy"), betas)

# Generate idiosyncratic volatilities based on risk type
if HOMOSKEDASTIC:
    # Homoskedastic: All assets have the same idiosyncratic volatility
    idio_stds = np.full(N_ASSETS, IDIO_STD_MEAN)
    print(f"Simulating with HOMOSKEDASTIC risks: σ_idio = {IDIO_STD_MEAN:.6f} (constant for all assets)")
else:
    # Heteroskedastic: Each asset has different idiosyncratic volatility
    idio_stds = np.clip(np.random.normal(IDIO_STD_MEAN, IDIO_STD_STD, size=N_ASSETS), 0.003, None)
    print(f"Simulating with HETEROSKEDASTIC risks: σ_idio ~ N({IDIO_STD_MEAN:.6f}, {IDIO_STD_STD:.6f})")
    print(f"  Idiosyncratic vol range: [{idio_stds.min():.6f}, {idio_stds.max():.6f}]")

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

    # Single-factor model (no alpha): r_it = beta_i * F_t + eps_it
    R = betas[:, None] * F[None, :] + eps  # shape (500, 63)

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
monthly_df = pd.DataFrame(monthly_mat, columns=["permno"] + monthly_dates)
monthly_df.to_csv(os.path.join(OUT_DIR, "monthly_ret.csv"), index=False, float_format="%.10f")

# Summary
print(f"\n{'='*80}")
print(f"Simulation completed successfully!")
print(f"{'='*80}")
print(f"Risk type: {'HOMOSKEDASTIC' if HOMOSKEDASTIC else 'HETEROSKEDASTIC'}")
print(f"Generated {len(monthly_dates)} months of data for {N_ASSETS} assets")
print(f"Output directory: {OUT_DIR}/")
print(f"Files: {len(monthly_dates)} daily return CSVs + 1 monthly_ret.csv")
print(f"{'='*80}")
