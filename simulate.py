# multi_factor_sim.py
"""
Multi-factor model simulation (without alpha, for covariance research).

Model: r_it = sum_k(beta_i,k * F_k,t) + eps_it

Single-factor (N_FACTORS=1):
  - r_it = beta_i * F_t + eps_it
  - F_t ~ N(FACTOR_MEAN, FACTOR_STD): common factor
  - beta_i ~ N(BETA_MEAN, BETA_STD): factor loading (cross-sectional)
  
Multi-factor (N_FACTORS>1):
  - r_it = beta_i,1 * F_1,t + beta_i,2 * F_2,t + ... + eps_it
  - F_1,t ~ N(FACTOR_MEAN, FACTOR_STD): primary factor
  - F_k,t ~ N(0, FACTOR_k_STD): additional factors (k >= 2)
  - beta_i,1 ~ N(BETA_MEAN, BETA_STD): loading for factor 1
  - beta_i,k ~ N(LOADING_MEAN, LOADING_STD): loadings for factors k >= 2
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
- true_betas.npy: factor 1 loadings (for validation)
- true_factor_loadings.npy: all factor loadings (N_ASSETS x N_FACTORS, only if N_FACTORS > 1)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
from calendar import monthrange

# ----------------------------
# Default Configuration
# ----------------------------
DEFAULT_SEED = 42
DEFAULT_N_ASSETS = 500
DEFAULT_DAYS_PER_MONTH = 21
DEFAULT_WINDOW_DAYS = DEFAULT_DAYS_PER_MONTH * 3  # 3 months = 63 days
DEFAULT_HOMOSKEDASTIC = True
DEFAULT_N_FACTORS = 2  # Set to 1 for single-factor model, 2 for two-factor model, etc.

# Factor 1 parameters (primary factor, similar to market factor)
# FACTOR_MEAN = 0.0003          # daily factor drift
FACTOR_MEAN = 0.0          # daily factor drift
# FACTOR_STD  = 0.01            # daily factor volatility
FACTOR_STD  = 0.16/np.sqrt(252)            # daily factor volatility
# ALPHA_MEAN  = 0.00005         # cross-sectional alpha mean (daily) - COMMENTED OUT: not used for covariance research
# ALPHA_STD   = 0.0002          # cross-sectional alpha std  (daily) - COMMENTED OUT: not used for covariance research
BETA_MEAN   = 1.0             # cross-sectional beta mean for factor 1
# BETA_STD    = 0.3             # cross-sectional beta std
BETA_STD    = 0.5             # cross-sectional beta std for factor 1

# Additional factors parameters (factors 2, 3, 4, etc.)
# These are used only when N_FACTORS > 1
# For 7-factor model, these match H_jsm.py: fvol = [16, 4, 2, 20, 15, 10, 5]
FACTOR_2_STD = 0.04/np.sqrt(252)  # daily volatility for factor 2 (4% annual)
FACTOR_3_STD = 0.02/np.sqrt(252)  # daily volatility for factor 3 (2% annual)
FACTOR_4_STD = 0.20/np.sqrt(252)  # daily volatility for factor 4 (20% annual)
FACTOR_5_STD = 0.15/np.sqrt(252)  # daily volatility for factor 5 (15% annual)
FACTOR_6_STD = 0.10/np.sqrt(252)  # daily volatility for factor 6 (10% annual)
FACTOR_7_STD = 0.05/np.sqrt(252)  # daily volatility for factor 7 (5% annual)

# Loadings for additional factors (mean=0, std=1 by default, similar to simulationJSE.py)
LOADING_MEAN = 0.0  # mean of loadings for factors 2+
LOADING_STD = 1.0   # std of loadings for factors 2+

# Factor correlation matrix
# Set correlations between factors. For independent factors, use identity matrix.
# For correlated factors, specify the correlation matrix below.
# Example for 2 factors with correlation 0.28:
#   factor_corr_matrix = np.array([[1.0, 0.28],
#                                   [0.28, 1.0]])
# For no correlation (independent factors):
#   factor_corr_matrix = np.eye(n_factors)
# This is the same idea from code H_jsm
# 
# Note: Factor correlation matrix is now built inside run_simulation() function
# based on the n_factors parameter passed to it.

# Idiosyncratic risk parameters
# IDIO_STD_MEAN = 0.012         # idiosyncratic daily vol mean (used for both homo- and heteroskedastic)
IDIO_STD_MEAN = 0.6/np.sqrt(252)         # idiosyncratic daily vol mean (used for both homo- and heteroskedastic)
# IDIO_STD_STD  = 0.004         # idiosyncratic daily vol dispersion (only used for heteroskedastic)
IDIO_STD_STD  = 0.1/np.sqrt(252)         # idiosyncratic daily vol dispersion (only used for heteroskedastic)

DEFAULT_OUT_DIR = "500_ret_sim"


def run_simulation(
    seed=DEFAULT_SEED,
    n_factors=DEFAULT_N_FACTORS,
    n_assets=DEFAULT_N_ASSETS,
    homoskedastic=DEFAULT_HOMOSKEDASTIC,
    out_dir=DEFAULT_OUT_DIR,
    days_per_month=DEFAULT_DAYS_PER_MONTH,
    start_ym="201501",
    end_ym="202412"
):
    """
    Run multi-factor model simulation with configurable parameters.
    
    Args:
        seed: Random seed for reproducibility
        n_factors: Number of common factors (1 for single-factor, 2+ for multi-factor)
        n_assets: Number of assets to simulate
        homoskedastic: If True, all assets have same idiosyncratic volatility
        out_dir: Output directory for generated files
        days_per_month: Trading days per month
        start_ym: Start month in YYYYMM format
        end_ym: End month in YYYYMM format
    
    Returns:
        Dictionary with simulation metadata
    """
    # Set random seed
    np.random.seed(seed)
    
    # Configuration
    WINDOW_DAYS = days_per_month * 3
    
    os.makedirs(out_dir, exist_ok=True)
    
    asset_ids = np.arange(1, n_assets + 1, dtype=int)
    
    # Fixed cross-sectional parameters for the entire experiment
    # alphas = np.random.normal(ALPHA_MEAN, ALPHA_STD, size=n_assets)  # COMMENTED OUT: not used for covariance research
    
    # Generate factor loadings based on number of factors
    if n_factors == 1:
        # Single-factor model: only betas for factor 1
        betas = np.random.normal(BETA_MEAN, BETA_STD, size=n_assets)
        factor_loadings = betas[:, None]  # shape (n_assets, 1)
    else:
        # Multi-factor model: betas for factor 1 + loadings for additional factors
        betas = np.random.normal(BETA_MEAN, BETA_STD, size=n_assets)
        # Additional loadings with mean 0, std 1 (similar to simulationJSE.py)
        additional_loadings = np.random.normal(LOADING_MEAN, LOADING_STD, size=(n_assets, n_factors - 1))
        factor_loadings = np.column_stack([betas, additional_loadings])  # shape (n_assets, n_factors)
    
    # Save true betas (factor 1 loadings) for validation tests
    np.save(os.path.join(out_dir, "true_betas.npy"), betas)
    
    # Save all factor loadings for multi-factor validation
    if n_factors > 1:
        np.save(os.path.join(out_dir, "true_factor_loadings.npy"), factor_loadings)
    
    # Build factor correlation matrix
    if n_factors == 1:
        factor_corr_matrix = np.array([[1.0]])
    elif n_factors == 2:
        # Default: uncorrelated factors
        factor_corr_matrix = np.array([[1.0, 0.0],
                                        [0.0, 1.0]])
    elif n_factors == 3:
        # Default: some correlations for 3 factors
        factor_corr_matrix = np.array([[1.00,  0.28, -0.30],
                                        [0.28,  1.00, -0.11],
                                        [-0.30, -0.11,  1.00]])
    elif n_factors == 7:
        # 7-factor model from H_jsm.py: 1 market + 2 style + 4 block factors
        # This correlation matrix is from jsm_code/H_jsm.py
        factor_corr_matrix = np.array([
            [ 1.00,  0.28, -0.30,  0.16,  0.08, 0.04,  0.02],
            [ 0.28,  1.00, -0.11,  0.00,  0.00, 0.00,  0.00],
            [-0.30, -0.11,  1.00,  0.00,  0.00, 0.00,  0.00],
            [ 0.16,  0.00,  0.00,  1.00,  0.00, 0.00,  0.00],
            [ 0.08,  0.00,  0.00,  0.00,  1.00, 0.00,  0.00],
            [ 0.04,  0.00,  0.00,  0.00,  0.00, 1.00,  0.00],
            [ 0.02,  0.00,  0.00,  0.00,  0.00, 0.00, 1.00]])
    else:
        # For 4-6 and 8+ factors, start with identity
        factor_corr_matrix = np.eye(n_factors)
    
    if n_factors > 1:
        np.save(os.path.join(out_dir, "factor_correlation_matrix.npy"), factor_corr_matrix)
    
    # Define factor standard deviations
    factor_stds = [FACTOR_STD]
    if n_factors >= 2:
        factor_stds.append(FACTOR_2_STD)
    if n_factors >= 3:
        factor_stds.append(FACTOR_3_STD)
    if n_factors >= 4:
        factor_stds.append(FACTOR_4_STD)
    if n_factors >= 5:
        factor_stds.append(FACTOR_5_STD)
    if n_factors >= 6:
        factor_stds.append(FACTOR_6_STD)
    if n_factors >= 7:
        factor_stds.append(FACTOR_7_STD)
    # For factors beyond 7, use a default value
    for i in range(7, n_factors):
        factor_stds.append(0.04/np.sqrt(252))  # default std for extra factors
    
    factor_stds = np.array(factor_stds[:n_factors])
    
    # Construct factor covariance matrix and transformation matrix (following H_jsm.py approach)
    # F = diag(factor_stds) @ factor_corr_matrix @ diag(factor_stds)
    # Then decompose F to get transformation matrix A
    if n_factors > 1:
        # Symmetrize correlation matrix just in case
        factor_corr_matrix = (factor_corr_matrix + factor_corr_matrix.T) / 2
        
        # Construct covariance matrix: F = outer(vol, vol) * Corr
        factor_cov_matrix = np.outer(factor_stds, factor_stds) * factor_corr_matrix
        
        # Eigendecomposition to get transformation matrix
        # This allows us to generate correlated factors from independent standard normals
        vals, vecs = np.linalg.eigh(factor_cov_matrix)
        A = vecs @ np.diag(np.sqrt(vals))  # Transformation matrix
        
        # Verify the covariance matrix is valid (positive definite)
        if np.any(vals < -1e-10):
            print("ERROR: Factor covariance matrix is not positive semi-definite!")
            print(f"Eigenvalues: {vals}")
            raise ValueError("Factor covariance matrix is not positive semi-definite")
        
        # Transform factor loadings to embed correlation structure
        # When we generate independent standard normal factors X,
        # the returns will be: r = (factor_loadings @ A) @ X + eps
        # This gives correlated factor returns with the desired covariance structure
        factor_loadings_transformed = factor_loadings @ A
        
        print(f"\nFactor correlation matrix:")
        print(np.round(factor_corr_matrix, 3))
        print(f"\nFactor covariance matrix:")
        print(np.round(factor_cov_matrix, 6))
    else:
        # Single factor case: no transformation needed
        A = np.array([[factor_stds[0]]])
        factor_loadings_transformed = factor_loadings * factor_stds[0]
        factor_cov_matrix = np.array([[factor_stds[0]**2]])
    
    # Generate idiosyncratic volatilities based on risk type
    if homoskedastic:
        # Homoskedastic: All assets have the same idiosyncratic volatility
        idio_stds = np.full(n_assets, IDIO_STD_MEAN)
        print(f"Simulating {n_factors}-factor model with HOMOSKEDASTIC risks")
        print(f"  σ_idio = {IDIO_STD_MEAN:.6f} (constant for all assets)")
    else:
        # Heteroskedastic: Each asset has different idiosyncratic volatility
        idio_stds = np.clip(np.random.normal(IDIO_STD_MEAN, IDIO_STD_STD, size=n_assets), 0.003, None)
        print(f"Simulating {n_factors}-factor model with HETEROSKEDASTIC risks")
        print(f"  σ_idio ~ N({IDIO_STD_MEAN:.6f}, {IDIO_STD_STD:.6f})")
        print(f"  Idiosyncratic vol range: [{idio_stds.min():.6f}, {idio_stds.max():.6f}]")
    
    print(f"\nFactor parameters:")
    for k in range(n_factors):
        print(f"  Factor {k+1}: σ = {factor_stds[k]:.6f}")
    if n_factors == 1:
        print(f"  Factor 1 loadings: mean = {BETA_MEAN:.2f}, std = {BETA_STD:.2f}")
    else:
        print(f"  Factor 1 loadings: mean = {BETA_MEAN:.2f}, std = {BETA_STD:.2f}")
        print(f"  Factors 2+ loadings: mean = {LOADING_MEAN:.2f}, std = {LOADING_STD:.2f}")
        print(f"\nFactor correlations:")
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                print(f"  Corr(Factor {i+1}, Factor {j+1}) = {factor_corr_matrix[i,j]:.3f}")
    
    def month_iter(start, end):
        ys, ms = int(start[:4]), int(start[4:])
        ye, me = int(end[:4]), int(end[4:])
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
    
    for (year, month) in month_iter(start_ym, end_ym):
        yyyymm = f"{year}{month:02d}"
        
        # Simulate INDEPENDENT standard normal factor returns (n_factors x WINDOW_DAYS)
        # These will be transformed to create correlated factors via the loadings
        X = np.random.normal(0.0, 1.0, size=(n_factors, WINDOW_DAYS))
        
        # Add mean drift to first factor only (optional)
        if FACTOR_MEAN != 0.0:
            X[0, :] = X[0, :] + FACTOR_MEAN
        
        # Idiosyncratic shocks (n_assets x WINDOW_DAYS)
        eps = np.random.normal(0.0, idio_stds[:, None], size=(n_assets, WINDOW_DAYS))
        
        # Multi-factor model with correlated factors:
        # r_it = (factor_loadings_transformed @ X) + eps_it
        # where factor_loadings_transformed = factor_loadings @ A
        # and A embeds the factor covariance structure
        # 
        # Mathematically: r = (B @ A) @ X + eps
        # where X ~ N(0, I), so (A @ X) gives correlated factor returns
        # with covariance A @ I @ A^T = A @ A^T = factor_cov_matrix
        #
        # factor_loadings_transformed: (n_assets, n_factors)
        # X: (n_factors, WINDOW_DAYS)
        # Result: (n_assets, WINDOW_DAYS)
        R = factor_loadings_transformed @ X + eps  # shape (500, 63)
        
        # Write yyyymm_full.csv: first col = id, then 63 daily returns (no header)
        df = pd.DataFrame(np.column_stack([asset_ids, R]))
        df.to_csv(os.path.join(out_dir, f"{yyyymm}_full.csv"), header=False, index=False, float_format="%.10f")
        
        # Monthly gross return = product of (1 + daily return) over the last days_per_month columns
        last_n = R[:, -days_per_month:]
        gross = np.prod(1.0 + last_n, axis=1)  # store the product (not minus 1), as requested
        monthly_rets.append(gross - 1)
        
        # Month-end date yyyy/mm/dd
        last_day = monthrange(year, month)[1]
        monthly_dates.append(f"{year}/{month}/{last_day}")
    
    # monthly_ret.csv: columns = ['id'] + monthly_dates; first row carries the dates as header
    monthly_mat = np.column_stack([asset_ids] + [np.array(col) for col in monthly_rets])
    monthly_df = pd.DataFrame(monthly_mat, columns=["permno"] + monthly_dates)
    monthly_df.to_csv(os.path.join(out_dir, "monthly_ret.csv"), index=False, float_format="%.10f")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Simulation completed successfully!")
    print(f"{'='*80}")
    print(f"Model type: {n_factors}-factor model")
    print(f"Risk type: {'HOMOSKEDASTIC' if homoskedastic else 'HETEROSKEDASTIC'}")
    print(f"Generated {len(monthly_dates)} months of data for {n_assets} assets")
    print(f"Output directory: {out_dir}/")
    print(f"Files: {len(monthly_dates)} daily return CSVs + 1 monthly_ret.csv + factor loadings")
    if n_factors == 1:
        print(f"  - true_betas.npy: Factor 1 loadings (shape: {betas.shape})")
    else:
        print(f"  - true_betas.npy: Factor 1 loadings (shape: {betas.shape})")
        print(f"  - true_factor_loadings.npy: All factor loadings (shape: {factor_loadings.shape})")
        print(f"  - factor_correlation_matrix.npy: Factor correlation matrix (shape: {factor_corr_matrix.shape})")
    print(f"{'='*80}")
    
    # Return metadata
    return {
        "n_assets": n_assets,
        "n_factors": n_factors,
        "n_months": len(monthly_dates),
        "homoskedastic": homoskedastic,
        "seed": seed,
        "out_dir": out_dir
    }


# Main execution when run as a script
if __name__ == "__main__":
    run_simulation()
