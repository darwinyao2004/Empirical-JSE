"""
Full Pipeline for Shrinkage Method Performance Testing
=======================================================

This script runs the complete workflow to test different covariance shrinkage methods:
1. simulate.py         - Generate simulated return data
2. construct_cov_sim.py - Construct covariance matrices (LW, PCA, JSE)
3. construct_portfolio.py - Build portfolio weights (GMV, TVMV)
4. portfolio_test.py   - Backtest and evaluate performance

Usage:
    python run_full_pipeline.py --num_runs 10

Configuration:
    NUM_RUNS: Number of simulation runs with different random seeds (default: 10)
    NUM_FACTORS: Number of factors for covariance estimation (default: 2)

Author: Automated pipeline wrapper
"""

import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# ========================================
# Configuration
# ========================================
NUM_RUNS = 10  # Number of simulation runs
NUM_FACTORS = 7  # Number of factors for covariance estimation

# ========================================
# Step 1: Simulate Returns
# ========================================
def run_simulation_step(seed: int = 42, num_factors: int = NUM_FACTORS):
    """Run the simulation to generate return data with specified random seed and number of factors
    
    Args:
        seed: Random seed for reproducibility
        num_factors: Number of common factors in the multi-factor model
    
    Returns:
        Boolean indicating success or failure
    """
    print("="*80)
    print(f"STEP 1: Generating Simulated Return Data (seed={seed}, K={num_factors} factors)")
    print("="*80)
    start_time = time.time()
    
    try:
        # Import the refactored simulate module
        from simulate import run_simulation
        
        # Call the simulation function with our parameters
        result = run_simulation(
            seed=seed,
            n_factors=num_factors,
            n_assets=500,
            homoskedastic=True,
            out_dir="500_ret_sim",
            days_per_month=21,
            start_ym="201501",
            end_ym="202412"
        )
        
        print(f"✓ Simulation completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ ERROR in simulation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# ========================================
# Step 2: Construct Covariance Matrices
# ========================================
def run_covariance_construction():
    """Construct covariance matrices using different shrinkage methods"""
    print("\n" + "="*80)
    print("STEP 2: Constructing Covariance Matrices (PCA, JSE)")
    print("="*80)
    start_time = time.time()
    
    try:
        # Import construct_cov_sim module
        import construct_cov_sim
        
        # Setup parameters
        in_dir = Path("500_ret_sim").expanduser().resolve()
        result_txt = Path("result_500_sim.txt").expanduser().resolve()
        out_root = Path("covariance_outputs_sim").expanduser().resolve()
        
        print(f"Input directory:      {in_dir}")
        print(f"Output root directory: {out_root}")
        
        if not in_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {in_dir}")
        
        construct_cov_sim.ensure_dir(out_root)
        construct_cov_sim.process_folder(
            in_dir, 
            result_txt, 
            out_root, 
            eps=1e-12, 
            num_factors=NUM_FACTORS
        )
        
        print(f"✓ Covariance construction completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ ERROR in covariance construction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# ========================================
# Step 3: Construct Portfolios
# ========================================
def run_portfolio_construction():
    """Construct portfolio weights from covariance matrices"""
    print("\n" + "="*80)
    print("STEP 3: Constructing Portfolio Weights (GMV)")
    print("="*80)
    start_time = time.time()
    
    try:
        # Import construct_portfolio module
        import construct_portfolio
        
        print(f"Input root:  {construct_portfolio.INPUT_ROOT}")
        print(f"Output root: {construct_portfolio.OUTPUT_ROOT}")
        print(f"Methods: {construct_portfolio.METHOD_FOLDERS}")
        
        # Run the main function
        construct_portfolio.main()
        
        print(f"✓ Portfolio construction completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ ERROR in portfolio construction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# ========================================
# Step 4: Backtest and Evaluate
# ========================================
def run_backtest():
    """Run backtest to evaluate portfolio performance"""
    print("\n" + "="*80)
    print("STEP 4: Backtesting Portfolio Performance")
    print("="*80)
    start_time = time.time()
    
    try:
        # Import portfolio_test module
        import portfolio_test
        
        print(f"Portfolio root: {portfolio_test.PORTFOLIO_ROOT}")
        print(f"Monthly returns: {portfolio_test.MONTHLY_RET_CSV}")
        print(f"Results root: {portfolio_test.RESULTS_ROOT}")
        
        # Run the main function
        portfolio_test.main()
        
        print(f"✓ Backtest completed in {time.time() - start_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"✗ ERROR in backtest: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# ========================================
# Single Run Pipeline
# ========================================
def run_single_pipeline(seed: int) -> Dict[str, float]:
    """Run the complete pipeline for a single seed and return variance results"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print(f"║" + " "*25 + f"RUNNING WITH SEED: {seed}" + " "*(78-25-len(f"RUNNING WITH SEED: {seed}")) + "║")
    print("╚" + "="*78 + "╝")
    
    # Run each step in sequence
    if not run_simulation_step(seed, num_factors=NUM_FACTORS):
        print(f"\n✗ Pipeline stopped at simulation (seed={seed})")
        return {}
    
    if not run_covariance_construction():
        print(f"\n✗ Pipeline stopped at covariance construction (seed={seed})")
        return {}
    
    if not run_portfolio_construction():
        print(f"\n✗ Pipeline stopped at portfolio construction (seed={seed})")
        return {}
    
    if not run_backtest():
        print(f"\n✗ Pipeline stopped at backtest (seed={seed})")
        return {}
    
    # Read the results
    try:
        summary_path = Path("results_sim/summary.csv")
        if not summary_path.exists():
            print(f"✗ Results file not found: {summary_path}")
            return {}
        
        summary_df = pd.read_csv(summary_path)
        
        # Extract var_return for each method
        results = {}
        for _, row in summary_df.iterrows():
            method = row['method']
            var_return = row['var_return']
            results[method] = var_return
        
        print(f"\n✓ Seed {seed} completed. Variance results: {results}")
        return results
    
    except Exception as e:
        print(f"✗ Error reading results for seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ========================================
# Visualization
# ========================================
def create_boxplot(all_results: Dict[str, List[float]], output_path: str = "variance_comparison_boxplot.png"):
    """Create box plot comparing variance across methods"""
    print("\n" + "="*80)
    print("Creating Box Plot Visualization")
    print("="*80)
    
    # Prepare data for plotting
    methods = list(all_results.keys())
    data = [all_results[method] for method in methods]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create box plot
    bp = ax.boxplot(data, labels=methods, patch_artist=True)
    
    # Customize colors (adjusted for PCA and JSE only)
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Labels and title
    ax.set_xlabel('Shrinkage Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variance of Returns', fontsize=12, fontweight='bold')
    ax.set_title(f'Out-of-Sample Return Variance Comparison\n({len(data[0])} simulations)', 
                 fontsize=14, fontweight='bold')
    
    # Grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Box plot saved to: {output_path}")
    
    # Print statistics
    print("\n" + "="*80)
    print("VARIANCE STATISTICS ACROSS ALL RUNS")
    print("="*80)
    for method, values in all_results.items():
        print(f"\n{method}:")
        print(f"  Mean:   {np.mean(values):.6f}")
        print(f"  Median: {np.median(values):.6f}")
        print(f"  Std:    {np.std(values, ddof=1):.6f}")
        print(f"  Min:    {np.min(values):.6f}")
        print(f"  Max:    {np.max(values):.6f}")
    print("="*80)


# ========================================
# Main Pipeline (Multiple Runs)
# ========================================
def main(num_runs: int = NUM_RUNS):
    """Run the complete pipeline multiple times with different seeds"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "MULTIPLE SIMULATION PERFORMANCE PIPELINE" + " "*22 + "║")
    print("╚" + "="*78 + "╝")
    print(f"\nNumber of runs: {num_runs}")
    print(f"Number of factors: {NUM_FACTORS}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    overall_start = time.time()
    
    # Storage for all results (LW disabled - may have issues)
    all_results: Dict[str, List[float]] = {
        "PCA": [],
        "JSE": []
    }
    
    successful_runs = 0
    
    # Run pipeline multiple times with different seeds
    for run_idx in range(num_runs):
        seed = 42 + run_idx  # Use different seeds: 42, 43, 44, ...
        
        print("\n" + "="*80)
        print(f"RUN {run_idx + 1}/{num_runs} (Seed: {seed})")
        print("="*80)
        
        run_start = time.time()
        results = run_single_pipeline(seed)
        run_time = time.time() - run_start
        
        if results:
            # Store results (LW disabled)
            for method in ["PCA", "JSE"]:
                if method in results:
                    all_results[method].append(results[method])
            successful_runs += 1
            print(f"\n✓ Run {run_idx + 1} completed in {run_time:.2f} seconds")
        else:
            print(f"\n✗ Run {run_idx + 1} failed")
    
    # Final summary
    total_time = time.time() - overall_start
    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print(f"Successful runs: {successful_runs}/{num_runs}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per run: {total_time/num_runs:.2f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create visualization if we have results (LW disabled)
    if successful_runs > 0 and all(len(all_results[m]) > 0 for m in ["PCA", "JSE"]):
        print("\n")
        create_boxplot(all_results)
        
        # Save detailed results to CSV
    #     results_df = pd.DataFrame(all_results)
    #     results_csv = "variance_results_all_runs.csv"
    #     results_df.to_csv(results_csv, index=False)
    #     print(f"✓ Detailed results saved to: {results_csv}")
    # else:
    #     print("\n✗ Insufficient results to create visualization")
    
    print("\n" + "="*80 + "\n")
    
    if successful_runs == num_runs:
        print("✓ All runs completed successfully!")
        return 0
    elif successful_runs > 0:
        print(f"⚠ {num_runs - successful_runs} run(s) failed")
        return 1
    else:
        print("✗ All runs failed")
        return 1


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run multiple simulations to compare shrinkage methods"
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of simulation runs (default: {NUM_RUNS})"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(num_runs=args.num_runs))

