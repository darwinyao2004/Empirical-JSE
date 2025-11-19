"""
The code is mostly based on paper "Portfolio optimisation via strategy-specific eigenvector shrinkage" https://link.springer.com/article/10.1007/s00780-025-00566-4,
Using formula 3.6/4.12 to test if JSE is moving eigenvector closer to the ground truth
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from construct_cov_sim import (
    demean_over_time,
    sample_cov,
    top_k_eigenpairs,
    js_eigvec_factor_cov
)
# Import LW shrinkage
from honeyShrinkage import covCor


def compute_angle_degrees(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle in degrees between two vectors.
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-18)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-18)
    
    cos_theta = abs(np.dot(v1_norm, v2_norm))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical safety
    
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_cos2(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute cosine of two vectors, to compare with the asymptotic delta later
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-18)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-18)
    cos_theta = abs(float(np.dot(v1_norm, v2_norm)))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return cos_theta ** 2


def compute_phi2(S: np.ndarray, n: int) -> float:
    """
    Compute phi^2, the average of the nonzero eigenvalues of S that are less than leading eigenvalue,
    as in eq. (3.3)-(3.6) of the paper Portfolio optimisation via strategy-specific eigenvector shrinkage
    """
    eigvals = np.linalg.eigvalsh(S)
    lambda_2 = float(eigvals[-1])

    # Average of other nonzero eigenvalues
    ell2 = (np.trace(S) - lambda_2) / max(n - 1, 1)

    # phi^2
    phi2 = (lambda_2 - ell2) / (ell2 + 1e-18)
    return phi2


def compute_cos_sin_theta2(true_eigvec: np.ndarray,
                           constraint_vecs: np.ndarray) -> tuple[float, float]:
    """
    Compute cos^2 theta and sin^2 theta where theta is the angle between the true eigenvector b
    and the constraint subspace C = span(constraint_vecs).

    Here with a full-investment constraint, constraint_vecs is 1_p.
    """
    b = true_eigvec / (np.linalg.norm(true_eigvec) + 1e-18)

    # Orthonormal basis of C by QR
    Q, _ = np.linalg.qr(constraint_vecs)
    # Projection of b onto C
    proj = Q @ (Q.T @ b)
    proj_norm = np.linalg.norm(proj)

    cos_theta = float(proj_norm)    # since ||b|| = 1, cos is just the projection
    cos_theta = max(min(cos_theta, 1.0), 0.0)
    cos2_theta = cos_theta ** 2
    sin2_theta = 1.0 - cos2_theta
    return cos2_theta, sin2_theta


def asymptotic_improvement(phi2: float,
                            cos2_theta: float,
                            sin2_theta: float) -> float:
    """
    RHS of eq. (3.6) / (4.12):

        (cos^2 θ_JSE - cos^2 θ_PCA)
        ≈ 1/(φ^2 + 1) * cos^2 Θ / (φ^2 sin^2 Θ + 1)
    """
    return (1.0 / (phi2 + 1.0)) * (cos2_theta / (phi2 * sin2_theta + 1.0))


def get_true_betas(data_dir: Path):
    """
    The betas are saved by simulate.py in 500_ret_sim/true_betas.npy, read directly from simulation.
    """
    betas_file = data_dir / "true_betas.npy"
    
    if not betas_file.exists():
        raise FileNotFoundError(
            f"True betas file not found: {betas_file}\n"
            f"Please run: python simulate.py"
        )
    
    betas = np.load(betas_file)
    true_eigvec = betas / np.linalg.norm(betas)
    
    return betas, true_eigvec


def analyze_all_months(data_dir: Path):
    """
    Analyze JSE vs PCA angles for all months.
    """
    # Get true eigenvector, should be beta_mean around 1.0, and beta_std around 0.5
    betas, true_eigvec_full = get_true_betas(data_dir)
    print(f"\nTrue betas loaded: {len(betas)} assets")
    print(f"Beta mean: {betas.mean():.3f}, std: {betas.std():.3f}")
    
    # Find all data files
    data_files = sorted(data_dir.glob("*_full.csv"))
    
    if not data_files:
        print(f"\nERROR: No data files found in {data_dir}")
        print("Please run: python simulate.py")
        return None
    
    print(f"\nFound {len(data_files)} months of data")
    print("-"*80)
    
    # Store results
    results = []
    
    # Process each month
    for i, data_file in enumerate(data_files):
        month = data_file.stem
        
        # Load data
        df = pd.read_csv(data_file, header=None)
        X_raw = df.iloc[:, 1:].values.astype(float)
        
        # Remove missing data
        mask = ~np.any(np.isnan(X_raw), axis=1)
        X = X_raw[mask, :]
        
        # Demean and compute sample covariance
        X = demean_over_time(X)
        S, n = sample_cov(X)
        p = X.shape[0]
        
        k = 1  # Single-factor model
        
        # PCA: Leading eigenvector from sample covariance
        U_pca, lam_pca = top_k_eigenpairs(S, k)
        h_pca = U_pca[:, 0]
        
        # LW: Leading eigenvector from LW-shrunk covariance
        Sigma_lw, _ = covCor(X.T)  # covCor expects (n, p) format
        U_lw, lam_lw = top_k_eigenpairs(Sigma_lw, k)
        h_lw = U_lw[:, 0]
        
        # JSE: Leading eigenvector from JSE-shrunk covariance
        Sigma_jse = js_eigvec_factor_cov(S, k, n)
        U_jse, lam_jse = top_k_eigenpairs(Sigma_jse, k)
        h_jse = U_jse[:, 0]
        
        # True eigenvector (aligned with kept assets)
        true_eigvec = true_eigvec_full[mask]
        true_eigvec = true_eigvec / np.linalg.norm(true_eigvec)
        
        # Compute angles to truth
        angle_pca = compute_angle_degrees(h_pca, true_eigvec)
        angle_lw = compute_angle_degrees(h_lw, true_eigvec)
        angle_jse = compute_angle_degrees(h_jse, true_eigvec)
        
        # Improvements relative to PCA
        improvement_jse = angle_pca - angle_jse
        improvement_lw = angle_pca - angle_lw
        pct_improvement_jse = (improvement_jse / angle_pca) * 100
        pct_improvement_lw = (improvement_lw / angle_pca) * 100

        # Squared cosines and asymptotic improvement (eq. 4.12 / 3.6) ---
        # Finite cos^2 values
        cos2_pca = compute_cos2(h_pca, true_eigvec)
        cos2_jse = compute_cos2(h_jse, true_eigvec)
        diff_finite = cos2_jse - cos2_pca

        # φ^2 from sample eigenvalues (uses n returned by sample_cov)
        phi2 = compute_phi2(S, n)

        # Θ between true eigenvector and constraint subspace C
        # For a single full-investment constraint: C = span(1_p)
        constraint_vec = np.ones((p, 1))
        cos2_theta, sin2_theta = compute_cos_sin_theta2(true_eigvec, constraint_vec)

        # Asymptotic difference from the paper
        diff_asymptotic = asymptotic_improvement(phi2, cos2_theta, sin2_theta)
        
        # Store results
        results.append({
            'month': month,
            'angle_pca': angle_pca,
            'angle_lw': angle_lw,
            'angle_jse': angle_jse,
            'improvement_jse_deg': improvement_jse,
            'improvement_lw_deg': improvement_lw,
            'improvement_jse_pct': pct_improvement_jse,
            'improvement_lw_pct': pct_improvement_lw,
            'jse_better': improvement_jse > 0,
            'lw_better': improvement_lw > 0,
            'cos2_pca': cos2_pca,
            'cos2_jse': cos2_jse,
            'diff_finite': diff_finite,
            'diff_asymptotic': diff_asymptotic,
            'diff_error': diff_finite - diff_asymptotic,
        })
        
        # Print result
        status_jse = "[+]" if improvement_jse > 0 else "[-]"
        status_lw = "[+]" if improvement_lw > 0 else "[-]"
        print(
            f"{month}: "
            f"PCA={angle_pca:6.3f}°, "
            f"LW={angle_lw:6.3f}° {status_lw}, "
            f"JSE={angle_jse:6.3f}° {status_jse}, "
            f"delta_cos²_fin={diff_finite:8.5f}, "
            f"delta_cos²_asy={diff_asymptotic:8.5f}"
        )
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    mean_angle_pca = df_results['angle_pca'].mean()
    mean_angle_lw = df_results['angle_lw'].mean()
    mean_angle_jse = df_results['angle_jse'].mean()
    
    mean_improvement_lw = df_results['improvement_lw_deg'].mean()
    mean_improvement_jse = df_results['improvement_jse_deg'].mean()
    median_improvement_lw = df_results['improvement_lw_deg'].median()
    median_improvement_jse = df_results['improvement_jse_deg'].median()
    
    mean_diff_fin = df_results['diff_finite'].mean()
    mean_diff_asy = df_results['diff_asymptotic'].mean()
    mean_diff_err = df_results['diff_error'].mean()
    std_diff_err = df_results['diff_error'].std()
    
    num_lw_improved = df_results['lw_better'].sum()
    num_jse_improved = df_results['jse_better'].sum()
    num_total = len(df_results)
    pct_lw_improved = (num_lw_improved / num_total) * 100
    pct_jse_improved = (num_jse_improved / num_total) * 100
    
    print(f"\nMean angles:")
    print(f"  PCA: {mean_angle_pca:.3f}°")
    print(f"  LW:  {mean_angle_lw:.3f}°")
    print(f"  JSE: {mean_angle_jse:.3f}°")
    
    print(f"\nMean improvement over PCA:")
    print(f"  LW:  {mean_improvement_lw:+.3f}° ({mean_improvement_lw/mean_angle_pca*100:+.2f}%)")
    print(f"  JSE: {mean_improvement_jse:+.3f}° ({mean_improvement_jse/mean_angle_pca*100:+.2f}%)")
    
    print(f"\nMedian improvement over PCA:")
    print(f"  LW:  {median_improvement_lw:+.3f}°")
    print(f"  JSE: {median_improvement_jse:+.3f}°")
    
    print(f"\nSuccess rate (better than PCA):")
    print(f"  LW:  {num_lw_improved}/{num_total} months ({pct_lw_improved:.1f}%)")
    print(f"  JSE: {num_jse_improved}/{num_total} months ({pct_jse_improved:.1f}%)")
    
    print(f"\nJSE asymptotic validation:")
    print(f"  Mean delta cos² (finite):      {mean_diff_fin:.6e}")
    print(f"  Mean delta cos² (asymptotic):  {mean_diff_asy:.6e}")
    print(f"  Mean (fin - asy) error:        {mean_diff_err:.6e}")
    print(f"  Std of error across months:    {std_diff_err:.6e}")
    
    if mean_improvement_jse > 0:
        print("\nVALIDATION PASSED: JSE improves over PCA on average!")
    else:
        print("\nWARNING: JSE does not improve on average")
    
    print("="*80)
    
    return df_results


def plot_results(df_results: pd.DataFrame,
                 output_file: str = "jse_angles_comparison.png") -> None:
    months = np.arange(len(df_results))

    # --- Create 2 stacked subplots ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 1) Angles Over Time
    ax1.plot(months, df_results['angle_pca'], 'o-', label='PCA',
             color='blue', alpha=0.7, linewidth=2, markersize=4)
    ax1.plot(months, df_results['angle_lw'], '^-', label='LW',
             color='pink', alpha=0.7, linewidth=2, markersize=4)
    ax1.plot(months, df_results['angle_jse'], 's-', label='JSE',
             color='red', alpha=0.7, linewidth=2, markersize=4)
    
    ax1.set_ylabel('Angle to True Eigenvector (degrees)', fontsize=12)
    ax1.set_title('PCA vs LW vs JSE Eigenvector Estimation: Angles Over Time',
                  fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend()
    
    # Add summary stats box for angles
    mean_angle_pca = df_results['angle_pca'].mean()
    mean_angle_lw = df_results['angle_lw'].mean()
    mean_angle_jse = df_results['angle_jse'].mean()
    improvement_lw = df_results['improvement_lw_deg'].mean()
    improvement_jse = df_results['improvement_jse_deg'].mean()
    
    stats_text = (
        f"Mean angle:\n"
        f"  PCA: {mean_angle_pca:.3f}°\n"
        f"  LW:  {mean_angle_lw:.3f}°\n"
        f"  JSE: {mean_angle_jse:.3f}°\n"
        f"Mean improvement:\n"
        f"  LW:  {improvement_lw:+.3f}°\n"
        f"  JSE: {improvement_jse:+.3f}°"
    )
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    # 2) delta cos² (finite vs asymptotic) - JSE only
    ax2.plot(months, df_results['diff_finite'], 'o-',
             label=r'$\Delta \cos^2$ (finite)',
             color='green', alpha=0.7, linewidth=2, markersize=4)
    ax2.plot(months, df_results['diff_asymptotic'], 's--',
             label=r'$\Delta \cos^2$ (asymptotic)',
             color='orange', alpha=0.7, linewidth=2, markersize=4)

    # Zero line for visual reference
    ax2.axhline(0.0, color='black', linewidth=1, linestyle=':')

    ax2.set_xlabel('Month Index', fontsize=12)
    ax2.set_ylabel(r'$\Delta \cos^2$ (JSE − PCA)', fontsize=12)
    ax2.set_title(r'JSE Improvement: '
                  r'$\Delta \cos^2_{\mathrm{fin}}$ vs '
                  r'$\Delta \cos^2_{\mathrm{asy}}$ (Theorem 4.5)',
                  fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend()

    # Optional: summary box for delta cos²
    mean_diff_fin = df_results['diff_finite'].mean()
    mean_diff_asy = df_results['diff_asymptotic'].mean()
    mean_diff_err = (df_results['diff_finite']
                     - df_results['diff_asymptotic']).mean()

    stats_text2 = (
        f"Mean delta cos² (fin): {mean_diff_fin:.3e}\n"
        f"Mean delta cos² (asy): {mean_diff_asy:.3e}\n"
        f"Mean (fin − asy):  {mean_diff_err:.3e}"
    )
    ax2.text(0.02, 0.98, stats_text2, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    """Main analysis pipeline."""
    data_dir = Path("500_ret_sim")
    
    if not data_dir.exists():
        print(f"ERROR: Data directory {data_dir} not found!")
        print("Please run: python simulate.py")
        return
    
    # Run analysis
    df_results = analyze_all_months(data_dir)
    
    if df_results is not None:
        # Create visualization
        plot_results(df_results)


if __name__ == "__main__":
    main()

