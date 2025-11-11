"""
Portfolio construction from monthly full-rank covariance matrices.

This module provides tools for constructing optimal portfolios using various
optimization strategies (GMV, target volatility) with flexible constraints.

Architecture:
    - PortfolioConfig: Immutable configuration object
    - CovarianceLoader: CSV file I/O handler
    - PortfolioOptimizer: Core optimization algorithms
    - PortfolioConstructor: High-level pipeline orchestrator

Author: Darwin Yao
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================================
# Solver Backend Detection
# ============================================================================

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

try:
    from scipy.optimize import minimize, LinearConstraint, Bounds
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if not (HAS_CVXPY or HAS_SCIPY):
    raise ImportError("Either cvxpy or scipy required for optimization")


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class PortfolioConfig:
    """Immutable configuration for portfolio optimization.
    
    Attributes:
        ridge_eps: Ridge regularization for covariance (Σ + εI)
        l2_lambda: L2 penalty on weights (λ||w||²)
        long_only: If True, enforce w >= 0
        weight_cap: Maximum weight per asset (None = no cap)
        allow_short: Use closed-form GMV when unconstrained
        target_vol: Target portfolio volatility for Vol-MV optimization
        gamma_min/max: Bisection bounds for gamma search
        gamma_tol: Relative tolerance for gamma convergence
        vol_tol: Absolute tolerance for target vol achievement
        max_bisect_iters: Maximum bisection iterations
    """
    ridge_eps: float = 0.0
    l2_lambda: float = 0.0
    long_only: bool = True
    weight_cap: Optional[float] = None
    allow_short: bool = False
    target_vol: float = 0.10
    gamma_min: float = 1e-6
    gamma_max: float = 1e6
    gamma_tol: float = 1e-4
    vol_tol: float = 1e-5
    max_bisect_iters: int = 50


# ============================================================================
# Data Loading
# ============================================================================

class CovarianceLoader:
    """Loads covariance matrices from CSV files.
    
    Expected format: First column is PERMNO identifiers, remaining p columns
    form a p×p covariance matrix (row-major order).
    """
    
    @staticmethod
    def load(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load covariance matrix and identifiers from CSV.
        
        Args:
            path: Path to CSV file
            
        Returns:
            (permnos, covariance_matrix) where permnos is 1D array and
            covariance_matrix is 2D symmetric p×p array
            
        Raises:
            ValueError: If matrix dimensions are inconsistent
        """
        # Try loading without header first, fallback to with-header
        for header in [None, 0]:
            try:
                df = pd.read_csv(path, header=header)
                permnos = df.iloc[:, 0].to_numpy()
                cov_matrix = df.iloc[:, 1:].to_numpy().astype(float)
                
                p = len(permnos)
                if cov_matrix.shape == (p, p):
                    return permnos, cov_matrix
            except Exception:
                continue
        
        raise ValueError(
            f"Cannot parse {path}: expected first column as IDs and "
            f"remaining columns as p×p covariance matrix"
        )


# ============================================================================
# Portfolio Optimization
# ============================================================================

class PortfolioOptimizer:
    """Solves portfolio optimization problems.
    
    Supports:
        - Global Minimum Variance (GMV)
        - Target Volatility Mean-Variance with zero-mean prior
        - Long-only and long-short constraints
        - Box constraints (weight caps)
        - L2 regularization
    
    Uses CVXPY if available, otherwise falls back to scipy.
    """
    
    def __init__(self, config: PortfolioConfig):
        """Initialize optimizer with configuration.
        
        Args:
            config: Portfolio optimization parameters
        """
        self.config = config
    
    def _prepare_covariance(self, sigma: np.ndarray) -> np.ndarray:
        """Symmetrize and regularize covariance matrix.
        
        Args:
            sigma: Raw covariance matrix
            
        Returns:
            Regularized symmetric covariance: (Σ + Σ')/2 + εI
        """
        sigma = 0.5 * (sigma + sigma.T)  # Ensure symmetry
        if self.config.ridge_eps > 0:
            sigma += self.config.ridge_eps * np.eye(len(sigma))
        return sigma
    
    def _gmv_analytical(self, sigma: np.ndarray) -> np.ndarray:
        """Closed-form GMV solution: w* = Σ⁻¹1 / (1'Σ⁻¹1).
        
        Used when no active constraints are present.
        
        Args:
            sigma: Covariance matrix
            
        Returns:
            Optimal weights (fully-invested)
            
        Raises:
            ValueError: If denominator is numerically zero
        """
        ones = np.ones(len(sigma))
        
        # Prefer solve() over inv() for numerical stability
        try:
            inv_sigma_ones = np.linalg.solve(sigma, ones)
        except np.linalg.LinAlgError:
            # For singular matrices, raise error instead of using lstsq
            raise ValueError("GMV analytical solution: singular covariance matrix")
        
        denominator = ones @ inv_sigma_ones
        if abs(denominator) < 1e-12:
            raise ValueError("GMV analytical solution: singular covariance")
        
        return inv_sigma_ones / denominator
    
    def _solve_qp(
        self,
        sigma: np.ndarray,
        gamma: float,
        objective_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Solve constrained quadratic program.
        
        Minimizes: (γ/2)w'Σw + λ||w||²
        Subject to: w'1 = 1, box constraints (if specified)
        
        Args:
            sigma: Covariance matrix
            gamma: Risk aversion parameter
            objective_fn: Optional custom objective (for scipy)
            
        Returns:
            Optimal weights satisfying constraints
            
        Raises:
            RuntimeError: If optimization fails
        """
        if HAS_CVXPY:
            return self._solve_with_cvxpy(sigma, gamma)
        elif HAS_SCIPY:
            return self._solve_with_scipy(sigma, gamma, objective_fn)
        else:
            raise RuntimeError("No optimization backend available")
    
    def _solve_with_cvxpy(self, sigma: np.ndarray, gamma: float) -> np.ndarray:
        """Solve QP using CVXPY (preferred for robustness).
        
        Args:
            sigma: Covariance matrix
            gamma: Risk aversion parameter
            
        Returns:
            Optimal weights
        """
        p = len(sigma)
        w = cp.Variable(p)
        
        # Build objective: (γ/2)w'Σw + λ||w||²
        objective = (gamma / 2.0) * cp.quad_form(w, sigma)
        if self.config.l2_lambda > 0:
            objective += self.config.l2_lambda * cp.sum_squares(w)
        
        # Constraints: budget constraint always active
        constraints = [cp.sum(w) == 1]
        
        # Box constraints (long-only and/or caps)
        if self.config.long_only:
            constraints.append(w >= 0)
            if self.config.weight_cap is not None:
                constraints.append(w <= self.config.weight_cap)
        
        # Solve with fallback between solvers
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        for solver in [cp.OSQP, cp.ECOS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if w.value is not None and problem.status in {
                    "optimal", "optimal_inaccurate"
                }:
                    return np.asarray(w.value).flatten()
            except Exception:
                continue
        
        raise RuntimeError(f"CVXPY failed: {problem.status}")
    
    def _solve_with_scipy(
        self,
        sigma: np.ndarray,
        gamma: float,
        objective_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Solve QP using scipy trust-constr (fallback).
        
        Args:
            sigma: Covariance matrix
            gamma: Risk aversion parameter
            objective_fn: Optional custom objective function
            
        Returns:
            Optimal weights
        """
        p = len(sigma)
        
        # Define objective and gradient
        if objective_fn is None:
            def objective_fn(w):
                return (0.5 * gamma * w @ sigma @ w + 
                        self.config.l2_lambda * w @ w)
        
        def gradient_fn(w):
            return gamma * sigma @ w + 2 * self.config.l2_lambda * w
        
        # Equality constraint: w'1 = 1
        budget_constraint = LinearConstraint(np.ones(p), lb=1.0, ub=1.0)
        
        # Box bounds
        if self.config.long_only:
            lower = np.zeros(p)
            upper = (np.full(p, self.config.weight_cap) 
                    if self.config.weight_cap else np.full(p, np.inf))
        else:
            lower = np.full(p, -np.inf)
            upper = np.full(p, np.inf)
        
        # Solve from equal-weight starting point
        result = minimize(
            objective_fn,
            x0=np.full(p, 1.0 / p),
            jac=gradient_fn,
            method="trust-constr",
            constraints=[budget_constraint],
            bounds=Bounds(lower, upper),
            options={"maxiter": 10_000, "gtol": 1e-10}
        )
        
        if not result.success:
            raise RuntimeError(f"Scipy optimization failed: {result.message}")
        
        return result.x
    
    def solve_gmv(self, sigma: np.ndarray) -> np.ndarray:
        """Solve Global Minimum Variance portfolio.
        
        Finds weights that minimize portfolio variance w'Σw subject to
        budget constraint and any specified box constraints.
        
        Args:
            sigma: Asset covariance matrix
            
        Returns:
            Optimal GMV weights (1D array summing to 1)
        """
        sigma = self._prepare_covariance(sigma)
        
        # Use analytical solution if unconstrained
        if (not self.config.long_only and 
            self.config.weight_cap is None and 
            self.config.allow_short and
            self.config.l2_lambda == 0):
            return self._gmv_analytical(sigma)
        
        # Otherwise solve constrained QP
        return self._solve_qp(sigma, gamma=1.0)
    
    def solve_target_vol(
        self,
        sigma: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Solve target volatility portfolio via gamma bisection.
        
        Finds portfolio satisfying: minimize (γ/2)w'Σw subject to
        √(w'Σw) ≈ target_vol, using bisection on risk aversion γ.
        
        Higher γ → more risk penalty → lower volatility
        Lower γ → less risk penalty → higher volatility
        
        Args:
            sigma: Asset covariance matrix
            
        Returns:
            Tuple of (weights, achieved_vol, gamma_used)
        """
        sigma = self._prepare_covariance(sigma)
        
        def compute_portfolio_vol(gamma: float) -> Tuple[np.ndarray, float]:
            """Helper: solve for given gamma and return (weights, vol)."""
            weights = self._solve_qp(sigma, gamma)
            variance = float(weights @ sigma @ weights)
            volatility = math.sqrt(max(variance, 0.0))
            return weights, volatility
        
        # Initialize bisection bounds
        lo, hi = self.config.gamma_min, self.config.gamma_max
        w_lo, vol_lo = compute_portfolio_vol(lo)
        w_hi, vol_hi = compute_portfolio_vol(hi)
        
        # Ensure monotonicity: higher gamma → lower vol
        # If violated, swap endpoints
        if vol_hi > vol_lo:
            lo, hi, w_lo, vol_lo, w_hi, vol_hi = (
                hi, lo, w_hi, vol_hi, w_lo, vol_lo
            )
        
        # Check if target already achieved at boundaries
        for w, vol, g in [(w_lo, vol_lo, lo), (w_hi, vol_hi, hi)]:
            if abs(vol - self.config.target_vol) <= self.config.vol_tol:
                return w, vol, g
        
        # Track best solution found
        best = min(
            [(w_lo, vol_lo, lo), (w_hi, vol_hi, hi)],
            key=lambda x: abs(x[1] - self.config.target_vol)
        )
        
        # Bisection search
        for _ in range(self.config.max_bisect_iters):
            # Use geometric mean for log-scale parameter
            mid = math.sqrt(lo * hi)
            w_mid, vol_mid = compute_portfolio_vol(mid)
            
            # Update best if improved
            if abs(vol_mid - self.config.target_vol) < abs(best[1] - self.config.target_vol):
                best = (w_mid, vol_mid, mid)
            
            # Check convergence
            if abs(vol_mid - self.config.target_vol) <= self.config.vol_tol:
                return w_mid, vol_mid, mid
            
            # Update bounds: need higher vol → lower gamma
            if vol_mid > self.config.target_vol:
                lo = mid
            else:
                hi = mid
            
            # Check gamma convergence
            if (hi / lo) < (1.0 + self.config.gamma_tol):
                break
        
        return best


# ============================================================================
# Pipeline Orchestration
# ============================================================================

class PortfolioConstructor:
    """High-level pipeline for batch portfolio construction.
    
    Processes covariance matrices from input directory tree and writes
    optimal portfolio weights to output directory tree, preserving the
    folder structure (e.g., raw/, LW/, PCA/, JSE/).
    """
    
    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        config: Optional[PortfolioConfig] = None
    ):
        """Initialize constructor with I/O paths and configuration.
        
        Args:
            input_root: Root directory containing method folders with CSVs
            output_root: Root directory for writing portfolio weights
            config: Optimization configuration (uses defaults if None)
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.config = config or PortfolioConfig()
        self.optimizer = PortfolioOptimizer(self.config)
        self.loader = CovarianceLoader()
    
    def _save_weights(self, permnos: np.ndarray, weights: np.ndarray, path: Path):
        """Save portfolio weights to CSV with columns [permno, weight].
        
        Args:
            permnos: Asset identifiers
            weights: Portfolio weights
            path: Output CSV path (parent directories created if needed)
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"permno": permnos, "weight": weights})
        df.to_csv(path, index=False)
    
    def process_method(
        self,
        method_name: str,
        file_suffix: str = "_full_cov.csv"
    ):
        """Process all covariance files for one estimation method.
        
        Reads all *{file_suffix} files from input_root/method_name/,
        computes optimal portfolios, and writes weights to
        output_root/method_name/PortfolioA_GMV/.
        
        Args:
            method_name: Folder name (e.g., 'LW', 'PCA', 'JSE')
            file_suffix: File pattern to match (default: '_full_cov.csv')
        """
        method_dir = self.input_root / method_name
        cov_files = sorted(method_dir.glob(f"*{file_suffix}"))
        
        if not cov_files:
            print(f"[{method_name}] No files found in {method_dir}")
            return
        
        # Prepare output directories
        gmv_dir = self.output_root / method_name / "PortfolioA_GMV"
        
        # Process each date
        for cov_file in cov_files:
            try:
                # Extract date identifier from filename
                # Remove the suffix to get the date part
                date_str = cov_file.stem  # Gets filename without extension
                if file_suffix.startswith("_"):
                    # Remove the suffix pattern from the stem
                    suffix_without_ext = file_suffix.replace(".csv", "")
                    date_str = date_str.replace(suffix_without_ext, "")
                
                # Load data
                permnos, sigma = self.loader.load(cov_file)
                
                # Optimize portfolio
                weights = self.optimizer.solve_gmv(sigma)
                
                # Save results
                self._save_weights(
                    permnos, 
                    weights, 
                    gmv_dir / f"{date_str}_weights.csv"
                )
                
                print(f"[{method_name}] {date_str}: GMV completed")
                
            except Exception as e:
                print(
                    f"[{method_name}] ERROR on {cov_file.name}: {e}",
                    file=sys.stderr
                )
    
    def run(self, methods: list[str]):
        """Process multiple estimation methods in sequence.
        
        Args:
            methods: List of method folder names to process
        """
        for method in methods:
            self.process_method(method)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Command-line entry point with example configuration."""
    config = PortfolioConfig(
        long_only=True,
        weight_cap=None,
        allow_short=False
    )
    
    constructor = PortfolioConstructor(
        input_root=Path("covariance_outputs_sim"),
        output_root=Path("portfolio_outputs_sim"),
        config=config
    )
    
    constructor.run(["LW", "PCA", "JSE"])


if __name__ == "__main__":
    main()
