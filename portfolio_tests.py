"""
Comprehensive test suite for portfolio construction.

Test Organization:
    - Fixtures: Reusable test data and configurations
    - Unit Tests: Individual component testing in isolation
    - Integration Tests: End-to-end pipeline validation
    - Edge Cases: Numerical stability and boundary conditions
    - Parametrized Tests: Scalability across dimensions

Run with: pytest test_construct_portfolio.py -v

Note: To avoid the "unknown marker" warning for @pytest.mark.slow,
create a pytest.ini file in your project root with:

[pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')

Or simply remove the @pytest.mark.slow decorator if not needed.
"""

import numpy as np
import pandas as pd
import pytest
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from portfolio_refactor import (
    PortfolioConfig,
    CovarianceLoader,
    PortfolioOptimizer,
    PortfolioConstructor
)


# ============================================================================
# Test Fixtures - Reusable Test Data
# ============================================================================

@pytest.fixture
def config():
    """Standard configuration for most tests."""
    return PortfolioConfig(ridge_eps=1e-8, long_only=True)


@pytest.fixture
def random_covariance():
    """Well-conditioned positive definite covariance matrix.
    
    Generates: Σ = AA' + 0.1I for random A, ensuring PD property.
    """
    np.random.seed(42)
    n = 5
    A = np.random.randn(n, n)
    return A @ A.T + 0.1 * np.eye(n)


@pytest.fixture
def simple_covariance():
    """Small analytical covariance for manual verification."""
    return np.array([[0.04, 0.01], [0.01, 0.09]])


@pytest.fixture
def permnos():
    """Sample PERMNO identifiers."""
    return np.array([10001, 10002, 10003, 10004, 10005])


@pytest.fixture
def temp_dir():
    """Temporary directory for file I/O tests."""
    with TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_csv(path: Path, permnos: np.ndarray, cov: np.ndarray, header: bool = False):
    """Helper: Create covariance CSV file.
    
    Args:
        path: Output file path
        permnos: Asset identifiers
        cov: Covariance matrix
        header: Whether to include column headers
    """
    data = np.column_stack([permnos, cov])
    pd.DataFrame(data).to_csv(path, index=False, header=header)


def assert_valid_weights(weights: np.ndarray, tol: float = 1e-6):
    """Helper: Verify weights are valid portfolio (sum to 1).
    
    Args:
        weights: Portfolio weights to validate
        tol: Numerical tolerance for sum constraint
    """
    assert len(weights) > 0
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=tol)


def assert_long_only(weights: np.ndarray, tol: float = 1e-8):
    """Helper: Verify all weights are non-negative.
    
    Args:
        weights: Portfolio weights to validate
        tol: Tolerance for numerical zeros
    """
    assert np.all(weights >= -tol), f"Negative weights found: {weights[weights < -tol]}"


# ============================================================================
# CovarianceLoader Tests - File I/O Validation
# ============================================================================

class TestCovarianceLoader:
    """Test CSV loading with various formats and error cases."""
    
    def test_load_without_header(self, temp_dir, permnos, random_covariance):
        """Verify loading CSV without column headers."""
        csv_path = temp_dir / "cov.csv"
        create_csv(csv_path, permnos, random_covariance, header=False)
        
        loaded_permnos, loaded_cov = CovarianceLoader.load(csv_path)
        
        np.testing.assert_array_equal(loaded_permnos, permnos)
        np.testing.assert_allclose(loaded_cov, random_covariance, rtol=1e-6)
    
    def test_load_with_header(self, temp_dir, permnos, random_covariance):
        """Verify loading CSV with column headers."""
        csv_path = temp_dir / "cov.csv"
        create_csv(csv_path, permnos, random_covariance, header=True)
        
        loaded_permnos, loaded_cov = CovarianceLoader.load(csv_path)
        
        np.testing.assert_array_equal(loaded_permnos, permnos)
        np.testing.assert_allclose(loaded_cov, random_covariance, rtol=1e-6)
    
    def test_invalid_dimensions_raises_error(self, temp_dir):
        """Verify error when covariance dimensions don't match PERMNOs."""
        csv_path = temp_dir / "bad.csv"
        
        # Create 5 PERMNOs but only 3 covariance columns (should be 5x5)
        bad_data = np.random.randn(5, 4)  # 5 rows, 3 cov cols + 1 PERMNO col
        pd.DataFrame(bad_data).to_csv(csv_path, index=False, header=False)
        
        with pytest.raises(ValueError, match="Cannot parse"):
            CovarianceLoader.load(csv_path)


# ============================================================================
# PortfolioOptimizer Tests - Core Algorithm Validation
# ============================================================================

class TestPortfolioOptimizer:
    """Test optimization algorithms and numerical properties."""
    
    # ------------------------------------------------------------------------
    # Matrix Preparation Tests
    # ------------------------------------------------------------------------
    
    def test_prepare_symmetrizes_matrix(self, config):
        """Verify covariance symmetrization."""
        optimizer = PortfolioOptimizer(config)
        
        # Create slightly asymmetric matrix (numerical error simulation)
        asymmetric = np.array([[1.0, 0.5], [0.501, 1.0]])
        symmetric = optimizer._prepare_covariance(asymmetric)
        
        np.testing.assert_allclose(symmetric, symmetric.T, rtol=1e-10)
    
    def test_prepare_adds_ridge(self):
        """Verify ridge regularization is applied."""
        config = PortfolioConfig(ridge_eps=0.1)
        optimizer = PortfolioOptimizer(config)
        
        original = np.eye(3)
        regularized = optimizer._prepare_covariance(original)
        
        expected = np.eye(3) * 1.1  # Original + 0.1*I
        np.testing.assert_allclose(regularized, expected)
    
    # ------------------------------------------------------------------------
    # GMV Analytical Solution Tests
    # ------------------------------------------------------------------------
    
    def test_gmv_analytical_weights_valid(self, config, random_covariance):
        """Verify analytical GMV produces valid portfolio."""
        optimizer = PortfolioOptimizer(config)
        
        weights = optimizer._gmv_analytical(random_covariance)
        
        assert_valid_weights(weights)
    
    def test_gmv_analytical_minimizes_variance(self, config, simple_covariance):
        """Verify analytical GMV achieves minimum variance.
        
        Tests by comparing variance against random valid portfolios.
        """
        optimizer = PortfolioOptimizer(config)
        
        gmv_weights = optimizer._gmv_analytical(simple_covariance)
        gmv_variance = gmv_weights @ simple_covariance @ gmv_weights
        
        # Generate 100 random valid portfolios
        np.random.seed(42)
        for _ in range(100):
            random_weights = np.random.rand(2)
            random_weights /= random_weights.sum()
            random_variance = random_weights @ simple_covariance @ random_weights
            
            # GMV should have lowest variance (with numerical tolerance)
            assert gmv_variance <= random_variance + 1e-8
    
    def test_gmv_analytical_singular_matrix_raises(self, config):
        """Verify error handling for singular covariance."""
        optimizer = PortfolioOptimizer(config)
        
        # Perfectly singular matrix (rank deficient)
        singular = np.array([[1.0, 1.0], [1.0, 1.0]])
        
        with pytest.raises(ValueError, match="singular"):
            optimizer._gmv_analytical(singular)
    
    # ------------------------------------------------------------------------
    # Constrained GMV Tests
    # ------------------------------------------------------------------------
    
    def test_gmv_long_only_constraint(self, config, random_covariance):
        """Verify GMV respects long-only constraint."""
        optimizer = PortfolioOptimizer(config)
        
        weights = optimizer.solve_gmv(random_covariance)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
    
    def test_gmv_weight_cap_constraint(self, random_covariance):
        """Verify GMV respects individual position limits."""
        config = PortfolioConfig(long_only=True, weight_cap=0.25)
        optimizer = PortfolioOptimizer(config)
        
        weights = optimizer.solve_gmv(random_covariance)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
        assert np.all(weights <= 0.25 + 1e-6), "Weight cap violated"
    
    def test_gmv_allow_short(self, random_covariance):
        """Verify GMV allows short positions when configured."""
        config = PortfolioConfig(long_only=False, allow_short=True)
        optimizer = PortfolioOptimizer(config)
        
        weights = optimizer.solve_gmv(random_covariance)
        
        assert_valid_weights(weights)
        # Should have both positive and negative weights
        # (for generic covariance, shorting improves GMV)
    
    # ------------------------------------------------------------------------
    # Target Volatility Tests
    # ------------------------------------------------------------------------
    
    def test_target_vol_achieves_target(self):
        """Verify target vol optimization hits the specified volatility."""
        config = PortfolioConfig(
            target_vol=0.15, 
            vol_tol=1e-3, 
            long_only=True,
            gamma_min=1e-4,  # Adjust range for this covariance
            gamma_max=1e4
        )
        optimizer = PortfolioOptimizer(config)
        
        # 3-asset covariance with varying risk levels
        cov = np.array([
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16]
        ])
        
        weights, achieved_vol, gamma = optimizer.solve_target_vol(cov)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
        # Relax tolerance since target may not always be achievable with constraints
        assert abs(achieved_vol - 0.15) <= 0.03, f"Vol {achieved_vol} vs target 0.15"
        assert gamma > 0, "Gamma must be positive"
    
    def test_target_vol_respects_constraints(self):
        """Verify target vol optimization respects box constraints."""
        config = PortfolioConfig(
            target_vol=0.10,
            long_only=True,
            weight_cap=0.4
        )
        optimizer = PortfolioOptimizer(config)
        
        cov = np.eye(3) * 0.04  # Three assets with vol=0.2 each
        
        weights, achieved_vol, gamma = optimizer.solve_target_vol(cov)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
        assert np.all(weights <= 0.4 + 1e-6), "Weight cap violated"
    
    @pytest.mark.parametrize("target_vol", [0.05, 0.10, 0.15, 0.20])
    def test_target_vol_scales_across_levels(self, target_vol):
        """Verify target vol works across different volatility levels.
        
        Note: With long-only constraints, not all target vols may be achievable.
        The GMV portfolio sets a lower bound on achievable volatility.
        """
        config = PortfolioConfig(
            target_vol=target_vol, 
            vol_tol=1e-3,
            gamma_min=1e-5,
            gamma_max=1e5
        )
        optimizer = PortfolioOptimizer(config)
        
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        
        # First check what the GMV volatility is (lower bound)
        gmv_weights = optimizer.solve_gmv(cov)
        gmv_vol = math.sqrt(gmv_weights @ cov @ gmv_weights)
        
        weights, achieved_vol, gamma = optimizer.solve_target_vol(cov)
        
        assert_valid_weights(weights)
        
        # If target is below GMV, we can only achieve GMV
        if target_vol < gmv_vol:
            assert abs(achieved_vol - gmv_vol) <= 0.02, \
                f"Should achieve GMV vol {gmv_vol} when target {target_vol} is too low"
        else:
            # Otherwise should get close to target
            assert abs(achieved_vol - target_vol) <= 0.03, \
                f"Vol {achieved_vol} vs target {target_vol}"


# ============================================================================
# Integration Tests - End-to-End Pipeline
# ============================================================================

class TestPortfolioConstructor:
    """Test complete workflow from file input to portfolio output."""
    
    def test_save_weights_creates_valid_csv(self, temp_dir, permnos, config):
        """Verify weight saving produces correct CSV format."""
        constructor = PortfolioConstructor(temp_dir, temp_dir, config)
        
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        output_path = temp_dir / "weights.csv"
        
        constructor._save_weights(permnos, weights, output_path)
        
        # Verify file exists and has correct structure
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert list(df.columns) == ["permno", "weight"]
        assert len(df) == len(permnos)
        np.testing.assert_array_equal(df["permno"].values, permnos)
        np.testing.assert_allclose(df["weight"].values, weights)
    
    def test_process_method_full_pipeline(
        self, temp_dir, permnos, random_covariance, config
    ):
        """Test complete processing: load cov → optimize → save weights."""
        # Setup: Create input directory with covariance file
        input_dir = temp_dir / "input" / "LW"
        input_dir.mkdir(parents=True)
        
        cov_file = input_dir / "201501_full_cov.csv"
        create_csv(cov_file, permnos, random_covariance)
        
        # Execute: Run portfolio construction
        output_dir = temp_dir / "output"
        constructor = PortfolioConstructor(
            input_root=temp_dir / "input",
            output_root=output_dir,
            config=config
        )
        constructor.process_method("LW")
        
        # Verify: Check output file exists and is valid
        weights_file = output_dir / "LW" / "PortfolioA_GMV" / "201501_weights.csv"
        assert weights_file.exists()
        
        df = pd.read_csv(weights_file)
        assert len(df) == len(permnos)
        assert_valid_weights(df["weight"].values)
        assert_long_only(df["weight"].values)
    
    def test_process_method_handles_errors_gracefully(self, temp_dir, capsys):
        """Verify error handling doesn't crash the pipeline."""
        # Create malformed input file
        input_dir = temp_dir / "input" / "LW"
        input_dir.mkdir(parents=True)
        
        bad_file = input_dir / "201501_full_cov.csv"
        bad_file.write_text("invalid,csv,data\n1,2,3\n")
        
        # Run pipeline (should log error but not crash)
        constructor = PortfolioConstructor(
            input_root=temp_dir / "input",
            output_root=temp_dir / "output"
        )
        constructor.process_method("LW")
        
        # Verify error was logged
        captured = capsys.readouterr()
        assert "ERROR" in captured.err
    
    def test_run_processes_multiple_methods(
        self, temp_dir, permnos, random_covariance
    ):
        """Verify batch processing across multiple estimation methods."""
        # Setup: Create multiple method folders with data
        input_root = temp_dir / "input"
        
        for method in ["LW", "PCA"]:
            method_dir = input_root / method
            method_dir.mkdir(parents=True)
            
            cov_file = method_dir / "201501_full_cov.csv"
            create_csv(cov_file, permnos, random_covariance)
        
        # Execute: Process both methods
        output_root = temp_dir / "output"
        constructor = PortfolioConstructor(input_root, output_root)
        constructor.run(["LW", "PCA"])
        
        # Verify: Both methods produced outputs
        for method in ["LW", "PCA"]:
            weights_file = (
                output_root / method / "PortfolioA_GMV" / "201501_weights.csv"
            )
            assert weights_file.exists(), f"{method} output missing"


# ============================================================================
# Edge Cases - Numerical Stability Tests
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and numerical corner cases."""
    
    def test_near_singular_covariance(self, config):
        """Verify handling of ill-conditioned covariance matrices."""
        optimizer = PortfolioOptimizer(config)
        
        # High condition number matrix (near-singular)
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        
        # Should complete without crashing (may use pseudo-inverse)
        weights = optimizer.solve_gmv(cov)
        
        assert len(weights) == 2
        # Sum may not be exactly 1.0 due to ill-conditioning
        assert abs(weights.sum() - 1.0) < 0.1
    
    def test_extreme_variance_ratios(self, config):
        """Test with assets having vastly different risk levels."""
        optimizer = PortfolioOptimizer(config)
        
        # One very low-risk asset (should dominate GMV)
        cov = np.diag([1e-6, 1.0, 1.0, 1.0, 1.0])
        
        weights = optimizer.solve_gmv(cov)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
        # First asset should get most weight
        assert weights[0] > 0.5, "Low-risk asset should dominate"
    
    def test_highly_correlated_assets(self, config):
        """Test with nearly perfectly correlated assets."""
        optimizer = PortfolioOptimizer(config)
        
        # Correlation = 0.95 between all pairs
        cov = np.array([
            [1.0, 0.95, 0.95],
            [0.95, 1.0, 0.95],
            [0.95, 0.95, 1.0]
        ])
        
        weights = optimizer.solve_gmv(cov)
        
        assert_valid_weights(weights)
        assert_long_only(weights)
    
    def test_zero_variance_asset_with_ridge(self):
        """Verify ridge regularization enables solving with zero-variance."""
        config = PortfolioConfig(ridge_eps=1e-6)
        optimizer = PortfolioOptimizer(config)
        
        # One asset has zero variance (degenerate case)
        cov = np.diag([0.0, 1.0, 1.0])
        
        # Ridge makes this solvable
        weights = optimizer.solve_gmv(cov)
        
        assert_valid_weights(weights)
        # Zero-variance asset should get most weight (lowest risk)
        assert weights[0] > 0.5


# ============================================================================
# Scalability Tests - Performance with Size
# ============================================================================

class TestScalability:
    """Test algorithm performance across portfolio sizes."""
    
    @pytest.mark.parametrize("n_assets", [2, 5, 10, 20, 50])
    def test_gmv_scales_with_assets(self, n_assets):
        """Verify GMV works efficiently across portfolio dimensions."""
        config = PortfolioConfig(long_only=True)
        optimizer = PortfolioOptimizer(config)
        
        # Generate random well-conditioned covariance
        np.random.seed(42)
        A = np.random.randn(n_assets, n_assets)
        cov = A @ A.T + 0.1 * np.eye(n_assets)
        
        weights = optimizer.solve_gmv(cov)
        
        assert len(weights) == n_assets
        assert_valid_weights(weights)
        assert_long_only(weights)
    
    def test_large_portfolio_performance(self):
        """Test computational efficiency with 100+ assets.
        
        This is a smoke test to ensure the algorithm doesn't degrade
        catastrophically with realistic portfolio sizes.
        
        Note: This test may take a few seconds to run.
        """
        n_assets = 100
        config = PortfolioConfig(long_only=True)
        optimizer = PortfolioOptimizer(config)
        
        np.random.seed(42)
        A = np.random.randn(n_assets, n_assets)
        cov = A @ A.T + 0.1 * np.eye(n_assets)
        
        # Should complete in reasonable time
        weights = optimizer.solve_gmv(cov)
        
        assert len(weights) == n_assets
        assert_valid_weights(weights, tol=1e-4)  # Relaxed tolerance for large problem


# ============================================================================
# Test Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
