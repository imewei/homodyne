"""
Tests for Direct Classical Least Squares Solver
===============================================

Comprehensive test suite for the JAX-accelerated direct solver.
Tests accuracy, performance, and numerical stability across
different dataset sizes and conditions.
"""

import time
import unittest

import numpy as np

# Handle optional dependencies
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from homodyne.core.fitting import DatasetSize, FitResult
from homodyne.optimization.direct_solver import (
    DirectLeastSquaresSolver,
    DirectSolverConfig,
    fit_homodyne_direct,
)


class TestDirectSolver(unittest.TestCase):
    """Test suite for DirectLeastSquaresSolver."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.solver = DirectLeastSquaresSolver()

    def test_small_dataset_accuracy(self):
        """Test accuracy on small datasets."""
        # Generate synthetic data
        n_samples = 1000
        theory = np.random.randn(n_samples)
        contrast_true = 0.5
        offset_true = 1.0
        noise = 0.01 * np.random.randn(n_samples)
        data = theory * contrast_true + offset_true + noise

        # Fit using direct solver
        result = self.solver.fit(data, theory)

        # Check accuracy
        self.assertAlmostEqual(result.contrast, contrast_true, places=2)
        self.assertAlmostEqual(result.offset, offset_true, places=2)
        self.assertTrue(result.converged)
        self.assertEqual(result.dataset_size, DatasetSize.SMALL)
        self.assertIsInstance(result, FitResult)

    def test_medium_dataset_chunking(self):
        """Test chunked processing for medium datasets."""
        # Generate medium-sized data
        n_samples = 5_000_000
        theory = np.random.randn(n_samples).astype(np.float32)
        contrast_true = 0.3
        offset_true = 0.8
        data = theory * contrast_true + offset_true

        # Fit with chunking
        result = fit_homodyne_direct(data, theory)

        # Check accuracy - tighter tolerance for noiseless data
        self.assertAlmostEqual(result.contrast, contrast_true, places=5)
        self.assertAlmostEqual(result.offset, offset_true, places=5)
        self.assertEqual(result.dataset_size, DatasetSize.MEDIUM)

    def test_large_dataset_handling(self):
        """Test handling of large datasets with chunking."""
        # Generate large dataset (but manageable for testing)
        n_samples = 11_000_000  # Just over 10M threshold
        theory = np.random.randn(n_samples).astype(np.float32)
        contrast_true = 0.7
        offset_true = 0.3
        data = theory * contrast_true + offset_true

        # Fit with large dataset optimizations
        solver = DirectLeastSquaresSolver()
        result = solver.fit(data, theory)

        # Check results
        self.assertAlmostEqual(result.contrast, contrast_true, places=4)
        self.assertAlmostEqual(result.offset, offset_true, places=4)
        self.assertEqual(result.dataset_size, DatasetSize.LARGE)

    def test_weighted_least_squares(self):
        """Test weighted least squares with sigma."""
        n_samples = 10000
        theory = np.random.randn(n_samples)
        contrast_true = 0.6
        offset_true = 0.4

        # Generate data with heteroscedastic noise
        sigma = np.abs(np.random.randn(n_samples)) + 0.1
        noise = sigma * np.random.randn(n_samples)
        data = theory * contrast_true + offset_true + noise

        # Fit with weights
        result = self.solver.fit(data, theory, sigma=sigma)

        # Check that weighted fit is reasonable
        self.assertAlmostEqual(result.contrast, contrast_true, places=1)
        self.assertAlmostEqual(result.offset, offset_true, places=1)
        self.assertTrue(result.chi_squared > 0)
        self.assertTrue(result.reduced_chi_squared > 0)

    def test_ill_conditioned_matrix(self):
        """Test handling of ill-conditioned matrices."""
        # Create ill-conditioned problem (nearly constant theory)
        n_samples = 100
        theory = np.ones(n_samples) + 1e-10 * np.random.randn(n_samples)
        data = np.random.randn(n_samples)

        # Should not raise error
        result = self.solver.fit(data, theory)

        # Check that result is valid
        self.assertTrue(result.converged)
        self.assertTrue(np.isfinite(result.contrast))
        self.assertTrue(np.isfinite(result.offset))

    def test_parameter_bounds_enforcement(self):
        """Test that parameter bounds are enforced."""
        n_samples = 1000
        theory = np.random.randn(n_samples)

        # Generate data that would lead to out-of-bounds parameters
        data = theory * 10.0 - 5.0  # Would give contrast=10, offset=-5

        # Fit with bounds enforcement
        result = self.solver.fit(data, theory)

        # Check bounds are respected
        param_space = self.solver.parameter_space
        self.assertGreaterEqual(result.contrast, param_space.contrast_bounds[0])
        self.assertLessEqual(result.contrast, param_space.contrast_bounds[1])
        self.assertGreaterEqual(result.offset, param_space.offset_bounds[0])
        self.assertLessEqual(result.offset, param_space.offset_bounds[1])

    def test_jax_numpy_consistency(self):
        """Test that JAX and NumPy implementations give same results."""
        # Test data
        n_samples = 10000
        theory = np.random.randn(n_samples)
        data = 0.4 * theory + 0.6 + 0.05 * np.random.randn(n_samples)

        # Create solvers
        solver_with_jax = DirectLeastSquaresSolver()

        # Temporarily disable JAX for comparison
        import homodyne.optimization.direct_solver as ds

        original_jax = ds.JAX_AVAILABLE

        # Test with JAX (if available)
        if original_jax:
            ds.JAX_AVAILABLE = True
            solver_jax = DirectLeastSquaresSolver()
            result_jax = solver_jax.fit(data, theory)

            # Test without JAX
            ds.JAX_AVAILABLE = False
            solver_numpy = DirectLeastSquaresSolver()
            result_numpy = solver_numpy.fit(data, theory)

            # Restore original state
            ds.JAX_AVAILABLE = original_jax

            # Compare results
            self.assertAlmostEqual(result_jax.contrast, result_numpy.contrast, places=6)
            self.assertAlmostEqual(result_jax.offset, result_numpy.offset, places=6)
        else:
            # If JAX not available, just check NumPy works
            result = solver_with_jax.fit(data, theory)
            self.assertTrue(result.converged)

    def test_different_analysis_modes(self):
        """Test solver works with different analysis modes."""
        n_samples = 1000
        theory = np.random.randn(n_samples)
        data = 0.5 * theory + 1.0

        # Test each analysis mode
        modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]

        for mode in modes:
            with self.subTest(mode=mode):
                solver = DirectLeastSquaresSolver(analysis_mode=mode)
                result = solver.fit(data, theory)
                self.assertTrue(result.converged)
                self.assertEqual(result.analysis_mode, mode)

    def test_fit_result_statistics(self):
        """Test that fit statistics are computed correctly."""
        n_samples = 10000
        theory = np.random.randn(n_samples)
        contrast_true = 0.5
        offset_true = 1.0
        sigma = 0.1
        noise = sigma * np.random.randn(n_samples)
        data = theory * contrast_true + offset_true + noise

        # Fit and check statistics
        result = self.solver.fit(data, theory)

        # Check statistics are reasonable
        self.assertTrue(result.chi_squared > 0)
        self.assertTrue(result.reduced_chi_squared > 0)
        self.assertEqual(result.degrees_of_freedom, n_samples - 2)
        self.assertAlmostEqual(result.residual_std, sigma, places=1)
        self.assertTrue(result.max_residual > 0)
        self.assertEqual(result.fit_iterations, 1)  # Direct solution

    def test_config_customization(self):
        """Test solver with custom configuration."""
        config = DirectSolverConfig(
            chunk_size_small=50000,
            chunk_size_medium=5000,
            chunk_size_large=500,
            regularization=1e-8,
            condition_threshold=1e8,
            use_parameter_space_bounds=False,
        )

        solver = DirectLeastSquaresSolver(config=config)

        # Test that config is applied
        self.assertEqual(solver.config.chunk_size_small, 50000)
        self.assertEqual(solver.config.regularization, 1e-8)
        self.assertFalse(solver.config.use_parameter_space_bounds)

        # Test fitting works with custom config
        n_samples = 1000
        theory = np.random.randn(n_samples)
        data = 0.5 * theory + 1.0

        result = solver.fit(data, theory)
        self.assertTrue(result.converged)

    def test_benchmark_functionality(self):
        """Test the benchmark method works."""
        # Run quick benchmark with small sizes
        data_sizes = [100, 1000]
        results = self.solver.benchmark(data_sizes=data_sizes, n_trials=2)

        # Check results structure
        self.assertEqual(len(results), len(data_sizes))
        for size in data_sizes:
            self.assertIn(size, results)
            self.assertIn("mean_time", results[size])
            self.assertIn("std_time", results[size])
            self.assertIn("backend", results[size])
            self.assertTrue(results[size]["mean_time"] > 0)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with single data point
        with self.subTest("single point"):
            data = np.array([1.0])
            theory = np.array([2.0])
            result = self.solver.fit(data, theory)
            self.assertTrue(result.converged)

        # Test with zero theory
        with self.subTest("zero theory"):
            n_samples = 100
            data = np.random.randn(n_samples)
            theory = np.zeros(n_samples)
            result = self.solver.fit(data, theory)
            self.assertTrue(result.converged)

        # Test with perfect fit (no noise)
        with self.subTest("perfect fit"):
            n_samples = 100
            theory = np.random.randn(n_samples)
            data = 0.5 * theory + 1.0
            result = self.solver.fit(data, theory)
            self.assertAlmostEqual(result.contrast, 0.5, places=10)
            self.assertAlmostEqual(result.offset, 1.0, places=10)
            self.assertAlmostEqual(result.chi_squared, 0.0, places=10)

    def test_performance_comparison(self):
        """Compare performance with numpy.linalg.lstsq."""
        if not JAX_AVAILABLE:
            self.skipTest("JAX not available for performance comparison")

        n_samples = 100000
        theory = np.random.randn(n_samples).astype(np.float32)
        data = 0.5 * theory + 1.0 + 0.01 * np.random.randn(n_samples)

        # Time JAX solver
        start = time.time()
        result_jax = self.solver.fit(data, theory)
        time_jax = time.time() - start

        # Time NumPy lstsq
        start = time.time()
        A = np.column_stack([theory, np.ones(n_samples)])
        params_numpy, _, _, _ = np.linalg.lstsq(A, data, rcond=None)
        time_numpy = time.time() - start

        # JAX should be competitive or faster
        print(f"JAX time: {time_jax:.4f}s, NumPy time: {time_numpy:.4f}s")
        print(f"Speedup: {time_numpy / time_jax:.2f}x")

        # Check results match
        self.assertAlmostEqual(result_jax.contrast, params_numpy[0], places=5)
        self.assertAlmostEqual(result_jax.offset, params_numpy[1], places=5)


class TestGeneralLeastSquares(unittest.TestCase):
    """Test general N-parameter least squares functions."""

    def test_general_solver_2d(self):
        """Test general solver with 2 parameters (contrast/offset)."""
        from homodyne.core.fitting import solve_least_squares_general_jax

        n_samples = 1000
        design_matrix = np.column_stack(
            [np.random.randn(n_samples), np.ones(n_samples)]
        )
        params_true = np.array([0.5, 1.0])
        target = design_matrix @ params_true + 0.01 * np.random.randn(n_samples)

        if JAX_AVAILABLE:
            design_jax = jnp.array(design_matrix)
            target_jax = jnp.array(target)
            params_fit = solve_least_squares_general_jax(design_jax, target_jax)
            params_fit = np.array(params_fit)
        else:
            params_fit = solve_least_squares_general_jax(design_matrix, target)

        np.testing.assert_allclose(params_fit, params_true, atol=0.01)

    def test_general_solver_5d(self):
        """Test general solver with 5 parameters."""
        from homodyne.core.fitting import solve_least_squares_general_jax

        n_samples = 10000
        n_params = 5
        design_matrix = np.random.randn(n_samples, n_params)
        params_true = np.random.randn(n_params)
        target = design_matrix @ params_true + 0.01 * np.random.randn(n_samples)

        if JAX_AVAILABLE:
            design_jax = jnp.array(design_matrix)
            target_jax = jnp.array(target)
            params_fit = solve_least_squares_general_jax(design_jax, target_jax)
            params_fit = np.array(params_fit)
        else:
            params_fit = solve_least_squares_general_jax(design_matrix, target)

        np.testing.assert_allclose(params_fit, params_true, atol=0.01)

    def test_chunked_solver(self):
        """Test chunked solver for large datasets."""
        from homodyne.core.fitting import solve_least_squares_chunked_jax

        # Create chunked data
        n_chunks = 10
        chunk_size = 1000
        theory_chunks = []
        exp_chunks = []

        contrast_true = 0.6
        offset_true = 0.4

        for _ in range(n_chunks):
            theory_chunk = np.random.randn(chunk_size)
            exp_chunk = theory_chunk * contrast_true + offset_true
            theory_chunks.append(theory_chunk)
            exp_chunks.append(exp_chunk)

        if JAX_AVAILABLE:
            theory_jax = jnp.array(theory_chunks)
            exp_jax = jnp.array(exp_chunks)
            contrast, offset = solve_least_squares_chunked_jax(theory_jax, exp_jax)
            contrast = float(contrast)
            offset = float(offset)
        else:
            contrast, offset = solve_least_squares_chunked_jax(
                np.array(theory_chunks), np.array(exp_chunks)
            )

        self.assertAlmostEqual(contrast, contrast_true, places=5)
        self.assertAlmostEqual(offset, offset_true, places=5)


if __name__ == "__main__":
    unittest.main()