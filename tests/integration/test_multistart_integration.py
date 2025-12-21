"""Integration tests for multi-start NLSQ optimization.

Tests cover:
- Full multi-start workflow with synthetic data (T011)
- Subsample strategy end-to-end (T039)
- Custom starting points inclusion
- Result reproducibility with fixed seeds
- Degeneracy detection in practice
"""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.nlsq.multistart import (
    MultiStartConfig,
    MultiStartResult,
    MultiStartStrategy,
    SingleStartResult,
    create_stratified_subsample,
    run_multistart_nlsq,
    select_multistart_strategy,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def synthetic_quadratic_data():
    """Create synthetic data for a simple quadratic function.

    The function is: f(x) = sum_i (params[i] - targets[i])^2
    This has a unique global minimum at params = targets.
    """
    np.random.seed(42)
    n_points = 1000
    n_angles = 3

    # Create synthetic phi angles
    phi = np.repeat([0.0, 30.0, 60.0], n_points // n_angles)

    # Pad to exact length
    if len(phi) < n_points:
        phi = np.concatenate([phi, np.full(n_points - len(phi), 0.0)])

    data = {
        "phi": phi,
        "g2": np.random.randn(n_points),
        "t1": np.arange(n_points, dtype=np.float64),
        "t2": np.arange(n_points, dtype=np.float64),
    }

    return data


@pytest.fixture
def simple_bounds():
    """Simple 3-parameter bounds."""
    return np.array(
        [
            [0.0, 10.0],  # param 0
            [0.0, 10.0],  # param 1
            [0.0, 10.0],  # param 2
        ]
    )


@pytest.fixture
def multistart_config():
    """Default multi-start configuration for tests."""
    return MultiStartConfig(
        enable=True,
        n_starts=5,
        seed=42,
        sampling_strategy="latin_hypercube",
        n_workers=1,  # Use single worker for deterministic testing
        use_screening=False,
        screen_keep_fraction=0.5,
    )


def create_quadratic_fit_function(target_params: np.ndarray):
    """Create a simple fit function that finds the minimum of a quadratic.

    The objective is: sum_i (params[i] - target[i])^2
    """

    def single_fit_func(data, initial_params):
        # Simple gradient descent to find minimum
        params = initial_params.copy()
        learning_rate = 0.1

        for _ in range(100):
            gradient = 2 * (params - target_params)
            params = params - learning_rate * gradient

        chi_squared = float(np.sum((params - target_params) ** 2))

        return SingleStartResult(
            start_idx=0,
            initial_params=initial_params,
            final_params=params,
            chi_squared=chi_squared,
            success=True,
            n_iterations=100,
        )

    return single_fit_func


# =============================================================================
# T011: Integration test for full multi-start workflow
# =============================================================================


class TestFullMultiStartWorkflow:
    """Integration tests for the complete multi-start workflow."""

    def test_full_workflow_finds_global_minimum(
        self, synthetic_quadratic_data, simple_bounds, multistart_config
    ):
        """Test that multi-start finds the global minimum."""
        target = np.array([3.0, 5.0, 7.0])
        fit_func = create_quadratic_fit_function(target)

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=multistart_config,
            single_fit_func=fit_func,
        )

        # Check structure
        assert isinstance(result, MultiStartResult)
        assert result.best is not None
        assert len(result.all_results) == multistart_config.n_starts

        # Check convergence to target
        np.testing.assert_allclose(result.best.final_params, target, rtol=0.1)
        assert result.best.chi_squared < 0.1

        # Check metadata
        assert result.n_successful == multistart_config.n_starts
        assert result.strategy_used == "full"
        assert result.total_wall_time > 0

    def test_full_workflow_reproducibility(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that results are reproducible with the same seed."""
        target = np.array([3.0, 5.0, 7.0])
        fit_func = create_quadratic_fit_function(target)

        config1 = MultiStartConfig(enable=True, n_starts=3, seed=42, n_workers=1)
        config2 = MultiStartConfig(enable=True, n_starts=3, seed=42, n_workers=1)

        result1 = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config1,
            single_fit_func=fit_func,
        )

        result2 = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config2,
            single_fit_func=fit_func,
        )

        # Same starting points should give same results
        np.testing.assert_allclose(
            result1.best.final_params, result2.best.final_params, rtol=1e-6
        )
        np.testing.assert_allclose(
            result1.best.chi_squared, result2.best.chi_squared, rtol=1e-6
        )

    def test_full_workflow_different_seeds(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that different seeds give different starting points."""
        target = np.array([3.0, 5.0, 7.0])
        fit_func = create_quadratic_fit_function(target)

        config1 = MultiStartConfig(enable=True, n_starts=3, seed=42, n_workers=1)
        config2 = MultiStartConfig(enable=True, n_starts=3, seed=123, n_workers=1)

        result1 = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config1,
            single_fit_func=fit_func,
        )

        result2 = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config2,
            single_fit_func=fit_func,
        )

        # Different seeds should give different initial params
        # (but both should converge to same final params)
        all_init_1 = [r.initial_params for r in result1.all_results]
        all_init_2 = [r.initial_params for r in result2.all_results]

        # At least one starting point should differ
        differs = False
        for p1, p2 in zip(all_init_1, all_init_2, strict=False):
            if not np.allclose(p1, p2):
                differs = True
                break
        assert differs, "Different seeds should produce different starting points"

    def test_full_workflow_handles_failed_starts(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that the workflow handles failed optimization attempts."""

        def sometimes_fails(data, initial_params):
            # Fail if first param < 2
            if initial_params[0] < 2:
                return SingleStartResult(
                    start_idx=0,
                    initial_params=initial_params,
                    final_params=initial_params,
                    chi_squared=np.inf,
                    success=False,
                    message="Simulated failure",
                )
            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=np.array([5.0, 5.0, 5.0]),
                chi_squared=1.0,
                success=True,
            )

        config = MultiStartConfig(enable=True, n_starts=10, seed=42, n_workers=1)

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=sometimes_fails,
        )

        # Should have some successes and some failures
        assert result.n_successful < config.n_starts
        assert result.n_successful > 0
        assert result.best.success

    def test_full_workflow_with_custom_starts(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that custom starting points are included."""
        target = np.array([3.0, 5.0, 7.0])
        fit_func = create_quadratic_fit_function(target)

        custom_starts = [[3.1, 4.9, 7.1], [2.9, 5.1, 6.9]]  # Close to target

        config = MultiStartConfig(
            enable=True,
            n_starts=3,
            seed=42,
            n_workers=1,
            custom_starts=custom_starts,
        )

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=fit_func,
        )

        # Should have 5 total results (3 generated + 2 custom)
        assert len(result.all_results) == 5

        # Custom starts (first 2) should be very close to target
        for i in range(2):
            assert result.all_results[i].chi_squared < 0.1


# =============================================================================
# T039: Integration test for subsample strategy end-to-end
# =============================================================================


class TestSubsampleStrategy:
    """Integration tests for the subsample multi-start strategy."""

    @pytest.fixture
    def large_synthetic_data(self):
        """Create synthetic data for subsample strategy testing.

        Creates 2M points to trigger subsample strategy (threshold is 1M).
        """
        np.random.seed(42)
        n_points = 2_000_000
        n_angles = 4

        # Create phi angles with even distribution
        angles = [0.0, 30.0, 60.0, 90.0]
        points_per_angle = n_points // n_angles
        phi = np.concatenate([np.full(points_per_angle, a) for a in angles])

        data = {
            "phi": phi,
            "g2": np.random.randn(n_points),
            "t1": np.arange(n_points, dtype=np.float64),
            "t2": np.arange(n_points, dtype=np.float64),
        }

        return data

    def test_strategy_selection_for_large_dataset(self, large_synthetic_data):
        """Test that subsample strategy is selected for large datasets."""
        config = MultiStartConfig(enable=True)

        strategy = select_multistart_strategy(len(large_synthetic_data["phi"]), config)

        assert strategy == MultiStartStrategy.SUBSAMPLE

    def test_stratified_subsample_preserves_angles(self, large_synthetic_data):
        """Test that stratified subsampling preserves angle distribution."""
        target_size = 100_000  # 100K subsample

        subsample = create_stratified_subsample(
            large_synthetic_data, target_size=target_size, seed=42
        )

        # Check size is approximately target
        actual_size = len(subsample["phi"])
        assert actual_size >= target_size * 0.9
        assert actual_size <= target_size

        # Check angle distribution is preserved
        original_angles = np.unique(large_synthetic_data["phi"])
        subsample_angles = np.unique(subsample["phi"])
        np.testing.assert_array_equal(original_angles, subsample_angles)

        # Check proportions are approximately equal
        for angle in original_angles:
            original_frac = np.mean(large_synthetic_data["phi"] == angle)
            subsample_frac = np.mean(subsample["phi"] == angle)
            # Within 10% of original proportion
            assert abs(original_frac - subsample_frac) < 0.1

    def test_subsample_strategy_end_to_end(self, large_synthetic_data, simple_bounds):
        """Test the complete subsample strategy workflow."""
        target = np.array([3.0, 5.0, 7.0])
        fit_func = create_quadratic_fit_function(target)

        config = MultiStartConfig(
            enable=True,
            n_starts=3,
            seed=42,
            n_workers=1,
            subsample_size=50_000,  # Small subsample for faster tests
        )

        result = run_multistart_nlsq(
            data=large_synthetic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=fit_func,
        )

        # Check strategy was used
        assert result.strategy_used == "subsample"

        # Check result quality
        assert result.best.success
        np.testing.assert_allclose(result.best.final_params, target, rtol=0.1)

        # Should have n_starts + 1 results (subsample results + final full fit)
        assert len(result.all_results) == config.n_starts + 1

        # Last result should be the final refinement (start_idx = -1)
        assert result.all_results[-1].start_idx == -1


# =============================================================================
# Degeneracy Detection Tests
# =============================================================================


class TestDegeneracyDetection:
    """Integration tests for parameter degeneracy detection."""

    def test_detects_degeneracy_with_multiple_minima(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that degeneracy is detected when multiple minima exist."""

        def multi_minimum_fit(data, initial_params):
            # Two minima: at [2, 2, 2] and [8, 8, 8]
            target1 = np.array([2.0, 2.0, 2.0])
            target2 = np.array([8.0, 8.0, 8.0])

            dist1 = np.linalg.norm(initial_params - target1)
            dist2 = np.linalg.norm(initial_params - target2)

            if dist1 < dist2:
                final = target1 + np.random.randn(3) * 0.01
            else:
                final = target2 + np.random.randn(3) * 0.01

            return SingleStartResult(
                start_idx=0,
                initial_params=initial_params,
                final_params=final,
                chi_squared=100.0 + np.random.randn() * 0.1,  # Similar chi-squared
                success=True,
            )

        config = MultiStartConfig(
            enable=True,
            n_starts=10,
            seed=42,
            n_workers=1,
            degeneracy_threshold=0.1,  # 10% chi-squared threshold
        )

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=multi_minimum_fit,
        )

        # Should detect degeneracy (2 basins)
        assert result.degeneracy_detected
        assert result.n_unique_basins == 2

    def test_no_degeneracy_with_single_minimum(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that no degeneracy is reported for a single minimum."""
        target = np.array([5.0, 5.0, 5.0])
        fit_func = create_quadratic_fit_function(target)

        config = MultiStartConfig(enable=True, n_starts=5, seed=42, n_workers=1)

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=fit_func,
        )

        # Should not detect degeneracy
        assert not result.degeneracy_detected
        assert result.n_unique_basins == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_volume_bounds_fallback(self, synthetic_quadratic_data):
        """Test fallback to single-start when bounds have zero volume."""
        # All parameters fixed
        fixed_bounds = np.array(
            [
                [5.0, 5.0],  # Fixed at 5
                [5.0, 5.0],  # Fixed at 5
                [5.0, 5.0],  # Fixed at 5
            ]
        )

        def simple_fit(data, params):
            return SingleStartResult(
                start_idx=0,
                initial_params=params,
                final_params=params,
                chi_squared=1.0,
                success=True,
            )

        config = MultiStartConfig(enable=True, n_starts=5, seed=42, n_workers=1)

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=fixed_bounds,
            config=config,
            single_fit_func=simple_fit,
        )

        # Should fall back to single-start
        assert result.strategy_used == "single_start_fallback"
        assert len(result.all_results) == 1
        np.testing.assert_allclose(result.best.final_params, [5.0, 5.0, 5.0])

    def test_custom_starts_outside_bounds(
        self, synthetic_quadratic_data, simple_bounds
    ):
        """Test that custom starts outside bounds are filtered."""
        target = np.array([5.0, 5.0, 5.0])
        fit_func = create_quadratic_fit_function(target)

        # Some custom starts are outside bounds [0, 10]
        custom_starts = [
            [5.0, 5.0, 5.0],  # Valid
            [-1.0, 5.0, 5.0],  # Invalid (< 0)
            [5.0, 15.0, 5.0],  # Invalid (> 10)
        ]

        config = MultiStartConfig(
            enable=True,
            n_starts=3,
            seed=42,
            n_workers=1,
            custom_starts=custom_starts,
        )

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=fit_func,
        )

        # Should have 4 results (3 generated + 1 valid custom)
        assert len(result.all_results) == 4

    def test_all_optimizations_fail(self, synthetic_quadratic_data, simple_bounds):
        """Test handling when all optimizations fail."""

        def always_fails(data, params):
            return SingleStartResult(
                start_idx=0,
                initial_params=params,
                final_params=params,
                chi_squared=np.inf,
                success=False,
                message="Always fails",
            )

        config = MultiStartConfig(enable=True, n_starts=3, seed=42, n_workers=1)

        result = run_multistart_nlsq(
            data=synthetic_quadratic_data,
            bounds=simple_bounds,
            config=config,
            single_fit_func=always_fails,
        )

        # Should still return a result, but with no successful optimizations
        assert result.n_successful == 0
        assert not result.best.success
        assert result.best.chi_squared == np.inf


# =============================================================================
# Regression Tests
# =============================================================================


class TestRegressionFixes:
    """Regression tests for previously identified bugs."""

    def test_xpcs_data_format_dataset_size_calculation(self):
        """Test that XPCS data format (c2_exp) calculates dataset size correctly.

        Regression test for bug where dataset size was calculated from
        len(phi_angles_list) (3 unique angles) instead of c2_exp.size (3M points).

        Bug symptom: Multi-start selected "full" strategy for 3M point dataset
        because it thought there were only 3 data points.

        Fixed in v2.6.1: _get_dataset_size() now correctly extracts size from
        g2/c2_exp arrays.
        """
        from homodyne.optimization.nlsq.multistart import (
            MultiStartStrategy,
            _get_dataset_size,
            select_multistart_strategy,
        )

        # Simulate XPCS data format (what CLI produces)
        xpcs_data = {
            "phi_angles_list": np.array([-5.8, 4.9, 90.0]),  # 3 unique angles
            "c2_exp": np.random.randn(3, 100, 100),  # 30K points (3 × 100 × 100)
        }

        # Should correctly identify 30,000 points, not 3
        n_points = _get_dataset_size(xpcs_data)
        assert n_points == 30_000, f"Expected 30,000 points, got {n_points}"

        # With default config, 30K points should use FULL strategy (< 1M threshold)
        config = MultiStartConfig(enable=True)
        strategy = select_multistart_strategy(n_points, config)
        assert strategy == MultiStartStrategy.FULL

    def test_xpcs_large_dataset_uses_subsample_strategy(self):
        """Test that large XPCS datasets correctly use subsample strategy.

        Ensures that a 2M+ point dataset (simulated by c2_exp shape) correctly
        triggers the SUBSAMPLE strategy rather than FULL.
        """
        from homodyne.optimization.nlsq.multistart import (
            MultiStartStrategy,
            _get_dataset_size,
            select_multistart_strategy,
        )

        # Simulate large XPCS data (> 1M points)
        large_xpcs_data = {
            "phi_angles_list": np.array([0.0, 30.0, 60.0, 90.0]),  # 4 angles
            "c2_exp": np.random.randn(4, 500, 500),  # 1M points (4 × 500 × 500)
        }

        n_points = _get_dataset_size(large_xpcs_data)
        assert n_points == 1_000_000, f"Expected 1,000,000 points, got {n_points}"

        # 1M points should trigger SUBSAMPLE strategy (>= 1M threshold)
        config = MultiStartConfig(enable=True)
        strategy = select_multistart_strategy(n_points, config)
        assert strategy == MultiStartStrategy.SUBSAMPLE

    def test_xpcs_3d_data_subsampling(self):
        """Test that 3D XPCS data (c2_exp) is subsampled correctly.

        Regression test for bug where create_stratified_subsample used
        len(phi_angles_list) (3 unique angles) instead of c2_exp.size (3M points).

        Bug symptom: "Dataset (3) <= target (500000), no subsampling" in logs
        when dataset actually had 3,000,000 points.

        Fixed in v2.6.1: create_stratified_subsample now correctly handles
        3D XPCS data by subsampling in (t1, t2) dimensions.
        """
        from homodyne.optimization.nlsq.multistart import create_stratified_subsample

        # Simulate XPCS data format (3D c2_exp)
        xpcs_data = {
            "phi_angles_list": np.array([-5.8, 4.9, 90.0]),  # 3 unique angles
            "c2_exp": np.random.randn(3, 100, 100),  # 30K points (3 × 100 × 100)
            "t1": np.arange(100, dtype=np.float64),
            "t2": np.arange(100, dtype=np.float64),
        }

        # Subsample to 5K target
        subsample = create_stratified_subsample(xpcs_data, target_size=5_000, seed=42)

        # Verify subsampling occurred
        original_size = xpcs_data["c2_exp"].size  # 30,000
        subsample_size = subsample["c2_exp"].size

        assert subsample_size < original_size, (
            f"Expected subsample ({subsample_size}) < original ({original_size})"
        )
        # Should be roughly target_size (5K), allowing for rounding
        assert subsample_size >= 3_000, f"Subsample too small: {subsample_size}"
        assert subsample_size <= 10_000, f"Subsample too large: {subsample_size}"

        # All angles should be preserved
        assert subsample["c2_exp"].shape[0] == 3, "All angles should be preserved"

        # t1, t2 should be reduced
        assert len(subsample["t1"]) < len(xpcs_data["t1"])
        assert len(subsample["t2"]) < len(xpcs_data["t2"])

    def test_custom_starts_included_in_multistart(self, simple_bounds):
        """Test that custom starting points are included in multi-start.

        Regression test for bug where fit_nlsq_multistart accepted initial_params
        but never passed them to run_multistart_nlsq as custom_starts.

        Fixed in v2.6.1: fit_nlsq_multistart now passes initial_params as
        custom_starts[0] to ensure user's initial guess is included.
        """
        from homodyne.optimization.nlsq.multistart import (
            include_custom_starts,
            generate_lhs_starts,
        )

        # Generate 5 LHS starts
        n_starts = 5
        n_params = 3
        lhs_starts = generate_lhs_starts(simple_bounds, n_starts, seed=42)

        # Add custom start (simulating user's initial parameters)
        custom_start = np.array([5.0, 5.0, 5.0])
        combined = include_custom_starts(lhs_starts, [custom_start.tolist()], simple_bounds)

        # Should have n_starts + 1 starts
        assert len(combined) == n_starts + 1, f"Expected {n_starts + 1}, got {len(combined)}"

        # First start should be the custom start
        np.testing.assert_allclose(combined[0], custom_start)
