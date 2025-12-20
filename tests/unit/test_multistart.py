"""Unit tests for multi-start NLSQ optimization.

Tests cover:
- LHS generation (coverage, bounds, reproducibility)
- Strategy selection by dataset size
- Screening phase
- Degeneracy detection
- Configuration parsing
- Stratified subsampling
"""

from __future__ import annotations

import numpy as np

from homodyne.optimization.nlsq.multistart import (
    THRESHOLD_LARGE,
    THRESHOLD_SMALL,
    MultiStartConfig,
    MultiStartResult,
    MultiStartStrategy,
    SingleStartResult,
    create_stratified_subsample,
    detect_degeneracy,
    generate_lhs_starts,
    generate_random_starts,
    screen_starts,
    select_multistart_strategy,
)


class TestMultiStartConfig:
    """Tests for MultiStartConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MultiStartConfig()

        assert config.enable is False  # Default OFF
        assert config.n_starts == 10
        assert config.seed == 42
        assert config.sampling_strategy == "latin_hypercube"
        assert config.n_workers == 0  # Auto
        assert config.use_screening is True
        assert config.screen_keep_fraction == 0.5
        assert config.subsample_size == 500_000
        assert config.warmup_only_threshold == 100_000_000
        assert config.refine_top_k == 3
        assert config.refinement_ftol == 1e-12
        assert config.degeneracy_threshold == 0.1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MultiStartConfig(
            enable=True,
            n_starts=20,
            seed=123,
            sampling_strategy="random",
            n_workers=4,
            use_screening=False,
            screen_keep_fraction=0.3,
            subsample_size=1_000_000,
            warmup_only_threshold=50_000_000,
            refine_top_k=5,
            refinement_ftol=1e-14,
            degeneracy_threshold=0.05,
        )

        assert config.enable is True
        assert config.n_starts == 20
        assert config.seed == 123
        assert config.sampling_strategy == "random"
        assert config.n_workers == 4
        assert config.use_screening is False
        assert config.screen_keep_fraction == 0.3
        assert config.subsample_size == 1_000_000
        assert config.warmup_only_threshold == 50_000_000
        assert config.refine_top_k == 5
        assert config.refinement_ftol == 1e-14
        assert config.degeneracy_threshold == 0.05


class TestLHSGeneration:
    """Tests for Latin Hypercube Sampling generation."""

    def test_generate_lhs_starts_shape(self):
        """Test LHS generates correct shape."""
        bounds = np.array(
            [
                [0.0, 1.0],  # param 0
                [10.0, 100.0],  # param 1
                [-1.0, 1.0],  # param 2
            ]
        )

        starts = generate_lhs_starts(bounds, n_starts=10, seed=42)

        assert starts.shape == (10, 3)

    def test_generate_lhs_starts_bounds(self):
        """Test LHS respects parameter bounds."""
        bounds = np.array(
            [
                [0.0, 1.0],
                [10.0, 100.0],
                [-1.0, 1.0],
            ]
        )

        starts = generate_lhs_starts(bounds, n_starts=100, seed=42)

        # Check all starts are within bounds
        for i in range(3):
            assert np.all(starts[:, i] >= bounds[i, 0])
            assert np.all(starts[:, i] <= bounds[i, 1])

    def test_generate_lhs_starts_reproducibility(self):
        """Test LHS is reproducible with same seed."""
        bounds = np.array(
            [
                [0.0, 1.0],
                [10.0, 100.0],
            ]
        )

        starts1 = generate_lhs_starts(bounds, n_starts=10, seed=42)
        starts2 = generate_lhs_starts(bounds, n_starts=10, seed=42)

        np.testing.assert_array_equal(starts1, starts2)

    def test_generate_lhs_starts_different_seeds(self):
        """Test LHS produces different results with different seeds."""
        bounds = np.array(
            [
                [0.0, 1.0],
                [10.0, 100.0],
            ]
        )

        starts1 = generate_lhs_starts(bounds, n_starts=10, seed=42)
        starts2 = generate_lhs_starts(bounds, n_starts=10, seed=123)

        assert not np.allclose(starts1, starts2)

    def test_generate_lhs_starts_coverage(self):
        """Test LHS provides good coverage (space-filling)."""
        bounds = np.array([[0.0, 1.0]])
        n_starts = 10

        starts = generate_lhs_starts(bounds, n_starts=n_starts, seed=42)

        # With LHS, each "stratum" should have exactly one sample
        # Sort and check spacing
        sorted_starts = np.sort(starts[:, 0])
        spacings = np.diff(sorted_starts)

        # Spacings should be relatively uniform (not clustered)
        # For 10 samples in [0,1], expect spacing ~0.1
        assert np.std(spacings) < 0.15  # Low variance in spacings


class TestRandomGeneration:
    """Tests for random uniform sampling."""

    def test_generate_random_starts_shape(self):
        """Test random generates correct shape."""
        bounds = np.array(
            [
                [0.0, 1.0],
                [10.0, 100.0],
            ]
        )

        starts = generate_random_starts(bounds, n_starts=10, seed=42)

        assert starts.shape == (10, 2)

    def test_generate_random_starts_bounds(self):
        """Test random respects parameter bounds."""
        bounds = np.array(
            [
                [0.0, 1.0],
                [10.0, 100.0],
            ]
        )

        starts = generate_random_starts(bounds, n_starts=100, seed=42)

        for i in range(2):
            assert np.all(starts[:, i] >= bounds[i, 0])
            assert np.all(starts[:, i] <= bounds[i, 1])

    def test_generate_random_starts_reproducibility(self):
        """Test random is reproducible with same seed."""
        bounds = np.array([[0.0, 1.0], [10.0, 100.0]])

        starts1 = generate_random_starts(bounds, n_starts=10, seed=42)
        starts2 = generate_random_starts(bounds, n_starts=10, seed=42)

        np.testing.assert_array_equal(starts1, starts2)


class TestStrategySelection:
    """Tests for dataset size-based strategy selection."""

    def test_select_full_strategy_small_dataset(self):
        """Test full strategy selected for < 1M points."""
        config = MultiStartConfig()

        # Small dataset
        strategy = select_multistart_strategy(500_000, config)
        assert strategy == MultiStartStrategy.FULL

        # Just below threshold
        strategy = select_multistart_strategy(THRESHOLD_SMALL - 1, config)
        assert strategy == MultiStartStrategy.FULL

    def test_select_subsample_strategy_medium_dataset(self):
        """Test subsample strategy selected for 1M-100M points."""
        config = MultiStartConfig()

        # At threshold
        strategy = select_multistart_strategy(THRESHOLD_SMALL, config)
        assert strategy == MultiStartStrategy.SUBSAMPLE

        # Middle of range
        strategy = select_multistart_strategy(50_000_000, config)
        assert strategy == MultiStartStrategy.SUBSAMPLE

        # Just below large threshold
        strategy = select_multistart_strategy(THRESHOLD_LARGE - 1, config)
        assert strategy == MultiStartStrategy.SUBSAMPLE

    def test_select_phase1_strategy_large_dataset(self):
        """Test phase1 strategy selected for > 100M points."""
        config = MultiStartConfig()

        # At threshold
        strategy = select_multistart_strategy(THRESHOLD_LARGE, config)
        assert strategy == MultiStartStrategy.PHASE1

        # Well above threshold
        strategy = select_multistart_strategy(500_000_000, config)
        assert strategy == MultiStartStrategy.PHASE1

    def test_custom_threshold(self):
        """Test custom warmup_only_threshold is respected."""
        config = MultiStartConfig(warmup_only_threshold=50_000_000)

        # Below custom threshold - subsample
        strategy = select_multistart_strategy(40_000_000, config)
        assert strategy == MultiStartStrategy.SUBSAMPLE

        # At custom threshold - phase1
        strategy = select_multistart_strategy(50_000_000, config)
        assert strategy == MultiStartStrategy.PHASE1


class TestScreening:
    """Tests for screening phase."""

    def test_screen_starts_basic(self):
        """Test basic screening functionality."""
        starts = np.array(
            [
                [0.1, 0.1],
                [0.5, 0.5],
                [0.9, 0.9],
                [0.3, 0.3],
            ]
        )

        # Cost function: distance from [0.5, 0.5]
        def cost_func(params):
            return np.sum((params - 0.5) ** 2)

        filtered, costs = screen_starts(
            cost_func, starts, keep_fraction=0.5, min_keep=1
        )

        # Should keep 2 out of 4 (min_keep=1, so 50% = 2)
        assert len(filtered) == 2
        # Best should be [0.5, 0.5]
        assert np.allclose(filtered[0], [0.5, 0.5])

    def test_screen_starts_min_keep(self):
        """Test minimum keep is respected."""
        starts = np.array(
            [
                [0.1, 0.1],
                [0.5, 0.5],
                [0.9, 0.9],
                [0.3, 0.3],
            ]
        )

        def cost_func(params):
            return np.sum(params**2)

        # Request very aggressive filtering, but min_keep=3
        filtered, costs = screen_starts(
            cost_func, starts, keep_fraction=0.1, min_keep=3
        )

        assert len(filtered) == 3

    def test_screen_starts_returns_costs(self):
        """Test screening returns all initial costs."""
        starts = np.array(
            [
                [0.1],
                [0.5],
                [0.9],
            ]
        )

        def cost_func(params):
            return params[0] ** 2

        filtered, costs = screen_starts(cost_func, starts, keep_fraction=1.0)

        # Should have costs for all original starts
        assert len(costs) == 3
        np.testing.assert_allclose(costs, [0.01, 0.25, 0.81])


class TestDegeneracyDetection:
    """Tests for parameter degeneracy detection."""

    def test_no_degeneracy_single_basin(self):
        """Test no degeneracy when all converge to same point."""
        results = [
            SingleStartResult(
                start_idx=0,
                initial_params=np.array([0.0]),
                final_params=np.array([1.0]),
                chi_squared=100.0,
                success=True,
            ),
            SingleStartResult(
                start_idx=1,
                initial_params=np.array([0.5]),
                final_params=np.array([1.01]),  # Very close
                chi_squared=100.5,
                success=True,
            ),
            SingleStartResult(
                start_idx=2,
                initial_params=np.array([1.0]),
                final_params=np.array([0.99]),  # Very close
                chi_squared=101.0,
                success=True,
            ),
        ]

        degeneracy, n_basins, labels = detect_degeneracy(results)

        assert degeneracy is False
        assert n_basins == 1

    def test_degeneracy_multiple_basins(self):
        """Test degeneracy detected with multiple distinct minima."""
        results = [
            SingleStartResult(
                start_idx=0,
                initial_params=np.array([0.0]),
                final_params=np.array([1.0]),
                chi_squared=100.0,
                success=True,
            ),
            SingleStartResult(
                start_idx=1,
                initial_params=np.array([0.5]),
                final_params=np.array([5.0]),  # Different basin
                chi_squared=100.5,  # Similar chi-squared
                success=True,
            ),
        ]

        degeneracy, n_basins, labels = detect_degeneracy(
            results, chi_sq_threshold=0.1, param_threshold=0.2
        )

        assert degeneracy is True
        assert n_basins == 2

    def test_no_degeneracy_different_chi_squared(self):
        """Test no degeneracy when chi-squared values are very different."""
        results = [
            SingleStartResult(
                start_idx=0,
                initial_params=np.array([0.0]),
                final_params=np.array([1.0]),
                chi_squared=100.0,
                success=True,
            ),
            SingleStartResult(
                start_idx=1,
                initial_params=np.array([0.5]),
                final_params=np.array([5.0]),  # Different point
                chi_squared=200.0,  # Very different chi-squared
                success=True,
            ),
        ]

        degeneracy, n_basins, labels = detect_degeneracy(results, chi_sq_threshold=0.1)

        # Second result has chi-squared too different, not counted as degeneracy
        assert degeneracy is False

    def test_degeneracy_with_failed_results(self):
        """Test degeneracy detection ignores failed results."""
        results = [
            SingleStartResult(
                start_idx=0,
                initial_params=np.array([0.0]),
                final_params=np.array([1.0]),
                chi_squared=100.0,
                success=True,
            ),
            SingleStartResult(
                start_idx=1,
                initial_params=np.array([0.5]),
                final_params=np.array([5.0]),
                chi_squared=np.inf,
                success=False,  # Failed
            ),
        ]

        degeneracy, n_basins, labels = detect_degeneracy(results)

        # Only one successful result
        assert degeneracy is False
        assert n_basins == 1


class TestStratifiedSubsample:
    """Tests for stratified subsampling."""

    def test_subsample_preserves_angles(self):
        """Test subsampling preserves angle distribution."""
        # Create data with 3 angles
        n_per_angle = 10000
        data = {
            "phi": np.concatenate(
                [
                    np.full(n_per_angle, 0.0),
                    np.full(n_per_angle, 30.0),
                    np.full(n_per_angle, 60.0),
                ]
            ),
            "g2": np.random.randn(3 * n_per_angle),
            "t1": np.random.randn(3 * n_per_angle),
            "t2": np.random.randn(3 * n_per_angle),
        }

        subsample = create_stratified_subsample(data, target_size=3000, seed=42)

        # Check each angle has ~1000 points
        phi_sub = subsample["phi"]
        for angle in [0.0, 30.0, 60.0]:
            count = np.sum(phi_sub == angle)
            assert 900 <= count <= 1100  # Within 10% of expected

    def test_subsample_size(self):
        """Test subsample achieves target size."""
        n_per_angle = 100000
        data = {
            "phi": np.concatenate(
                [
                    np.full(n_per_angle, 0.0),
                    np.full(n_per_angle, 30.0),
                ]
            ),
            "g2": np.random.randn(2 * n_per_angle),
        }

        target = 10000
        subsample = create_stratified_subsample(data, target_size=target, seed=42)

        # Should be close to target (may be slightly less due to integer division)
        assert len(subsample["phi"]) >= target * 0.9
        assert len(subsample["phi"]) <= target

    def test_subsample_no_op_small_dataset(self):
        """Test no subsampling when dataset is smaller than target."""
        data = {
            "phi": np.array([0.0, 30.0, 60.0]),
            "g2": np.array([1.0, 2.0, 3.0]),
        }

        subsample = create_stratified_subsample(data, target_size=10000, seed=42)

        # Should return original data unchanged
        assert len(subsample["phi"]) == 3
        np.testing.assert_array_equal(subsample["phi"], data["phi"])


class TestSingleStartResult:
    """Tests for SingleStartResult dataclass."""

    def test_creation(self):
        """Test SingleStartResult creation."""
        result = SingleStartResult(
            start_idx=5,
            initial_params=np.array([1.0, 2.0]),
            final_params=np.array([1.5, 2.5]),
            chi_squared=100.0,
            reduced_chi_squared=1.1,
            success=True,
            status=0,
            message="Converged",
            n_iterations=50,
            n_fev=200,
            wall_time=1.5,
        )

        assert result.start_idx == 5
        np.testing.assert_array_equal(result.initial_params, [1.0, 2.0])
        np.testing.assert_array_equal(result.final_params, [1.5, 2.5])
        assert result.chi_squared == 100.0
        assert result.success is True


class TestMultiStartResult:
    """Tests for MultiStartResult dataclass."""

    def test_creation(self):
        """Test MultiStartResult creation."""
        best = SingleStartResult(
            start_idx=2,
            initial_params=np.array([1.0]),
            final_params=np.array([2.0]),
            chi_squared=50.0,
            success=True,
        )

        all_results = [best]
        config = MultiStartConfig()

        result = MultiStartResult(
            best=best,
            all_results=all_results,
            config=config,
            strategy_used="full",
            n_successful=1,
            n_unique_basins=1,
            degeneracy_detected=False,
            total_wall_time=10.0,
        )

        assert result.best.chi_squared == 50.0
        assert result.strategy_used == "full"
        assert result.n_successful == 1
        assert result.degeneracy_detected is False


class TestConfigFromNLSQConfig:
    """Tests for creating MultiStartConfig from NLSQConfig."""

    def test_from_nlsq_config(self):
        """Test MultiStartConfig creation from NLSQConfig."""
        from homodyne.optimization.nlsq.config import NLSQConfig

        nlsq_config = NLSQConfig(
            enable_multi_start=True,
            multi_start_n_starts=20,
            multi_start_seed=123,
            multi_start_sampling_strategy="random",
            multi_start_n_workers=8,
            multi_start_use_screening=False,
            multi_start_screen_keep_fraction=0.3,
            multi_start_subsample_size=1_000_000,
            multi_start_warmup_only_threshold=50_000_000,
            multi_start_refine_top_k=5,
            multi_start_refinement_ftol=1e-14,
            multi_start_degeneracy_threshold=0.05,
        )

        ms_config = MultiStartConfig.from_nlsq_config(nlsq_config)

        assert ms_config.enable is True
        assert ms_config.n_starts == 20
        assert ms_config.seed == 123
        assert ms_config.sampling_strategy == "random"
        assert ms_config.n_workers == 8
        assert ms_config.use_screening is False
        assert ms_config.screen_keep_fraction == 0.3
        assert ms_config.subsample_size == 1_000_000
        assert ms_config.warmup_only_threshold == 50_000_000
        assert ms_config.refine_top_k == 5
        assert ms_config.refinement_ftol == 1e-14
        assert ms_config.degeneracy_threshold == 0.05
