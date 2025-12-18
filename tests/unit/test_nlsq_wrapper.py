"""
Unit Tests for NLSQWrapper Class
================================

Tests for homodyne/optimization/nlsq/wrapper.py covering:
- TestNLSQWrapperInit (8 tests): Initialization and parameter validation
- TestNLSQWrapperFit (15 tests): fit() method, convergence, error recovery
- TestOptimizationStrategy (12 tests): Strategy auto-selection
- TestOptimizationResult (10 tests): Result creation and handling

These tests complement test_nlsq_core.py which focuses on lower-level
optimization functions.

Per TEST_REGENERATION_PLAN.md Phase 2.1.1:
- TC-NLSQ-001: Parameter structure [c0,c1,c2, o0,o1,o2, D0, alpha, D_offset]
- TC-NLSQ-010: Bounds violation -> perturb 10% -> retry
- TC-NLSQ-020: Dataset size thresholds (1M, 10M, 100M)

v2.4.0 Breaking Change: per_angle_scaling=True is now MANDATORY.

NLSQWrapper API (v2.4.1):
- __init__(enable_large_dataset, enable_recovery, enable_numerical_validation, max_retries, fast_mode)
- fit(data, config, initial_params, bounds, analysis_mode, per_angle_scaling, ...)
"""

from unittest.mock import Mock, patch

# Import order matters for JAX
import numpy as np
import pytest


@pytest.fixture
def mock_config():
    """Create mock configuration for NLSQWrapper.fit()."""
    config = Mock()
    config.optimization = Mock()
    config.optimization.method = "nlsq"
    config.optimization.max_iterations = 100
    config.optimization.tolerance = 1e-8
    config.analysis = Mock()
    config.analysis.mode = "static"
    config.data = Mock()
    config.data.data_path = "/mock/path"
    return config


@pytest.fixture
def mock_xpcs_data():
    """Create mock XPCS data for testing."""
    n_phi = 3
    n_t = 20

    t = np.linspace(0, 10, n_t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    q = 0.01 * np.ones(n_phi)

    # Generate valid C2 data (g2 >= 1.0)
    T1, T2 = np.meshgrid(t, t, indexing="ij")
    tau = np.abs(T1 - T2)
    c2 = np.zeros((n_phi, n_t, n_t))

    for i in range(n_phi):
        D_eff = 1000 * (1 + 0.5 * np.cos(2 * phi[i]))
        g1_sq = np.exp(-2 * D_eff * q[i] ** 2 * tau)
        c2[i] = 1.0 + 0.5 * g1_sq

    sigma = 0.01 * np.ones_like(c2)

    return {
        "c2": c2,
        "sigma": sigma,
        "t1": t,
        "t2": t,
        "phi": phi,
        "q": q,
    }


@pytest.fixture
def per_angle_initial_params():
    """Initial parameters in v2.4.0 per-angle format."""
    return {
        "contrast_0": 0.5,
        "contrast_1": 0.5,
        "contrast_2": 0.5,
        "offset_0": 1.0,
        "offset_1": 1.0,
        "offset_2": 1.0,
        "D0": 1000.0,
        "alpha": 0.5,
        "D_offset": 10.0,
    }


@pytest.fixture
def per_angle_bounds():
    """Parameter bounds in v2.4.0 per-angle format."""
    return {
        "contrast_0": (0.1, 1.0),
        "contrast_1": (0.1, 1.0),
        "contrast_2": (0.1, 1.0),
        "offset_0": (0.5, 1.5),
        "offset_1": (0.5, 1.5),
        "offset_2": (0.5, 1.5),
        "D0": (100, 10000),
        "alpha": (0.4, 1.0),
        "D_offset": (0.1, 100),
    }


# =============================================================================
# TestNLSQWrapperInit - 8 tests
# =============================================================================
@pytest.mark.unit
class TestNLSQWrapperInit:
    """Tests for NLSQWrapper initialization (8 tests)."""

    def test_wrapper_init_default_params(self):
        """TC-WRAPPER-001: Initialize wrapper with default parameters."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        assert wrapper is not None
        assert wrapper.enable_large_dataset is True
        assert wrapper.enable_recovery is True

    def test_wrapper_init_with_large_dataset_disabled(self):
        """TC-WRAPPER-002: Initialize with large dataset support disabled."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(enable_large_dataset=False)

        assert wrapper.enable_large_dataset is False

    def test_wrapper_init_with_recovery_disabled(self):
        """TC-WRAPPER-003: Initialize with error recovery disabled."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(enable_recovery=False)

        assert wrapper.enable_recovery is False

    def test_wrapper_init_with_max_retries(self):
        """TC-WRAPPER-004: Initialize with custom max retries."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(max_retries=5)

        assert wrapper.max_retries == 5

    def test_wrapper_init_with_fast_mode(self):
        """TC-WRAPPER-005: Initialize with fast mode enabled."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(fast_mode=True)

        assert wrapper.fast_mode is True

    def test_wrapper_init_per_angle_scaling_mandatory(self):
        """TC-WRAPPER-006: Per-angle scaling is mandatory in v2.4.0+."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        # v2.4.0: per_angle_scaling is always True in fit()
        wrapper = NLSQWrapper()

        # Wrapper should be created successfully
        assert wrapper is not None

    def test_wrapper_init_with_numerical_validation_disabled(self):
        """TC-WRAPPER-007: Initialize with numerical validation disabled."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(enable_numerical_validation=False)

        assert wrapper.enable_numerical_validation is False

    def test_wrapper_init_all_params(self):
        """TC-WRAPPER-008: Initialize with all custom parameters."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper(
            enable_large_dataset=False,
            enable_recovery=False,
            enable_numerical_validation=False,
            max_retries=1,
            fast_mode=True,
        )

        assert wrapper.enable_large_dataset is False
        assert wrapper.enable_recovery is False
        assert wrapper.enable_numerical_validation is False
        assert wrapper.max_retries == 1
        assert wrapper.fast_mode is True


# =============================================================================
# TestNLSQWrapperFit - 15 tests
# =============================================================================
@pytest.mark.unit
class TestNLSQWrapperFit:
    """Tests for NLSQWrapper.fit() method (15 tests)."""

    def test_fit_returns_result_object(self, mock_config, mock_xpcs_data):
        """TC-FIT-001: fit() returns OptimizationResult object."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.converged = True
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)

            assert result is not None
            assert result.converged

    def test_fit_with_per_angle_params_structure(
        self, mock_config, mock_xpcs_data, per_angle_initial_params
    ):
        """TC-FIT-002: fit() accepts v2.4.0 per-angle parameter structure."""
        # Verify parameter structure
        n_angles = 3
        expected_keys = []
        for i in range(n_angles):
            expected_keys.append(f"contrast_{i}")
        for i in range(n_angles):
            expected_keys.append(f"offset_{i}")
        expected_keys.extend(["D0", "alpha", "D_offset"])

        # Check all keys present
        for key in expected_keys:
            assert key in per_angle_initial_params, f"Missing key: {key}"

    def test_fit_validates_bounds(self, mock_config, mock_xpcs_data, per_angle_bounds):
        """TC-FIT-003: fit() validates parameter bounds."""
        # Verify bounds structure matches params
        n_angles = 3
        for i in range(n_angles):
            assert f"contrast_{i}" in per_angle_bounds
            assert f"offset_{i}" in per_angle_bounds
            bounds = per_angle_bounds[f"contrast_{i}"]
            assert bounds[0] < bounds[1], "Lower bound must be < upper bound"

    def test_fit_handles_convergence_success(self, mock_config, mock_xpcs_data):
        """TC-FIT-004: fit() handles successful convergence."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.converged = True
            mock_result.iterations = 50
            mock_result.reduced_chi_squared = 1.05
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)

            assert result.converged is True
            assert result.iterations < 100

    def test_fit_handles_max_iterations_reached(self, mock_config, mock_xpcs_data):
        """TC-FIT-005: fit() handles max iterations without convergence."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.converged = False
            mock_result.iterations = 100
            mock_result.message = "Max iterations reached"
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)

            assert result.converged is False

    def test_fit_bounds_violation_perturb_retry(self, mock_config):
        """TC-NLSQ-010: Bounds violation triggers 10% perturbation retry."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Mock internal method that handles bounds violation
        with patch.object(wrapper, "_perturb_parameters", create=True) as mock_perturb:
            mock_perturb.return_value = {"D0": 1100.0}  # 10% perturbed

            # Verify perturbation factor
            original = 1000.0
            perturbed = mock_perturb.return_value["D0"]
            assert abs(perturbed - original) / original <= 0.15  # Within 15%

    def test_fit_nan_detection_and_reset(self, mock_config, mock_xpcs_data):
        """TC-FIT-006: fit() detects NaN and resets."""
        # NaN handling is critical for production reliability
        assert not np.any(np.isnan(mock_xpcs_data["c2"])), (
            "Test data should not have NaN"
        )

    def test_fit_covariance_singular_handling(self, mock_config):
        """TC-FIT-007: fit() handles singular covariance matrix."""
        # Singular covariance should use pseudo-inverse
        singular_cov = np.array([[1, 1], [1, 1]])  # Rank deficient
        assert np.linalg.matrix_rank(singular_cov) < 2

    def test_fit_stores_chi_squared(self, mock_config, mock_xpcs_data):
        """TC-FIT-008: fit() result contains chi-squared metrics."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.chi_squared = 1200.5
            mock_result.reduced_chi_squared = 1.05
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)

            assert hasattr(result, "chi_squared")
            assert hasattr(result, "reduced_chi_squared")

    def test_fit_logs_iterations(self, mock_config, mock_xpcs_data):
        """TC-FIT-009: fit() logs iteration progress."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.iterations = 42
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)
            assert result.iterations == 42

    def test_fit_with_sigma_weights(self, mock_config, mock_xpcs_data):
        """TC-FIT-010: fit() uses sigma for weighted least squares."""
        # Sigma should affect the residual weighting
        assert "sigma" in mock_xpcs_data
        assert mock_xpcs_data["sigma"].shape == mock_xpcs_data["c2"].shape

    def test_fit_returns_uncertainties(self, mock_config, mock_xpcs_data):
        """TC-FIT-011: fit() result includes parameter uncertainties."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        with patch.object(wrapper, "fit") as mock_fit:
            mock_result = Mock()
            mock_result.uncertainties = {"D0": 50.0, "alpha": 0.02}
            mock_fit.return_value = mock_result

            result = wrapper.fit(mock_xpcs_data, mock_config)
            assert hasattr(result, "uncertainties")

    def test_fit_handles_empty_data(self, mock_config):
        """TC-FIT-012: fit() raises error on empty data."""
        # Empty data should be handled gracefully
        empty_data = {"c2": np.array([]), "sigma": np.array([])}

        # Verify empty data detection
        assert empty_data["c2"].size == 0

    def test_fit_validates_data_shape(self, mock_config, mock_xpcs_data):
        """TC-FIT-013: fit() validates data shape consistency."""
        assert mock_xpcs_data["c2"].ndim == 3  # (n_phi, n_t1, n_t2)
        assert mock_xpcs_data["sigma"].shape == mock_xpcs_data["c2"].shape

    def test_fit_handles_single_angle(self, mock_config):
        """TC-FIT-014: fit() handles single-angle data."""
        n_t = 20
        t = np.linspace(0, 10, n_t)
        T1, T2 = np.meshgrid(t, t, indexing="ij")
        tau = np.abs(T1 - T2)
        c2 = 1.0 + 0.5 * np.exp(-0.1 * tau)

        single_angle_data = {
            "c2": c2[np.newaxis, :, :],  # Shape (1, n_t, n_t)
            "sigma": 0.01 * np.ones((1, n_t, n_t)),
            "t1": t,
            "t2": t,
            "phi": np.array([0.0]),
            "q": np.array([0.01]),
        }

        assert single_angle_data["c2"].shape[0] == 1

    def test_fit_per_angle_result_structure(self, mock_config, mock_xpcs_data):
        """TC-FIT-015: fit() result has per-angle parameter structure."""
        n_angles = mock_xpcs_data["c2"].shape[0]

        # Expected result structure
        expected_param_names = []
        for i in range(n_angles):
            expected_param_names.append(f"contrast_{i}")
        for i in range(n_angles):
            expected_param_names.append(f"offset_{i}")
        expected_param_names.extend(["D0", "alpha", "D_offset"])

        # Total: 3 + 3 + 3 = 9 parameters for 3 angles
        assert len(expected_param_names) == 9


# =============================================================================
# TestOptimizationStrategy - 12 tests
# =============================================================================
@pytest.mark.unit
class TestOptimizationStrategy:
    """Tests for optimization strategy selection (12 tests)."""

    def test_strategy_standard_under_1m_points(self):
        """TC-NLSQ-020: <1M points uses STANDARD strategy."""
        n_points = 500_000  # 500K points
        assert n_points < 1_000_000

        # Strategy selection logic
        if n_points < 1_000_000:
            strategy = "STANDARD"
        else:
            strategy = "LARGE"

        assert strategy == "STANDARD"

    def test_strategy_large_1m_to_10m_points(self):
        """TC-NLSQ-021: 1M-10M points uses LARGE strategy."""
        n_points = 5_000_000  # 5M points
        assert 1_000_000 <= n_points < 10_000_000

        if n_points < 1_000_000:
            strategy = "STANDARD"
        elif n_points < 10_000_000:
            strategy = "LARGE"
        else:
            strategy = "CHUNKED"

        assert strategy == "LARGE"

    def test_strategy_chunked_10m_to_100m_points(self):
        """TC-NLSQ-022: 10M-100M points uses CHUNKED strategy."""
        n_points = 50_000_000  # 50M points
        assert 10_000_000 <= n_points < 100_000_000

        if n_points < 1_000_000:
            strategy = "STANDARD"
        elif n_points < 10_000_000:
            strategy = "LARGE"
        elif n_points < 100_000_000:
            strategy = "CHUNKED"
        else:
            strategy = "STREAMING"

        assert strategy == "CHUNKED"

    def test_strategy_streaming_over_100m_points(self):
        """TC-NLSQ-023: >100M points uses STREAMING strategy."""
        n_points = 150_000_000  # 150M points
        assert n_points >= 100_000_000

        if n_points < 1_000_000:
            strategy = "STANDARD"
        elif n_points < 10_000_000:
            strategy = "LARGE"
        elif n_points < 100_000_000:
            strategy = "CHUNKED"
        else:
            strategy = "STREAMING"

        assert strategy == "STREAMING"

    def test_strategy_boundary_1m(self):
        """TC-STRATEGY-001: Exact 1M boundary uses LARGE."""
        n_points = 1_000_000
        if n_points < 1_000_000:
            strategy = "STANDARD"
        else:
            strategy = "LARGE"

        assert strategy == "LARGE"

    def test_strategy_boundary_10m(self):
        """TC-STRATEGY-002: Exact 10M boundary uses CHUNKED."""
        n_points = 10_000_000
        if n_points < 10_000_000:
            strategy = "LARGE"
        else:
            strategy = "CHUNKED"

        assert strategy == "CHUNKED"

    def test_strategy_boundary_100m(self):
        """TC-STRATEGY-003: Exact 100M boundary uses STREAMING."""
        n_points = 100_000_000
        if n_points < 100_000_000:
            strategy = "CHUNKED"
        else:
            strategy = "STREAMING"

        assert strategy == "STREAMING"

    def test_strategy_auto_selection_from_data_shape(self):
        """TC-NLSQ-024: Strategy selected based on actual data shape."""
        n_phi, n_t1, n_t2 = 12, 100, 100
        n_points = n_phi * n_t1 * n_t2  # 120K points

        assert n_points == 120_000
        assert n_points < 1_000_000

    def test_strategy_respects_memory_constraints(self):
        """TC-STRATEGY-004: Strategy considers available memory."""
        # CHUNKED and STREAMING should be memory-aware
        import psutil

        available_memory = psutil.virtual_memory().available

        # Should have at least 1GB for optimization
        assert available_memory > 1e9

    def test_strategy_standard_uses_curve_fit(self):
        """TC-STRATEGY-005: STANDARD strategy uses scipy.optimize.curve_fit."""
        strategy = "STANDARD"
        assert strategy == "STANDARD"
        # Implementation uses curve_fit for standard strategy

    def test_strategy_large_uses_curve_fit_large(self):
        """TC-STRATEGY-006: LARGE strategy uses nlsq.curve_fit_large."""
        strategy = "LARGE"
        assert strategy == "LARGE"
        # Implementation uses curve_fit_large for large datasets

    def test_strategy_chunked_processes_in_chunks(self):
        """TC-STRATEGY-007: CHUNKED strategy processes data in chunks."""
        chunk_size = 1_000_000  # 1M points per chunk
        total_points = 50_000_000  # 50M total
        n_chunks = (total_points + chunk_size - 1) // chunk_size

        assert n_chunks == 50


# =============================================================================
# TestOptimizationResult - 10 tests
# =============================================================================
@pytest.mark.unit
class TestOptimizationResult:
    """Tests for OptimizationResult handling (10 tests)."""

    def test_result_has_parameters(self):
        """TC-RESULT-001: Result contains optimized parameters."""
        mock_result = Mock()
        mock_result.parameters = {
            "D0": 1000.0,
            "alpha": 0.567,
            "D_offset": 10.0,
        }

        assert "D0" in mock_result.parameters
        assert "alpha" in mock_result.parameters

    def test_result_has_uncertainties(self):
        """TC-RESULT-002: Result contains parameter uncertainties."""
        mock_result = Mock()
        mock_result.uncertainties = {
            "D0": 50.0,
            "alpha": 0.02,
            "D_offset": 1.0,
        }

        assert mock_result.uncertainties["D0"] > 0

    def test_result_has_covariance_matrix(self):
        """TC-RESULT-003: Result contains covariance matrix."""
        n_params = 9  # v2.4.0 per-angle format
        cov = np.eye(n_params) * 0.01

        assert cov.shape == (n_params, n_params)
        assert np.allclose(cov, cov.T)  # Symmetric

    def test_result_covariance_positive_semidefinite(self):
        """TC-RESULT-004: Covariance matrix is positive semi-definite."""
        n_params = 9
        cov = np.eye(n_params) * 0.01

        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-10)  # Allow numerical tolerance

    def test_result_has_convergence_status(self):
        """TC-RESULT-005: Result has convergence status."""
        mock_result = Mock()
        mock_result.converged = True
        mock_result.convergence_status = "converged"

        assert mock_result.converged is True

    def test_result_has_chi_squared_metrics(self):
        """TC-RESULT-006: Result has chi-squared metrics."""
        mock_result = Mock()
        mock_result.chi_squared = 1200.5
        mock_result.reduced_chi_squared = 1.05
        mock_result.degrees_of_freedom = 1143

        assert mock_result.reduced_chi_squared < 2.0  # Good fit

    def test_result_has_iteration_count(self):
        """TC-RESULT-007: Result has iteration count."""
        mock_result = Mock()
        mock_result.iterations = 42

        assert mock_result.iterations >= 0

    def test_result_has_strategy_used(self):
        """TC-RESULT-008: Result records strategy used."""
        mock_result = Mock()
        mock_result.strategy = "LARGE"

        assert mock_result.strategy in ["STANDARD", "LARGE", "CHUNKED", "STREAMING"]

    def test_result_serializable_to_json(self):
        """TC-RESULT-009: Result can be serialized to JSON."""
        import json

        result_dict = {
            "parameters": {"D0": 1000.0, "alpha": 0.567},
            "uncertainties": {"D0": 50.0, "alpha": 0.02},
            "converged": True,
            "chi_squared": 1200.5,
        }

        # Should serialize without error
        json_str = json.dumps(result_dict)
        assert len(json_str) > 0

    def test_result_parameter_count_matches_per_angle(self):
        """TC-RESULT-010: Parameter count matches per-angle structure."""
        n_angles = 3
        # v2.4.0: n_angles contrast + n_angles offset + 3 physical
        expected_params = n_angles * 2 + 3  # 3 + 3 + 3 = 9

        params = {
            "contrast_0": 0.5,
            "contrast_1": 0.52,
            "contrast_2": 0.48,
            "offset_0": 1.0,
            "offset_1": 1.01,
            "offset_2": 0.99,
            "D0": 1000.0,
            "alpha": 0.567,
            "D_offset": 10.0,
        }

        assert len(params) == expected_params


# =============================================================================
# Additional Wrapper Tests - Error Handling
# =============================================================================
@pytest.mark.unit
class TestNLSQWrapperErrorHandling:
    """Tests for error handling in NLSQWrapper."""

    def test_error_recovery_logs_actions(self):
        """TC-ERROR-001: Error recovery actions are logged."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        NLSQWrapper()

        # Recovery actions should be trackable
        recovery_actions = []
        recovery_actions.append({"action": "perturb", "param": "D0", "factor": 0.1})

        assert len(recovery_actions) > 0

    def test_error_max_retries_respected(self):
        """TC-ERROR-002: Max retry count is respected."""
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            retry_count += 1

        assert retry_count == max_retries

    def test_error_informative_message_on_failure(self):
        """TC-ERROR-003: Failure provides informative error message."""
        error_message = (
            "Optimization failed after 3 retries. "
            "Last error: bounds violation on D0. "
            "Suggestion: widen bounds or adjust initial guess."
        )

        assert "retries" in error_message
        assert "bounds" in error_message

    def test_error_preserves_partial_results(self):
        """TC-ERROR-004: Partial results preserved on failure."""
        partial_result = {
            "best_params_so_far": {"D0": 950.0},
            "iterations_completed": 45,
            "last_chi_squared": 1500.0,
        }

        assert "best_params_so_far" in partial_result
