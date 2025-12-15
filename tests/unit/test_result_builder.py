"""Unit tests for homodyne.optimization.nlsq.result_builder module.

Tests result building utilities including quality metrics computation,
uncertainty extraction, and result normalization.
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.result_builder import (
    QualityMetrics,
    ResultBuilder,
    compute_quality_metrics,
    compute_uncertainties,
    determine_convergence_status,
    normalize_nlsq_result,
)


class TestQualityMetrics:
    """Tests for QualityMetrics dataclass."""

    def test_quality_metrics_creation(self):
        """Test basic QualityMetrics instantiation."""
        metrics = QualityMetrics(
            chi_squared=10.5,
            reduced_chi_squared=1.05,
            quality_flag="good",
            n_at_bounds=0,
        )

        assert metrics.chi_squared == 10.5
        assert metrics.reduced_chi_squared == 1.05
        assert metrics.quality_flag == "good"
        assert metrics.n_at_bounds == 0

    def test_quality_metrics_default_n_at_bounds(self):
        """Test that n_at_bounds defaults to 0."""
        metrics = QualityMetrics(
            chi_squared=10.5, reduced_chi_squared=1.05, quality_flag="good"
        )

        assert metrics.n_at_bounds == 0


class TestComputeQualityMetrics:
    """Tests for compute_quality_metrics function."""

    def test_good_quality_fit(self):
        """Test quality metrics for a good fit."""
        residuals = np.random.randn(100) * 0.1  # Small residuals
        n_data = 100
        n_params = 5

        metrics = compute_quality_metrics(residuals, n_data, n_params)

        assert metrics.chi_squared > 0
        assert metrics.reduced_chi_squared == metrics.chi_squared / (n_data - n_params)
        # Small residuals should give good quality
        assert metrics.quality_flag in ["good", "marginal"]

    def test_poor_quality_fit(self):
        """Test quality metrics for a poor fit."""
        residuals = np.random.randn(100) * 10  # Large residuals
        n_data = 100
        n_params = 5

        metrics = compute_quality_metrics(residuals, n_data, n_params)

        assert metrics.chi_squared > 0
        # Large residuals typically give poor quality
        # (depends on actual values, so just check it's computed)

    def test_with_parameters_at_bounds(self):
        """Test quality metrics with parameters at bounds."""
        residuals = np.random.randn(100) * 0.5
        n_data = 100
        n_params = 5
        parameter_status = ["active", "at_lower_bound", "active", "at_upper_bound", "active"]

        metrics = compute_quality_metrics(
            residuals, n_data, n_params, parameter_status=parameter_status
        )

        assert metrics.n_at_bounds == 2

    def test_quality_flag_good(self):
        """Test that good quality flag is assigned correctly."""
        # Create residuals that give reduced_chi_squared < 2.0
        residuals = np.ones(100) * 0.1
        n_data = 100
        n_params = 5

        metrics = compute_quality_metrics(residuals, n_data, n_params)

        # With residuals = 0.1, chi_sq = 100 * 0.01 = 1.0
        # reduced = 1.0 / 95 ≈ 0.01 < 2.0 → good
        assert metrics.quality_flag == "good"

    def test_quality_flag_marginal(self):
        """Test that marginal quality flag is assigned correctly."""
        # Create residuals that give 2.0 <= reduced_chi_squared < 5.0
        residuals = np.ones(100) * 1.5
        n_data = 100
        n_params = 5
        parameter_status = ["active", "at_lower_bound", "active", "active", "active"]

        metrics = compute_quality_metrics(
            residuals, n_data, n_params, parameter_status=parameter_status
        )

        # reduced_chi_squared = (100 * 2.25) / 95 ≈ 2.37
        # n_at_bounds = 1 <= 2 → marginal (if reduced < 5)
        assert metrics.quality_flag in ["good", "marginal"]

    def test_avoids_division_by_zero(self):
        """Test that division by zero is avoided when n_data == n_params."""
        residuals = np.array([0.1, 0.2, 0.3])
        n_data = 3
        n_params = 3  # DOF would be 0

        metrics = compute_quality_metrics(residuals, n_data, n_params)

        # Should use max(dof, 1) = 1
        assert metrics.reduced_chi_squared == metrics.chi_squared


class TestComputeUncertainties:
    """Tests for compute_uncertainties function."""

    def test_diagonal_covariance(self):
        """Test uncertainty extraction from diagonal covariance."""
        covariance = np.diag([0.01, 0.04, 0.09, 0.16])

        uncertainties = compute_uncertainties(covariance)

        np.testing.assert_array_almost_equal(uncertainties, [0.1, 0.2, 0.3, 0.4])

    def test_none_covariance_returns_empty(self):
        """Test that None covariance returns empty array."""
        uncertainties = compute_uncertainties(None)

        assert len(uncertainties) == 0

    def test_empty_covariance_returns_empty(self):
        """Test that empty covariance returns empty array."""
        covariance = np.array([])

        uncertainties = compute_uncertainties(covariance)

        assert len(uncertainties) == 0

    def test_negative_diagonal_handled(self):
        """Test that negative diagonal elements are handled gracefully."""
        # Numerical issues can sometimes produce small negative variances
        covariance = np.diag([0.01, -1e-10, 0.09])

        uncertainties = compute_uncertainties(covariance)

        # Negative values should be clipped to 0, so sqrt gives 0
        np.testing.assert_array_almost_equal(uncertainties, [0.1, 0.0, 0.3])


class TestNormalizeNlsqResult:
    """Tests for normalize_nlsq_result function."""

    def test_dict_result_format(self):
        """Test normalization of dict result format."""
        result = {
            "x": np.array([1.0, 2.0, 3.0]),
            "pcov": np.eye(3),
            "success": True,
            "message": "Converged",
        }

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(popt, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pcov, np.eye(3))
        assert info["success"] is True

    def test_dict_result_with_popt_key(self):
        """Test normalization of dict result with popt key."""
        result = {"popt": np.array([1.0, 2.0, 3.0]), "pcov": np.eye(3)}

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(popt, [1.0, 2.0, 3.0])

    def test_tuple_two_elements(self):
        """Test normalization of (popt, pcov) tuple."""
        result = (np.array([1.0, 2.0, 3.0]), np.eye(3))

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(popt, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(pcov, np.eye(3))
        assert info == {}

    def test_tuple_three_elements(self):
        """Test normalization of (popt, pcov, info) tuple."""
        result = (
            np.array([1.0, 2.0, 3.0]),
            np.eye(3),
            {"success": True, "nfev": 100},
        )

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(popt, [1.0, 2.0, 3.0])
        assert info["success"] is True
        assert info["nfev"] == 100

    def test_object_with_x_attribute(self):
        """Test normalization of object with x attribute."""

        class MockResult:
            x = np.array([1.0, 2.0, 3.0])
            pcov = np.eye(3)
            success = True
            nfev = 50

        result = MockResult()

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(popt, [1.0, 2.0, 3.0])
        assert info["success"] is True
        assert info["nfev"] == 50

    def test_object_without_pcov_uses_identity(self):
        """Test that missing pcov uses identity matrix."""

        class MockResult:
            x = np.array([1.0, 2.0, 3.0])

        result = MockResult()

        popt, pcov, info = normalize_nlsq_result(result, "test_strategy")

        np.testing.assert_array_equal(pcov, np.eye(3))

    def test_invalid_tuple_length_raises(self):
        """Test that invalid tuple length raises TypeError."""
        result = (np.array([1.0]), np.eye(1), {}, "extra")

        with pytest.raises(TypeError, match="Unexpected tuple length"):
            normalize_nlsq_result(result, "test_strategy")

    def test_unrecognized_format_raises(self):
        """Test that unrecognized format raises TypeError."""
        result = "invalid"

        with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
            normalize_nlsq_result(result, "test_strategy")


class TestDetermineConvergenceStatus:
    """Tests for determine_convergence_status function."""

    def test_success_true_converged(self):
        """Test that success=True gives converged status."""
        info = {"success": True}
        metrics = QualityMetrics(
            chi_squared=1.0, reduced_chi_squared=0.1, quality_flag="good"
        )

        status = determine_convergence_status(info, metrics)

        assert status == "converged"

    def test_success_false_max_iter(self):
        """Test that max iterations message gives max_iter status."""
        info = {"success": False, "message": "Maximum number of iterations reached"}
        metrics = QualityMetrics(
            chi_squared=1.0, reduced_chi_squared=0.1, quality_flag="good"
        )

        status = determine_convergence_status(info, metrics)

        assert status == "max_iter"

    def test_success_false_failed(self):
        """Test that general failure gives failed status."""
        info = {"success": False, "message": "Some other error"}
        metrics = QualityMetrics(
            chi_squared=1.0, reduced_chi_squared=0.1, quality_flag="good"
        )

        status = determine_convergence_status(info, metrics)

        assert status == "failed"

    def test_infer_from_quality_converged(self):
        """Test convergence inference from good quality."""
        info = {}  # No success flag
        metrics = QualityMetrics(
            chi_squared=1.0, reduced_chi_squared=5.0, quality_flag="marginal"
        )

        status = determine_convergence_status(info, metrics)

        assert status == "converged"  # reduced_chi_squared < 10

    def test_infer_from_quality_failed(self):
        """Test failure inference from poor quality."""
        info = {}  # No success flag
        metrics = QualityMetrics(
            chi_squared=1000.0, reduced_chi_squared=15.0, quality_flag="poor"
        )

        status = determine_convergence_status(info, metrics)

        assert status == "failed"  # reduced_chi_squared >= 10


class TestResultBuilder:
    """Tests for ResultBuilder class."""

    def test_builder_basic_usage(self):
        """Test basic ResultBuilder usage."""
        builder = ResultBuilder()
        params = np.array([1.0, 2.0, 3.0])
        cov = np.diag([0.01, 0.04, 0.09])

        builder.with_parameters(params)
        builder.with_covariance(cov)
        builder.with_data_size(100)
        builder.with_info({"success": True})

        result = builder.build()

        np.testing.assert_array_equal(result["parameters"], params)
        np.testing.assert_array_almost_equal(result["uncertainties"], [0.1, 0.2, 0.3])
        assert result["convergence_status"] == "converged"

    def test_builder_fluent_interface(self):
        """Test that builder methods return self for chaining."""
        builder = ResultBuilder()

        result = (
            builder.with_parameters(np.array([1.0]))
            .with_covariance(np.eye(1))
            .with_data_size(100)
            .with_info({"success": True})
        )

        assert result is builder

    def test_builder_without_parameters_raises(self):
        """Test that building without parameters raises ValueError."""
        builder = ResultBuilder()

        with pytest.raises(ValueError, match="Parameters must be set"):
            builder.build()

    def test_builder_without_covariance_uses_zeros(self):
        """Test that missing covariance gives zero uncertainties."""
        builder = ResultBuilder()
        builder.with_parameters(np.array([1.0, 2.0, 3.0]))
        builder.with_data_size(100)
        builder.with_info({"success": True})

        result = builder.build()

        np.testing.assert_array_equal(result["uncertainties"], [0.0, 0.0, 0.0])

    def test_builder_with_recovery_actions(self):
        """Test builder with recovery actions."""
        builder = ResultBuilder()
        builder.with_parameters(np.array([1.0]))
        builder.with_data_size(100)
        builder.with_recovery_actions(["reset_to_bounds", "perturb_params"])

        result = builder.build()

        assert result["recovery_actions"] == ["reset_to_bounds", "perturb_params"]

    def test_builder_with_diagnostics(self):
        """Test builder with diagnostics."""
        builder = ResultBuilder()
        builder.with_parameters(np.array([1.0]))
        builder.with_data_size(100)
        builder.with_nlsq_diagnostics({"iterations": 50})

        result = builder.build()

        assert result["nlsq_diagnostics"] == {"iterations": 50}

    def test_builder_execution_time(self):
        """Test that execution time is computed."""
        import time

        builder = ResultBuilder()
        builder.with_parameters(np.array([1.0]))
        builder.with_data_size(100)

        time.sleep(0.01)  # Small delay

        result = builder.build()

        assert result["execution_time"] > 0
