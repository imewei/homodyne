"""
Unit Tests for NLSQ Validation Utilities
========================================

Tests for homodyne/optimization/nlsq/validation/ covering:
- TestInputValidator: Input data validation
- TestValidateArrayDimensions: Array dimension checks
- TestValidateNoNanInf: NaN/Inf detection
- TestValidateBoundsConsistency: Bounds validation
- TestValidateInitialParams: Initial parameter bounds check
- TestResultValidator: Result validation
- TestValidateOptimizedParams: Optimized parameter checks
- TestValidateCovariance: Covariance matrix validation
- TestValidateResultConsistency: Result consistency checks

Part of v2.14.0 architecture refactoring tests.
"""

import numpy as np
import pytest


# =============================================================================
# TestInputValidator
# =============================================================================
@pytest.mark.unit
class TestInputValidator:
    """Tests for InputValidator class."""

    def test_validate_all_passes_valid_data(self):
        """Valid data passes all validation."""
        from homodyne.optimization.nlsq.validation import InputValidator

        validator = InputValidator(strict_mode=True)

        xdata = np.linspace(0, 10, 100)
        ydata = np.ones(100)
        initial_params = np.array([1.0, 2.0, 3.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([10.0, 10.0, 10.0]))

        result = validator.validate_all(xdata, ydata, initial_params, bounds)

        assert result is True
        assert len(validator.validation_errors) == 0

    def test_validate_all_rejects_mismatched_dimensions(self):
        """Mismatched dimensions are rejected."""
        from homodyne.optimization.nlsq.validation import InputValidator

        validator = InputValidator(strict_mode=True)

        xdata = np.linspace(0, 10, 100)
        ydata = np.ones(50)  # Wrong length
        initial_params = np.array([1.0])
        bounds = None

        with pytest.raises(ValueError, match="Array dimension mismatch"):
            validator.validate_all(xdata, ydata, initial_params, bounds)

    def test_validate_all_rejects_nan_in_xdata(self):
        """NaN in xdata is rejected."""
        from homodyne.optimization.nlsq.validation import InputValidator

        validator = InputValidator(strict_mode=True)

        xdata = np.array([1.0, np.nan, 3.0])
        ydata = np.ones(3)
        initial_params = np.array([1.0])
        bounds = None

        with pytest.raises(ValueError, match="xdata contains NaN"):
            validator.validate_all(xdata, ydata, initial_params, bounds)

    def test_validate_all_non_strict_returns_false(self):
        """Non-strict mode returns False but doesn't raise."""
        from homodyne.optimization.nlsq.validation import InputValidator

        validator = InputValidator(strict_mode=False)

        xdata = np.array([1.0, np.nan, 3.0])
        ydata = np.ones(3)
        initial_params = np.array([1.0])
        bounds = None

        result = validator.validate_all(xdata, ydata, initial_params, bounds)

        assert result is False
        assert len(validator.validation_errors) > 0

    def test_validate_all_rejects_params_outside_bounds(self):
        """Initial params outside bounds are rejected."""
        from homodyne.optimization.nlsq.validation import InputValidator

        validator = InputValidator(strict_mode=True)

        xdata = np.ones(10)
        ydata = np.ones(10)
        initial_params = np.array([5.0])  # Outside bounds
        bounds = (np.array([0.0]), np.array([1.0]))

        with pytest.raises(ValueError, match="Initial parameters outside bounds"):
            validator.validate_all(xdata, ydata, initial_params, bounds)


# =============================================================================
# TestValidateArrayDimensions
# =============================================================================
@pytest.mark.unit
class TestValidateArrayDimensions:
    """Tests for validate_array_dimensions function."""

    def test_valid_dimensions_pass(self):
        """Matching array lengths pass."""
        from homodyne.optimization.nlsq.validation import validate_array_dimensions

        xdata = np.ones(100)
        ydata = np.ones(100)

        assert validate_array_dimensions(xdata, ydata) is True

    def test_empty_xdata_fails(self):
        """Empty xdata fails."""
        from homodyne.optimization.nlsq.validation import validate_array_dimensions

        xdata = np.array([])
        ydata = np.ones(10)

        assert validate_array_dimensions(xdata, ydata) is False

    def test_empty_ydata_fails(self):
        """Empty ydata fails."""
        from homodyne.optimization.nlsq.validation import validate_array_dimensions

        xdata = np.ones(10)
        ydata = np.array([])

        assert validate_array_dimensions(xdata, ydata) is False

    def test_mismatched_lengths_fail(self):
        """Different array lengths fail."""
        from homodyne.optimization.nlsq.validation import validate_array_dimensions

        xdata = np.ones(100)
        ydata = np.ones(50)

        assert validate_array_dimensions(xdata, ydata) is False


# =============================================================================
# TestValidateNoNanInf
# =============================================================================
@pytest.mark.unit
class TestValidateNoNanInf:
    """Tests for validate_no_nan_inf function."""

    def test_finite_array_passes(self):
        """Array with finite values passes."""
        from homodyne.optimization.nlsq.validation import validate_no_nan_inf

        arr = np.array([1.0, 2.0, 3.0])

        assert validate_no_nan_inf(arr, "test") is True

    def test_nan_detected(self):
        """NaN values are detected."""
        from homodyne.optimization.nlsq.validation import validate_no_nan_inf

        arr = np.array([1.0, np.nan, 3.0])

        assert validate_no_nan_inf(arr, "test") is False

    def test_inf_detected(self):
        """Inf values are detected."""
        from homodyne.optimization.nlsq.validation import validate_no_nan_inf

        arr = np.array([1.0, np.inf, 3.0])

        assert validate_no_nan_inf(arr, "test") is False

    def test_neg_inf_detected(self):
        """Negative Inf values are detected."""
        from homodyne.optimization.nlsq.validation import validate_no_nan_inf

        arr = np.array([1.0, -np.inf, 3.0])

        assert validate_no_nan_inf(arr, "test") is False


# =============================================================================
# TestValidateBoundsConsistency
# =============================================================================
@pytest.mark.unit
class TestValidateBoundsConsistency:
    """Tests for validate_bounds_consistency function."""

    def test_valid_bounds_pass(self):
        """Consistent bounds pass."""
        from homodyne.optimization.nlsq.validation import validate_bounds_consistency

        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        params = np.array([0.5, 0.5])

        assert validate_bounds_consistency(bounds, params) is True

    def test_lower_greater_than_upper_fails(self):
        """lower > upper fails."""
        from homodyne.optimization.nlsq.validation import validate_bounds_consistency

        bounds = (np.array([2.0, 0.0]), np.array([1.0, 1.0]))  # First lower > upper
        params = np.array([0.5, 0.5])

        assert validate_bounds_consistency(bounds, params) is False

    def test_bounds_length_mismatch_fails(self):
        """Bounds with wrong length fail."""
        from homodyne.optimization.nlsq.validation import validate_bounds_consistency

        bounds = (np.array([0.0]), np.array([1.0]))  # Length 1
        params = np.array([0.5, 0.5])  # Length 2

        assert validate_bounds_consistency(bounds, params) is False


# =============================================================================
# TestValidateInitialParams
# =============================================================================
@pytest.mark.unit
class TestValidateInitialParams:
    """Tests for validate_initial_params function."""

    def test_params_within_bounds_pass(self):
        """Parameters within bounds pass."""
        from homodyne.optimization.nlsq.validation import validate_initial_params

        params = np.array([0.5, 0.5])
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        assert validate_initial_params(params, bounds) is True

    def test_params_at_bounds_pass(self):
        """Parameters at exact bounds pass."""
        from homodyne.optimization.nlsq.validation import validate_initial_params

        params = np.array([0.0, 1.0])  # At boundaries
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        assert validate_initial_params(params, bounds) is True

    def test_params_below_lower_fail(self):
        """Parameters below lower bound fail."""
        from homodyne.optimization.nlsq.validation import validate_initial_params

        params = np.array([-0.1, 0.5])  # First below lower
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        assert validate_initial_params(params, bounds) is False

    def test_params_above_upper_fail(self):
        """Parameters above upper bound fail."""
        from homodyne.optimization.nlsq.validation import validate_initial_params

        params = np.array([0.5, 1.1])  # Second above upper
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))

        assert validate_initial_params(params, bounds) is False

    def test_none_bounds_always_passes(self):
        """None bounds always passes."""
        from homodyne.optimization.nlsq.validation import validate_initial_params

        params = np.array([1000.0, -1000.0])
        bounds = None

        assert validate_initial_params(params, bounds) is True


# =============================================================================
# TestResultValidator
# =============================================================================
@pytest.mark.unit
class TestResultValidator:
    """Tests for ResultValidator class."""

    def test_validate_all_passes_valid_result(self):
        """Valid result passes all validation."""
        from homodyne.optimization.nlsq.validation import ResultValidator

        validator = ResultValidator(strict_mode=False)

        params = np.array([0.5, 0.5])
        covariance = np.array([[0.01, 0.0], [0.0, 0.01]])
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        chi_squared = 0.05

        result = validator.validate_all(params, covariance, bounds, chi_squared)

        assert result is True
        assert len(validator.validation_warnings) == 0

    def test_validate_all_warns_invalid_covariance(self):
        """Invalid covariance triggers warning."""
        from homodyne.optimization.nlsq.validation import ResultValidator

        validator = ResultValidator(strict_mode=False)

        params = np.array([0.5, 0.5])
        covariance = np.array([[-0.01, 0.0], [0.0, 0.01]])  # Negative diagonal
        bounds = None
        chi_squared = 0.05

        result = validator.validate_all(params, covariance, bounds, chi_squared)

        assert result is False
        assert "Covariance matrix invalid" in validator.validation_warnings

    def test_validate_all_strict_raises(self):
        """Strict mode raises on failure."""
        from homodyne.optimization.nlsq.validation import ResultValidator

        validator = ResultValidator(strict_mode=True)

        params = np.array([np.nan])  # Invalid
        covariance = None
        bounds = None

        with pytest.raises(ValueError, match="Result validation failed"):
            validator.validate_all(params, covariance, bounds)


# =============================================================================
# TestValidateOptimizedParams
# =============================================================================
@pytest.mark.unit
class TestValidateOptimizedParams:
    """Tests for validate_optimized_params function."""

    def test_finite_params_pass(self):
        """Finite parameters pass."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([1.0, 2.0, 3.0])

        assert validate_optimized_params(params, bounds=None) is True

    def test_nan_params_fail(self):
        """NaN parameters fail."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([1.0, np.nan, 3.0])

        assert validate_optimized_params(params, bounds=None) is False

    def test_inf_params_fail(self):
        """Inf parameters fail."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([1.0, np.inf, 3.0])

        assert validate_optimized_params(params, bounds=None) is False

    def test_params_within_bounds_pass(self):
        """Parameters within bounds pass."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([0.5])
        bounds = (np.array([0.0]), np.array([1.0]))

        assert validate_optimized_params(params, bounds) is True

    def test_params_outside_bounds_fail(self):
        """Parameters outside bounds fail."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([1.5])  # Outside
        bounds = (np.array([0.0]), np.array([1.0]))

        assert validate_optimized_params(params, bounds) is False

    def test_tolerance_allows_small_violations(self):
        """Small boundary violations within tolerance pass."""
        from homodyne.optimization.nlsq.validation import validate_optimized_params

        params = np.array([1.0 + 1e-11])  # Tiny violation
        bounds = (np.array([0.0]), np.array([1.0]))

        # Default tolerance is 1e-10, so 1e-11 should pass
        assert validate_optimized_params(params, bounds, tolerance=1e-10) is True


# =============================================================================
# TestValidateCovariance
# =============================================================================
@pytest.mark.unit
class TestValidateCovariance:
    """Tests for validate_covariance function."""

    def test_valid_covariance_passes(self):
        """Valid covariance matrix passes."""
        from homodyne.optimization.nlsq.validation import validate_covariance

        covariance = np.array([[1.0, 0.5], [0.5, 1.0]])

        assert validate_covariance(covariance, n_params=2) is True

    def test_wrong_shape_fails(self):
        """Wrong shape covariance fails."""
        from homodyne.optimization.nlsq.validation import validate_covariance

        covariance = np.array([[1.0, 0.0], [0.0, 1.0]])

        assert validate_covariance(covariance, n_params=3) is False

    def test_nan_in_covariance_fails(self):
        """NaN in covariance fails."""
        from homodyne.optimization.nlsq.validation import validate_covariance

        covariance = np.array([[1.0, np.nan], [np.nan, 1.0]])

        assert validate_covariance(covariance, n_params=2) is False

    def test_asymmetric_covariance_fails(self):
        """Asymmetric covariance fails."""
        from homodyne.optimization.nlsq.validation import validate_covariance

        covariance = np.array([[1.0, 0.5], [0.3, 1.0]])  # Asymmetric

        assert validate_covariance(covariance, n_params=2) is False

    def test_negative_diagonal_fails(self):
        """Negative diagonal element fails."""
        from homodyne.optimization.nlsq.validation import validate_covariance

        covariance = np.array([[-0.1, 0.0], [0.0, 1.0]])  # Negative diagonal

        assert validate_covariance(covariance, n_params=2) is False


# =============================================================================
# TestValidateResultConsistency
# =============================================================================
@pytest.mark.unit
class TestValidateResultConsistency:
    """Tests for validate_result_consistency function."""

    def test_valid_chi_squared_passes(self):
        """Valid chi-squared passes."""
        from homodyne.optimization.nlsq.validation import validate_result_consistency

        params = np.array([1.0])
        chi_squared = 0.5

        assert validate_result_consistency(params, chi_squared) is True

    def test_nan_chi_squared_fails(self):
        """NaN chi-squared fails."""
        from homodyne.optimization.nlsq.validation import validate_result_consistency

        params = np.array([1.0])
        chi_squared = np.nan

        assert validate_result_consistency(params, chi_squared) is False

    def test_inf_chi_squared_fails(self):
        """Inf chi-squared fails."""
        from homodyne.optimization.nlsq.validation import validate_result_consistency

        params = np.array([1.0])
        chi_squared = np.inf

        assert validate_result_consistency(params, chi_squared) is False

    def test_negative_chi_squared_fails(self):
        """Negative chi-squared fails."""
        from homodyne.optimization.nlsq.validation import validate_result_consistency

        params = np.array([1.0])
        chi_squared = -0.5

        assert validate_result_consistency(params, chi_squared) is False


# =============================================================================
# TestModuleExports
# =============================================================================
@pytest.mark.unit
class TestModuleExports:
    """Tests for module __all__ exports."""

    def test_input_validator_exports(self):
        """InputValidator module exports all expected symbols."""
        from homodyne.optimization.nlsq.validation import input_validator

        expected = [
            "InputValidator",
            "validate_array_dimensions",
            "validate_bounds_consistency",
            "validate_initial_params",
            "validate_no_nan_inf",
        ]

        for name in expected:
            assert hasattr(input_validator, name)
            assert name in input_validator.__all__

    def test_result_validator_exports(self):
        """ResultValidator module exports all expected symbols."""
        from homodyne.optimization.nlsq.validation import result_validator

        expected = [
            "ResultValidator",
            "validate_covariance",
            "validate_optimized_params",
            "validate_result_consistency",
        ]

        for name in expected:
            assert hasattr(result_validator, name)
            assert name in result_validator.__all__

    def test_package_exports(self):
        """Validation package exports key classes."""
        from homodyne.optimization.nlsq import validation

        assert hasattr(validation, "InputValidator")
        assert hasattr(validation, "ResultValidator")
