"""
Demonstration of NumericalValidator exception-based fixtures.

This test module shows how to use the three new fixtures added to conftest.py
for testing the v2.4.0+ exception-based NumericalValidator API:

1. numerical_error_types - Maps error type names to exception classes
2. mock_numerical_validator - Mock validator with exception-based methods
3. numerical_validation_context - Context manager for capturing validation results

These fixtures help update tests from the old return-value based API
(v2.3.0 and earlier) to the new exception-based API (v2.4.0+).

Author: Testing Engineer (Fixture Migration)
Date: 2025-12-03
"""

import numpy as np
import pytest

from homodyne.optimization.exceptions import NLSQNumericalError

# ============================================================================
# Test Group 1: Using numerical_error_types fixture
# ============================================================================


class TestNumericalErrorTypes:
    """Tests demonstrating numerical_error_types fixture usage."""

    def test_fixture_provides_exception_classes(self, numerical_error_types):
        """Verify fixture provides expected exception classes."""
        assert "NumericalError" in numerical_error_types
        assert "OptimizationError" in numerical_error_types
        assert "ConvergenceError" in numerical_error_types

    def test_exception_classes_are_usable(self, numerical_error_types):
        """Verify exception classes can be used with pytest.raises."""
        NumericalError = numerical_error_types["NumericalError"]
        numerical_error_types["OptimizationError"]

        # Can instantiate exceptions
        try:
            raise NumericalError("Test error", detection_point="test")
        except NumericalError as e:
            assert "Test error" in str(e)
            assert e.detection_point == "test"

    def test_exception_hierarchy(self, numerical_error_types):
        """Verify exception hierarchy is correct."""
        NumericalError = numerical_error_types["NumericalError"]
        OptimizationError = numerical_error_types["OptimizationError"]

        # NumericalError should be an OptimizationError
        assert issubclass(NumericalError, OptimizationError)


# ============================================================================
# Test Group 2: Using mock_numerical_validator fixture
# ============================================================================


class TestMockNumericalValidator:
    """Tests demonstrating mock_numerical_validator fixture usage."""

    def test_mock_validator_has_required_methods(self, mock_numerical_validator):
        """Verify mock validator has all required methods."""
        validator = mock_numerical_validator

        assert hasattr(validator, "validate_values")
        assert hasattr(validator, "validate_gradients")
        assert hasattr(validator, "validate_parameters")
        assert hasattr(validator, "validate_loss")
        assert hasattr(validator, "enable")
        assert hasattr(validator, "disable")

    def test_valid_parameters_pass_without_exception(self, mock_numerical_validator):
        """Valid parameters should pass validation without raising exception."""
        validator = mock_numerical_validator

        valid_params = np.array([0.5, 1.0, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        # Should not raise
        validator.validate_parameters(valid_params, bounds)

    def test_nan_in_parameters_raises_exception(self, mock_numerical_validator):
        """NaN in parameters should raise exception."""
        validator = mock_numerical_validator

        invalid_params = np.array([0.3, np.nan, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(invalid_params, bounds)

    def test_inf_in_parameters_raises_exception(self, mock_numerical_validator):
        """Inf in parameters should raise exception."""
        validator = mock_numerical_validator

        invalid_params = np.array([0.3, 1.0, np.inf])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(invalid_params, bounds)

    def test_nan_in_gradients_raises_exception(self, mock_numerical_validator):
        """NaN in gradients should raise exception."""
        validator = mock_numerical_validator

        gradients = np.array([1.0, 2.0, np.nan, 4.0])

        with pytest.raises(NLSQNumericalError):
            validator.validate_gradients(gradients)

    def test_nan_in_loss_raises_exception(self, mock_numerical_validator):
        """NaN in loss should raise exception."""
        validator = mock_numerical_validator

        with pytest.raises(NLSQNumericalError):
            validator.validate_loss(np.nan)

    def test_bounds_violations_raise_exception(self, mock_numerical_validator):
        """Parameter bounds violations should raise exception."""
        validator = mock_numerical_validator

        # Parameters outside bounds
        invalid_params = np.array(
            [0.3, 1.0, 50000.0]
        )  # Last param > upper bound (10000)
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(invalid_params, bounds)

    def test_validation_can_be_disabled(self, mock_numerical_validator):
        """Validation should be skipped when disabled."""
        validator = mock_numerical_validator

        # Disable validation
        validator.disable()

        # NaN should not raise when validation is disabled
        invalid_params = np.array([0.3, np.nan, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        validator.validate_parameters(invalid_params, bounds)  # Should not raise

        # Re-enable validation
        validator.enable()

        # Now it should raise
        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(invalid_params, bounds)

    def test_set_bounds_on_validator(self, mock_numerical_validator):
        """Validator should use set bounds for validation."""
        validator = mock_numerical_validator

        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        validator.set_bounds(bounds)

        # Now parameters are checked against these bounds
        valid_params = np.array([0.5, 1.0, 5000.0])
        validator.validate_parameters(valid_params)  # Should not raise

        # Out of bounds should raise
        invalid_params = np.array([0.3, 1.0, 50000.0])

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(invalid_params)


# ============================================================================
# Test Group 3: Using numerical_validation_context fixture
# ============================================================================


class TestNumericalValidationContext:
    """Tests demonstrating numerical_validation_context fixture usage."""

    def test_context_manager_is_callable(self, numerical_validation_context):
        """Fixture should return a callable context manager factory."""
        context_mgr = numerical_validation_context()

        # Should be usable as context manager
        assert hasattr(context_mgr, "__enter__")
        assert hasattr(context_mgr, "__exit__")

    def test_context_captures_successful_validation(
        self, mock_numerical_validator, numerical_validation_context
    ):
        """Context manager should capture successful validation."""
        validator = mock_numerical_validator
        context_mgr = numerical_validation_context()

        valid_params = np.array([0.5, 1.0, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        try:
            with context_mgr as result:
                validator.validate_parameters(valid_params, bounds)

            assert result.passed is True
            assert result.exception is None
        except Exception:
            pytest.fail("No exception should be raised for valid parameters")

    def test_context_captures_failed_validation(
        self, mock_numerical_validator, numerical_validation_context
    ):
        """Context manager should capture failed validation."""
        validator = mock_numerical_validator
        context_mgr = numerical_validation_context()

        invalid_params = np.array([0.3, np.nan, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        try:
            with context_mgr as result:
                validator.validate_parameters(invalid_params, bounds)
        except Exception:
            # Exception is raised but also captured
            assert result.passed is False
            assert result.exception is not None
            assert result.detection_point == "parameter"

    def test_context_captures_detection_point(
        self, mock_numerical_validator, numerical_validation_context
    ):
        """Context manager should capture detection point."""
        validator = mock_numerical_validator
        context_mgr = numerical_validation_context()

        gradients = np.array([1.0, np.nan, 3.0])

        try:
            with context_mgr as result:
                validator.validate_gradients(gradients)
        except Exception:
            assert result.detection_point == "gradient"

    def test_context_captures_error_context(
        self, mock_numerical_validator, numerical_validation_context
    ):
        """Context manager should capture error context."""
        validator = mock_numerical_validator
        context_mgr = numerical_validation_context()

        invalid_params = np.array([0.3, np.nan, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        try:
            with context_mgr as result:
                validator.validate_parameters(invalid_params, bounds)
        except Exception:
            assert result.error_context is not None
            assert "n_invalid" in result.error_context


# ============================================================================
# Test Group 4: Integration tests combining all fixtures
# ============================================================================


class TestFixtureIntegration:
    """Integration tests combining multiple fixtures."""

    def test_comprehensive_validation_with_all_fixtures(
        self,
        mock_numerical_validator,
        numerical_error_types,
        numerical_validation_context,
    ):
        """Comprehensive test using all three fixtures together."""
        validator = mock_numerical_validator
        NumericalError = numerical_error_types["NumericalError"]

        test_params = np.array([0.5, 1.0, 1000.0])
        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        # Test 1: Valid parameters with context manager
        context_mgr = numerical_validation_context()

        try:
            with context_mgr as result:
                validator.validate_parameters(test_params, bounds)

            assert result.passed
        except Exception:
            pytest.fail("Valid parameters should pass")

        # Test 2: Invalid parameters with pytest.raises and error type fixture
        invalid_params = np.array([0.3, np.nan, 1000.0])

        with pytest.raises(NumericalError):
            validator.validate_parameters(invalid_params, bounds)

        # Test 3: Invalid parameters with context manager for inspection
        context_mgr = numerical_validation_context()

        try:
            with context_mgr as result:
                validator.validate_parameters(invalid_params, bounds)
        except NumericalError:
            assert not result.passed
            assert isinstance(result.exception, NumericalError)
            assert result.detection_point == "parameter"

    def test_mixed_validation_scenarios(
        self, mock_numerical_validator, numerical_error_types
    ):
        """Test multiple validation scenarios in sequence."""
        validator = mock_numerical_validator
        NumericalError = numerical_error_types["NumericalError"]

        bounds = (
            np.array([0.1, 0.5, 100.0]),
            np.array([1.0, 2.0, 10000.0]),
        )

        # Valid parameters
        validator.validate_parameters(np.array([0.5, 1.0, 1000.0]), bounds)

        # Invalid: NaN
        with pytest.raises(NumericalError):
            validator.validate_parameters(np.array([0.3, np.nan, 1000.0]), bounds)

        # Invalid: Inf
        with pytest.raises(NumericalError):
            validator.validate_parameters(np.array([0.3, 1.0, np.inf]), bounds)

        # Invalid: Out of bounds
        with pytest.raises(NumericalError):
            validator.validate_parameters(np.array([0.3, 1.0, 50000.0]), bounds)


# ============================================================================
# Test Group 5: Edge cases and special scenarios
# ============================================================================


class TestEdgeCases:
    """Edge case tests for fixtures."""

    def test_empty_bounds_handling(self, mock_numerical_validator):
        """Validator should handle None bounds gracefully."""
        validator = mock_numerical_validator

        params = np.array([0.5, 1.0, 1000.0])

        # Should pass with None bounds (no bounds checking)
        validator.validate_parameters(params, bounds=None)

    def test_multiple_nan_values(self, mock_numerical_validator):
        """Validator should detect multiple NaN values."""
        validator = mock_numerical_validator

        params = np.array([np.nan, 1.0, np.nan])

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(params)

    def test_mixed_nan_inf_values(self, mock_numerical_validator):
        """Validator should detect mixed NaN and Inf values."""
        validator = mock_numerical_validator

        params = np.array([np.nan, np.inf, 1000.0])

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(params)

    def test_negative_infinity(self, mock_numerical_validator):
        """Validator should detect negative infinity."""
        validator = mock_numerical_validator

        params = np.array([0.5, -np.inf, 1000.0])

        with pytest.raises(NLSQNumericalError):
            validator.validate_parameters(params)
