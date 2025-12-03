"""Comprehensive tests for recovery_strategies.py to achieve >80% coverage.

This test suite covers:
- All recovery strategy types
- Strategy application logic
- Parameter perturbation
- Bounds handling
- Retry logic
- Edge cases
"""

import numpy as np

from homodyne.optimization.exceptions import (
    NLSQConvergenceError,
    NLSQNumericalError,
    NLSQOptimizationError,
)
from homodyne.optimization.recovery_strategies import (
    ERROR_RECOVERY_STRATEGIES,
    RecoveryStrategyApplicator,
)


class TestRecoveryStrategyApplicator:
    """Test RecoveryStrategyApplicator class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        applicator = RecoveryStrategyApplicator()
        assert applicator.max_retries == 2

    def test_init_custom_max_retries(self):
        """Test initialization with custom max retries."""
        applicator = RecoveryStrategyApplicator(max_retries=5)
        assert applicator.max_retries == 5

    def test_should_retry_within_limit(self):
        """Test should_retry returns True when under limit."""
        applicator = RecoveryStrategyApplicator(max_retries=3)
        assert applicator.should_retry(0) is True
        assert applicator.should_retry(1) is True
        assert applicator.should_retry(2) is True

    def test_should_retry_at_limit(self):
        """Test should_retry returns False when at limit."""
        applicator = RecoveryStrategyApplicator(max_retries=2)
        assert applicator.should_retry(2) is False
        assert applicator.should_retry(3) is False

    def test_get_recovery_strategy_convergence_error_first_attempt(self):
        """Test recovery strategy for convergence error, first attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQConvergenceError("Failed to converge")
        params = np.array([1.0, 2.0, 3.0])

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=0
        )

        assert strategy_name == "perturb_parameters"
        assert modified_params.shape == params.shape
        # Should have some perturbation
        assert not np.allclose(modified_params, params)

    def test_get_recovery_strategy_convergence_error_second_attempt(self):
        """Test recovery strategy for convergence error, second attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQConvergenceError("Failed to converge")
        params = np.array([1.0, 2.0, 3.0])

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=1
        )

        assert strategy_name == "increase_iterations"
        # Parameters unchanged for this strategy
        assert np.allclose(modified_params, params)

    def test_get_recovery_strategy_convergence_error_third_attempt(self):
        """Test recovery strategy for convergence error, third attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQConvergenceError("Failed to converge")
        params = np.array([1.0, 2.0, 3.0])

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=2
        )

        assert strategy_name == "relax_tolerance"
        # Parameters unchanged for this strategy
        assert np.allclose(modified_params, params)

    def test_get_recovery_strategy_no_more_strategies(self):
        """Test get_recovery_strategy returns None when exhausted."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQConvergenceError("Failed to converge")
        params = np.array([1.0, 2.0, 3.0])

        # Attempt beyond available strategies
        result = applicator.get_recovery_strategy(error, params, attempt=3)

        assert result is None

    def test_get_recovery_strategy_numerical_error_first_attempt(self):
        """Test recovery strategy for numerical error, first attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQNumericalError("NaN detected", detection_point="gradient")
        params = np.array([1.0, 2.0, 3.0])

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=0
        )

        assert strategy_name == "reduce_step_size"
        # Parameters unchanged for this strategy
        assert np.allclose(modified_params, params)

    def test_get_recovery_strategy_numerical_error_second_attempt(self):
        """Test recovery strategy for numerical error, second attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQNumericalError("NaN detected", detection_point="gradient")
        params = np.array([100.0, 200.0, 300.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([1000.0, 1000.0, 1000.0]))

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=1, bounds=bounds
        )

        assert strategy_name == "tighten_bounds"
        # Parameters should be clipped to tightened bounds
        assert modified_params.shape == params.shape
        # Should be closer to center
        assert np.all(modified_params >= bounds[0])
        assert np.all(modified_params <= bounds[1])

    def test_get_recovery_strategy_numerical_error_third_attempt(self):
        """Test recovery strategy for numerical error, third attempt."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQNumericalError("NaN detected", detection_point="gradient")
        params = np.array([1.0, 2.0, 3.0])

        strategy_name, modified_params = applicator.get_recovery_strategy(
            error, params, attempt=2
        )

        assert strategy_name == "rescale_data"
        # Parameters unchanged for this strategy
        assert np.allclose(modified_params, params)

    def test_get_recovery_strategy_unknown_error_type(self):
        """Test recovery strategy for unknown error type."""
        applicator = RecoveryStrategyApplicator()
        error = NLSQOptimizationError("Unknown error")
        params = np.array([1.0, 2.0, 3.0])

        # Unknown error types have no strategies
        result = applicator.get_recovery_strategy(error, params, attempt=0)

        assert result is None

    def test_perturb_parameters_adds_noise(self):
        """Test parameter perturbation adds random noise."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        # Set random seed for reproducibility
        np.random.seed(42)
        perturbed = applicator._perturb_parameters(params, perturbation_fraction=0.1)

        assert perturbed.shape == params.shape
        # Should be different from original
        assert not np.allclose(perturbed, params)
        # Should be roughly within 10% (may vary due to randomness)
        relative_diff = np.abs((perturbed - params) / params)
        assert np.all(relative_diff < 0.5)  # Allow some variation

    def test_perturb_parameters_larger_fraction(self):
        """Test parameter perturbation with larger fraction."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        # Larger perturbation
        np.random.seed(42)
        perturbed = applicator._perturb_parameters(params, perturbation_fraction=0.5)

        assert perturbed.shape == params.shape
        assert not np.allclose(perturbed, params)

    def test_apply_strategy_tighten_bounds_no_bounds(self):
        """Test tighten_bounds strategy when no bounds provided."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "tighten_bounds", params, strategy_param=0.9, bounds=None
        )

        # Should return unchanged params when no bounds
        assert np.allclose(result, params)

    def test_apply_strategy_tighten_bounds_with_bounds(self):
        """Test tighten_bounds strategy with bounds."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([100.0, 200.0, 300.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([1000.0, 1000.0, 1000.0]))

        result = applicator._apply_strategy(
            "tighten_bounds", params, strategy_param=0.9, bounds=bounds
        )

        assert result.shape == params.shape
        # Should be within original bounds
        assert np.all(result >= bounds[0])
        assert np.all(result <= bounds[1])

    def test_apply_strategy_unknown_strategy(self):
        """Test unknown strategy returns unchanged params."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "unknown_strategy", params, strategy_param=None, bounds=None
        )

        # Should return copy of params
        assert np.allclose(result, params)
        assert result is not params  # Should be a copy

    def test_apply_strategy_increase_iterations(self):
        """Test increase_iterations strategy."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "increase_iterations", params, strategy_param=1.5, bounds=None
        )

        # Should return copy of params unchanged
        assert np.allclose(result, params)

    def test_apply_strategy_relax_tolerance(self):
        """Test relax_tolerance strategy."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "relax_tolerance", params, strategy_param=10.0, bounds=None
        )

        # Should return copy of params unchanged
        assert np.allclose(result, params)

    def test_apply_strategy_reduce_step_size(self):
        """Test reduce_step_size strategy."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "reduce_step_size", params, strategy_param=0.5, bounds=None
        )

        # Should return copy of params unchanged
        assert np.allclose(result, params)

    def test_apply_strategy_rescale_data(self):
        """Test rescale_data strategy."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([10.0, 20.0, 30.0])

        result = applicator._apply_strategy(
            "rescale_data", params, strategy_param="normalize", bounds=None
        )

        # Should return copy of params unchanged
        assert np.allclose(result, params)


class TestErrorRecoveryStrategies:
    """Test ERROR_RECOVERY_STRATEGIES dictionary."""

    def test_convergence_error_strategies_defined(self):
        """Test convergence error has strategies defined."""
        strategies = ERROR_RECOVERY_STRATEGIES[NLSQConvergenceError]
        assert len(strategies) == 3
        assert strategies[0] == ("perturb_parameters", 0.05)
        assert strategies[1] == ("increase_iterations", 1.5)
        assert strategies[2] == ("relax_tolerance", 10.0)

    def test_numerical_error_strategies_defined(self):
        """Test numerical error has strategies defined."""
        strategies = ERROR_RECOVERY_STRATEGIES[NLSQNumericalError]
        assert len(strategies) == 3
        assert strategies[0] == ("reduce_step_size", 0.5)
        assert strategies[1] == ("tighten_bounds", 0.9)
        assert strategies[2] == ("rescale_data", "normalize")


class TestRecoveryStrategyIntegration:
    """Integration tests for recovery strategies."""

    def test_full_retry_sequence_convergence_error(self):
        """Test full retry sequence for convergence error."""
        applicator = RecoveryStrategyApplicator(max_retries=3)
        error = NLSQConvergenceError("Failed to converge")
        params = np.array([1.0, 2.0, 3.0])

        # Attempt 0
        assert applicator.should_retry(0)
        result = applicator.get_recovery_strategy(error, params, 0)
        assert result is not None
        assert result[0] == "perturb_parameters"

        # Attempt 1
        assert applicator.should_retry(1)
        result = applicator.get_recovery_strategy(error, params, 1)
        assert result is not None
        assert result[0] == "increase_iterations"

        # Attempt 2
        assert applicator.should_retry(2)
        result = applicator.get_recovery_strategy(error, params, 2)
        assert result is not None
        assert result[0] == "relax_tolerance"

        # Attempt 3 - exhausted
        assert not applicator.should_retry(3)
        result = applicator.get_recovery_strategy(error, params, 3)
        assert result is None

    def test_bounds_respected_throughout(self):
        """Test that bounds are respected in all strategies."""
        applicator = RecoveryStrategyApplicator()
        params = np.array([5.0, 10.0, 15.0])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([20.0, 20.0, 20.0]))
        error = NLSQNumericalError("NaN", detection_point="loss")

        # Try all numerical error strategies
        for attempt in range(3):
            result = applicator.get_recovery_strategy(error, params, attempt, bounds)
            if result is not None:
                _, modified_params = result
                # All modified params should be within bounds
                assert np.all(modified_params >= bounds[0])
                assert np.all(modified_params <= bounds[1])
