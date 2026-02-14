"""Integration Tests for Error Recovery
=======================================

Error injection -> verification of correct exception types and recovery paths.

Tests cover:
- NLSQStrategy selection for various dataset sizes
- Exception hierarchy integrity
- Error context preservation through exception chain

These tests verify the error-handling infrastructure without running
actual optimization (which would be too slow for integration tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from homodyne.optimization.exceptions import (
    NLSQCheckpointError,
    NLSQConvergenceError,
    NLSQNumericalError,
    NLSQOptimizationError,
)
from homodyne.optimization.nlsq.memory import (
    NLSQStrategy,
    StrategyDecision,
    estimate_peak_memory_gb,
    select_nlsq_strategy,
)

# ---------------------------------------------------------------------------
# Tests: Strategy Selection
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNLSQStrategySelection:
    """Verify memory-based strategy selection for various dataset sizes."""

    def test_small_dataset_selects_standard(self) -> None:
        """Small datasets (< 1M points) should use STANDARD strategy."""
        decision = select_nlsq_strategy(n_points=100_000, n_params=9)

        assert isinstance(decision, StrategyDecision)
        assert decision.strategy == NLSQStrategy.STANDARD
        assert decision.peak_memory_gb < decision.threshold_gb

    def test_strategy_decision_has_all_fields(self) -> None:
        """StrategyDecision should expose all documented fields."""
        decision = select_nlsq_strategy(n_points=1_000, n_params=3)

        assert hasattr(decision, "strategy")
        assert hasattr(decision, "threshold_gb")
        assert hasattr(decision, "index_memory_gb")
        assert hasattr(decision, "peak_memory_gb")
        assert hasattr(decision, "reason")

        assert isinstance(decision.strategy, NLSQStrategy)
        assert isinstance(decision.threshold_gb, float)
        assert isinstance(decision.reason, str)
        assert decision.threshold_gb > 0

    def test_peak_memory_increases_with_data_size(self) -> None:
        """Larger datasets must produce larger peak memory estimates."""
        small_mem = estimate_peak_memory_gb(n_points=100_000, n_params=9)
        large_mem = estimate_peak_memory_gb(n_points=10_000_000, n_params=9)

        assert large_mem > small_mem, (
            f"Expected large_mem ({large_mem:.2f} GB) > small_mem ({small_mem:.2f} GB)"
        )

    def test_peak_memory_increases_with_param_count(self) -> None:
        """More parameters must produce larger peak memory estimates."""
        few_params = estimate_peak_memory_gb(n_points=1_000_000, n_params=3)
        many_params = estimate_peak_memory_gb(n_points=1_000_000, n_params=53)

        assert many_params > few_params, (
            f"Expected many_params ({many_params:.2f} GB) > "
            f"few_params ({few_params:.2f} GB)"
        )

    def test_strategy_enum_values(self) -> None:
        """NLSQStrategy enum should have the expected member values."""
        assert NLSQStrategy.STANDARD.value == "standard"
        assert NLSQStrategy.OUT_OF_CORE.value == "out_of_core"
        assert NLSQStrategy.HYBRID_STREAMING.value == "hybrid_streaming"

    def test_zero_params_returns_standard(self) -> None:
        """Edge case: n_params=0 should not crash and should select STANDARD."""
        decision = select_nlsq_strategy(n_points=1_000, n_params=0)
        assert decision.strategy == NLSQStrategy.STANDARD
        assert decision.peak_memory_gb == 0.0


# ---------------------------------------------------------------------------
# Tests: Exception Hierarchy
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestNLSQExceptionHierarchy:
    """Verify exception inheritance chain."""

    def test_convergence_error_is_optimization_error(self) -> None:
        """NLSQConvergenceError must be a subclass of NLSQOptimizationError."""
        assert issubclass(NLSQConvergenceError, NLSQOptimizationError)

    def test_numerical_error_is_optimization_error(self) -> None:
        """NLSQNumericalError must be a subclass of NLSQOptimizationError."""
        assert issubclass(NLSQNumericalError, NLSQOptimizationError)

    def test_checkpoint_error_is_optimization_error(self) -> None:
        """NLSQCheckpointError must be a subclass of NLSQOptimizationError."""
        assert issubclass(NLSQCheckpointError, NLSQOptimizationError)

    def test_all_errors_are_exceptions(self) -> None:
        """All custom exceptions must be subclasses of Exception."""
        for exc_class in (
            NLSQOptimizationError,
            NLSQConvergenceError,
            NLSQNumericalError,
            NLSQCheckpointError,
        ):
            assert issubclass(exc_class, Exception), (
                f"{exc_class.__name__} is not a subclass of Exception"
            )

    def test_base_catches_all_subtypes(self) -> None:
        """Catching NLSQOptimizationError must catch all subtypes."""
        subtypes = [
            NLSQConvergenceError("convergence failed"),
            NLSQNumericalError("NaN detected"),
            NLSQCheckpointError("checkpoint corrupted"),
        ]

        for exc in subtypes:
            with pytest.raises(NLSQOptimizationError):
                raise exc


# ---------------------------------------------------------------------------
# Tests: Exception Attribute Preservation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestExceptionAttributesPreserved:
    """Verify error context is maintained through the exception chain."""

    def test_convergence_error_attributes(self) -> None:
        """NLSQConvergenceError should preserve iteration_count and final_loss."""
        params = np.array([1.0, 2.0, 3.0])
        exc = NLSQConvergenceError(
            "Failed to converge after 100 iterations",
            iteration_count=100,
            final_loss=0.42,
            parameters=params,
        )

        assert exc.iteration_count == 100
        assert exc.final_loss == 0.42
        assert exc.parameters is params
        assert "100" in str(exc) or "iteration_count" in str(exc)

    def test_numerical_error_attributes(self) -> None:
        """NLSQNumericalError should preserve detection_point and invalid_values."""
        exc = NLSQNumericalError(
            "NaN in gradient",
            detection_point="gradient",
            invalid_values=["param[0]=NaN", "param[2]=Inf"],
        )

        assert exc.detection_point == "gradient"
        assert len(exc.invalid_values) == 2
        assert "gradient" in str(exc) or "detection_point" in str(exc)

    def test_checkpoint_error_attributes(self) -> None:
        """NLSQCheckpointError should preserve checkpoint_path and operation."""
        original_error = OSError("disk full")
        exc = NLSQCheckpointError(
            "Failed to save checkpoint",
            checkpoint_path="/tmp/checkpoint.h5",
            operation="save",
            io_error=original_error,
        )

        assert exc.checkpoint_path == "/tmp/checkpoint.h5"
        assert exc.operation == "save"
        assert exc.io_error is original_error

    def test_error_context_dict(self) -> None:
        """error_context dict should contain all supplied metadata."""
        exc = NLSQOptimizationError(
            "generic failure",
            error_context={"n_points": 1_000_000, "strategy": "streaming"},
        )

        assert exc.error_context["n_points"] == 1_000_000
        assert exc.error_context["strategy"] == "streaming"
        assert "n_points" in str(exc) or "1000000" in str(exc)

    def test_empty_context_is_valid(self) -> None:
        """Creating an exception without error_context should not fail."""
        exc = NLSQOptimizationError("simple message")
        assert exc.error_context == {}
        assert str(exc) == "simple message"

    def test_convergence_error_context_auto_populated(self) -> None:
        """Iteration count and final loss should appear in error_context."""
        exc = NLSQConvergenceError(
            "max iterations exceeded",
            iteration_count=500,
            final_loss=1.23,
        )

        assert "iteration_count" in exc.error_context
        assert exc.error_context["iteration_count"] == 500
        assert "final_loss" in exc.error_context
        assert exc.error_context["final_loss"] == 1.23
