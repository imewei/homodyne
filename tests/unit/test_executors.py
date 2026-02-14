"""Unit tests for homodyne.optimization.nlsq.strategies.executors module.

Tests optimization strategy executors including the Strategy pattern
implementation and executor factory.
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.strategies.executors import (
    ExecutionResult,
    LargeDatasetExecutor,
    OptimizationExecutor,
    StandardExecutor,
    StreamingExecutor,
    get_executor,
)


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_execution_result_creation(self):
        """Test basic ExecutionResult instantiation."""
        popt = np.array([1.0, 2.0, 3.0])
        pcov = np.eye(3)
        info = {"success": True, "nfev": 100}

        result = ExecutionResult(
            popt=popt,
            pcov=pcov,
            info=info,
            recovery_actions=["action1"],
            convergence_status="converged",
        )

        np.testing.assert_array_equal(result.popt, popt)
        np.testing.assert_array_equal(result.pcov, pcov)
        assert result.info["success"] is True
        assert result.recovery_actions == ["action1"]
        assert result.convergence_status == "converged"


class TestOptimizationExecutorInterface:
    """Tests for OptimizationExecutor abstract base class."""

    def test_executor_is_abstract(self):
        """Test that OptimizationExecutor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OptimizationExecutor()

    def test_concrete_executor_has_required_methods(self):
        """Test that concrete executors implement required methods."""
        executor = StandardExecutor()

        assert hasattr(executor, "execute")
        assert hasattr(executor, "name")
        assert hasattr(executor, "supports_progress")
        assert callable(executor.execute)


class TestStandardExecutor:
    """Tests for StandardExecutor class."""

    def test_standard_executor_name(self):
        """Test StandardExecutor name property."""
        executor = StandardExecutor()
        assert executor.name == "standard"

    def test_standard_executor_no_progress(self):
        """Test StandardExecutor doesn't support progress."""
        executor = StandardExecutor()
        assert executor.supports_progress is False

    def test_standard_executor_execute_simple(self):
        """Test StandardExecutor execution with simple function."""
        executor = StandardExecutor()

        # Simple linear model: y = a*x + b
        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 100)
        ydata = 2.5 * xdata + 1.0 + np.random.randn(100) * 0.1

        import logging

        logger = logging.getLogger(__name__)

        result = executor.execute(
            residual_fn=model,
            xdata=xdata,
            ydata=ydata,
            initial_params=np.array([1.0, 0.0]),
            bounds=(np.array([0.0, -10.0]), np.array([10.0, 10.0])),
            loss_name="linear",
            x_scale_value=1.0,
            logger=logger,
        )

        assert isinstance(result, ExecutionResult)
        assert result.convergence_status == "converged"
        # Parameters should be close to true values
        np.testing.assert_allclose(result.popt[0], 2.5, rtol=0.1)
        np.testing.assert_allclose(result.popt[1], 1.0, rtol=0.5)


class TestLargeDatasetExecutor:
    """Tests for LargeDatasetExecutor class."""

    def test_large_executor_name(self):
        """Test LargeDatasetExecutor name property."""
        executor = LargeDatasetExecutor()
        assert executor.name == "large"

    def test_large_executor_supports_progress(self):
        """Test LargeDatasetExecutor supports progress."""
        executor = LargeDatasetExecutor()
        assert executor.supports_progress is True

    def test_large_executor_execute(self):
        """Test LargeDatasetExecutor execution."""
        executor = LargeDatasetExecutor()

        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 1000)
        ydata = 2.0 * xdata + 0.5 + np.random.randn(1000) * 0.1

        import logging

        logger = logging.getLogger(__name__)

        result = executor.execute(
            residual_fn=model,
            xdata=xdata,
            ydata=ydata,
            initial_params=np.array([1.0, 0.0]),
            bounds=(np.array([0.0, -10.0]), np.array([10.0, 10.0])),
            loss_name="linear",
            x_scale_value=1.0,
            logger=logger,
        )

        assert isinstance(result, ExecutionResult)
        assert result.info["strategy"] == "large"


class TestStreamingExecutor:
    """Tests for StreamingExecutor class."""

    def test_streaming_executor_name(self):
        """Test StreamingExecutor name property."""
        executor = StreamingExecutor()
        assert executor.name == "streaming"

    def test_streaming_executor_supports_progress(self):
        """Test StreamingExecutor supports progress."""
        executor = StreamingExecutor()
        assert executor.supports_progress is True

    def test_streaming_executor_with_config(self):
        """Test StreamingExecutor with checkpoint config."""
        config = {"chunk_size": 50000, "checkpoint_interval": 5}
        executor = StreamingExecutor(checkpoint_config=config)

        assert executor.checkpoint_config == config


class TestGetExecutor:
    """Tests for get_executor factory function."""

    def test_get_standard_executor(self):
        """Test getting standard executor."""
        executor = get_executor("standard")

        assert isinstance(executor, StandardExecutor)
        assert executor.name == "standard"

    def test_get_large_executor(self):
        """Test getting large dataset executor."""
        executor = get_executor("large")

        assert isinstance(executor, LargeDatasetExecutor)
        assert executor.name == "large"

    def test_get_chunked_executor_maps_to_large(self):
        """Test that chunked strategy maps to large executor."""
        executor = get_executor("chunked")

        assert isinstance(executor, LargeDatasetExecutor)

    def test_get_streaming_executor(self):
        """Test getting streaming executor."""
        executor = get_executor("streaming")

        assert isinstance(executor, StreamingExecutor)
        assert executor.name == "streaming"

    def test_get_streaming_with_config(self):
        """Test getting streaming executor with config."""
        config = {"chunk_size": 100000}
        executor = get_executor("streaming", checkpoint_config=config)

        assert isinstance(executor, StreamingExecutor)
        assert executor.checkpoint_config == config

    def test_unknown_strategy_raises(self):
        """Test that unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_executor("unknown_strategy")

    def test_available_strategies_in_error(self):
        """Test that available strategies are listed in error message."""
        with pytest.raises(ValueError, match="standard.*large.*chunked.*streaming"):
            get_executor("invalid")


class TestExecutorXScaleHandling:
    """Tests for x_scale parameter handling in executors."""

    def test_standard_executor_scalar_x_scale(self):
        """Test StandardExecutor with scalar x_scale."""
        executor = StandardExecutor()

        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 50)
        ydata = 2.0 * xdata + 1.0

        import logging

        logger = logging.getLogger(__name__)

        # Should not raise with scalar x_scale
        result = executor.execute(
            residual_fn=model,
            xdata=xdata,
            ydata=ydata,
            initial_params=np.array([1.0, 0.0]),
            bounds=(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])),
            loss_name="linear",
            x_scale_value=1.0,  # Scalar
            logger=logger,
        )

        assert isinstance(result, ExecutionResult)

    def test_standard_executor_array_x_scale(self):
        """Test StandardExecutor with array x_scale."""
        executor = StandardExecutor()

        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 50)
        ydata = 2.0 * xdata + 1.0

        import logging

        logger = logging.getLogger(__name__)

        result = executor.execute(
            residual_fn=model,
            xdata=xdata,
            ydata=ydata,
            initial_params=np.array([1.0, 0.0]),
            bounds=(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])),
            loss_name="linear",
            x_scale_value=np.array([1.0, 1.0]),  # Array
            logger=logger,
        )

        assert isinstance(result, ExecutionResult)

    def test_standard_executor_jac_x_scale(self):
        """Test StandardExecutor with 'jac' x_scale."""
        executor = StandardExecutor()

        def model(x, a, b):
            return a * x + b

        xdata = np.linspace(0, 10, 50)
        ydata = 2.0 * xdata + 1.0

        import logging

        logger = logging.getLogger(__name__)

        result = executor.execute(
            residual_fn=model,
            xdata=xdata,
            ydata=ydata,
            initial_params=np.array([1.0, 0.5]),
            bounds=(np.array([-np.inf, -np.inf]), np.array([np.inf, np.inf])),
            loss_name="linear",
            x_scale_value="jac",  # String 'jac'
            logger=logger,
        )

        assert isinstance(result, ExecutionResult)


class TestExecutorErrorHandling:
    """Tests for error handling in executors."""

    def test_standard_executor_propagates_errors(self):
        """Test that StandardExecutor propagates optimization errors."""
        executor = StandardExecutor()

        def bad_model(x, a):
            raise ValueError("Intentional error")

        import logging

        logger = logging.getLogger(__name__)

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            executor.execute(
                residual_fn=bad_model,
                xdata=np.array([1.0, 2.0]),
                ydata=np.array([1.0, 2.0]),
                initial_params=np.array([1.0]),
                bounds=None,
                loss_name="linear",
                x_scale_value=1.0,
                logger=logger,
            )
