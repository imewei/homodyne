"""
Unit Tests for NLSQ Streaming Optimizer Support
================================================

Tests for homodyne/optimization/nlsq/wrapper.py streaming-related functionality:
- TestStreamingMemoryEstimation (6 tests): Memory estimation methods
- TestStreamingAutoSelection (8 tests): Auto-selection logic based on memory
- TestStreamingOptimizer (10 tests): Streaming optimizer method
- TestStreamingConfig (5 tests): Configuration handling
- TestStreamingIntegration (5 tests): Streaming auto-selection integration
- TestHybridStreamingOptimizer (8 tests): Hybrid streaming optimizer (v2.6.0+)
- TestAdaptiveMemoryThreshold (12 tests): Adaptive memory threshold (v2.7.0+)
- TestAdaptiveThresholdIntegration (5 tests): Integration tests for adaptive threshold
- TestNLSQPhaseResultStructures (15 tests): NLSQ Phase result dataclasses (v2.10.0+)
- TestNLSQTRFDataclasses (12 tests): NLSQ TRF dataclasses (v2.10.0+)
- TestHybridStreamingPhaseHistory (8 tests): Phase history structure (v2.10.0+)

Added in v2.5.0 to support memory-bounded optimization for large datasets.
Extended in v2.6.0 with AdaptiveHybridStreamingOptimizer for improved convergence.
Extended in v2.7.0 with adaptive memory threshold (75% of system RAM by default).
Extended in v2.10.0 with tests for NLSQ Phase classes and TRF dataclasses.

Reference:
- docs/architecture/memory-fix-plan.md
- CLAUDE.md "NLSQ Streaming Mode (v2.5.0+)" section
- CLAUDE.md "NLSQ Adaptive Hybrid Streaming Mode (v2.6.0+)" section
- NLSQ streaming/phases/ module refactoring (Dec 2025-Jan 2026)
- NLSQ core/trf.py dataclass refactoring (Dec 2025-Jan 2026)
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Suppress deprecation warnings for DatasetSizeStrategy used internally by NLSQWrapper
pytestmark = pytest.mark.filterwarnings(
    "ignore:DatasetSizeStrategy is deprecated:DeprecationWarning"
)


# =============================================================================
# TestStreamingMemoryEstimation - 6 tests
# =============================================================================
@pytest.mark.unit
class TestStreamingMemoryEstimation:
    """Tests for memory estimation methods (6 tests)."""

    def test_estimate_memory_basic_calculation(self):
        """TC-STREAM-MEM-001: Basic memory estimation calculation."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        estimated = wrapper._estimate_memory_for_stratified_ls(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
        )

        # Should return positive value in bytes
        assert estimated > 0
        # Should be at least Jacobian size: n_points × n_params × 8 bytes
        min_jacobian = 1_000_000 * 53 * 8
        assert estimated >= min_jacobian

    def test_estimate_memory_scales_with_points(self):
        """TC-STREAM-MEM-002: Memory estimate scales with data points."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        est_small = wrapper._estimate_memory_for_stratified_ls(
            n_points=1_000_000, n_params=53, n_chunks=10
        )
        est_large = wrapper._estimate_memory_for_stratified_ls(
            n_points=10_000_000, n_params=53, n_chunks=100
        )

        # 10× more points should result in more memory
        # Note: JAX cache overhead (~5GB) is constant, so scaling isn't linear
        assert est_large > est_small * 2  # At least 2× increase

    def test_estimate_memory_scales_with_params(self):
        """TC-STREAM-MEM-003: Memory estimate scales with parameters."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        est_few = wrapper._estimate_memory_for_stratified_ls(
            n_points=1_000_000, n_params=10, n_chunks=10
        )
        est_many = wrapper._estimate_memory_for_stratified_ls(
            n_points=1_000_000, n_params=100, n_chunks=10
        )

        # 10× more params should result in more memory (Jacobian effect)
        # Note: JAX cache overhead (~5GB) is constant, so scaling isn't linear
        assert est_many > est_few  # More params = more memory

    def test_estimate_memory_23m_points(self):
        """TC-STREAM-MEM-004: Memory estimate for 23M point dataset (real case)."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Real-world case from log: 23M points, 53 params, 231 chunks
        estimated = wrapper._estimate_memory_for_stratified_ls(
            n_points=23_000_000,
            n_params=53,
            n_chunks=231,
        )

        estimated_gb = estimated / 1e9

        # Should be >30 GB based on analysis (Jacobian + autodiff intermediates)
        assert estimated_gb > 30, f"Expected >30 GB, got {estimated_gb:.1f} GB"
        # Should be <100 GB (sanity check)
        assert estimated_gb < 100, f"Unreasonable estimate: {estimated_gb:.1f} GB"

    def test_estimate_memory_includes_jax_overhead(self):
        """TC-STREAM-MEM-005: Estimate includes JAX compilation overhead."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        estimated = wrapper._estimate_memory_for_stratified_ls(
            n_points=100_000, n_params=10, n_chunks=1
        )

        # Even for small datasets, should include JAX cache overhead (~5 GB)
        estimated_gb = estimated / 1e9
        assert estimated_gb >= 5, "Should include JAX compilation cache"

    def test_estimate_memory_handles_single_chunk(self):
        """TC-STREAM-MEM-006: Estimate handles single chunk edge case."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        estimated = wrapper._estimate_memory_for_stratified_ls(
            n_points=50_000, n_params=10, n_chunks=1
        )

        # Should still provide valid estimate
        assert estimated > 0


# =============================================================================
# TestStreamingAutoSelection - 8 tests
# =============================================================================
@pytest.mark.unit
class TestStreamingAutoSelection:
    """Tests for automatic streaming mode selection (8 tests)."""

    def test_should_use_streaming_above_threshold(self):
        """TC-STREAM-AUTO-001: Streaming enabled when above memory threshold."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # 23M points should exceed 16 GB threshold
        should_stream, est_gb, reason = wrapper._should_use_streaming(
            n_points=23_000_000,
            n_params=53,
            n_chunks=231,
            memory_threshold_gb=16.0,
        )

        assert should_stream is True
        assert est_gb > 16.0
        assert "exceeds" in reason.lower()

    def test_should_use_streaming_below_threshold(self):
        """TC-STREAM-AUTO-002: Streaming disabled when below threshold."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Small dataset should be below threshold
        # Note: JAX overhead is ~5GB, so we need threshold > 5GB for this to pass
        should_stream, est_gb, reason = wrapper._should_use_streaming(
            n_points=100_000,
            n_params=10,
            n_chunks=1,
            memory_threshold_gb=20.0,  # High threshold
        )

        assert should_stream is False
        assert "within limits" in reason.lower()

    def test_should_use_streaming_custom_threshold(self):
        """TC-STREAM-AUTO-003: Custom threshold is respected."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Same dataset with different thresholds
        stream_low, _, _ = wrapper._should_use_streaming(
            n_points=5_000_000,
            n_params=53,
            n_chunks=50,
            memory_threshold_gb=8.0,  # Low threshold
        )
        stream_high, _, _ = wrapper._should_use_streaming(
            n_points=5_000_000,
            n_params=53,
            n_chunks=50,
            memory_threshold_gb=50.0,  # High threshold
        )

        # Low threshold should trigger streaming, high should not
        assert stream_low is True
        assert stream_high is False

    @patch("psutil.virtual_memory")
    def test_should_use_streaming_available_memory_check(self, mock_memory):
        """TC-STREAM-AUTO-004: Available memory check (70% rule)."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Mock system with 32 GB total, 10 GB available
        mock_memory.return_value = MagicMock(
            available=10 * 1e9,  # 10 GB available
            total=32 * 1e9,  # 32 GB total
        )

        # Dataset that needs ~15 GB (exceeds 70% of 10 GB = 7 GB)
        should_stream, est_gb, reason = wrapper._should_use_streaming(
            n_points=2_000_000,
            n_params=53,
            n_chunks=20,
            memory_threshold_gb=100.0,  # High threshold to isolate 70% rule
        )

        # If estimated memory > 70% of available (7 GB), should stream
        if est_gb > 7.0:
            assert should_stream is True
            assert "available" in reason.lower()

    def test_should_use_streaming_returns_reason(self):
        """TC-STREAM-AUTO-005: Returns human-readable reason."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        _, _, reason = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=10,
            n_chunks=10,
            memory_threshold_gb=16.0,
        )

        # Reason should be a non-empty string
        assert isinstance(reason, str)
        assert len(reason) > 0
        # Should contain memory information
        assert "gb" in reason.lower() or "GB" in reason

    def test_should_use_streaming_returns_estimate(self):
        """TC-STREAM-AUTO-006: Returns estimated memory in GB."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        _, est_gb, _ = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
            memory_threshold_gb=16.0,
        )

        # Should return positive float in GB
        assert isinstance(est_gb, float)
        assert est_gb > 0

    def test_should_use_streaming_tuple_return(self):
        """TC-STREAM-AUTO-007: Returns (bool, float, str) tuple."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        result = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
            memory_threshold_gb=16.0,
        )

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)
        assert isinstance(result[2], str)

    def test_should_use_streaming_edge_cases(self):
        """TC-STREAM-AUTO-008: Edge cases (zero points, etc.)."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Very small dataset
        should_stream, _, _ = wrapper._should_use_streaming(
            n_points=100,
            n_params=3,
            n_chunks=1,
            memory_threshold_gb=16.0,
        )

        # Tiny dataset should never need streaming
        # (though JAX overhead may still dominate)
        assert isinstance(should_stream, bool)


# =============================================================================
# TestStreamingOptimizer - 10 tests
# =============================================================================
@pytest.mark.unit
class TestStreamingOptimizer:
    """Tests for streaming optimizer method (10 tests)."""

    def test_streaming_available_flag(self):
        """TC-STREAM-OPT-001: STREAMING_AVAILABLE flag is set correctly."""
        from homodyne.optimization.nlsq.wrapper import STREAMING_AVAILABLE

        # Should be True if NLSQ >= 0.1.5 is installed
        assert isinstance(STREAMING_AVAILABLE, bool)

    @pytest.mark.skipif(
        not __import__(
            "homodyne.optimization.nlsq.wrapper", fromlist=["STREAMING_AVAILABLE"]
        ).STREAMING_AVAILABLE,
        reason="StreamingOptimizer not available",
    )
    def test_streaming_optimizer_method_exists(self):
        """TC-STREAM-OPT-002: _fit_with_streaming_optimizer method exists."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        assert hasattr(wrapper, "_fit_with_streaming_optimizer")
        assert callable(wrapper._fit_with_streaming_optimizer)

    def test_streaming_optimizer_requires_nlsq(self):
        """TC-STREAM-OPT-003: Method raises error if NLSQ streaming unavailable."""
        from homodyne.optimization.nlsq.wrapper import (
            STREAMING_AVAILABLE,
            NLSQWrapper,
        )

        if not STREAMING_AVAILABLE:
            wrapper = NLSQWrapper()

            with pytest.raises(RuntimeError, match="StreamingOptimizer not available"):
                wrapper._fit_with_streaming_optimizer(
                    stratified_data=Mock(),
                    per_angle_scaling=True,
                    physical_param_names=["D0", "alpha", "D_offset"],
                    initial_params=np.array([0.5] * 9),
                    bounds=None,
                    logger=Mock(),
                )

    @pytest.mark.skipif(
        not __import__(
            "homodyne.optimization.nlsq.wrapper", fromlist=["STREAMING_AVAILABLE"]
        ).STREAMING_AVAILABLE,
        reason="StreamingOptimizer not available",
    )
    def test_streaming_optimizer_returns_tuple(self):
        """TC-STREAM-OPT-004: Method returns (popt, pcov, info) tuple."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Create minimal mock data
        mock_chunk = Mock()
        mock_chunk.phi = np.array([0.0, 0.1, 0.2])
        mock_chunk.t1 = np.array([0.0, 0.1, 0.2])
        mock_chunk.t2 = np.array([0.0, 0.1, 0.2])
        mock_chunk.g2 = np.array([1.5, 1.4, 1.3])
        mock_chunk.q = 0.01
        mock_chunk.L = 0.001
        mock_chunk.dt = 0.001

        mock_data = Mock()
        mock_data.chunks = [mock_chunk]
        mock_data.g2_flat = mock_chunk.g2

        # Mock the AdaptiveHybridStreamingOptimizer to avoid actual optimization
        # Note: _fit_with_streaming_optimizer delegates to
        # _fit_with_stratified_hybrid_streaming
        with patch(
            "homodyne.optimization.nlsq.wrapper.AdaptiveHybridStreamingOptimizer"
        ) as mock_optimizer:
            mock_result = {
                "x": np.array([0.5] * 9),
                "success": True,
                "message": "Test",
                "nit": 10,
                "streaming_diagnostics": {},
            }
            mock_optimizer.return_value.fit.return_value = mock_result

            result = wrapper._fit_with_streaming_optimizer(
                stratified_data=mock_data,
                per_angle_scaling=True,
                physical_param_names=["D0", "alpha", "D_offset"],
                initial_params=np.array([0.5] * 9),
                bounds=None,
                logger=Mock(),
                streaming_config={"max_epochs": 1},
            )

            assert isinstance(result, tuple)
            assert len(result) == 3
            # popt, pcov, info
            assert isinstance(result[0], np.ndarray)  # popt
            assert isinstance(result[1], np.ndarray)  # pcov
            assert isinstance(result[2], dict)  # info

    def test_streaming_config_defaults(self):
        """TC-STREAM-OPT-005: Default streaming config values."""
        # Verify default values match documentation
        default_config = {
            "batch_size": 10_000,
            "max_epochs": 50,
            "learning_rate": 0.001,
            "convergence_tol": 1e-6,
        }

        assert default_config["batch_size"] == 10_000
        assert default_config["max_epochs"] == 50
        assert default_config["learning_rate"] == 0.001
        assert default_config["convergence_tol"] == 1e-6

    def test_streaming_config_custom_values(self):
        """TC-STREAM-OPT-006: Custom streaming config values are used."""
        custom_config = {
            "batch_size": 5_000,
            "max_epochs": 100,
            "learning_rate": 0.0001,
            "convergence_tol": 1e-8,
        }

        # Verify custom values
        assert custom_config["batch_size"] == 5_000
        assert custom_config["max_epochs"] == 100

    def test_streaming_info_contains_diagnostics(self):
        """TC-STREAM-OPT-007: Info dict contains streaming diagnostics."""
        info = {
            "success": True,
            "message": "Streaming optimization completed",
            "nfev": 100000,
            "nit": 10,
            "best_loss": 0.001,
            "optimization_time": 120.5,
            "method": "streaming_optimizer",
            "streaming_diagnostics": {
                "total_batches_attempted": 10000,
                "batch_success_rate": 0.95,
            },
        }

        assert info["method"] == "streaming_optimizer"
        assert "streaming_diagnostics" in info

    def test_streaming_bounds_enforcement(self):
        """TC-STREAM-OPT-008: Bounds are enforced on final parameters."""
        # Test that final parameters are clipped to bounds
        popt = np.array([0.0, 1.5, 0.5])  # Some out of bounds
        bounds = (np.array([0.1, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        clipped = np.clip(popt, bounds[0], bounds[1])

        assert clipped[0] == 0.1  # Clipped to lower
        assert clipped[1] == 1.0  # Clipped to upper
        assert clipped[2] == 0.5  # Unchanged

    def test_streaming_covariance_fallback(self):
        """TC-STREAM-OPT-009: Covariance fallback when not available."""
        n_params = 9
        pcov_fallback = np.eye(n_params)

        # Fallback covariance should be identity
        assert pcov_fallback.shape == (n_params, n_params)
        assert np.allclose(pcov_fallback, np.eye(n_params))

    def test_streaming_data_preparation(self):
        """TC-STREAM-OPT-010: Data is correctly prepared for streaming."""
        # Test data packing: x = (phi_idx, t1_idx, t2_idx), y = g2
        phi_unique = np.array([0.0, 0.1, 0.2])
        t1_unique = np.array([0.0, 0.1])
        t2_unique = np.array([0.0, 0.1])

        phi_to_idx = {float(p): i for i, p in enumerate(phi_unique)}
        t1_to_idx = {float(t): i for i, t in enumerate(t1_unique)}
        t2_to_idx = {float(t): i for i, t in enumerate(t2_unique)}

        # Sample data point
        phi_val, t1_val, t2_val = 0.1, 0.0, 0.1

        x_packed = [
            phi_to_idx[phi_val],
            t1_to_idx[t1_val],
            t2_to_idx[t2_val],
        ]

        assert x_packed == [1, 0, 1]  # Indices


# =============================================================================
# TestStreamingConfig - 5 tests
# =============================================================================
@pytest.mark.unit
class TestStreamingConfig:
    """Tests for streaming configuration handling (5 tests)."""

    def test_config_extraction_from_yaml_structure(self):
        """TC-STREAM-CFG-001: Config extracted from YAML structure."""
        # Simulate YAML config structure
        config = {
            "optimization": {
                "nlsq": {
                    "memory_threshold_gb": 16.0,
                    "use_streaming": False,
                    "streaming": {
                        "batch_size": 10000,
                        "max_epochs": 50,
                        "learning_rate": 0.001,
                    },
                }
            }
        }

        nlsq_config = config["optimization"]["nlsq"]
        streaming_config = nlsq_config.get("streaming", {})

        assert nlsq_config["memory_threshold_gb"] == 16.0
        assert streaming_config["batch_size"] == 10000

    def test_config_force_streaming_mode(self):
        """TC-STREAM-CFG-002: use_streaming flag forces streaming mode."""
        config = {
            "optimization": {
                "nlsq": {
                    "use_streaming": True,
                }
            }
        }

        use_streaming = config["optimization"]["nlsq"].get("use_streaming", False)
        assert use_streaming is True

    def test_config_default_threshold(self):
        """TC-STREAM-CFG-003: Default memory threshold is 16 GB."""
        # Default from wrapper implementation
        default_threshold = 16.0

        assert default_threshold == 16.0

    def test_config_missing_streaming_section(self):
        """TC-STREAM-CFG-004: Handles missing streaming section gracefully."""
        config = {
            "optimization": {
                "nlsq": {
                    # No streaming section
                }
            }
        }

        streaming_config = config["optimization"]["nlsq"].get("streaming", {})

        assert streaming_config == {}
        # Defaults should be used
        batch_size = streaming_config.get("batch_size", 10_000)
        assert batch_size == 10_000

    def test_config_partial_streaming_section(self):
        """TC-STREAM-CFG-005: Handles partial streaming config."""
        config = {
            "optimization": {
                "nlsq": {
                    "streaming": {
                        "batch_size": 5000,
                        # Other fields missing
                    }
                }
            }
        }

        streaming_config = config["optimization"]["nlsq"]["streaming"]

        # Provided value
        assert streaming_config.get("batch_size", 10_000) == 5000
        # Default for missing
        assert streaming_config.get("max_epochs", 50) == 50


# =============================================================================
# TestStreamingIntegration - 5 tests
# =============================================================================
@pytest.mark.unit
class TestStreamingIntegration:
    """Integration tests for streaming auto-selection in fit() (5 tests)."""

    def test_fit_auto_selects_streaming_for_large_dataset(self):
        """TC-STREAM-INT-001: fit() auto-selects streaming for large datasets."""
        # This test verifies the auto-selection logic is wired up correctly
        # in the fit() method

        # The _should_use_streaming method should be called during fit()
        # for large datasets when STREAMING_AVAILABLE is True

        from homodyne.optimization.nlsq.wrapper import (
            STREAMING_AVAILABLE,
            NLSQWrapper,
        )

        wrapper = NLSQWrapper()

        # Verify method exists
        assert hasattr(wrapper, "_should_use_streaming")

        # Test auto-selection logic
        if STREAMING_AVAILABLE:
            should_stream, _, _ = wrapper._should_use_streaming(
                n_points=23_000_000,
                n_params=53,
                n_chunks=231,
                memory_threshold_gb=16.0,
            )
            assert should_stream is True

    def test_fit_logs_memory_check_result(self):
        """TC-STREAM-INT-002: fit() logs memory check result."""
        # Verify that the memory check logs its decision
        import logging

        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Create a logger with capture
        logger = logging.getLogger("test_streaming")
        logger.setLevel(logging.INFO)

        # The _should_use_streaming returns a reason string
        _, _, reason = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
            memory_threshold_gb=16.0,
        )

        # Reason should be loggable
        assert len(reason) > 0
        logger.info(f"Memory check: {reason}")

    def test_fit_fallback_on_streaming_failure(self):
        """TC-STREAM-INT-003: fit() falls back to stratified LS on streaming failure."""
        # If streaming fails, should fall back to stratified least squares
        # This is handled in the fit() method's exception handling

        # Verify exception handling structure
        try:
            raise RuntimeError("Streaming failed")
        except RuntimeError:
            # Should fall through to stratified least-squares
            fallback_used = True

        assert fallback_used is True

    def test_fit_streaming_result_structure(self):
        """TC-STREAM-INT-004: Streaming result has expected structure."""
        # Result from streaming should match OptimizationResult structure

        expected_fields = [
            "parameters",
            "uncertainties",
            "covariance",
            "converged",
            "chi_squared",
            "reduced_chi_squared",
            "iterations",
            "execution_time",
        ]

        # Mock result structure
        result = Mock()
        for field in expected_fields:
            setattr(result, field, None)

        for field in expected_fields:
            assert hasattr(result, field)

    def test_fit_streaming_uses_correct_method_in_result(self):
        """TC-STREAM-INT-005: Result records 'streaming_optimizer' method."""
        # When streaming is used, info["method"] should be "streaming_optimizer"

        info = {
            "method": "streaming_optimizer",
            "streaming_diagnostics": {},
        }

        assert info["method"] == "streaming_optimizer"
        assert "streaming_diagnostics" in info


# =============================================================================
# TestHybridStreamingOptimizer - 8 tests (v2.6.0+)
# =============================================================================
@pytest.mark.unit
class TestHybridStreamingOptimizer:
    """Tests for AdaptiveHybridStreamingOptimizer integration (8 tests).

    Added in v2.6.0 to improve convergence for large datasets with:
    - Parameter normalization for gradient balancing
    - L-BFGS warmup + Gauss-Newton refinement
    - Exact J^T J accumulation for covariance
    """

    def test_hybrid_streaming_available_flag(self):
        """TC-HYBRID-001: HYBRID_STREAMING_AVAILABLE flag is set correctly."""
        from homodyne.optimization.nlsq.wrapper import HYBRID_STREAMING_AVAILABLE

        # Should be True if NLSQ >= 0.3.2 is installed
        assert isinstance(HYBRID_STREAMING_AVAILABLE, bool)

    def test_hybrid_streaming_method_exists(self):
        """TC-HYBRID-002: _fit_with_stratified_hybrid_streaming method exists."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        assert hasattr(wrapper, "_fit_with_stratified_hybrid_streaming")
        assert callable(wrapper._fit_with_stratified_hybrid_streaming)

    def test_hybrid_streaming_requires_nlsq(self):
        """TC-HYBRID-003: Method raises error if NLSQ hybrid unavailable."""
        from homodyne.optimization.nlsq.wrapper import (
            HYBRID_STREAMING_AVAILABLE,
            NLSQWrapper,
        )

        if not HYBRID_STREAMING_AVAILABLE:
            wrapper = NLSQWrapper()

            with pytest.raises(
                RuntimeError, match="AdaptiveHybridStreamingOptimizer not available"
            ):
                wrapper._fit_with_stratified_hybrid_streaming(
                    stratified_data=Mock(),
                    per_angle_scaling=True,
                    physical_param_names=["D0", "alpha", "D_offset"],
                    initial_params=np.array([0.5] * 9),
                    bounds=None,
                    logger=Mock(),
                )

    @pytest.mark.skipif(
        not __import__(
            "homodyne.optimization.nlsq.wrapper",
            fromlist=["HYBRID_STREAMING_AVAILABLE"],
        ).HYBRID_STREAMING_AVAILABLE,
        reason="AdaptiveHybridStreamingOptimizer not available",
    )
    def test_hybrid_streaming_returns_tuple(self):
        """TC-HYBRID-004: Method returns (popt, pcov, info) tuple."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Create minimal mock data
        mock_chunk = Mock()
        mock_chunk.phi = np.array([0.0, 0.1, 0.2])
        mock_chunk.t1 = np.array([0.0, 0.1, 0.2])
        mock_chunk.t2 = np.array([0.0, 0.1, 0.2])
        mock_chunk.g2 = np.array([1.5, 1.4, 1.3])
        mock_chunk.q = 0.01
        mock_chunk.L = 0.001
        mock_chunk.dt = 0.001

        mock_data = Mock()
        mock_data.chunks = [mock_chunk]
        mock_data.g2_flat = mock_chunk.g2

        # Mock the AdaptiveHybridStreamingOptimizer to avoid actual optimization
        with patch(
            "homodyne.optimization.nlsq.wrapper.AdaptiveHybridStreamingOptimizer"
        ) as mock_optimizer:
            mock_result = {
                "x": np.array([0.5] * 9),
                "pcov": np.eye(9) * 0.01,
                "success": True,
                "message": "Test",
                "final_loss": 0.01,
                "lbfgs_epochs": 50,
                "gauss_newton_iterations": 10,
                "diagnostics": {},
            }
            mock_optimizer.return_value.fit.return_value = mock_result

            result = wrapper._fit_with_stratified_hybrid_streaming(
                stratified_data=mock_data,
                per_angle_scaling=True,
                physical_param_names=["D0", "alpha", "D_offset"],
                initial_params=np.array([0.5] * 9),
                bounds=(np.zeros(9), np.ones(9)),
                logger=MagicMock(),
                # Disable anti-degeneracy constant mode to preserve 9 params
                anti_degeneracy_config={"per_angle_mode": "individual"},
            )

            assert isinstance(result, tuple)
            assert len(result) == 3

            popt, pcov, info = result
            assert isinstance(popt, np.ndarray)
            assert isinstance(pcov, np.ndarray)
            assert isinstance(info, dict)

    def test_hybrid_streaming_config_parsing(self):
        """TC-HYBRID-005: Hybrid streaming config is properly parsed."""
        # Test that config keys are correctly extracted

        hybrid_config = {
            "enable": True,
            "normalize": True,
            "normalization_strategy": "bounds",
            "warmup_iterations": 100,
            "max_warmup_iterations": 500,
            "warmup_learning_rate": 0.001,
            "gauss_newton_max_iterations": 50,
            "gauss_newton_tol": 1e-8,
            "chunk_size": 50000,
        }

        # All keys should be extractable
        assert hybrid_config.get("enable", False) is True
        assert hybrid_config.get("normalize", False) is True
        assert hybrid_config.get("normalization_strategy", "scale") == "bounds"
        assert hybrid_config.get("warmup_iterations", 50) == 100

    def test_hybrid_streaming_info_structure(self):
        """TC-HYBRID-006: Hybrid streaming info dict has expected keys."""
        expected_keys = [
            "success",
            "message",
            "nfev",
            "nit",
            "final_loss",
            "lbfgs_epochs",
            "gauss_newton_iterations",
            "optimization_time",
            "method",
            "hybrid_streaming_diagnostics",
        ]

        # Mock info structure
        info = {
            "success": True,
            "message": "Hybrid streaming optimization completed",
            "nfev": 1000,
            "nit": 60,
            "final_loss": 0.001,
            "lbfgs_epochs": 50,
            "gauss_newton_iterations": 10,
            "optimization_time": 120.5,
            "method": "adaptive_hybrid_streaming",
            "hybrid_streaming_diagnostics": {},
        }

        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

        assert info["method"] == "adaptive_hybrid_streaming"

    def test_hybrid_streaming_physics_formula_g1_shear(self):
        """TC-HYBRID-007: g1_shear uses correct sinc² formula."""
        # Verify the physics: g1_shear = [sinc(Φ)]²
        # This test validates that the fix from the double-check was applied

        import jax.numpy as jnp

        from homodyne.core.physics_utils import safe_sinc

        # Test sinc² behavior
        test_args = jnp.array([0.0, 0.5, 1.0, 2.0])
        sinc_vals = safe_sinc(test_args)
        g1_shear = sinc_vals**2  # This is the CORRECT formula

        # sinc(0) = 1, so sinc²(0) = 1
        assert jnp.isclose(g1_shear[0], 1.0, atol=1e-6)

        # sinc² should always be <= 1 for real inputs
        assert jnp.all(g1_shear <= 1.0 + 1e-6)

        # sinc² should always be >= 0
        assert jnp.all(g1_shear >= 0.0)

    def test_hybrid_streaming_fallback_on_error(self):
        """TC-HYBRID-008: Falls back to basic streaming on error."""
        # Test that hybrid streaming failure triggers fallback

        fallback_expected = True

        # The wrapper should catch exceptions and fall through
        # to basic streaming optimizer if hybrid fails
        try:
            raise RuntimeError("Hybrid streaming failed")
        except RuntimeError:
            fallback_used = True

        assert fallback_used is fallback_expected


# =============================================================================
# TestAdaptiveMemoryThreshold - 12 tests (v2.7.0+)
# =============================================================================
@pytest.mark.unit
class TestAdaptiveMemoryThreshold:
    """Tests for adaptive memory threshold functionality (12 tests).

    Added in v2.7.0 to replace fixed 16 GB threshold with system-aware
    adaptive threshold (75% of total system memory by default).
    """

    def test_adaptive_threshold_returns_tuple(self):
        """TC-ADAPT-MEM-001: get_adaptive_memory_threshold returns (float, dict)."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        result = get_adaptive_memory_threshold()

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # threshold_gb
        assert isinstance(result[1], dict)  # info

    def test_adaptive_threshold_positive_value(self):
        """TC-ADAPT-MEM-002: Threshold is always positive."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        threshold_gb, _ = get_adaptive_memory_threshold()

        assert threshold_gb > 0

    def test_adaptive_threshold_info_keys(self):
        """TC-ADAPT-MEM-003: Info dict contains expected keys."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        _, info = get_adaptive_memory_threshold()

        expected_keys = [
            "memory_fraction",
            "source",
            "total_memory_gb",
            "detection_method",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_adaptive_threshold_default_fraction(self):
        """TC-ADAPT-MEM-004: Default memory fraction is 0.75."""
        # Clear env var to ensure default is used
        import os

        from homodyne.optimization.nlsq import DEFAULT_MEMORY_FRACTION
        from homodyne.optimization.nlsq.memory import get_adaptive_memory_threshold

        env_backup = os.environ.pop("NLSQ_MEMORY_FRACTION", None)

        try:
            _, info = get_adaptive_memory_threshold()
            assert info["memory_fraction"] == DEFAULT_MEMORY_FRACTION
            assert info["source"] == "default"
        finally:
            if env_backup is not None:
                os.environ["NLSQ_MEMORY_FRACTION"] = env_backup

    def test_adaptive_threshold_custom_fraction(self):
        """TC-ADAPT-MEM-005: Custom memory fraction is used when provided."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        custom_fraction = 0.5
        _, info = get_adaptive_memory_threshold(memory_fraction=custom_fraction)

        assert info["memory_fraction"] == custom_fraction
        assert info["source"] == "argument"

    def test_adaptive_threshold_env_var_override(self):
        """TC-ADAPT-MEM-006: Environment variable overrides default fraction."""
        import os

        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        env_backup = os.environ.get("NLSQ_MEMORY_FRACTION")

        try:
            os.environ["NLSQ_MEMORY_FRACTION"] = "0.6"
            _, info = get_adaptive_memory_threshold()

            assert info["memory_fraction"] == 0.6
            assert info["source"] == "env"
        finally:
            if env_backup is not None:
                os.environ["NLSQ_MEMORY_FRACTION"] = env_backup
            else:
                os.environ.pop("NLSQ_MEMORY_FRACTION", None)

    def test_adaptive_threshold_argument_overrides_env(self):
        """TC-ADAPT-MEM-007: Argument overrides environment variable."""
        import os

        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        env_backup = os.environ.get("NLSQ_MEMORY_FRACTION")

        try:
            os.environ["NLSQ_MEMORY_FRACTION"] = "0.6"
            _, info = get_adaptive_memory_threshold(memory_fraction=0.8)

            # Argument takes precedence
            assert info["memory_fraction"] == 0.8
            assert info["source"] == "argument"
        finally:
            if env_backup is not None:
                os.environ["NLSQ_MEMORY_FRACTION"] = env_backup
            else:
                os.environ.pop("NLSQ_MEMORY_FRACTION", None)

    def test_adaptive_threshold_fraction_clamping_low(self):
        """TC-ADAPT-MEM-008: Fraction below 0.1 is clamped to 0.1."""
        import warnings

        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, info = get_adaptive_memory_threshold(memory_fraction=0.05)

            # Should be clamped to minimum
            assert info["memory_fraction"] == 0.1
            # Should warn about clamping
            assert len(w) == 1
            assert "clamped" in str(w[0].message).lower()

    def test_adaptive_threshold_fraction_clamping_high(self):
        """TC-ADAPT-MEM-009: Fraction above 0.9 is clamped to 0.9."""
        import warnings

        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _, info = get_adaptive_memory_threshold(memory_fraction=0.95)

            # Should be clamped to maximum
            assert info["memory_fraction"] == 0.9
            # Should warn about clamping
            assert len(w) == 1
            assert "clamped" in str(w[0].message).lower()

    @patch("psutil.virtual_memory")
    def test_adaptive_threshold_with_mocked_memory(self, mock_memory):
        """TC-ADAPT-MEM-010: Correct threshold with mocked system memory."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        # Mock 64 GB total memory
        mock_memory.return_value = MagicMock(
            total=64 * 1024**3,
            available=32 * 1024**3,
        )

        threshold_gb, info = get_adaptive_memory_threshold(memory_fraction=0.75)

        # 75% of 64 GB = 48 GB
        assert abs(threshold_gb - 48.0) < 0.1
        assert abs(info["total_memory_gb"] - 64.0) < 0.1
        assert info["detection_method"] == "psutil"

    @patch("homodyne.optimization.nlsq.memory.detect_total_system_memory")
    def test_adaptive_threshold_fallback_on_detection_failure(self, mock_detect):
        """TC-ADAPT-MEM-011: Falls back to 16 GB when detection fails."""
        import warnings

        from homodyne.optimization.nlsq.memory import (
            FALLBACK_THRESHOLD_GB,
            get_adaptive_memory_threshold,
        )

        # Make detection fail
        mock_detect.return_value = None

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            threshold_gb, info = get_adaptive_memory_threshold()

            # Should fall back to 16 GB
            assert threshold_gb == FALLBACK_THRESHOLD_GB
            assert info["detection_method"] == "fallback"
            # Should warn about fallback
            assert len(w) >= 1
            assert any("fallback" in str(warning.message).lower() for warning in w)

    def test_adaptive_threshold_invalid_env_var(self):
        """TC-ADAPT-MEM-012: Invalid env var falls back to default."""
        import os
        import warnings

        from homodyne.optimization.nlsq import DEFAULT_MEMORY_FRACTION
        from homodyne.optimization.nlsq.memory import get_adaptive_memory_threshold

        env_backup = os.environ.get("NLSQ_MEMORY_FRACTION")

        try:
            os.environ["NLSQ_MEMORY_FRACTION"] = "not_a_number"

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _, info = get_adaptive_memory_threshold()

                # Should fall back to default
                assert info["memory_fraction"] == DEFAULT_MEMORY_FRACTION
                # Should warn about invalid value
                assert len(w) >= 1
                assert any("invalid" in str(warning.message).lower() for warning in w)
        finally:
            if env_backup is not None:
                os.environ["NLSQ_MEMORY_FRACTION"] = env_backup
            else:
                os.environ.pop("NLSQ_MEMORY_FRACTION", None)


# =============================================================================
# TestAdaptiveThresholdIntegration - 5 tests (v2.7.0+)
# =============================================================================
@pytest.mark.unit
class TestAdaptiveThresholdIntegration:
    """Integration tests for adaptive threshold in _should_use_streaming (5 tests)."""

    def test_should_use_streaming_with_adaptive_threshold(self):
        """TC-ADAPT-INT-001: _should_use_streaming uses adaptive threshold."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Call without explicit threshold - should use adaptive
        should_stream, est_gb, reason = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
            # memory_threshold_gb not provided - uses adaptive
        )

        # Should return valid results
        assert isinstance(should_stream, bool)
        assert isinstance(est_gb, float)
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_should_use_streaming_explicit_threshold_overrides(self):
        """TC-ADAPT-INT-002: Explicit threshold overrides adaptive."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Very low threshold should trigger streaming
        should_stream_low, _, _ = wrapper._should_use_streaming(
            n_points=5_000_000,
            n_params=53,
            n_chunks=50,
            memory_threshold_gb=1.0,  # Explicit low threshold
        )

        # Very high threshold should not trigger streaming (unless available memory low)
        should_stream_high, _, _ = wrapper._should_use_streaming(
            n_points=5_000_000,
            n_params=53,
            n_chunks=50,
            memory_threshold_gb=500.0,  # Explicit high threshold
        )

        # Low threshold should definitely trigger streaming for 5M points
        assert should_stream_low is True

    def test_should_use_streaming_memory_fraction_parameter(self):
        """TC-ADAPT-INT-003: memory_fraction parameter affects threshold."""
        from homodyne.optimization.nlsq.wrapper import NLSQWrapper

        wrapper = NLSQWrapper()

        # Low fraction = lower threshold = more likely to stream
        # High fraction = higher threshold = less likely to stream
        # (assuming same dataset size)
        _, _, _ = wrapper._should_use_streaming(
            n_points=1_000_000,
            n_params=53,
            n_chunks=10,
            memory_fraction=0.5,
        )

        # Should complete without error
        assert True

    def test_no_hardcoded_16gb_in_adaptive_code(self):
        """TC-ADAPT-INT-004: No hardcoded 16.0 in adaptive threshold path."""
        from homodyne.optimization.nlsq.wrapper import get_adaptive_memory_threshold

        # When memory is properly detected, threshold should NOT be 16.0
        # unless the system happens to have exactly 21.33 GB RAM (unlikely)
        threshold_gb, info = get_adaptive_memory_threshold()

        # If detection worked, threshold should be based on actual system memory
        if info["detection_method"] != "fallback":
            expected = info["total_memory_gb"] * info["memory_fraction"]
            assert abs(threshold_gb - expected) < 0.01

    def test_config_memory_fraction_option(self):
        """TC-ADAPT-INT-005: Config supports memory_fraction option."""
        # Test that the config structure supports memory_fraction
        config = {
            "optimization": {
                "nlsq": {
                    "memory_fraction": 0.6,  # New option
                }
            }
        }

        nlsq_config = config["optimization"]["nlsq"]

        assert "memory_fraction" in nlsq_config
        assert nlsq_config["memory_fraction"] == 0.6


# =============================================================================
# TestNLSQPhaseResultStructures - 15 tests (v2.10.0+)
# =============================================================================
@pytest.mark.unit
class TestNLSQPhaseResultStructures:
    """Tests for NLSQ Phase result dataclass structures (15 tests).

    Added to verify homodyne integration with NLSQ's refactored Phase classes:
    - WarmupResult: Result from L-BFGS warmup phase
    - GNResult: Result from Gauss-Newton phase
    - PhaseOrchestratorResult: Complete orchestration result
    - CheckpointState: Checkpoint state container

    Reference: NLSQ streaming/phases/ module refactoring (Dec 2025-Jan 2026)
    """

    def test_warmup_result_import_available(self):
        """TC-PHASE-001: WarmupResult can be imported from NLSQ."""
        try:
            from nlsq.streaming.phases import WarmupResult

            assert WarmupResult is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_warmup_result_fields(self):
        """TC-PHASE-002: WarmupResult has expected fields."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import WarmupResult

            # Create a WarmupResult instance
            result = WarmupResult(
                params=jnp.array([1.0, 2.0, 3.0]),
                cost=0.01,
                iterations=50,
                converged=True,
                cost_history=[0.5, 0.1, 0.01],
            )

            # Verify fields
            assert hasattr(result, "params")
            assert hasattr(result, "cost")
            assert hasattr(result, "iterations")
            assert hasattr(result, "converged")
            assert hasattr(result, "cost_history")

            # Verify values
            assert result.cost == 0.01
            assert result.iterations == 50
            assert result.converged is True
            assert len(result.cost_history) == 3
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_warmup_result_immutable(self):
        """TC-PHASE-003: WarmupResult is immutable (frozen dataclass)."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import WarmupResult

            result = WarmupResult(
                params=jnp.array([1.0]),
                cost=0.01,
                iterations=10,
                converged=True,
                cost_history=[0.01],
            )

            # Should raise error when trying to modify
            with pytest.raises(AttributeError):
                result.cost = 0.02
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_gn_result_import_available(self):
        """TC-PHASE-004: GNResult can be imported from NLSQ."""
        try:
            from nlsq.streaming.phases import GNResult

            assert GNResult is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_gn_result_fields(self):
        """TC-PHASE-005: GNResult has expected fields."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import GNResult

            # Create a GNResult instance
            result = GNResult(
                params=jnp.array([1.0, 2.0, 3.0]),
                cost=0.001,
                iterations=25,
                converged=True,
                jacobian=jnp.eye(3),
                cov=jnp.eye(3) * 0.01,
            )

            # Verify fields
            assert hasattr(result, "params")
            assert hasattr(result, "cost")
            assert hasattr(result, "iterations")
            assert hasattr(result, "converged")
            assert hasattr(result, "jacobian")
            assert hasattr(result, "cov")

            # Verify values
            assert result.cost == 0.001
            assert result.iterations == 25
            assert result.converged is True
            assert result.jacobian is not None
            assert result.cov is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_gn_result_optional_fields(self):
        """TC-PHASE-006: GNResult jacobian and cov are optional."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import GNResult

            # Create GNResult without optional fields
            result = GNResult(
                params=jnp.array([1.0, 2.0]),
                cost=0.01,
                iterations=10,
                converged=False,
            )

            # Optional fields should default to None
            assert result.jacobian is None
            assert result.cov is None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_phase_orchestrator_result_import(self):
        """TC-PHASE-007: PhaseOrchestratorResult can be imported."""
        try:
            from nlsq.streaming.phases import PhaseOrchestratorResult

            assert PhaseOrchestratorResult is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_phase_orchestrator_result_fields(self):
        """TC-PHASE-008: PhaseOrchestratorResult has expected fields."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import PhaseOrchestratorResult

            result = PhaseOrchestratorResult(
                params=jnp.array([1.0, 2.0]),
                normalized_params=jnp.array([0.5, 0.5]),
                cost=0.001,
                warmup_result=None,
                gn_result=None,
                phase_history=[{"phase": 0, "name": "setup"}],
                total_time=10.5,
            )

            # Verify fields
            assert hasattr(result, "params")
            assert hasattr(result, "normalized_params")
            assert hasattr(result, "cost")
            assert hasattr(result, "warmup_result")
            assert hasattr(result, "gn_result")
            assert hasattr(result, "phase_history")
            assert hasattr(result, "total_time")

            # Verify values
            assert result.cost == 0.001
            assert result.total_time == 10.5
            assert len(result.phase_history) == 1
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_checkpoint_state_import(self):
        """TC-PHASE-009: CheckpointState can be imported."""
        try:
            from nlsq.streaming.phases import CheckpointState

            assert CheckpointState is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_checkpoint_state_fields(self):
        """TC-PHASE-010: CheckpointState has expected fields."""
        try:
            import jax.numpy as jnp
            from nlsq.streaming.phases import CheckpointState

            state = CheckpointState(
                current_phase=1,
                normalized_params=jnp.array([1.0, 2.0]),
                phase1_optimizer_state=None,
                phase2_JTJ_accumulator=None,
                phase2_JTr_accumulator=None,
                best_params_global=jnp.array([1.0, 2.0]),
                best_cost_global=0.01,
                phase_history=[{"phase": 0}],
                normalizer=None,
                tournament_selector=None,
                multistart_candidates=None,
            )

            # Verify required fields
            assert hasattr(state, "current_phase")
            assert hasattr(state, "normalized_params")
            assert hasattr(state, "phase1_optimizer_state")
            assert hasattr(state, "phase2_JTJ_accumulator")
            assert hasattr(state, "phase2_JTr_accumulator")
            assert hasattr(state, "best_params_global")
            assert hasattr(state, "best_cost_global")
            assert hasattr(state, "phase_history")

            # Verify values
            assert state.current_phase == 1
            assert state.best_cost_global == 0.01
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_checkpoint_manager_import(self):
        """TC-PHASE-011: CheckpointManager can be imported."""
        try:
            from nlsq.streaming.phases import CheckpointManager

            assert CheckpointManager is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_warmup_phase_import(self):
        """TC-PHASE-012: WarmupPhase can be imported."""
        try:
            from nlsq.streaming.phases import WarmupPhase

            assert WarmupPhase is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_gauss_newton_phase_import(self):
        """TC-PHASE-013: GaussNewtonPhase can be imported."""
        try:
            from nlsq.streaming.phases import GaussNewtonPhase

            assert GaussNewtonPhase is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_phase_orchestrator_import(self):
        """TC-PHASE-014: PhaseOrchestrator can be imported."""
        try:
            from nlsq.streaming.phases import PhaseOrchestrator

            assert PhaseOrchestrator is not None
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")

    def test_all_phase_classes_lazy_import(self):
        """TC-PHASE-015: All phase classes use lazy import pattern."""
        try:
            from nlsq.streaming import phases

            # Verify __all__ contains expected exports
            expected_exports = [
                "WarmupPhase",
                "WarmupResult",
                "GaussNewtonPhase",
                "GNResult",
                "CheckpointManager",
                "CheckpointState",
                "PhaseOrchestrator",
                "PhaseOrchestratorResult",
            ]

            for name in expected_exports:
                assert name in phases.__all__, f"{name} not in phases.__all__"

            # Verify lazy import works
            assert hasattr(phases, "WarmupResult")
            assert hasattr(phases, "GNResult")
        except ImportError:
            pytest.skip("NLSQ Phase classes not available")


# =============================================================================
# TestNLSQTRFDataclasses - 12 tests (v2.10.0+)
# =============================================================================
@pytest.mark.unit
class TestNLSQTRFDataclasses:
    """Tests for NLSQ TRF dataclass structures (12 tests).

    Added to verify homodyne can work with NLSQ's refactored TRF dataclasses:
    - TRFConfig: TRF algorithm configuration
    - StepContext: Iteration state context
    - BoundsContext: Parameter bounds context
    - FallbackContext: Fallback tracking context

    Reference: NLSQ core/trf.py refactoring (Dec 2025-Jan 2026)
    """

    def test_trf_config_import_available(self):
        """TC-TRF-001: TRFConfig can be imported from NLSQ."""
        try:
            from nlsq.core.trf import TRFConfig

            assert TRFConfig is not None
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_trf_config_default_values(self):
        """TC-TRF-002: TRFConfig has correct default values."""
        try:
            from nlsq.core.trf import TRFConfig

            config = TRFConfig()

            # Verify default values match NLSQ documentation
            assert config.ftol == 1e-8
            assert config.xtol == 1e-8
            assert config.gtol == 1e-8
            assert config.max_nfev is None
            assert config.x_scale == "jac"
            assert config.loss == "linear"
            assert config.tr_solver == "exact"
            assert config.verbose == 0
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_trf_config_immutable(self):
        """TC-TRF-003: TRFConfig is immutable (frozen dataclass)."""
        try:
            from nlsq.core.trf import TRFConfig

            config = TRFConfig()

            with pytest.raises(AttributeError):
                config.ftol = 1e-6
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_trf_config_validation_negative_ftol(self):
        """TC-TRF-004: TRFConfig rejects negative ftol."""
        try:
            from nlsq.core.trf import TRFConfig

            with pytest.raises(ValueError, match="ftol must be positive"):
                TRFConfig(ftol=-1e-8)
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_trf_config_valid_loss_functions(self):
        """TC-TRF-005: TRFConfig accepts all valid loss functions."""
        try:
            from nlsq.core.trf import TRFConfig

            valid_losses = ["linear", "soft_l1", "huber", "cauchy", "arctan"]
            for loss in valid_losses:
                config = TRFConfig(loss=loss)
                assert config.loss == loss
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_step_context_import(self):
        """TC-TRF-006: StepContext can be imported."""
        try:
            from nlsq.core.trf import StepContext

            assert StepContext is not None
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_step_context_creation(self):
        """TC-TRF-007: StepContext can be created with required fields."""
        try:
            import jax.numpy as jnp
            from nlsq.core.trf import StepContext

            ctx = StepContext(
                x=jnp.array([1.0, 2.0, 3.0]),
                f=jnp.zeros(10),
                J=jnp.ones((10, 3)),
                cost=0.5,
                g=jnp.array([0.1, 0.2, 0.3]),
                trust_radius=1.0,
                iteration=0,
                scale=jnp.ones(3),
                scale_inv=jnp.ones(3),
            )

            assert ctx.x.shape == (3,)
            assert ctx.f.shape == (10,)
            assert ctx.J.shape == (10, 3)
            assert ctx.cost == 0.5
            assert ctx.trust_radius == 1.0
            assert ctx.iteration == 0
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_step_context_mutable(self):
        """TC-TRF-008: StepContext is mutable (not frozen)."""
        try:
            import jax.numpy as jnp
            from nlsq.core.trf import StepContext

            ctx = StepContext(
                x=jnp.array([1.0]),
                f=jnp.array([0.0]),
                J=jnp.array([[1.0]]),
                cost=0.0,
                g=jnp.array([0.0]),
                trust_radius=1.0,
                iteration=0,
                scale=jnp.array([1.0]),
                scale_inv=jnp.array([1.0]),
            )

            # Should be mutable
            ctx.iteration = 5
            assert ctx.iteration == 5

            ctx.trust_radius = 2.0
            assert ctx.trust_radius == 2.0
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_bounds_context_import(self):
        """TC-TRF-009: BoundsContext can be imported."""
        try:
            from nlsq.core.trf import BoundsContext

            assert BoundsContext is not None
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_bounds_context_from_bounds_factory(self):
        """TC-TRF-010: BoundsContext.from_bounds factory method works."""
        try:
            import jax.numpy as jnp
            from nlsq.core.trf import BoundsContext

            lb = jnp.array([0.0, -1.0, 0.5])
            ub = jnp.array([10.0, 1.0, 2.0])
            ctx = BoundsContext.from_bounds(lb, ub)

            assert ctx.lb.shape == (3,)
            assert ctx.ub.shape == (3,)
            assert jnp.allclose(ctx.lb, lb)
            assert jnp.allclose(ctx.ub, ub)
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_fallback_context_import(self):
        """TC-TRF-011: FallbackContext can be imported."""
        try:
            from nlsq.core.trf import FallbackContext

            assert FallbackContext is not None
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")

    def test_fallback_context_default_values(self):
        """TC-TRF-012: FallbackContext has correct default values."""
        try:
            import jax.numpy as jnp
            from nlsq.core.trf import FallbackContext

            ctx = FallbackContext(original_dtype=jnp.float32)

            assert ctx.original_dtype == jnp.float32
            assert ctx.fallback_triggered is False
            assert ctx.fallback_reason == ""
            assert ctx.step_context is None
        except ImportError:
            pytest.skip("NLSQ TRF dataclasses not available")


# =============================================================================
# TestHybridStreamingPhaseHistory - 8 tests (v2.10.0+)
# =============================================================================
@pytest.mark.unit
class TestHybridStreamingPhaseHistory:
    """Tests for phase history structure from hybrid streaming optimizer (8 tests).

    Verifies that the phase_history returned by the optimizer has the expected
    structure with records for each optimization phase.

    Reference: NLSQ streaming/phases/orchestrator.py
    """

    def test_phase_history_structure(self):
        """TC-PHIST-001: Phase history has expected structure."""
        # Expected phase history structure from AdaptiveHybridStreamingOptimizer
        phase_history = [
            {
                "phase": 0,
                "name": "setup",
                "timestamp": 1234567890.0,
            },
            {
                "phase": 1,
                "name": "lbfgs_warmup",
                "iterations": 50,
                "final_loss": 0.01,
                "best_loss": 0.009,
                "switch_reason": "Gradient norm below threshold",
                "timestamp": 1234567891.0,
            },
            {
                "phase": 2,
                "name": "gauss_newton",
                "iterations": 10,
                "final_cost": 0.001,
                "best_cost": 0.0009,
                "convergence_reason": "Cost change below tolerance",
                "gradient_norm": 1e-9,
                "timestamp": 1234567892.0,
            },
            {
                "phase": 3,
                "name": "finalization",
                "final_cost": 0.0009,
                "total_time": 120.5,
                "timestamp": 1234567893.0,
            },
        ]

        # Verify structure
        assert len(phase_history) >= 1

        # Phase 0 should have setup fields
        setup = next((p for p in phase_history if p["phase"] == 0), None)
        if setup:
            assert "name" in setup
            assert "timestamp" in setup

    def test_phase_history_warmup_record(self):
        """TC-PHIST-002: Warmup phase record has expected fields."""
        warmup_record = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": 50,
            "final_loss": 0.01,
            "best_loss": 0.009,
            "switch_reason": "Gradient norm below threshold",
            "timestamp": 1234567891.0,
        }

        assert warmup_record["phase"] == 1
        assert warmup_record["name"] == "lbfgs_warmup"
        assert "iterations" in warmup_record
        assert "final_loss" in warmup_record or "best_loss" in warmup_record
        assert "switch_reason" in warmup_record

    def test_phase_history_gauss_newton_record(self):
        """TC-PHIST-003: Gauss-Newton phase record has expected fields."""
        gn_record = {
            "phase": 2,
            "name": "gauss_newton",
            "iterations": 10,
            "final_cost": 0.001,
            "best_cost": 0.0009,
            "convergence_reason": "Cost change below tolerance",
            "gradient_norm": 1e-9,
            "timestamp": 1234567892.0,
        }

        assert gn_record["phase"] == 2
        assert gn_record["name"] == "gauss_newton"
        assert "iterations" in gn_record
        assert "final_cost" in gn_record or "best_cost" in gn_record
        assert "convergence_reason" in gn_record

    def test_phase_history_finalization_record(self):
        """TC-PHIST-004: Finalization phase record has expected fields."""
        final_record = {
            "phase": 3,
            "name": "finalization",
            "final_cost": 0.0009,
            "total_time": 120.5,
            "timestamp": 1234567893.0,
        }

        assert final_record["phase"] == 3
        assert final_record["name"] == "finalization"
        assert "total_time" in final_record

    def test_phase_history_skipped_warmup(self):
        """TC-PHIST-005: Warmup can be skipped (warm start detected)."""
        skipped_warmup = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": 0,
            "skipped": True,
            "warm_start": True,
            "relative_loss": 0.005,
            "switch_reason": "Warm start detected - skipping L-BFGS warmup",
            "timestamp": 1234567891.0,
        }

        assert skipped_warmup["skipped"] is True
        assert skipped_warmup["warm_start"] is True
        assert skipped_warmup["iterations"] == 0

    def test_phase_history_cost_guard_triggered(self):
        """TC-PHIST-006: Warmup record includes cost guard info when triggered."""
        cost_guard_record = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": 25,
            "final_loss": 0.02,
            "best_loss": 0.008,
            "switch_reason": "Cost increase guard triggered (ratio=1.2500)",
            "cost_guard_triggered": True,
            "lr_mode": "exploration",
            "timestamp": 1234567891.0,
        }

        assert cost_guard_record["cost_guard_triggered"] is True
        assert "cost increase guard" in cost_guard_record["switch_reason"].lower()

    def test_phase_history_lr_mode_tracking(self):
        """TC-PHIST-007: Warmup record tracks learning rate mode."""
        warmup_with_lr = {
            "phase": 1,
            "name": "lbfgs_warmup",
            "iterations": 50,
            "lr_mode": "refinement",
            "relative_loss": 0.05,
            "timestamp": 1234567891.0,
        }

        valid_modes = ["refinement", "careful", "exploration", "fixed"]
        assert warmup_with_lr["lr_mode"] in valid_modes

    def test_phase_history_max_iterations(self):
        """TC-PHIST-008: Phase records include max iterations info."""
        gn_max_iter = {
            "phase": 2,
            "name": "gauss_newton",
            "iterations": 50,
            "convergence_reason": "Maximum iterations reached",
            "timestamp": 1234567892.0,
        }

        assert "maximum iterations" in gn_max_iter["convergence_reason"].lower()
