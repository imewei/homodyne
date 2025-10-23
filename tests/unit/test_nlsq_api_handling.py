"""Comprehensive tests for NLSQ API return type handling.

Tests the _handle_nlsq_result() method in NLSQWrapper to ensure it correctly
normalizes all possible NLSQ v0.1.5 return formats to a consistent
(popt, pcov, info) tuple.

Test Coverage:
--------------
1. Dict format (StreamingOptimizer results)
2. Tuple (2 elements) - (popt, pcov)
3. Tuple (3 elements) - (popt, pcov, info)
4. Object with attributes - CurveFitResult / OptimizeResult
5. Edge cases and error conditions
6. Mock different NLSQ version behaviors
"""

import numpy as np
import pytest
from typing import Any
from unittest.mock import Mock

from homodyne.optimization.nlsq_wrapper import NLSQWrapper, OptimizationStrategy


class TestHandleNLSQResult:
    """Test suite for _handle_nlsq_result() method."""

    # =========================================================================
    # Case 1: Dict Format (StreamingOptimizer Results)
    # =========================================================================

    def test_dict_with_x_and_pcov(self):
        """Test dict with 'x' and 'pcov' keys (StreamingOptimizer standard)."""
        result = {
            'x': np.array([1.0, 2.0, 3.0]),
            'pcov': np.eye(3) * 0.1,
            'success': True,
            'message': 'Optimization succeeded',
            'streaming_diagnostics': {'batches_processed': 100}
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        np.testing.assert_array_equal(popt, result['x'])
        np.testing.assert_array_equal(pcov, result['pcov'])
        assert info['success'] is True
        assert info['message'] == 'Optimization succeeded'
        assert 'streaming_diagnostics' in info
        assert info['streaming_diagnostics']['batches_processed'] == 100

    def test_dict_with_popt_fallback(self):
        """Test dict with 'popt' key instead of 'x' (alternative format)."""
        result = {
            'popt': np.array([10.0, 20.0]),
            'pcov': np.eye(2) * 0.05,
            'success': False,
            'message': 'Max iterations reached'
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        np.testing.assert_array_equal(popt, result['popt'])
        np.testing.assert_array_equal(pcov, result['pcov'])
        assert info['success'] is False

    def test_dict_missing_pcov_creates_identity(self):
        """Test dict without 'pcov' creates identity matrix."""
        result = {
            'x': np.array([5.0, 10.0, 15.0, 20.0]),
            'success': True,
            'message': 'Converged'
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.CHUNKED
        )

        np.testing.assert_array_equal(popt, result['x'])
        # Should create identity matrix with correct size
        expected_pcov = np.eye(4)
        np.testing.assert_array_equal(pcov, expected_pcov)

    def test_dict_with_partial_streaming_diagnostics(self):
        """Test dict with partial streaming diagnostics."""
        result = {
            'x': np.array([1.0, 2.0]),
            'success': True,
            'best_loss': 0.001,
            'final_epoch': 50
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        # Dict format always creates these keys
        assert 'streaming_diagnostics' in info
        assert info['streaming_diagnostics'] == {}  # Empty dict when not provided
        assert info['best_loss'] == 0.001
        assert info['final_epoch'] == 50

    # =========================================================================
    # Case 2: Tuple (2 elements) - (popt, pcov)
    # =========================================================================

    def test_tuple_two_elements_standard(self):
        """Test standard (popt, pcov) tuple return."""
        popt_in = np.array([100.0, 200.0, 300.0])
        pcov_in = np.eye(3) * 0.2
        result = (popt_in, pcov_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_equal(popt, popt_in)
        np.testing.assert_array_equal(pcov, pcov_in)
        assert info == {}  # Empty dict for 2-element tuple

    def test_tuple_two_elements_curve_fit_large(self):
        """Test curve_fit_large returning (popt, pcov) only."""
        popt_in = np.array([1e3, 1e4])
        pcov_in = np.array([[1.0, 0.1], [0.1, 2.0]])
        result = (popt_in, pcov_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        np.testing.assert_array_equal(popt, popt_in)
        np.testing.assert_array_equal(pcov, pcov_in)
        assert isinstance(info, dict)
        assert len(info) == 0

    # =========================================================================
    # Case 3: Tuple (3 elements) - (popt, pcov, info)
    # =========================================================================

    def test_tuple_three_elements_with_info(self):
        """Test (popt, pcov, info) tuple with full_output=True."""
        popt_in = np.array([50.0, 100.0])
        pcov_in = np.eye(2) * 0.15
        info_in = {
            'nfev': 42,
            'njev': 20,
            'mesg': 'Converged successfully',
            'ier': 1
        }
        result = (popt_in, pcov_in, info_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_equal(popt, popt_in)
        np.testing.assert_array_equal(pcov, pcov_in)
        assert info == info_in
        assert info['nfev'] == 42

    def test_tuple_three_elements_empty_info(self):
        """Test (popt, pcov, info) with empty info dict."""
        popt_in = np.array([1.0])
        pcov_in = np.array([[0.01]])
        info_in = {}
        result = (popt_in, pcov_in, info_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_equal(popt, popt_in)
        np.testing.assert_array_equal(pcov, pcov_in)
        assert info == {}

    # =========================================================================
    # Case 4: Object with Attributes (CurveFitResult / OptimizeResult)
    # =========================================================================

    def test_object_with_x_attribute(self):
        """Test object with 'x' attribute (OptimizeResult format)."""
        result = Mock()
        result.x = np.array([10.0, 20.0, 30.0])
        result.pcov = np.eye(3) * 0.5
        result.success = True
        result.message = 'Optimization terminated successfully'
        result.nfev = 100
        result.njev = 50

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        np.testing.assert_array_equal(popt, result.x)
        np.testing.assert_array_equal(pcov, result.pcov)
        assert info['success'] is True
        assert info['message'] == 'Optimization terminated successfully'
        assert info['nfev'] == 100
        assert info['njev'] == 50

    def test_object_with_popt_attribute(self):
        """Test object with 'popt' attribute (CurveFitResult format)."""
        result = Mock(spec=['popt', 'pcov', 'success', 'message'])
        result.popt = np.array([5.0, 15.0])
        result.pcov = np.eye(2) * 0.25
        result.success = False
        result.message = 'Maximum iterations exceeded'

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_equal(popt, result.popt)
        np.testing.assert_array_equal(pcov, result.pcov)
        assert info['success'] is False
        assert info['message'] == 'Maximum iterations exceeded'

    def test_object_missing_pcov_creates_identity(self):
        """Test object without 'pcov' attribute creates identity matrix."""
        result = Mock()
        result.x = np.array([1.0, 2.0, 3.0, 4.0])
        result.success = True
        result.message = 'Converged'
        # Simulate missing pcov attribute
        delattr(result, 'pcov')

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.CHUNKED
        )

        np.testing.assert_array_equal(popt, result.x)
        expected_pcov = np.eye(4)
        np.testing.assert_array_equal(pcov, expected_pcov)

    def test_object_with_all_common_attributes(self):
        """Test object with all common optimization result attributes."""
        result = Mock(spec=['x', 'pcov', 'success', 'message', 'fun', 'jac', 'nfev', 'njev', 'optimality'])
        result.x = np.array([100.0])
        result.pcov = np.array([[1.0]])
        result.success = True
        result.message = 'All good'
        result.fun = 0.123
        result.jac = np.array([0.01])
        result.nfev = 25
        result.njev = 12
        result.optimality = 1e-6

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        # Check all standard attributes extracted
        # Implementation extracts: message, success, nfev, njev, fun, jac, optimality
        assert info['success'] is True
        assert info['message'] == 'All good'
        assert info['fun'] == 0.123
        np.testing.assert_array_equal(info['jac'], np.array([0.01]))
        assert info['nfev'] == 25
        assert info['njev'] == 12
        assert info['optimality'] == 1e-6

    # =========================================================================
    # Edge Cases and Error Conditions
    # =========================================================================

    def test_invalid_tuple_length_raises_error(self):
        """Test that tuple with wrong length raises TypeError."""
        result = (np.array([1.0]), np.eye(1), {}, "extra")  # 4 elements

        with pytest.raises(TypeError, match="Unexpected tuple length"):
            NLSQWrapper._handle_nlsq_result(result, OptimizationStrategy.STANDARD)

    def test_empty_tuple_raises_error(self):
        """Test that empty tuple raises TypeError."""
        result = ()

        with pytest.raises(TypeError, match="Unexpected tuple length"):
            NLSQWrapper._handle_nlsq_result(result, OptimizationStrategy.STANDARD)

    def test_unrecognized_type_raises_error(self):
        """Test that unrecognized type raises TypeError."""
        result = "invalid_result_string"

        with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
            NLSQWrapper._handle_nlsq_result(result, OptimizationStrategy.STANDARD)

    def test_none_result_raises_error(self):
        """Test that None result raises TypeError."""
        with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
            NLSQWrapper._handle_nlsq_result(None, OptimizationStrategy.STANDARD)

    def test_dict_missing_both_x_and_popt_raises_error(self):
        """Test dict without 'x' or 'popt' raises error when converting None to array."""
        result = {'pcov': np.eye(2), 'success': True}

        # np.asarray(None) will raise TypeError or ValueError
        with pytest.raises((TypeError, ValueError)):
            NLSQWrapper._handle_nlsq_result(result, OptimizationStrategy.STREAMING)

    def test_object_missing_both_x_and_popt_raises_error(self):
        """Test object without 'x' or 'popt' raises TypeError."""
        result = Mock(spec=['pcov'])  # Only has pcov, no x or popt
        result.pcov = np.eye(3)

        # Should fall through to unrecognized format error
        with pytest.raises(TypeError, match="Unrecognized NLSQ result format"):
            NLSQWrapper._handle_nlsq_result(result, OptimizationStrategy.STANDARD)

    # =========================================================================
    # Type Conversions and Array Handling
    # =========================================================================

    def test_list_converted_to_array(self):
        """Test that list parameters are converted to numpy arrays."""
        result = {
            'x': [1.0, 2.0, 3.0],  # List instead of array
            'pcov': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # List of lists
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        assert isinstance(popt, np.ndarray)
        assert isinstance(pcov, np.ndarray)
        np.testing.assert_array_equal(popt, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(pcov, np.eye(3))

    def test_scalar_popt_converted_to_array(self):
        """Test single parameter optimization (scalar to array)."""
        result = (np.array([42.0]), np.array([[0.1]]))

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        assert popt.shape == (1,)
        assert pcov.shape == (1, 1)

    # =========================================================================
    # Strategy-Specific Handling
    # =========================================================================

    def test_standard_strategy_with_curve_fit_result(self):
        """Test STANDARD strategy typically returns tuple."""
        popt_in = np.array([10.0, 20.0])
        pcov_in = np.eye(2) * 0.1
        result = (popt_in, pcov_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_equal(popt, popt_in)
        assert info == {}

    def test_large_strategy_with_optimize_result_object(self):
        """Test LARGE strategy can return object."""
        result = Mock()
        result.x = np.array([100.0, 200.0, 300.0])
        result.pcov = np.eye(3)
        result.success = True
        result.message = "Success"

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        assert info['success'] is True
        assert info['message'] == "Success"

    def test_chunked_strategy_progress_info(self):
        """Test CHUNKED strategy extracts standard attributes."""
        result = Mock(spec=['x', 'pcov', 'success', 'message'])
        result.x = np.array([5.0, 10.0])
        result.pcov = np.eye(2)
        result.success = True
        result.message = "Chunked optimization complete"

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.CHUNKED
        )

        # Standard attributes extracted
        assert info['success'] is True
        assert info['message'] == "Chunked optimization complete"

    def test_streaming_strategy_with_full_diagnostics(self):
        """Test STREAMING strategy with comprehensive diagnostics."""
        result = {
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'pcov': np.eye(5) * 0.01,
            'success': True,
            'message': 'Streaming optimization converged',
            'fun': 0.00123,
            'best_loss': 0.00123,
            'final_epoch': 100,
            'streaming_diagnostics': {
                'batches_processed': 1000,
                'batches_succeeded': 995,
                'batches_failed': 5,
                'best_epoch': 87,
                'convergence_rate': 0.95
            }
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        assert 'streaming_diagnostics' in info
        assert info['streaming_diagnostics']['batches_processed'] == 1000
        assert info['streaming_diagnostics']['batches_succeeded'] == 995
        assert info['best_loss'] == 0.00123
        assert info['final_epoch'] == 100

    # =========================================================================
    # Mock Different NLSQ Versions
    # =========================================================================

    def test_nlsq_v015_curve_fit_standard(self):
        """Mock NLSQ v0.1.5 curve_fit standard behavior."""
        # curve_fit returns (popt, pcov) by default
        result = (
            np.array([1.0, 2.0, 3.0]),
            np.eye(3) * 0.1
        )

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        assert isinstance(popt, np.ndarray)
        assert isinstance(pcov, np.ndarray)
        assert isinstance(info, dict)

    def test_nlsq_v015_curve_fit_with_full_output(self):
        """Mock NLSQ v0.1.5 curve_fit with full_output=True."""
        # curve_fit with full_output=True returns (popt, pcov, info)
        result = (
            np.array([10.0, 20.0]),
            np.eye(2),
            {'nfev': 50, 'mesg': 'Success'}
        )

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        assert info['nfev'] == 50
        assert info['mesg'] == 'Success'

    def test_nlsq_v015_curve_fit_large(self):
        """Mock NLSQ v0.1.5 curve_fit_large behavior."""
        # curve_fit_large returns (popt, pcov) only
        result = (
            np.array([100.0, 200.0, 300.0]),
            np.eye(3) * 0.5
        )

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        assert info == {}  # No info dict from curve_fit_large

    def test_nlsq_v015_streaming_optimizer(self):
        """Mock NLSQ v0.1.5 StreamingOptimizer.fit() behavior."""
        # StreamingOptimizer returns dict
        result = {
            'x': np.array([5.0, 10.0, 15.0]),
            'success': True,
            'message': 'Optimization succeeded',
            'fun': 0.001,
            'best_loss': 0.001,
            'final_epoch': 50,
            'streaming_diagnostics': {
                'batches_processed': 500,
                'batches_succeeded': 490,
                'batches_failed': 10
            }
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        assert info['success'] is True
        assert 'streaming_diagnostics' in info

    def test_future_nlsq_version_with_info_attribute(self):
        """Test forward compatibility with objects that have 'info' dict attribute."""
        result = Mock(spec=['x', 'pcov', 'success', 'message', 'info'])
        result.x = np.array([1.0, 2.0])
        result.pcov = np.eye(2)
        result.success = True
        result.message = "Success"
        # Future version might nest additional info in 'info' attribute
        result.info = {
            'convergence_score': 0.95,
            'optimization_time': 5.3,
            'new_feature': "value"
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        # Should extract known attributes
        assert info['success'] is True
        assert info['message'] == "Success"
        # Info dict should be merged
        assert 'convergence_score' in info
        assert info['convergence_score'] == 0.95
        assert info['optimization_time'] == 5.3
        assert info['new_feature'] == "value"

    # =========================================================================
    # Data Integrity and Consistency
    # =========================================================================

    def test_popt_pcov_dimension_consistency(self):
        """Test that pcov dimensions match popt length."""
        n_params = 7
        result = {
            'x': np.random.randn(n_params),
            'pcov': np.eye(n_params) * 0.1
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        assert popt.shape == (n_params,)
        assert pcov.shape == (n_params, n_params)

    def test_identity_pcov_when_missing_has_correct_size(self):
        """Test identity matrix created with correct size when pcov missing."""
        n_params = 9
        result = {'x': np.random.randn(n_params), 'success': True}

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        assert pcov.shape == (n_params, n_params)
        np.testing.assert_array_equal(pcov, np.eye(n_params))

    def test_preserves_array_dtype(self):
        """Test that array dtypes are preserved correctly."""
        popt_in = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        pcov_in = np.eye(3, dtype=np.float64)
        result = (popt_in, pcov_in)

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        assert popt.dtype == np.float64
        assert pcov.dtype == np.float64

    def test_large_parameter_count(self):
        """Test handling of optimization with many parameters."""
        n_params = 50
        result = {
            'x': np.random.randn(n_params),
            'pcov': np.eye(n_params) * 0.01,
            'success': True
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.CHUNKED
        )

        assert len(popt) == n_params
        assert pcov.shape == (n_params, n_params)


class TestNLSQAPIIntegration:
    """Integration tests simulating real NLSQ API calls."""

    def test_simulate_curve_fit_success(self):
        """Simulate successful curve_fit call."""
        # This is what NLSQ curve_fit actually returns
        popt = np.array([1.5, 2.3, 0.8])
        pcov = np.array([
            [0.1, 0.01, 0.0],
            [0.01, 0.2, 0.01],
            [0.0, 0.01, 0.15]
        ])
        result = (popt, pcov)

        popt_out, pcov_out, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STANDARD
        )

        np.testing.assert_array_almost_equal(popt_out, popt)
        np.testing.assert_array_almost_equal(pcov_out, pcov)
        assert info == {}

    def test_simulate_curve_fit_large_success(self):
        """Simulate successful curve_fit_large call."""
        # curve_fit_large returns only (popt, pcov)
        popt = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        pcov = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        result = (popt, pcov)

        popt_out, pcov_out, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        np.testing.assert_array_almost_equal(popt_out, popt)
        np.testing.assert_array_almost_equal(pcov_out, pcov)
        assert isinstance(info, dict)

    def test_simulate_streaming_optimizer_success(self):
        """Simulate successful StreamingOptimizer.fit() call."""
        # StreamingOptimizer returns dict with detailed diagnostics
        result = {
            'x': np.array([10.5, 20.3, 30.1]),
            'success': True,
            'message': 'Optimization terminated successfully.',
            'fun': 0.00234,
            'best_loss': 0.00234,
            'final_epoch': 75,
            'streaming_diagnostics': {
                'batches_processed': 750,
                'batches_succeeded': 748,
                'batches_failed': 2,
                'best_epoch': 73,
                'success_rate': 0.997
            }
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.STREAMING
        )

        np.testing.assert_array_almost_equal(popt, result['x'])
        assert info['success'] is True
        assert info['streaming_diagnostics']['success_rate'] == 0.997

    def test_simulate_convergence_failure(self):
        """Simulate optimization that fails to converge."""
        result = {
            'x': np.array([1.0, 2.0]),  # Last attempted parameters
            'success': False,
            'message': 'Maximum iterations exceeded without convergence',
            'fun': 10.5,  # High loss value
            'final_epoch': 100
        }

        popt, pcov, info = NLSQWrapper._handle_nlsq_result(
            result, OptimizationStrategy.LARGE
        )

        assert info['success'] is False
        assert 'Maximum iterations' in info['message']
        # Identity pcov created when missing
        np.testing.assert_array_equal(pcov, np.eye(2))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
