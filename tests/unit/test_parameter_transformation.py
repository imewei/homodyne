"""
Unit tests for parameter and data transformations in NLSQ wrapper.

Tests cover:
- Multi-dimensional XPCS data → flattened 1D arrays (T006)
- Bounds format conversion (T010)
- Parameter transformation between homodyne and NLSQ formats
"""

import pytest
import numpy as np
import jax.numpy as jnp
from homodyne.optimization.nlsq_wrapper import NLSQWrapper


class TestDataFlattening:
    """Test data flattening transformation (T006)."""

    def test_data_flattening_shape_transformation(self):
        """
        Verify multi-dimensional XPCS data (n_phi, n_t1, n_t2) → flattened 1D arrays.

        Test data: (23, 1001, 1001) → 23,023,023 elements
        Verify meshgrid indexing='ij', flatten() preserves order.
        """
        # Create mock XPCS data with known structure
        n_phi, n_t1, n_t2 = 23, 1001, 1001
        expected_size = n_phi * n_t1 * n_t2  # 23,023,023

        # Mock data object
        class MockXPCSData:
            def __init__(self):
                self.phi = np.linspace(0, 2*np.pi, n_phi)
                self.t1 = np.linspace(0, 1, n_t1)
                self.t2 = np.linspace(0, 1, n_t2)
                # 3D correlation data
                self.g2 = np.random.rand(n_phi, n_t1, n_t2)

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        # Execute transformation
        xdata, ydata = wrapper._prepare_data(mock_data)

        # Assertions
        assert xdata.shape[0] == expected_size, \
            f"xdata should have {expected_size} elements, got {xdata.shape[0]}"
        assert ydata.shape[0] == expected_size, \
            f"ydata should have {expected_size} elements, got {ydata.shape[0]}"
        assert xdata.ndim == 1, f"xdata should be 1D, got {xdata.ndim}D"
        assert ydata.ndim == 1, f"ydata should be 1D, got {ydata.ndim}D"

    def test_meshgrid_indexing_order(self):
        """
        Verify meshgrid uses indexing='ij' to preserve correct ordering.

        This ensures compatibility with homodyne's physics calculations.
        """
        # Small test case for manual verification
        n_phi, n_t1, n_t2 = 2, 3, 4

        class MockXPCSData:
            def __init__(self):
                self.phi = np.array([0.0, 1.0])
                self.t1 = np.array([0.0, 0.5, 1.0])
                self.t2 = np.array([0.0, 0.33, 0.67, 1.0])
                self.g2 = np.arange(n_phi * n_t1 * n_t2).reshape(n_phi, n_t1, n_t2)

        mock_data = MockXPCSData()
        wrapper = NLSQWrapper()

        xdata, ydata = wrapper._prepare_data(mock_data)

        # Verify first few elements match expected indexing='ij' order
        # With indexing='ij', phi varies slowest, t2 varies fastest
        expected_ydata_start = mock_data.g2.flatten()  # NumPy default is C-order

        assert len(ydata) == n_phi * n_t1 * n_t2
        np.testing.assert_array_equal(
            ydata[:5],
            expected_ydata_start[:5],
            err_msg="Flattening order doesn't match expected indexing='ij'"
        )

    def test_empty_data_handling(self):
        """Test graceful handling of empty data."""
        class MockEmptyData:
            def __init__(self):
                self.phi = np.array([])
                self.t1 = np.array([])
                self.t2 = np.array([])
                self.g2 = np.array([])

        mock_data = MockEmptyData()
        wrapper = NLSQWrapper()

        with pytest.raises((ValueError, IndexError)):
            wrapper._prepare_data(mock_data)


class TestBoundsFormatConversion:
    """Test bounds format conversion (T010)."""

    def test_bounds_tuple_unchanged(self):
        """
        Verify homodyne bounds tuple format → NLSQ format conversion.

        Test: (lower_array, upper_array) tuple unchanged, verify shapes match n_params.
        """
        n_params = 5
        lower = np.array([0.0, 0.0, 100.0, 0.3, 1.0])
        upper = np.array([1.0, 2.0, 1e5, 1.5, 1000.0])
        homodyne_bounds = (lower, upper)

        wrapper = NLSQWrapper()
        nlsq_bounds = wrapper._convert_bounds(homodyne_bounds)

        # NLSQ expects same tuple format
        assert isinstance(nlsq_bounds, tuple)
        assert len(nlsq_bounds) == 2
        np.testing.assert_array_equal(nlsq_bounds[0], lower)
        np.testing.assert_array_equal(nlsq_bounds[1], upper)
        assert nlsq_bounds[0].shape == (n_params,)
        assert nlsq_bounds[1].shape == (n_params,)

    def test_bounds_validation_lower_less_than_upper(self):
        """Verify bounds validation checks lower < upper elementwise."""
        # Invalid bounds: some lower > upper
        lower = np.array([0.0, 2.0, 100.0])  # lower[1] = 2.0
        upper = np.array([1.0, 1.0, 1e5])    # upper[1] = 1.0 (invalid!)
        invalid_bounds = (lower, upper)

        wrapper = NLSQWrapper()

        with pytest.raises(ValueError, match="lower.*upper|bound"):
            wrapper._convert_bounds(invalid_bounds)

    def test_none_bounds_handling(self):
        """Test handling of None bounds (unbounded optimization)."""
        wrapper = NLSQWrapper()

        # None bounds should return None or appropriate default
        result = wrapper._convert_bounds(None)
        assert result is None or result == (-np.inf, np.inf)
