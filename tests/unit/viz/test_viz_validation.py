"""Unit tests for visualization validation utilities.

Tests for pre-plot data validation to detect NaN/Inf values.
"""

import numpy as np

from homodyne.viz.validation import validate_plot_arrays


class TestValidatePlotArrays:
    """Tests for validate_plot_arrays function."""

    def test_clean_array_returns_true(self):
        """Test that clean arrays return True."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_multiple_clean_arrays_return_true(self):
        """Test that multiple clean arrays return True."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, 5.0, 6.0])
        arr3 = np.array([7.0, 8.0, 9.0])
        result = validate_plot_arrays(arr1, arr2, arr3)
        assert result is True

    def test_array_with_nan_returns_false(self):
        """Test that arrays with NaN return False."""
        arr = np.array([1.0, np.nan, 3.0])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_array_with_inf_returns_false(self):
        """Test that arrays with Inf return False."""
        arr = np.array([1.0, np.inf, 3.0])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_array_with_negative_inf_returns_false(self):
        """Test that arrays with -Inf return False."""
        arr = np.array([1.0, -np.inf, 3.0])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_multiple_arrays_one_with_nan(self):
        """Test multiple arrays where one contains NaN."""
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([4.0, np.nan, 6.0])
        result = validate_plot_arrays(arr1, arr2)
        assert result is False

    def test_empty_array_is_clean(self):
        """Test that empty arrays are considered clean."""
        arr = np.array([])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_with_named_arrays(self):
        """Test validation with named arrays for logging."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, np.nan])
        names = ["x_data", "y_data"]
        result = validate_plot_arrays(arr1, arr2, names=names)
        assert result is False

    def test_2d_array_with_nan(self):
        """Test validation of 2D arrays with NaN."""
        arr = np.array([[1.0, 2.0], [3.0, np.nan]])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_2d_array_clean(self):
        """Test validation of clean 2D arrays."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_all_nan_array(self):
        """Test array with all NaN values."""
        arr = np.full(5, np.nan)
        result = validate_plot_arrays(arr)
        assert result is False

    def test_all_inf_array(self):
        """Test array with all Inf values."""
        arr = np.full(5, np.inf)
        result = validate_plot_arrays(arr)
        assert result is False

    def test_mixed_nan_and_inf(self):
        """Test array with both NaN and Inf values."""
        arr = np.array([1.0, np.nan, np.inf, 4.0])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_large_array_performance(self):
        """Test validation of large arrays."""
        arr = np.random.randn(10000, 100)
        result = validate_plot_arrays(arr)
        assert result is True

    def test_integer_array_is_clean(self):
        """Test that integer arrays are considered clean."""
        arr = np.array([1, 2, 3, 4, 5])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_zero_values_are_valid(self):
        """Test that zero values are considered valid."""
        arr = np.array([0.0, 0.0, 0.0])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_negative_values_are_valid(self):
        """Test that negative values are considered valid."""
        arr = np.array([-1.0, -2.0, -3.0])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_very_small_values_are_valid(self):
        """Test that very small values are considered valid."""
        arr = np.array([1e-100, 1e-200, 1e-300])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_very_large_values_are_valid(self):
        """Test that very large (but finite) values are valid."""
        arr = np.array([1e100, 1e200, 1e300])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_single_element_array(self):
        """Test validation of single-element array."""
        arr = np.array([1.0])
        result = validate_plot_arrays(arr)
        assert result is True

    def test_single_nan_element(self):
        """Test validation of single NaN element."""
        arr = np.array([np.nan])
        result = validate_plot_arrays(arr)
        assert result is False

    def test_names_longer_than_arrays(self):
        """Test with more names than arrays."""
        arr1 = np.array([1.0, 2.0])
        names = ["arr1", "arr2", "arr3"]
        result = validate_plot_arrays(arr1, names=names)
        assert result is True

    def test_names_shorter_than_arrays(self):
        """Test with fewer names than arrays."""
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([3.0, 4.0])
        names = ["arr1"]
        result = validate_plot_arrays(arr1, arr2, names=names)
        assert result is True

    def test_none_names_parameter(self):
        """Test with explicit None for names parameter."""
        arr = np.array([1.0, 2.0])
        result = validate_plot_arrays(arr, names=None)
        assert result is True
