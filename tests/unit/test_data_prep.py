"""Unit tests for homodyne.optimization.nlsq.data_prep module.

Tests data preparation utilities for NLSQ optimization including
parameter expansion, bounds validation, and parameter labeling.
"""

import numpy as np
import pytest

from homodyne.optimization.nlsq.data_prep import (
    ExpandedParameters,
    PreparedData,
    build_parameter_labels,
    classify_parameter_status,
    convert_bounds_to_nlsq_format,
    expand_per_angle_parameters,
    validate_bounds,
    validate_initial_params,
)


class TestPreparedData:
    """Tests for PreparedData dataclass."""

    def test_prepared_data_creation(self):
        """Test basic PreparedData instantiation."""
        xdata = np.array([1.0, 2.0, 3.0])
        ydata = np.array([0.5, 1.0, 1.5])
        phi_unique = np.array([0.0, 45.0, 90.0])

        data = PreparedData(
            xdata=xdata,
            ydata=ydata,
            n_data=3,
            n_phi=3,
            phi_unique=phi_unique,
        )

        assert data.n_data == 3
        assert data.n_phi == 3
        np.testing.assert_array_equal(data.xdata, xdata)
        np.testing.assert_array_equal(data.ydata, ydata)
        np.testing.assert_array_equal(data.phi_unique, phi_unique)


class TestExpandedParameters:
    """Tests for ExpandedParameters dataclass."""

    def test_expanded_parameters_creation(self):
        """Test basic ExpandedParameters instantiation."""
        params = np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1e-11, 0.5, 1e-14])
        bounds = (np.zeros(9), np.ones(9))

        expanded = ExpandedParameters(
            params=params,
            bounds=bounds,
            n_params=9,
            n_physical=3,
            n_angles=3,
        )

        assert expanded.n_params == 9
        assert expanded.n_physical == 3
        assert expanded.n_angles == 3
        assert expanded.bounds is not None


class TestExpandPerAngleParameters:
    """Tests for expand_per_angle_parameters function."""

    def test_basic_expansion_static_mode(self):
        """Test parameter expansion for static mode (3 physical params)."""
        # compact: [contrast, offset, D0, alpha, D_offset]
        compact = np.array([0.8, 1.0, 1e-11, 0.5, 1e-14])
        n_angles = 3
        n_physical = 3

        result = expand_per_angle_parameters(compact, None, n_angles, n_physical)

        # Expected: [c0, c1, c2, o0, o1, o2, D0, alpha, D_offset]
        assert result.n_params == 2 * n_angles + n_physical  # 9
        assert result.n_angles == 3
        assert result.n_physical == 3
        assert result.bounds is None

        # Check contrast values (first n_angles)
        np.testing.assert_array_equal(result.params[:3], [0.8, 0.8, 0.8])
        # Check offset values (next n_angles)
        np.testing.assert_array_equal(result.params[3:6], [1.0, 1.0, 1.0])
        # Check physical params (last n_physical)
        np.testing.assert_array_equal(result.params[6:], [1e-11, 0.5, 1e-14])

    def test_expansion_with_bounds(self):
        """Test parameter expansion with bounds."""
        compact = np.array([0.8, 1.0, 1e-11, 0.5, 1e-14])
        compact_bounds = (
            np.array([0.1, 0.5, 1e-13, 0.1, 1e-16]),
            np.array([1.5, 2.0, 1e-9, 0.9, 1e-12]),
        )
        n_angles = 2
        n_physical = 3

        result = expand_per_angle_parameters(
            compact, compact_bounds, n_angles, n_physical
        )

        assert result.bounds is not None
        lower, upper = result.bounds

        # Check bounds shape
        assert len(lower) == 2 * n_angles + n_physical  # 7
        assert len(upper) == 2 * n_angles + n_physical

        # Check contrast bounds expanded
        np.testing.assert_array_equal(lower[:2], [0.1, 0.1])
        np.testing.assert_array_equal(upper[:2], [1.5, 1.5])

        # Check offset bounds expanded
        np.testing.assert_array_equal(lower[2:4], [0.5, 0.5])
        np.testing.assert_array_equal(upper[2:4], [2.0, 2.0])

    def test_laminar_flow_mode(self):
        """Test parameter expansion for laminar flow mode (7 physical params)."""
        # compact: [contrast, offset, D0, alpha, D_offset, gamma_dot_t0, beta, gamma_dot_t_offset, phi0]
        compact = np.array([0.8, 1.0, 1e-11, 0.5, 1e-14, 100.0, 0.3, 10.0, 0.0])
        n_angles = 4
        n_physical = 7

        result = expand_per_angle_parameters(compact, None, n_angles, n_physical)

        assert result.n_params == 2 * n_angles + n_physical  # 15
        assert result.n_physical == 7
        assert len(result.params) == 15

    def test_invalid_parameter_count_raises(self):
        """Test that mismatched parameter count raises ValueError."""
        compact = np.array([0.8, 1.0, 1e-11])  # Only 3 params
        n_angles = 2
        n_physical = 3  # Expects 3 + 2 = 5 params

        with pytest.raises(ValueError, match="Parameter count mismatch"):
            expand_per_angle_parameters(compact, None, n_angles, n_physical)

    def test_single_angle(self):
        """Test expansion with single angle."""
        compact = np.array([0.8, 1.0, 1e-11, 0.5, 1e-14])
        n_angles = 1
        n_physical = 3

        result = expand_per_angle_parameters(compact, None, n_angles, n_physical)

        assert result.n_params == 5  # 2*1 + 3
        assert result.params[0] == 0.8  # contrast
        assert result.params[1] == 1.0  # offset


class TestValidateBounds:
    """Tests for validate_bounds function."""

    def test_valid_bounds(self):
        """Test validation of valid bounds."""
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        n_params = 3

        result = validate_bounds(bounds, n_params)

        assert result is not None
        np.testing.assert_array_equal(result[0], bounds[0])
        np.testing.assert_array_equal(result[1], bounds[1])

    def test_none_bounds_returns_none(self):
        """Test that None bounds returns None."""
        result = validate_bounds(None, 3)
        assert result is None

    def test_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises ValueError."""
        bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0, 1.0]))
        n_params = 3

        with pytest.raises(ValueError, match="Bounds dimension mismatch"):
            validate_bounds(bounds, n_params)

    def test_invalid_bounds_lower_ge_upper_raises(self):
        """Test that lower >= upper raises ValueError."""
        bounds = (np.array([0.0, 1.5, 0.0]), np.array([1.0, 1.0, 1.0]))  # 1.5 >= 1.0
        n_params = 3

        with pytest.raises(ValueError, match="Invalid bounds at indices"):
            validate_bounds(bounds, n_params)

    def test_converts_to_float(self):
        """Test that bounds are converted to float dtype."""
        bounds = (np.array([0, 0, 0], dtype=int), np.array([1, 1, 1], dtype=int))
        n_params = 3

        result = validate_bounds(bounds, n_params)

        assert result[0].dtype == float
        assert result[1].dtype == float


class TestValidateInitialParams:
    """Tests for validate_initial_params function."""

    def test_params_within_bounds_unchanged(self):
        """Test that params within bounds are unchanged."""
        params = np.array([0.5, 0.5, 0.5])
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        result = validate_initial_params(params, bounds)

        np.testing.assert_array_equal(result, params)

    def test_params_clipped_to_bounds(self):
        """Test that params outside bounds are clipped."""
        params = np.array([-0.5, 0.5, 1.5])  # First below, last above
        bounds = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        result = validate_initial_params(params, bounds)

        np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])

    def test_none_bounds_returns_params(self):
        """Test that None bounds returns params unchanged."""
        params = np.array([0.5, 0.5, 0.5])

        result = validate_initial_params(params, None)

        np.testing.assert_array_equal(result, params)

    def test_converts_to_float(self):
        """Test that params are converted to float dtype."""
        params = np.array([1, 2, 3], dtype=int)

        result = validate_initial_params(params, None)

        assert result.dtype == float


class TestConvertBoundsToNlsqFormat:
    """Tests for convert_bounds_to_nlsq_format function."""

    def test_converts_lists_to_arrays(self):
        """Test conversion of lists to numpy arrays."""
        bounds = ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

        result = convert_bounds_to_nlsq_format(bounds)

        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
        assert result[0].dtype == np.float64
        assert result[1].dtype == np.float64

    def test_none_returns_none(self):
        """Test that None input returns None."""
        result = convert_bounds_to_nlsq_format(None)
        assert result is None

    def test_arrays_converted_to_float64(self):
        """Test that arrays are converted to float64."""
        bounds = (np.array([0, 0, 0], dtype=int), np.array([1, 1, 1], dtype=int))

        result = convert_bounds_to_nlsq_format(bounds)

        assert result[0].dtype == np.float64
        assert result[1].dtype == np.float64


class TestBuildParameterLabels:
    """Tests for build_parameter_labels function."""

    def test_per_angle_scaling_labels(self):
        """Test label generation with per-angle scaling."""
        labels = build_parameter_labels(
            per_angle_scaling=True,
            n_phi=3,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        expected = [
            "contrast[0]",
            "contrast[1]",
            "contrast[2]",
            "offset[0]",
            "offset[1]",
            "offset[2]",
            "D0",
            "alpha",
            "D_offset",
        ]
        assert labels == expected

    def test_scalar_scaling_labels(self):
        """Test label generation with scalar scaling."""
        labels = build_parameter_labels(
            per_angle_scaling=False,
            n_phi=3,
            physical_param_names=["D0", "alpha", "D_offset"],
        )

        expected = ["contrast", "offset", "D0", "alpha", "D_offset"]
        assert labels == expected

    def test_laminar_flow_labels(self):
        """Test label generation for laminar flow mode."""
        physical_names = [
            "D0",
            "alpha",
            "D_offset",
            "gamma_dot_t0",
            "beta",
            "gamma_dot_t_offset",
            "phi0",
        ]
        labels = build_parameter_labels(
            per_angle_scaling=True, n_phi=2, physical_param_names=physical_names
        )

        assert len(labels) == 2 * 2 + 7  # 11
        assert labels[4:] == physical_names


class TestClassifyParameterStatus:
    """Tests for classify_parameter_status function."""

    def test_all_active(self):
        """Test classification when all params are active (not at bounds)."""
        values = np.array([0.5, 0.5, 0.5])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        statuses = classify_parameter_status(values, lower, upper)

        assert statuses == ["active", "active", "active"]

    def test_at_lower_bound(self):
        """Test classification when params are at lower bound."""
        values = np.array([0.0, 0.5, 0.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        statuses = classify_parameter_status(values, lower, upper)

        assert statuses == ["at_lower_bound", "active", "at_lower_bound"]

    def test_at_upper_bound(self):
        """Test classification when params are at upper bound."""
        values = np.array([1.0, 0.5, 1.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        statuses = classify_parameter_status(values, lower, upper)

        assert statuses == ["at_upper_bound", "active", "at_upper_bound"]

    def test_none_bounds_all_active(self):
        """Test that None bounds returns all active."""
        values = np.array([0.5, 0.5, 0.5])

        statuses = classify_parameter_status(values, None, None)

        assert statuses == ["active", "active", "active"]

    def test_mixed_statuses(self):
        """Test mixed bound statuses."""
        values = np.array([0.0, 0.5, 1.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        statuses = classify_parameter_status(values, lower, upper)

        assert statuses == ["at_lower_bound", "active", "at_upper_bound"]

    def test_tolerance_respected(self):
        """Test that tolerance is respected for bound comparison."""
        values = np.array([1e-12, 0.5, 1.0 - 1e-12])  # Very close to bounds
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])

        statuses = classify_parameter_status(values, lower, upper, atol=1e-9)

        assert statuses == ["at_lower_bound", "active", "at_upper_bound"]
