"""
Unit tests for JAX array compatibility in angle filtering.

These tests ensure that angle filtering works correctly with JAX arrays,
which have stricter indexing requirements than NumPy arrays.

Regression test for: "Using a non-tuple sequence for multidimensional
indexing is not allowed" error when using JAX arrays.
"""

import numpy as np
import pytest

# JAX import with graceful fallback
try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

from homodyne.cli.commands import _apply_angle_filtering
from tests.factories.config_factory import (
    create_anisotropic_filtering_config,
    create_disabled_filtering_config,
    create_phi_filtering_config,
)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJAXArrayCompatibility:
    """Test angle filtering with JAX arrays."""

    def test_jax_array_indexing_with_filtering(self):
        """Test that filtered indexing works with JAX arrays.

        Regression test for JAX indexing error:
        "Using a non-tuple sequence for multidimensional indexing is not allowed"

        This error occurred because JAX arrays don't accept Python list indexing,
        but the angle filtering code was using a Python list for indices.
        """
        # Arrange - Create JAX arrays
        angles_np = np.array([0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 120.0, 180.0])
        phi_angles = jnp.array(angles_np)

        # Create 3D correlation data (n_phi, n_t1, n_t2)
        n_phi = len(angles_np)
        n_t = 20
        c2_data_np = np.random.rand(n_phi, n_t, n_t) + 1.0  # Values around 1-2
        c2_exp = jnp.array(c2_data_np)

        # Config with two ranges: [-10, 10] and [80, 100]
        config = create_anisotropic_filtering_config()

        # Act - This should NOT raise an error with JAX arrays
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert
        # Should match: 0.0, 10.0 (in [-10, 10]) and 85.0, 90.0 (in [80, 100])
        expected_indices = [0, 1, 5, 6]
        expected_angles = [0.0, 10.0, 85.0, 90.0]

        assert filtered_indices == expected_indices
        assert list(filtered_phi) == pytest.approx(expected_angles)
        assert filtered_c2.shape == (4, n_t, n_t)

        # Verify arrays are still JAX arrays (not converted to NumPy)
        assert isinstance(filtered_phi, jnp.ndarray)
        assert isinstance(filtered_c2, jnp.ndarray)

    def test_jax_array_no_filtering(self):
        """Test JAX arrays when filtering is disabled."""
        # Arrange
        angles_np = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        phi_angles = jnp.array(angles_np)

        n_phi = len(angles_np)
        n_t = 15
        c2_data_np = np.random.rand(n_phi, n_t, n_t) + 1.0
        c2_exp = jnp.array(c2_data_np)

        config = create_disabled_filtering_config()

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert - All angles returned
        assert filtered_indices == list(range(n_phi))
        assert filtered_phi.shape == phi_angles.shape
        assert filtered_c2.shape == c2_exp.shape

    def test_jax_array_single_angle_match(self):
        """Test JAX array with single angle matching."""
        # Arrange
        angles_np = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        phi_angles = jnp.array(angles_np)

        n_phi = len(angles_np)
        n_t = 10
        c2_data_np = np.random.rand(n_phi, n_t, n_t) + 1.0
        c2_exp = jnp.array(c2_data_np)

        # Config that only matches 90 degrees
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": 85.0, "max_angle": 95.0}],
        )

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert - Only 90° matched
        assert filtered_indices == [2]
        assert float(filtered_phi[0]) == 90.0
        assert filtered_c2.shape == (1, n_t, n_t)

    def test_jax_array_all_angles_match(self):
        """Test JAX array when all angles match filter."""
        # Arrange
        angles_np = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        phi_angles = jnp.array(angles_np)

        n_phi = len(angles_np)
        n_t = 10
        c2_data_np = np.random.rand(n_phi, n_t, n_t) + 1.0
        c2_exp = jnp.array(c2_data_np)

        # Config that matches all angles
        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -180.0, "max_angle": 180.0}],
        )

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert - All angles returned
        assert filtered_indices == list(range(n_phi))
        assert filtered_phi.shape == phi_angles.shape
        assert filtered_c2.shape == c2_exp.shape

    def test_numpy_arrays_still_work(self):
        """Verify that NumPy arrays still work correctly (backward compatibility)."""
        # Arrange - Use NumPy arrays (not JAX)
        phi_angles = np.array([0.0, 10.0, 30.0, 45.0, 60.0, 85.0, 90.0, 120.0, 180.0])

        n_phi = len(phi_angles)
        n_t = 20
        c2_exp = np.random.rand(n_phi, n_t, n_t) + 1.0

        config = create_anisotropic_filtering_config()

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert
        expected_indices = [0, 1, 5, 6]
        expected_angles = [0.0, 10.0, 85.0, 90.0]

        assert filtered_indices == expected_indices
        assert list(filtered_phi) == pytest.approx(expected_angles)
        assert filtered_c2.shape == (4, n_t, n_t)

        # Verify arrays are NumPy (not converted to JAX)
        assert isinstance(filtered_phi, np.ndarray)
        assert isinstance(filtered_c2, np.ndarray)


# Test with NumPy arrays (always runs, even without JAX)
class TestNumpyArrayCompatibility:
    """Test angle filtering with NumPy arrays (baseline)."""

    def test_numpy_array_filtering_baseline(self):
        """Baseline test with NumPy arrays."""
        # Arrange
        phi_angles = np.array([0.0, 45.0, 90.0, 135.0, 180.0])
        n_phi = len(phi_angles)
        n_t = 15
        c2_exp = np.random.rand(n_phi, n_t, n_t) + 1.0

        config = create_phi_filtering_config(
            enabled=True,
            target_ranges=[{"min_angle": -10.0, "max_angle": 10.0}],
        )

        # Act
        filtered_indices, filtered_phi, filtered_c2 = _apply_angle_filtering(
            phi_angles, c2_exp, config
        )

        # Assert - Only 0° should match [-10, 10] range
        assert filtered_indices == [0]
        assert float(filtered_phi[0]) == 0.0
        assert filtered_c2.shape == (1, n_t, n_t)
        assert isinstance(filtered_phi, np.ndarray)
        assert isinstance(filtered_c2, np.ndarray)
