"""Tests for quantile-based per-angle contrast/offset estimation.

This module tests the compute_quantile_per_angle_scaling function added in v2.17.0
for the "constant" mode in the anti-degeneracy defense system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from homodyne.optimization.nlsq.parameter_utils import (
    compute_quantile_per_angle_scaling,
)


@dataclass
class MockChunk:
    """Mock chunk for testing."""

    phi: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    g2: np.ndarray
    q: float = 1.0e-3
    L: float = 1000.0
    dt: float | None = None


@dataclass
class MockChunkedData:
    """Mock chunked data for testing."""

    chunks: list[MockChunk]
    sigma: np.ndarray


@dataclass
class MockStratifiedData:
    """Mock stratified data with flat arrays for testing."""

    phi_flat: np.ndarray
    t1_flat: np.ndarray
    t2_flat: np.ndarray
    g2_flat: np.ndarray
    sigma: np.ndarray
    q: float = 1.0e-3
    L: float = 1000.0
    dt: float | None = None


class TestComputeQuantilePerAngleScaling:
    """Test suite for compute_quantile_per_angle_scaling function."""

    def _create_synthetic_c2_data(
        self,
        n_phi: int,
        n_points_per_angle: int,
        true_contrast: float | np.ndarray,
        true_offset: float | np.ndarray,
        noise_std: float = 0.01,
        seed: int = 42,
    ) -> MockStratifiedData:
        """Create synthetic C2 data with known contrast/offset.

        The C2 data follows the physics: C2 = contrast * g1^2 + offset
        where g1^2 decays from 1 (small lag) to 0 (large lag).
        """
        rng = np.random.default_rng(seed)

        # Create per-angle contrast/offset arrays
        if isinstance(true_contrast, float):
            contrast_arr = np.full(n_phi, true_contrast)
        else:
            contrast_arr = np.asarray(true_contrast)

        if isinstance(true_offset, float):
            offset_arr = np.full(n_phi, true_offset)
        else:
            offset_arr = np.asarray(true_offset)

        # Create unique phi angles
        phi_unique = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

        # Create time arrays - logarithmically spaced for realistic lag distribution
        t1_unique = np.linspace(0, 100, int(np.sqrt(n_points_per_angle)))
        t2_unique = np.linspace(0, 100, int(np.sqrt(n_points_per_angle)))

        # Build flat arrays
        phi_list = []
        t1_list = []
        t2_list = []
        g2_list = []

        for i, phi in enumerate(phi_unique):
            for t1 in t1_unique:
                for t2 in t2_unique:
                    phi_list.append(phi)
                    t1_list.append(t1)
                    t2_list.append(t2)

                    # Compute synthetic g2 value
                    # g1^2 decays with time lag: g1^2 = exp(-2 * D * q^2 * |t1 - t2|)
                    delta_t = abs(t1 - t2)
                    g1_sq = np.exp(-0.02 * delta_t)  # Simple decay model

                    # C2 = contrast * g1^2 + offset
                    c2_true = contrast_arr[i] * g1_sq + offset_arr[i]
                    c2_noisy = c2_true + rng.normal(0, noise_std)
                    g2_list.append(c2_noisy)

        n_total = len(phi_list)
        return MockStratifiedData(
            phi_flat=np.array(phi_list),
            t1_flat=np.array(t1_list),
            t2_flat=np.array(t2_list),
            g2_flat=np.array(g2_list),
            sigma=np.full(n_total, 0.01),
        )

    def test_uniform_contrast_offset_recovery(self):
        """Test that uniform contrast/offset is recovered accurately."""
        true_contrast = 0.3
        true_offset = 0.8
        n_phi = 5
        n_points_per_angle = 2500  # 50x50 time grid

        data = self._create_synthetic_c2_data(
            n_phi=n_phi,
            n_points_per_angle=n_points_per_angle,
            true_contrast=true_contrast,
            true_offset=true_offset,
            noise_std=0.005,
        )

        contrast_est, offset_est = compute_quantile_per_angle_scaling(
            stratified_data=data,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Check shape
        assert len(contrast_est) == n_phi
        assert len(offset_est) == n_phi

        # Check recovered values are close to true values
        # Allow 20% relative error due to quantile estimation
        np.testing.assert_allclose(
            np.mean(contrast_est),
            true_contrast,
            rtol=0.2,
            err_msg="Mean contrast not recovered accurately",
        )
        np.testing.assert_allclose(
            np.mean(offset_est),
            true_offset,
            rtol=0.1,
            err_msg="Mean offset not recovered accurately",
        )

    def test_varying_per_angle_recovery(self):
        """Test that varying per-angle contrast/offset is recovered."""
        n_phi = 5
        # Varying contrast and offset per angle
        true_contrast = np.array([0.25, 0.30, 0.35, 0.28, 0.32])
        true_offset = np.array([0.75, 0.80, 0.85, 0.78, 0.82])

        data = self._create_synthetic_c2_data(
            n_phi=n_phi,
            n_points_per_angle=2500,
            true_contrast=true_contrast,
            true_offset=true_offset,
            noise_std=0.005,
        )

        contrast_est, offset_est = compute_quantile_per_angle_scaling(
            stratified_data=data,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Check that per-angle variation is captured
        # The standard deviation should be non-zero
        assert np.std(contrast_est) > 0.01, "Per-angle contrast variation not captured"
        assert np.std(offset_est) > 0.01, "Per-angle offset variation not captured"

        # Check means are reasonable
        np.testing.assert_allclose(
            np.mean(contrast_est), np.mean(true_contrast), rtol=0.2
        )
        np.testing.assert_allclose(np.mean(offset_est), np.mean(true_offset), rtol=0.1)

    def test_bounds_clipping(self):
        """Test that estimates are clipped to bounds."""
        # Create data with extreme values that would exceed bounds
        n_phi = 3
        data = self._create_synthetic_c2_data(
            n_phi=n_phi,
            n_points_per_angle=2500,
            true_contrast=0.9,  # High contrast
            true_offset=1.4,  # High offset
            noise_std=0.001,
        )

        # Use tight bounds that should clip
        contrast_bounds = (0.0, 0.5)  # Max 0.5, but true is 0.9
        offset_bounds = (0.5, 1.0)  # Max 1.0, but true is 1.4

        contrast_est, offset_est = compute_quantile_per_angle_scaling(
            stratified_data=data,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        # All values should be within bounds
        assert np.all(contrast_est >= contrast_bounds[0])
        assert np.all(contrast_est <= contrast_bounds[1])
        assert np.all(offset_est >= offset_bounds[0])
        assert np.all(offset_est <= offset_bounds[1])

    def test_insufficient_data_uses_midpoint(self):
        """Test that angles with insufficient data use midpoint defaults."""
        # Create data with very few points for some angles
        phi_flat = np.array(
            [0.0] * 50 + [1.0] * 200
        )  # Angle 0: 50 pts, Angle 1: 200 pts
        t1_flat = np.concatenate([np.linspace(0, 10, 50), np.linspace(0, 100, 200)])
        t2_flat = t1_flat.copy()
        g2_flat = np.full(250, 0.9)  # Dummy values
        sigma = np.full(250, 0.01)

        data = MockStratifiedData(
            phi_flat=phi_flat,
            t1_flat=t1_flat,
            t2_flat=t2_flat,
            g2_flat=g2_flat,
            sigma=sigma,
        )

        contrast_bounds = (0.0, 1.0)
        offset_bounds = (0.5, 1.5)

        contrast_est, offset_est = compute_quantile_per_angle_scaling(
            stratified_data=data,
            contrast_bounds=contrast_bounds,
            offset_bounds=offset_bounds,
        )

        # Angle with insufficient data (< 100 points) should get midpoint
        # Midpoint contrast = 0.5, midpoint offset = 1.0
        # Note: angle 0 has only 50 points, should get midpoint
        assert contrast_est[0] == pytest.approx(0.5, rel=0.01)
        assert offset_est[0] == pytest.approx(1.0, rel=0.01)

    def test_chunked_data_format(self):
        """Test that chunked data format is handled correctly."""
        n_phi = 3
        n_points = 500

        # Create mock chunks
        chunks = []
        phi_values = np.array([0.0, 1.0, 2.0])

        for _i in range(3):  # 3 chunks
            chunk_size = n_points // 3
            phi = np.repeat(phi_values, chunk_size // 3)
            t1 = np.tile(np.linspace(0, 100, chunk_size // 3), 3)
            t2 = t1.copy()
            # Synthetic g2 with known contrast/offset
            delta_t = np.abs(t1 - t2)
            g1_sq = np.exp(-0.02 * delta_t)
            g2 = 0.3 * g1_sq + 0.8 + np.random.normal(0, 0.01, len(phi))

            chunks.append(
                MockChunk(
                    phi=phi,
                    t1=t1,
                    t2=t2,
                    g2=g2,
                )
            )

        data = MockChunkedData(
            chunks=chunks,
            sigma=np.full(n_points, 0.01),
        )

        contrast_est, offset_est = compute_quantile_per_angle_scaling(
            stratified_data=data,
            contrast_bounds=(0.0, 1.0),
            offset_bounds=(0.5, 1.5),
        )

        # Should return n_phi estimates
        assert len(contrast_est) == n_phi
        assert len(offset_est) == n_phi

        # All estimates should be within bounds
        assert np.all(contrast_est >= 0.0) and np.all(contrast_est <= 1.0)
        assert np.all(offset_est >= 0.5) and np.all(offset_est <= 1.5)


class TestAntiDegeneracyControllerFixedScaling:
    """Test the fixed per-angle scaling methods in AntiDegeneracyController."""

    def test_has_fixed_scaling_initially_false(self):
        """Test that has_fixed_per_angle_scaling returns False initially."""
        from homodyne.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        controller = AntiDegeneracyController.from_config(
            config_dict={"enable": True, "per_angle_mode": "constant"},
            n_phi=5,
            phi_angles=np.linspace(0, 2 * np.pi, 5, endpoint=False),
            n_physical=7,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )

        assert controller.has_fixed_per_angle_scaling() is False
        assert controller.get_fixed_per_angle_scaling() is None

    def test_diagnostics_includes_fixed_scaling_info(self):
        """Test that diagnostics includes has_fixed_per_angle_scaling field."""
        from homodyne.optimization.nlsq.anti_degeneracy_controller import (
            AntiDegeneracyController,
        )

        controller = AntiDegeneracyController.from_config(
            config_dict={"enable": True, "per_angle_mode": "constant"},
            n_phi=5,
            phi_angles=np.linspace(0, 2 * np.pi, 5, endpoint=False),
            n_physical=7,
            per_angle_scaling=True,
            is_laminar_flow=True,
        )

        diagnostics = controller.get_diagnostics()

        assert "has_fixed_per_angle_scaling" in diagnostics
        assert diagnostics["has_fixed_per_angle_scaling"] is False
        assert diagnostics["version"] == "2.18.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
