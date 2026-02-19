"""Physics equivalence test between CMA-ES and NLSQ computation paths.

Validates that CMA-ES (core.py, using _compute_g1_total_core from jax_backend.py)
and NLSQ (wrapper.py, using compute_g2_scaled from physics_nlsq.py) produce
identical physics results for the same inputs.

Root cause analysis (Feb 2026) identified these paths diverge in:
1. g1 sub-functions: _compute_g1_total_core (jax_backend) vs _compute_g1_total_meshgrid (physics_nlsq)
2. g2 clipping: NLSQ clips g2 to [0.5, 2.5], CMA-ES does not
3. Diagonal correction: NLSQ wrapper applies it, CMA-ES does not

This test verifies the physics formulas are mathematically equivalent
and documents the known behavioral differences.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from homodyne.core.jax_backend import _compute_g1_total_core
from homodyne.core.physics_nlsq import (
    _compute_g1_total_meshgrid,
    compute_g2_scaled,
)


class TestCMAESvsNLSQPhysicsEquivalence:
    """Verify CMA-ES and NLSQ use the same underlying physics."""

    @pytest.fixture
    def laminar_params(self):
        """Standard laminar_flow test parameters."""
        return jnp.array(
            [
                1e10,  # D0
                -0.5,  # alpha
                1e9,  # D_offset
                0.001,  # gamma_dot_t0
                0.0,  # beta
                0.0,  # gamma_dot_t_offset
                0.0,  # phi0
            ]
        )

    @pytest.fixture
    def static_params(self):
        """Standard static test parameters (3 params, no shear)."""
        return jnp.array(
            [
                1e10,  # D0
                -0.5,  # alpha
                1e9,  # D_offset
            ]
        )

    @pytest.fixture
    def physics_config(self):
        """Pre-computed physics factors matching typical XPCS config."""
        q = 0.01  # nm^-1
        L = 2e6  # nm (2mm stator-rotor gap)
        dt = 0.1  # seconds
        return {
            "q": q,
            "L": L,
            "dt": dt,
            "wavevector_q_squared_half_dt": 0.5 * q**2 * dt,
            "sinc_prefactor": 0.5 / jnp.pi * q * L * dt,
        }

    @pytest.fixture
    def time_grid(self, physics_config):
        """Small time grid for meshgrid comparison."""
        dt = physics_config["dt"]
        n_times = 20
        # Physical time in seconds (as produced by data loader)
        t = jnp.linspace(0.0, dt * (n_times - 1), n_times)
        return t

    @pytest.fixture
    def phi_angles(self):
        """Test phi angles (degrees)."""
        return jnp.array([0.0, 45.0, 90.0, 135.0, 180.0])

    def test_g1_matrix_mode_equivalence(
        self, laminar_params, physics_config, time_grid, phi_angles
    ):
        """Both g1 functions should agree in matrix mode for identical inputs.

        _compute_g1_total_core (jax_backend.py) in matrix mode should produce
        the same result as _compute_g1_total_meshgrid (physics_nlsq.py).
        """
        t1_grid, t2_grid = jnp.meshgrid(time_grid, time_grid, indexing="ij")
        wq = physics_config["wavevector_q_squared_half_dt"]
        sp = physics_config["sinc_prefactor"]
        dt = physics_config["dt"]

        # NLSQ path: physics_nlsq.py
        g1_nlsq = _compute_g1_total_meshgrid(
            laminar_params, t1_grid, t2_grid, phi_angles, wq, sp, dt
        )

        # CMA-ES path: jax_backend.py (matrix mode — small data, ndim=2)
        g1_cmaes = _compute_g1_total_core(
            laminar_params, t1_grid, t2_grid, phi_angles, wq, sp, dt
        )

        np.testing.assert_allclose(
            np.asarray(g1_cmaes),
            np.asarray(g1_nlsq),
            rtol=1e-10,
            err_msg=(
                "CMA-ES (jax_backend) and NLSQ (physics_nlsq) g1 functions "
                "diverge in matrix mode"
            ),
        )

    def test_g1_static_mode_equivalence(self, static_params, physics_config, time_grid):
        """Static mode (no shear) should be identical between both paths."""
        t1_grid, t2_grid = jnp.meshgrid(time_grid, time_grid, indexing="ij")
        wq = physics_config["wavevector_q_squared_half_dt"]
        sp = physics_config["sinc_prefactor"]
        dt = physics_config["dt"]
        phi = jnp.array([0.0])

        g1_nlsq = _compute_g1_total_meshgrid(
            static_params, t1_grid, t2_grid, phi, wq, sp, dt
        )
        g1_cmaes = _compute_g1_total_core(
            static_params, t1_grid, t2_grid, phi, wq, sp, dt
        )

        np.testing.assert_allclose(
            np.asarray(g1_cmaes),
            np.asarray(g1_nlsq),
            rtol=1e-10,
            err_msg="Static mode g1 diverges between CMA-ES and NLSQ paths",
        )

    def test_g2_formula_equivalence(
        self, laminar_params, physics_config, time_grid, phi_angles
    ):
        """g2 = offset + contrast * g1^2 should be identical before clipping.

        Both paths use the same Siegert relation. The NLSQ path (compute_g2_scaled)
        clips to [0.5, 2.5], while CMA-ES computes inline without clipping.
        This test verifies the pre-clipping values match.
        """
        t1_grid, t2_grid = jnp.meshgrid(time_grid, time_grid, indexing="ij")
        wq = physics_config["wavevector_q_squared_half_dt"]
        sp = physics_config["sinc_prefactor"]
        dt = physics_config["dt"]
        contrast = 0.5
        offset = 1.0

        # CMA-ES inline g2: offset + contrast * g1^2
        g1_cmaes = _compute_g1_total_core(
            laminar_params, t1_grid, t2_grid, phi_angles, wq, sp, dt
        )
        g2_cmaes = offset + contrast * g1_cmaes**2

        # NLSQ g2 via compute_g2_scaled (includes [0.5, 2.5] clipping)
        g2_nlsq = compute_g2_scaled(
            laminar_params,
            time_grid,  # 1D — compute_g2_scaled creates meshgrid internally
            time_grid,
            phi_angles,
            physics_config["q"],
            physics_config["L"],
            contrast,
            offset,
            dt,
        )

        # For typical parameters, g2 values should be in [0.5, 2.5],
        # so clipping should not affect the comparison
        in_clip_range = (g2_cmaes >= 0.5) & (g2_cmaes <= 2.5)
        assert jnp.all(in_clip_range), (
            f"Some g2 values fall outside NLSQ clip range [0.5, 2.5]: "
            f"min={float(jnp.min(g2_cmaes)):.4f}, max={float(jnp.max(g2_cmaes)):.4f}"
        )

        np.testing.assert_allclose(
            np.asarray(g2_cmaes),
            np.asarray(g2_nlsq),
            rtol=1e-10,
            err_msg="g2 values diverge between CMA-ES inline and NLSQ compute_g2_scaled",
        )

    def test_g2_no_clipping_equivalence(
        self, laminar_params, physics_config, phi_angles
    ):
        """Verify NLSQ and CMA-ES produce identical g2 (no clipping in either path).

        Previously NLSQ clipped g2 to [0.5, 2.5] (removed in NEW-P0 fix).
        Both paths should now produce unclipped, physically correct g2 values.
        """
        # Use very short time grid where g1 ≈ 1.0
        dt = physics_config["dt"]
        t_short = jnp.array([0.0, dt])
        contrast = 1.8  # High contrast to push g2 > 2.5
        offset = 1.0

        g2_nlsq = compute_g2_scaled(
            laminar_params,
            t_short,
            t_short,
            phi_angles,
            physics_config["q"],
            physics_config["L"],
            contrast,
            offset,
            dt,
        )

        # Compute CMA-ES g2 (inline)
        wq = physics_config["wavevector_q_squared_half_dt"]
        sp = physics_config["sinc_prefactor"]
        t1_grid, t2_grid = jnp.meshgrid(t_short, t_short, indexing="ij")
        g1 = _compute_g1_total_core(
            laminar_params, t1_grid, t2_grid, phi_angles, wq, sp, dt
        )
        g2_cmaes = offset + contrast * g1**2

        # Both should now exceed 2.5 (no clipping) and be equivalent
        nlsq_max = float(jnp.max(g2_nlsq))
        cmaes_max = float(jnp.max(g2_cmaes))

        assert nlsq_max > 2.5, f"NLSQ g2 should be unclipped (>2.5), got {nlsq_max}"
        assert cmaes_max > 2.5, f"CMA-ES g2 should exceed 2.5, got {cmaes_max}"
        np.testing.assert_allclose(
            np.asarray(g2_nlsq),
            np.asarray(g2_cmaes),
            rtol=1e-10,
            err_msg="NLSQ and CMA-ES g2 should be identical (no clipping in either)",
        )


class TestCMAESElementwiseVsMeshgrid:
    """Test element-wise mode (used by CMA-ES for large data) vs meshgrid mode."""

    @pytest.fixture
    def laminar_params(self):
        return jnp.array([1e10, -0.5, 1e9, 0.001, 0.0, 0.0, 0.0])

    @pytest.fixture
    def physics_config(self):
        q = 0.01
        L = 2e6
        dt = 0.1
        return {
            "q": q,
            "L": L,
            "dt": dt,
            "wavevector_q_squared_half_dt": 0.5 * q**2 * dt,
            "sinc_prefactor": 0.5 / jnp.pi * q * L * dt,
        }

    def test_elementwise_matches_meshgrid_for_offdiagonal(
        self, laminar_params, physics_config
    ):
        """Element-wise g1 values should match corresponding meshgrid entries.

        CMA-ES uses element-wise mode for large datasets (>2000 points).
        This test creates >2000 (t1, t2, phi) triples from a meshgrid,
        verifies they match the meshgrid computation.
        """
        dt = physics_config["dt"]
        wq = physics_config["wavevector_q_squared_half_dt"]
        sp = physics_config["sinc_prefactor"]
        n_times = 30
        phi_angles = jnp.array([0.0, 45.0, 90.0, 135.0, 180.0])

        t = jnp.linspace(0.0, dt * (n_times - 1), n_times)
        t1_grid, t2_grid = jnp.meshgrid(t, t, indexing="ij")

        # Meshgrid g1: shape (n_phi, n_times, n_times)
        g1_mesh = _compute_g1_total_core(
            laminar_params, t1_grid, t2_grid, phi_angles, wq, sp, dt
        )

        # Create element-wise arrays from meshgrid
        # Flatten all (phi, t1, t2) combinations into 1D arrays
        t1_flat_list = []
        t2_flat_list = []
        phi_flat_list = []
        g1_expected_list = []

        for p_idx, phi_val in enumerate(phi_angles):
            for i in range(n_times):
                for j in range(n_times):
                    if i != j:  # Skip diagonal (t1==t2) for cleaner comparison
                        t1_flat_list.append(float(t[i]))
                        t2_flat_list.append(float(t[j]))
                        phi_flat_list.append(float(phi_val))
                        g1_expected_list.append(float(g1_mesh[p_idx, i, j]))

        # Ensure >2000 points to trigger element-wise mode
        t1_flat = jnp.array(t1_flat_list)
        t2_flat = jnp.array(t2_flat_list)
        phi_flat = jnp.array(phi_flat_list)
        g1_expected = np.array(g1_expected_list)

        assert len(t1_flat) > 2000, (
            f"Need >2000 points for element-wise mode, got {len(t1_flat)}"
        )

        # Element-wise g1
        g1_elementwise = _compute_g1_total_core(
            laminar_params, t1_flat, t2_flat, phi_flat, wq, sp, dt
        )

        np.testing.assert_allclose(
            np.asarray(g1_elementwise),
            g1_expected,
            rtol=1e-4,
            atol=1e-6,
            err_msg=(
                "Element-wise g1 (CMA-ES mode) diverges from meshgrid g1 "
                "for corresponding (t1, t2, phi) triples. "
                "This indicates a numerical inconsistency between integration methods."
            ),
        )


class TestCMAESDiagonalFiltering:
    """Verify CMA-ES filters diagonal (t1==t2) points before optimization."""

    def test_diagonal_mask_excludes_t1_eq_t2(self):
        """Non-diagonal mask must exclude all i==j index pairs."""
        n_t = 10
        idx1, idx2 = np.meshgrid(np.arange(n_t), np.arange(n_t), indexing="ij")
        non_diag = (idx1 != idx2).flatten()

        # Total grid is n_t * n_t = 100, diagonal has n_t = 10 points
        assert non_diag.sum() == n_t * (n_t - 1)
        assert (~non_diag).sum() == n_t

        # Verify excluded points are exactly where indices match
        t1_flat = idx1.flatten()
        t2_flat = idx2.flatten()
        assert np.all(t1_flat[~non_diag] == t2_flat[~non_diag])
        assert np.all(t1_flat[non_diag] != t2_flat[non_diag])

    def test_tiled_mask_covers_all_phi_angles(self):
        """Tiled mask must filter diagonal for every phi angle independently."""
        n_t = 5
        n_phi = 3
        idx1, idx2 = np.meshgrid(np.arange(n_t), np.arange(n_t), indexing="ij")
        non_diag_single = (idx1 != idx2).flatten()
        non_diag_all = np.tile(non_diag_single, n_phi)

        # Each phi block removes n_t diagonal points
        expected_remaining = n_phi * n_t * (n_t - 1)
        assert non_diag_all.sum() == expected_remaining

    def test_ydata_sigma_length_matches_time_arrays(self):
        """After filtering, ydata/sigma must have same length as tiled time arrays."""
        n_t = 8
        n_phi = 4
        n_grid = n_t * n_t

        # Simulate g2 data: (n_phi, n_t, n_t) flattened
        ydata = np.random.default_rng(42).random(n_phi * n_grid)
        sigma = np.ones(n_phi * n_grid)

        # Build mask
        idx1, idx2 = np.meshgrid(np.arange(n_t), np.arange(n_t), indexing="ij")
        non_diag_single = (idx1 != idx2).flatten()
        non_diag_all = np.tile(non_diag_single, n_phi)

        # Filter data
        ydata_filtered = ydata[non_diag_all]
        sigma_filtered = sigma[non_diag_all]

        # Filter time arrays (per-phi block, then tile)
        t = np.arange(n_t, dtype=float)
        t1_mesh, t2_mesh = np.meshgrid(t, t, indexing="ij")
        t1_flat = t1_mesh.flatten()[non_diag_single]
        t2_flat = t2_mesh.flatten()[non_diag_single]
        n_time_points = len(t1_flat)
        t1_all = np.tile(t1_flat, n_phi)
        t2_all = np.tile(t2_flat, n_phi)

        assert len(ydata_filtered) == len(t1_all)
        assert len(sigma_filtered) == len(t2_all)
        assert len(t1_all) == n_phi * n_time_points
