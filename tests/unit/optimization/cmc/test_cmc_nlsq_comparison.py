"""Direct comparison test between CMC and NLSQ physics computations.

This test verifies that CMC element-wise physics produces the same results
as NLSQ meshgrid physics when given equivalent inputs.

Created: Dec 2025
Purpose: Diagnose CMC constant C2 output bug
"""

import numpy as np
import jax.numpy as jnp
import pytest

from homodyne.core.physics_cmc import (
    compute_g1_total as compute_g1_total_cmc,
    _compute_g1_diffusion_elementwise,
)
from homodyne.core.physics_nlsq import (
    compute_g2_scaled as compute_g2_scaled_nlsq,
    _compute_g1_diffusion_meshgrid,
    _compute_g1_total_meshgrid,
)
from homodyne.core.physics_utils import (
    trapezoid_cumsum,
    create_time_integral_matrix,
    calculate_diffusion_coefficient,
)


class TestCMCNLSQDiffusionComparison:
    """Test diffusion computation parity between CMC and NLSQ."""

    @pytest.fixture
    def physics_params(self):
        """Standard test parameters."""
        return {
            "D0": 1e10,
            "alpha": -0.5,
            "D_offset": 1e9,
            "q": 0.01,  # nm^-1
            "dt": 0.1,  # seconds
            "n_times": 50,
        }

    def test_cumsum_vs_matrix_integral(self, physics_params):
        """Verify trapezoid_cumsum produces same results as create_time_integral_matrix."""
        D0 = physics_params["D0"]
        alpha = physics_params["alpha"]
        D_offset = physics_params["D_offset"]
        dt = physics_params["dt"]
        n_times = physics_params["n_times"]

        # Create time grid (physical time, starting from 0)
        time_grid = np.linspace(0, dt * (n_times - 1), n_times)

        # Compute D(t) at each time point
        D_t = calculate_diffusion_coefficient(jnp.array(time_grid), D0, alpha, D_offset)
        D_t_np = np.array(D_t)

        # Method 1: NLSQ uses create_time_integral_matrix
        D_matrix = np.array(create_time_integral_matrix(jnp.array(D_t_np)))

        # Method 2: CMC uses trapezoid_cumsum + searchsorted
        D_cumsum = np.array(trapezoid_cumsum(jnp.array(D_t_np)))

        # Compare: D_matrix[i,j] should equal |D_cumsum[i] - D_cumsum[j]|
        for i in range(n_times):
            for j in range(n_times):
                expected = np.sqrt((D_cumsum[i] - D_cumsum[j]) ** 2 + 1e-20)
                actual = D_matrix[i, j]
                assert np.isclose(expected, actual, rtol=1e-10), (
                    f"Integral mismatch at ({i},{j}): matrix={actual:.6g}, "
                    f"cumsum_diff={expected:.6g}"
                )

        print(f"✓ D_matrix and cumsum method produce identical integrals")
        print(f"  D_matrix range: [{D_matrix.min():.4g}, {D_matrix.max():.4g}]")
        print(f"  D_cumsum range: [{D_cumsum.min():.4g}, {D_cumsum.max():.4g}]")

    def test_g1_diffusion_cmc_vs_nlsq(self, physics_params):
        """Compare g1_diffusion between CMC element-wise and NLSQ meshgrid."""
        D0 = physics_params["D0"]
        alpha = physics_params["alpha"]
        D_offset = physics_params["D_offset"]
        q = physics_params["q"]
        dt = physics_params["dt"]
        n_times = physics_params["n_times"]

        # Create time grid
        time_grid = np.linspace(0, dt * (n_times - 1), n_times)

        # Create meshgrid for NLSQ
        t1_mesh, t2_mesh = np.meshgrid(time_grid, time_grid, indexing="ij")

        # Create pooled arrays for CMC (flatten meshgrid)
        t1_pooled = t1_mesh.flatten()
        t2_pooled = t2_mesh.flatten()

        # Physics parameters
        params = jnp.array([D0, alpha, D_offset])
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

        # NLSQ: Compute g1_diffusion using meshgrid
        g1_nlsq = np.array(
            _compute_g1_diffusion_meshgrid(
                params,
                jnp.array(t1_mesh),
                jnp.array(t2_mesh),
                wavevector_q_squared_half_dt,
                dt,
            )
        )

        # CMC: Compute g1_diffusion using element-wise
        g1_cmc_flat = np.array(
            _compute_g1_diffusion_elementwise(
                params,
                jnp.array(t1_pooled),
                jnp.array(t2_pooled),
                jnp.array(time_grid),
                wavevector_q_squared_half_dt,
            )
        )

        # Reshape CMC result to match NLSQ shape
        g1_cmc = g1_cmc_flat.reshape(n_times, n_times)

        # Compare
        max_diff = np.abs(g1_nlsq - g1_cmc).max()
        mean_diff = np.abs(g1_nlsq - g1_cmc).mean()

        print(f"\n=== g1_diffusion comparison ===")
        print(f"  NLSQ g1 range: [{g1_nlsq.min():.6f}, {g1_nlsq.max():.6f}]")
        print(f"  CMC g1 range:  [{g1_cmc.min():.6f}, {g1_cmc.max():.6f}]")
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")

        # Sample points for detailed comparison
        sample_indices = [(0, 0), (0, 10), (10, 10), (10, 40), (49, 49)]
        print(f"\n  Sample comparisons (i,j): NLSQ vs CMC")
        for i, j in sample_indices:
            print(f"    ({i},{j}): {g1_nlsq[i, j]:.6f} vs {g1_cmc[i, j]:.6f}")

        # Assert they're close (allowing small numerical differences)
        assert max_diff < 1e-5, f"g1_diffusion mismatch: max_diff={max_diff:.2e}"

    def test_g1_total_with_shear(self, physics_params):
        """Compare g1_total (diffusion + shear) between CMC and NLSQ."""
        D0 = physics_params["D0"]
        alpha = physics_params["alpha"]
        D_offset = physics_params["D_offset"]
        q = physics_params["q"]
        dt = physics_params["dt"]
        n_times = physics_params["n_times"]
        L = 2e6  # nm

        # Full laminar flow parameters
        gamma_dot_0 = 1e3
        beta = 0.0
        gamma_dot_offset = 0.0
        phi0 = 45.0

        # Create time grid
        time_grid = np.linspace(0, dt * (n_times - 1), n_times)

        # Test with multiple phi angles
        phi_unique = np.array([0.0, 45.0, 90.0])

        # Create meshgrid for NLSQ
        t1_mesh, t2_mesh = np.meshgrid(time_grid, time_grid, indexing="ij")

        # Create pooled arrays for CMC
        t1_pooled = t1_mesh.flatten()
        t2_pooled = t2_mesh.flatten()

        # Physics parameters (7 params for laminar flow)
        params = jnp.array(
            [D0, alpha, D_offset, gamma_dot_0, beta, gamma_dot_offset, phi0]
        )
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
        sinc_prefactor = 0.5 / np.pi * q * L * dt

        # NLSQ: Compute g1_total using meshgrid
        g1_nlsq = np.array(
            _compute_g1_total_meshgrid(
                params,
                jnp.array(t1_mesh),
                jnp.array(t2_mesh),
                jnp.array(phi_unique),
                wavevector_q_squared_half_dt,
                sinc_prefactor,
                dt,
            )
        )  # Shape: (n_phi, n_times, n_times)

        # CMC: Compute g1_total using element-wise
        g1_cmc_2d = np.array(
            compute_g1_total_cmc(
                params,
                jnp.array(t1_pooled),
                jnp.array(t2_pooled),
                jnp.array(phi_unique),
                q,
                L,
                dt,
                time_grid=jnp.array(time_grid),
                _debug=False,
            )
        )  # Shape: (n_phi, n_points)

        print(f"\n=== g1_total comparison (with shear) ===")
        print(f"  NLSQ shape: {g1_nlsq.shape}")
        print(f"  CMC shape: {g1_cmc_2d.shape}")

        # Reshape CMC result: (n_phi, n_points) -> (n_phi, n_times, n_times)
        n_phi = len(phi_unique)
        g1_cmc = g1_cmc_2d.reshape(n_phi, n_times, n_times)

        for phi_idx, phi_val in enumerate(phi_unique):
            g1_nlsq_phi = g1_nlsq[phi_idx]
            g1_cmc_phi = g1_cmc[phi_idx]

            max_diff = np.abs(g1_nlsq_phi - g1_cmc_phi).max()
            mean_diff = np.abs(g1_nlsq_phi - g1_cmc_phi).mean()

            print(f"\n  phi={phi_val}°:")
            print(
                f"    NLSQ g1 range: [{g1_nlsq_phi.min():.6f}, {g1_nlsq_phi.max():.6f}]"
            )
            print(
                f"    CMC g1 range:  [{g1_cmc_phi.min():.6f}, {g1_cmc_phi.max():.6f}]"
            )
            print(f"    Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

            # Sample points
            print(f"    Sample (i,j): NLSQ vs CMC")
            for i, j in [(0, 10), (25, 25), (10, 40)]:
                print(
                    f"      ({i},{j}): {g1_nlsq_phi[i, j]:.6f} vs {g1_cmc_phi[i, j]:.6f}"
                )

            assert (
                max_diff < 1e-4
            ), f"g1_total mismatch at phi={phi_val}: max_diff={max_diff:.2e}"


class TestCMCSearchsortedIndexing:
    """Test that searchsorted correctly maps physical times to grid indices."""

    def test_searchsorted_exact_match(self):
        """Verify searchsorted finds exact matches correctly."""
        time_grid = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        # Test exact matches
        test_times = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        expected_indices = [0, 1, 2, 3, 4, 5]

        indices = jnp.searchsorted(time_grid, test_times, side="left")

        print(f"\nSearchsorted exact match test:")
        for t, idx, expected in zip(test_times, indices, expected_indices):
            print(f"  t={float(t):.1f} -> idx={int(idx)} (expected={expected})")
            assert int(idx) == expected

    def test_searchsorted_between_values(self):
        """Verify searchsorted behavior for values between grid points."""
        time_grid = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

        # Test values between grid points
        test_times = jnp.array([0.05, 0.15, 0.25])
        # side="left" returns index where value would be inserted
        expected_indices = [1, 2, 3]  # 0.05 goes before 0.1, 0.15 before 0.2, etc.

        indices = jnp.searchsorted(time_grid, test_times, side="left")

        print(f"\nSearchsorted between-values test:")
        for t, idx, expected in zip(test_times, indices, expected_indices):
            print(f"  t={float(t):.2f} -> idx={int(idx)} (expected={expected})")
            assert int(idx) == expected, f"Mismatch at t={t}"

    def test_cumsum_integral_sample(self):
        """Verify cumsum integral computation for specific time pairs."""
        # Simple case: constant D(t) = 1
        D_t = jnp.array([1.0, 1.0, 1.0, 1.0, 1.0])  # 5 time points
        D_cumsum = trapezoid_cumsum(D_t)

        print(f"\nCumsum integral test (D=1 constant):")
        print(f"  D_t = {np.array(D_t)}")
        print(f"  D_cumsum = {np.array(D_cumsum)}")

        # For constant D=1, trapezoidal integral from 0 to t should be t
        # But trapezoid_cumsum doesn't include dt scaling
        # So cumsum[i] = number of trapezoid steps = i steps
        # For D=1: cumsum = [0, 1, 2, 3, 4]
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(D_cumsum, expected, rtol=1e-10)

        # Test integral from t1=1 to t2=3
        idx1, idx2 = 1, 3
        integral_steps = float(np.abs(D_cumsum[idx2] - D_cumsum[idx1]))
        print(f"  Integral from idx {idx1} to {idx2}: {integral_steps} steps")
        assert integral_steps == 2.0  # 2 steps from index 1 to 3


class TestCMCPhysicsValues:
    """Test that CMC physics produces physically reasonable values."""

    def test_g1_decay_at_large_times(self):
        """Verify g1 decays to small values at large time separations."""
        # Parameters that should cause significant decay
        D0 = 1e10
        alpha = -0.5
        D_offset = 1e9
        q = 0.01
        dt = 0.1
        n_times = 100

        time_grid = np.linspace(0, dt * (n_times - 1), n_times)
        params = jnp.array([D0, alpha, D_offset])
        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt

        # CMC element-wise at diagonal (t1 = t2 = varying)
        t1_diag = time_grid
        t2_diag = time_grid
        g1_diagonal = np.array(
            _compute_g1_diffusion_elementwise(
                params,
                jnp.array(t1_diag),
                jnp.array(t2_diag),
                jnp.array(time_grid),
                wavevector_q_squared_half_dt,
            )
        )

        # CMC element-wise at off-diagonal (t1 = 0, t2 = varying)
        t1_off = np.zeros_like(time_grid)
        t2_off = time_grid
        g1_off_diagonal = np.array(
            _compute_g1_diffusion_elementwise(
                params,
                jnp.array(t1_off),
                jnp.array(t2_off),
                jnp.array(time_grid),
                wavevector_q_squared_half_dt,
            )
        )

        print(f"\n=== g1 decay test ===")
        print(f"  wavevector_q_squared_half_dt = {wavevector_q_squared_half_dt:.6g}")
        print(f"\n  Diagonal (t1=t2): g1 should be ~1")
        print(f"    g1 at t=0: {g1_diagonal[0]:.6f}")
        print(f"    g1 at t=5: {g1_diagonal[50]:.6f}")
        print(f"    g1 at t=10: {g1_diagonal[-1]:.6f}")

        print(f"\n  Off-diagonal (t1=0, t2=varying): g1 should decay")
        print(f"    g1 at delta_t=0: {g1_off_diagonal[0]:.6f}")
        print(f"    g1 at delta_t=1: {g1_off_diagonal[10]:.6f}")
        print(f"    g1 at delta_t=5: {g1_off_diagonal[50]:.6f}")
        print(f"    g1 at delta_t=10: {g1_off_diagonal[-1]:.6f}")

        # Diagonal should be ~1 (no time separation)
        assert (
            g1_diagonal[0] > 0.99
        ), f"g1 at t1=t2=0 should be ~1, got {g1_diagonal[0]}"

        # Off-diagonal should decay as time separation increases
        assert (
            g1_off_diagonal[-1] < g1_off_diagonal[0]
        ), f"g1 should decay: g1(0)={g1_off_diagonal[0]:.4f} vs g1(max)={g1_off_diagonal[-1]:.4f}"

    def test_prefactor_magnitude(self):
        """Debug test to check physics prefactor magnitudes."""
        # Realistic XPCS parameters
        q = 0.01  # nm^-1
        L = 2e6  # nm (2mm)
        dt = 0.1  # seconds

        # D(t) typical values
        D0 = 1e10  # nm²/s
        alpha = -0.5
        D_offset = 1e9  # nm²/s

        wavevector_q_squared_half_dt = 0.5 * (q**2) * dt
        sinc_prefactor = 0.5 / np.pi * q * L * dt

        print(f"\n=== Physics prefactor check ===")
        print(f"  q = {q} nm^-1")
        print(f"  L = {L:.2e} nm")
        print(f"  dt = {dt} s")
        print(f"  D0 = {D0:.2e} nm²/s")
        print(f"  alpha = {alpha}")
        print(f"  D_offset = {D_offset:.2e} nm²/s")
        print(
            f"\n  wavevector_q_squared_half_dt = 0.5 * {q}² * {dt} = {wavevector_q_squared_half_dt:.6g}"
        )
        print(f"  sinc_prefactor = 0.5/π * {q} * {L:.2e} * {dt} = {sinc_prefactor:.6g}")

        # Estimate D values at different times
        t_samples = np.array([0.0, 1.0, 5.0, 10.0])
        D_samples = np.array(
            calculate_diffusion_coefficient(
                jnp.array(t_samples + 1e-10), D0, alpha, D_offset
            )
        )

        print(f"\n  D(t) values:")
        for t, D in zip(t_samples, D_samples):
            print(f"    D(t={t:.0f}s) = {D:.4e} nm²/s")

        # Estimate integral for large time separation
        # For constant D: integral ≈ D * n_steps
        n_steps = 100
        avg_D = D_samples.mean()
        estimated_integral = avg_D * n_steps

        # Estimated exponent for g1
        exponent = -wavevector_q_squared_half_dt * estimated_integral
        estimated_g1 = np.exp(max(exponent, -700))

        print(f"\n  Estimated integral for {n_steps} steps: {estimated_integral:.4e}")
        print(
            f"  Exponent = -{wavevector_q_squared_half_dt:.6g} * {estimated_integral:.4e} = {exponent:.4g}"
        )
        print(f"  Estimated g1 = exp({exponent:.4g}) = {estimated_g1:.6g}")

        # This should give us insight into whether the physics produces reasonable decay


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
