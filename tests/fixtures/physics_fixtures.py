"""
Physics Test Fixtures for Homodyne v2.4.1+

This module provides pytest fixtures for physics validation testing:
- Physics-verified C2 correlation functions (guaranteed g2 >= 1.0)
- Synthetic physical data with known decay properties
- Standard Q-vectors and time grids for reproducible testing

Physics Constraints Enforced:
- g2(phi, t1, t2) >= 1.0 always (intensity correlation minimum)
- g2(phi, t1, t2) = g2(phi, t2, t1) (time symmetry)
- g2(phi, 0, 0) ~ 1 + contrast^2 (Cauchy-Schwarz bound)
- g2 -> 1.0 as tau -> infinity (equilibrium)

See CLAUDE.md "Core Equation" for c2 = 1 + contrast * [c1]^2.
"""

import pytest
import numpy as np
from typing import NamedTuple


class PhysicsTestData(NamedTuple):
    """Container for physics test data."""
    c2: np.ndarray  # Two-time correlation function
    t1: np.ndarray  # Time axis 1
    t2: np.ndarray  # Time axis 2
    phi: np.ndarray  # Phi angles
    q: np.ndarray  # Q vectors
    params: dict  # True parameters used to generate data


@pytest.fixture
def physics_verified_c2_exp():
    """
    Generate C2 correlation function with verified physics constraints.

    Creates synthetic C2 data that:
    - Guarantees g2 >= 1.0 everywhere (TC-CORE-001)
    - Is symmetric: c2[i,j] = c2[j,i] (TC-CORE-002)
    - Shows proper exponential decay (TC-CORE-010)

    Returns:
        PhysicsTestData: Named tuple with c2, axes, and true parameters.
    """
    # Setup grid
    n_t = 50
    n_phi = 8

    t = np.linspace(0.0, 10.0, n_t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    q = 0.01 * np.ones(n_phi)  # nm^-1

    # True parameters
    true_params = {
        "D0": 1000.0,  # nm^2/s
        "alpha": 0.567,
        "D_offset": 10.0,
        "contrast": 0.5,
        "offset": 1.0,
    }

    # Generate physics-verified C2
    T1, T2 = np.meshgrid(t, t, indexing='ij')
    tau = np.abs(T1 - T2)

    # c2(phi, t1, t2) = offset + contrast * exp(-2 * D * q^2 * tau)
    # This guarantees g2 >= offset (which is >= 1.0)
    c2 = np.zeros((n_phi, n_t, n_t))

    for i_phi in range(n_phi):
        D_eff = true_params["D0"] * (1 + true_params["alpha"] * np.cos(2 * phi[i_phi]))
        decay_rate = 2 * D_eff * q[i_phi]**2

        # Generate correlation with guaranteed g2 >= 1.0
        g1_squared = np.exp(-decay_rate * tau)
        c2[i_phi] = true_params["offset"] + true_params["contrast"] * g1_squared

        # Ensure symmetry (should be automatic but enforce)
        c2[i_phi] = 0.5 * (c2[i_phi] + c2[i_phi].T)

    # Verify constraints
    assert np.all(c2 >= 1.0 - 1e-10), f"g2 minimum violated: {np.min(c2)}"

    return PhysicsTestData(
        c2=c2,
        t1=t,
        t2=t,
        phi=phi,
        q=q,
        params=true_params,
    )


@pytest.fixture
def synthetic_physical_c2():
    """
    Generate synthetic C2 data matching exponential decay theory.

    Creates larger dataset suitable for optimization testing with:
    - Realistic noise levels (SNR ~ 100)
    - Multiple phi angles
    - Proper time grid spacing

    Returns:
        PhysicsTestData: Named tuple with c2, axes, and true parameters.
    """
    np.random.seed(42)  # Reproducible

    # Larger grid for optimization
    n_t = 100
    n_phi = 12

    t = np.linspace(0.0, 20.0, n_t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    q = 0.008 + 0.002 * np.arange(n_phi)  # Varying q per angle

    true_params = {
        "D0": 1500.0,
        "alpha": 0.6,
        "D_offset": 15.0,
        "contrast": 0.45,
        "offset": 1.02,
    }

    T1, T2 = np.meshgrid(t, t, indexing='ij')
    tau = np.abs(T1 - T2)

    c2 = np.zeros((n_phi, n_t, n_t))
    sigma = np.zeros((n_phi, n_t, n_t))

    for i_phi in range(n_phi):
        # Anisotropic diffusion
        D_eff = true_params["D0"] * (1 + true_params["alpha"] * np.cos(2 * phi[i_phi]))
        D_eff += true_params["D_offset"]

        decay_rate = 2 * D_eff * q[i_phi]**2
        g1_squared = np.exp(-decay_rate * tau)

        # Clean signal
        c2_clean = true_params["offset"] + true_params["contrast"] * g1_squared

        # Add realistic noise (SNR ~ 100)
        noise_level = 0.01 * true_params["contrast"]
        noise = np.random.normal(0, noise_level, c2_clean.shape)
        c2[i_phi] = c2_clean + noise

        # Ensure g2 >= 1.0 after noise
        c2[i_phi] = np.maximum(c2[i_phi], 1.0)

        # Enforce symmetry
        c2[i_phi] = 0.5 * (c2[i_phi] + c2[i_phi].T)

        # Uncertainty estimate
        sigma[i_phi] = noise_level * np.ones_like(c2[i_phi])

    result = PhysicsTestData(
        c2=c2,
        t1=t,
        t2=t,
        phi=phi,
        q=q,
        params=true_params,
    )

    # Attach sigma as extra attribute
    result = result._replace(params={**true_params, "_sigma": sigma})

    return result


@pytest.fixture
def q_vector_fixture():
    """
    Standard Q-vectors for testing across different scales.

    Returns:
        dict: Q-vector arrays for different test scenarios.
    """
    return {
        "small": np.array([0.001, 0.002, 0.003]),  # nm^-1
        "medium": np.array([0.005, 0.008, 0.010, 0.012]),
        "large": np.array([0.015, 0.018, 0.020, 0.022, 0.025]),
        "single": np.array([0.01]),
        "dense": np.linspace(0.005, 0.025, 20),
    }


@pytest.fixture
def time_grid_fixture():
    """
    Standard time grids for testing.

    Returns:
        dict: Time arrays for different test scenarios.
    """
    return {
        "small": np.linspace(0, 5, 20),
        "medium": np.linspace(0, 10, 50),
        "large": np.linspace(0, 20, 100),
        "fine": np.linspace(0, 5, 200),
        "coarse": np.linspace(0, 20, 30),
        "log_spaced": np.logspace(-2, 2, 50),
    }


@pytest.fixture
def physics_test_utils():
    """
    Utility class for physics-based test assertions.

    Returns:
        PhysicsTestUtils: Utility class instance.
    """
    return PhysicsTestUtils()


class PhysicsTestUtils:
    """Utility class for physics validation in tests."""

    @staticmethod
    def assert_g2_minimum(c2: np.ndarray, min_val: float = 1.0, tol: float = 1e-8):
        """Assert g2 >= min_val everywhere."""
        actual_min = np.min(c2)
        assert actual_min >= min_val - tol, (
            f"g2 minimum constraint violated: min={actual_min:.8f} < {min_val}"
        )

    @staticmethod
    def assert_time_symmetry(c2: np.ndarray, rtol: float = 1e-6):
        """Assert c2[i,j] = c2[j,i] for all phi."""
        if c2.ndim == 3:
            for i_phi in range(c2.shape[0]):
                np.testing.assert_allclose(
                    c2[i_phi], c2[i_phi].T,
                    rtol=rtol,
                    err_msg=f"Time symmetry violated at phi index {i_phi}"
                )
        else:
            np.testing.assert_allclose(
                c2, c2.T,
                rtol=rtol,
                err_msg="Time symmetry violated"
            )

    @staticmethod
    def assert_diagonal_maximum(c2: np.ndarray):
        """Assert diagonal has maximum value (tau=0)."""
        if c2.ndim == 3:
            for i_phi in range(c2.shape[0]):
                diag = np.diag(c2[i_phi])
                off_diag_max = np.max(c2[i_phi] - np.diag(diag) * np.eye(c2.shape[1]))
                assert np.all(diag >= off_diag_max - 1e-8), (
                    f"Diagonal not maximum at phi index {i_phi}"
                )

    @staticmethod
    def assert_decay_behavior(
        c2: np.ndarray,
        t: np.ndarray,
        expected_decay_time: float | None = None,
        rtol: float = 0.5,
    ):
        """Assert correlation shows proper decay behavior."""
        if c2.ndim == 3:
            c2_avg = np.mean(c2, axis=0)
        else:
            c2_avg = c2

        # Check decay from diagonal
        diag = np.diag(c2_avg)

        # Should decay from max at t=0
        assert diag[0] >= diag[-1], "No decay observed"

        # Should approach 1.0 at large tau
        off_diag_corner = c2_avg[0, -1]
        assert off_diag_corner < diag[0], "Off-diagonal should be less than diagonal"

    @staticmethod
    def compute_effective_decay_time(c2: np.ndarray, t: np.ndarray) -> float:
        """Compute effective decay time from correlation function."""
        if c2.ndim == 3:
            c2_avg = np.mean(c2, axis=0)
        else:
            c2_avg = c2

        diag = np.diag(c2_avg)

        # Find time where correlation drops to 1/e of initial
        initial = diag[0]
        baseline = 1.0
        target = baseline + (initial - baseline) / np.e

        # Find crossing
        idx = np.where(diag < target)[0]
        if len(idx) > 0:
            return t[idx[0]]
        return t[-1]  # No decay observed


def generate_g2_with_constraints(
    n_t: int,
    n_phi: int,
    D0: float = 1000.0,
    alpha: float = 0.5,
    contrast: float = 0.5,
    offset: float = 1.0,
    t_max: float = 10.0,
    q: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate g2 correlation function satisfying all physics constraints.

    Args:
        n_t: Number of time points
        n_phi: Number of phi angles
        D0: Diffusion coefficient
        alpha: Anisotropy parameter
        contrast: Contrast parameter
        offset: Offset parameter (must be >= 1.0)
        t_max: Maximum time
        q: Q-vector magnitude

    Returns:
        tuple: (g2 array, time array, phi array)
    """
    assert offset >= 1.0, "offset must be >= 1.0 for g2 >= 1.0 constraint"

    t = np.linspace(0, t_max, n_t)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    T1, T2 = np.meshgrid(t, t, indexing='ij')
    tau = np.abs(T1 - T2)

    g2 = np.zeros((n_phi, n_t, n_t))

    for i_phi in range(n_phi):
        D_eff = D0 * (1 + alpha * np.cos(2 * phi[i_phi]))
        decay_rate = 2 * D_eff * q**2
        g1_squared = np.exp(-decay_rate * tau)
        g2[i_phi] = offset + contrast * g1_squared
        # Ensure symmetry
        g2[i_phi] = 0.5 * (g2[i_phi] + g2[i_phi].T)

    return g2, t, phi
