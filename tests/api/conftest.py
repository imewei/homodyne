"""API test fixtures — lightweight overrides for interface contract tests.

API compatibility tests verify return types, attribute names, and function
signatures, not numerical accuracy. These minimal fixtures reduce JAX JIT
compilation and memory pressure on CI runners where the preceding unit test
stage (2700+ tests) leaves limited headroom.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def test_config():
    """Minimal config for API contract tests.

    Uses fewer iterations so the optimizer finishes quickly
    even without cached JIT artifacts.
    """
    return {
        "analysis_mode": "static",
        "optimization": {
            "method": "nlsq",
            "lsq": {"max_iterations": 10, "tolerance": 1e-4},
        },
        "hardware": {"force_cpu": True},
        "output": {"save_plots": False, "verbose": False},
    }


@pytest.fixture(scope="module")
def synthetic_xpcs_data():
    """Minimal synthetic data for API contract tests.

    10x10x3 (300 elements) vs main conftest's 50x50x36 (90K elements).
    Sufficient for verifying return type structure.
    """
    n_times = 10
    n_angles = 3

    t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing="ij")
    phi = np.linspace(0, 2 * np.pi, n_angles)
    tau = np.abs(t1 - t2) + 1e-6
    c2_base = 1 + 0.5 * np.exp(-tau / 5.0)
    c2_exp = np.tile(c2_base, (n_angles, 1, 1))

    np.random.seed(42)
    c2_exp += 0.01 * np.random.randn(*c2_exp.shape)

    return {
        "t1": t1,
        "t2": t2,
        "phi_angles_list": phi,
        "c2_exp": c2_exp,
        "wavevector_q_list": np.array([0.01]),
        "sigma": np.ones_like(c2_exp) * 0.01,
        "dt": 0.1,
    }
