"""Shared mock configuration objects for tests.

Provides factory functions for creating mock configs used across
multiple test files, eliminating inline MockConfig class duplication.
"""

from __future__ import annotations

from unittest.mock import Mock


def create_mock_nlsq_config(overrides: dict | None = None) -> Mock:
    """Create a mock NLSQ configuration for testing.

    Args:
        overrides: Optional dict of attribute overrides.

    Returns:
        Mock object with standard NLSQ config attributes.
    """
    config = Mock()
    config.analysis_mode = "static"
    config.n_params = 3
    config.loss_function = "linear"
    config.x_scale = "jac"
    config.max_iterations = 100
    config.ftol = 1e-8
    config.gtol = 1e-8
    config.xtol = 1e-8
    config.verbose = 0

    if overrides:
        for key, value in overrides.items():
            setattr(config, key, value)

    return config


def create_mock_xpcs_data(n_phi: int = 3, n_t1: int = 20, n_t2: int = 20) -> dict:
    """Create mock XPCS data dictionary for testing.

    Args:
        n_phi: Number of phi angles.
        n_t1: Number of t1 time points.
        n_t2: Number of t2 time points.

    Returns:
        Dict with standard XPCS data keys.
    """
    import numpy as np

    return {
        "phi_angles_list": np.linspace(0, 180, n_phi),
        "c2_exp": np.random.rand(n_phi, n_t1, n_t2) + 1.0,
        "wavevector_q_list": np.array([0.01]),
        "t1": np.linspace(0.001, 1.0, n_t1),
        "t2": np.linspace(0.001, 1.0, n_t2),
    }
