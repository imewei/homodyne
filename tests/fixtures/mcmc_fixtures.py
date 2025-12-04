"""
MCMC Test Fixtures for Homodyne v2.4.1+

This module provides pytest fixtures for MCMC testing, particularly:
- Parameter ordering validation (critical for NumPyro compatibility)
- Per-angle scaling parameter structures
- Mock MCMC samples for testing
- Convergence diagnostics fixtures

CRITICAL: NumPyro init_to_value() requires parameters in EXACT order as model.sample().
The correct order is:
1. Per-angle contrast params: contrast_0, contrast_1, ..., contrast_{n_phi-1}
2. Per-angle offset params:   offset_0, offset_1, ..., offset_{n_phi-1}
3. Physical parameters:        D0, alpha, D_offset, [gamma_dot_t0, beta, ...]

See CLAUDE.md "Known Issues > MCMC Initialization" for details.
"""

import pytest
import numpy as np
from typing import Any


@pytest.fixture
def per_angle_params_static():
    """
    Per-angle parameter structure for static isotropic mode with 3 angles.

    v2.4.0 format: [c0, c1, c2, o0, o1, o2, D0, alpha, D_offset]

    Returns:
        dict: Parameter dictionary with correct ordering and values.
    """
    return {
        # Per-angle contrast (first)
        "contrast_0": 0.5,
        "contrast_1": 0.52,
        "contrast_2": 0.48,
        # Per-angle offset (second)
        "offset_0": 1.0,
        "offset_1": 1.01,
        "offset_2": 0.99,
        # Physical parameters (last)
        "D0": 1000.0,
        "alpha": 0.567,
        "D_offset": 10.0,
    }


@pytest.fixture
def per_angle_params_laminar():
    """
    Per-angle parameter structure for laminar flow mode with 3 angles.

    v2.4.0 format: [c0, c1, c2, o0, o1, o2, D0, alpha, D_offset,
                    gamma_dot_t0, beta, gamma_dot_t_offset, phi0]

    Returns:
        dict: Parameter dictionary with correct ordering and values.
    """
    return {
        # Per-angle contrast (first)
        "contrast_0": 0.5,
        "contrast_1": 0.52,
        "contrast_2": 0.48,
        # Per-angle offset (second)
        "offset_0": 1.0,
        "offset_1": 1.01,
        "offset_2": 0.99,
        # Physical parameters (last)
        "D0": 1000.0,
        "alpha": 0.567,
        "D_offset": 10.0,
        "gamma_dot_t0": 100.0,
        "beta": 0.8,
        "gamma_dot_t_offset": 5.0,
        "phi0": 0.0,
    }


@pytest.fixture
def mcmc_parameter_ordering_static():
    """
    Correct parameter ordering for NumPyro init_to_value() in static mode.

    This fixture provides the EXACT order that NumPyro expects parameters
    when using init_to_value(). Getting this wrong causes:
    "Cannot find valid initial parameters" errors.

    Returns:
        list: Parameter names in correct sampling order.
    """
    n_angles = 3
    order = []

    # 1. Per-angle contrast params FIRST
    for i in range(n_angles):
        order.append(f"contrast_{i}")

    # 2. Per-angle offset params SECOND
    for i in range(n_angles):
        order.append(f"offset_{i}")

    # 3. Physical params LAST
    order.extend(["D0", "alpha", "D_offset"])

    return order


@pytest.fixture
def mcmc_parameter_ordering_laminar():
    """
    Correct parameter ordering for NumPyro init_to_value() in laminar flow mode.

    Returns:
        list: Parameter names in correct sampling order.
    """
    n_angles = 3
    order = []

    # 1. Per-angle contrast params FIRST
    for i in range(n_angles):
        order.append(f"contrast_{i}")

    # 2. Per-angle offset params SECOND
    for i in range(n_angles):
        order.append(f"offset_{i}")

    # 3. Physical params LAST (including laminar flow params)
    order.extend([
        "D0", "alpha", "D_offset",
        "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
    ])

    return order


@pytest.fixture
def mock_mcmc_samples():
    """
    Generate mock MCMC samples for testing posterior analysis.

    Provides 2 chains with 500 samples each, simulating converged chains.

    Returns:
        dict: Dictionary of parameter samples with shape (2, 500).
    """
    n_chains = 2
    n_samples = 500
    np.random.seed(42)  # Reproducible

    # Generate samples around true values with realistic spread
    samples = {
        "D0": np.random.normal(1000.0, 50.0, (n_chains, n_samples)),
        "alpha": np.random.normal(0.567, 0.02, (n_chains, n_samples)),
        "D_offset": np.random.normal(10.0, 1.0, (n_chains, n_samples)),
        "contrast_0": np.random.normal(0.5, 0.02, (n_chains, n_samples)),
        "contrast_1": np.random.normal(0.52, 0.02, (n_chains, n_samples)),
        "contrast_2": np.random.normal(0.48, 0.02, (n_chains, n_samples)),
        "offset_0": np.random.normal(1.0, 0.01, (n_chains, n_samples)),
        "offset_1": np.random.normal(1.01, 0.01, (n_chains, n_samples)),
        "offset_2": np.random.normal(0.99, 0.01, (n_chains, n_samples)),
    }

    # Ensure physical constraints
    samples["D0"] = np.clip(samples["D0"], 100.0, 10000.0)
    samples["alpha"] = np.clip(samples["alpha"], 0.4, 1.0)
    samples["D_offset"] = np.clip(samples["D_offset"], 0.1, 100.0)

    return samples


@pytest.fixture
def convergence_diagnostics_fixture():
    """
    Pre-computed convergence diagnostics for testing.

    Provides Gelman-Rubin R-hat, ESS, and other diagnostic values
    that represent a well-converged MCMC run.

    Returns:
        dict: Convergence diagnostic values.
    """
    return {
        "r_hat": {
            "D0": 1.01,
            "alpha": 1.02,
            "D_offset": 1.01,
            "contrast_0": 1.00,
            "contrast_1": 1.01,
            "contrast_2": 1.01,
            "offset_0": 1.00,
            "offset_1": 1.00,
            "offset_2": 1.01,
        },
        "ess": {
            "D0": 450,
            "alpha": 480,
            "D_offset": 420,
            "contrast_0": 500,
            "contrast_1": 490,
            "contrast_2": 485,
            "offset_0": 510,
            "offset_1": 505,
            "offset_2": 495,
        },
        "is_converged": True,
        "n_chains": 2,
        "n_samples": 500,
        "n_warmup": 200,
    }


def validate_parameter_ordering(params_dict: dict[str, Any], n_angles: int) -> bool:
    """
    Validate that a parameter dictionary has correct ordering for NumPyro.

    Args:
        params_dict: Dictionary of parameter values
        n_angles: Number of angles (determines per-angle param count)

    Returns:
        bool: True if ordering is correct, False otherwise.

    Raises:
        ValueError: If parameters are in wrong order with detailed message.
    """
    keys = list(params_dict.keys())

    # Expected order: contrast_0..n, offset_0..n, then physical params
    expected_contrast = [f"contrast_{i}" for i in range(n_angles)]
    expected_offset = [f"offset_{i}" for i in range(n_angles)]

    # Check contrast params come first
    for i, key in enumerate(expected_contrast):
        if i >= len(keys) or keys[i] != key:
            raise ValueError(
                f"Parameter ordering error at position {i}: "
                f"expected '{key}' but got '{keys[i] if i < len(keys) else 'MISSING'}'. "
                f"NumPyro requires contrast params first."
            )

    # Check offset params come second
    offset_start = len(expected_contrast)
    for i, key in enumerate(expected_offset):
        idx = offset_start + i
        if idx >= len(keys) or keys[idx] != key:
            raise ValueError(
                f"Parameter ordering error at position {idx}: "
                f"expected '{key}' but got '{keys[idx] if idx < len(keys) else 'MISSING'}'. "
                f"NumPyro requires offset params after contrast params."
            )

    # Physical params should come after per-angle params
    physical_start = len(expected_contrast) + len(expected_offset)
    if physical_start >= len(keys):
        raise ValueError("Missing physical parameters after per-angle params.")

    # D0 should be first physical param
    if keys[physical_start] != "D0":
        raise ValueError(
            f"Physical parameter ordering error: expected 'D0' at position {physical_start}, "
            f"got '{keys[physical_start]}'."
        )

    return True
