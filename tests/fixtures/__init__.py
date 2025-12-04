"""
Centralized test fixtures for homodyne test suite.

This module provides reusable pytest fixtures for:
- MCMC parameter ordering and validation
- Physics-verified correlation function generation
- Performance benchmark baselines

Usage:
    Import fixtures directly in test files or conftest.py:

    from tests.fixtures.mcmc_fixtures import (
        per_angle_params_static,
        per_angle_params_laminar,
        mcmc_parameter_ordering_static,
    )

    from tests.fixtures.physics_fixtures import (
        physics_verified_c2_exp,
        synthetic_physical_c2,
    )
"""

from tests.fixtures.mcmc_fixtures import (
    per_angle_params_static,
    per_angle_params_laminar,
    mcmc_parameter_ordering_static,
    mcmc_parameter_ordering_laminar,
    mock_mcmc_samples,
    convergence_diagnostics_fixture,
)

from tests.fixtures.physics_fixtures import (
    physics_verified_c2_exp,
    synthetic_physical_c2,
    q_vector_fixture,
    time_grid_fixture,
    physics_test_utils,
)

__all__ = [
    # MCMC fixtures
    "per_angle_params_static",
    "per_angle_params_laminar",
    "mcmc_parameter_ordering_static",
    "mcmc_parameter_ordering_laminar",
    "mock_mcmc_samples",
    "convergence_diagnostics_fixture",
    # Physics fixtures
    "physics_verified_c2_exp",
    "synthetic_physical_c2",
    "q_vector_fixture",
    "time_grid_fixture",
    "physics_test_utils",
]
