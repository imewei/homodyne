"""Unified Backend API Facade for Homodyne Physics Computations.

This module provides a unified API for accessing physics computations,
automatically selecting between NLSQ (meshgrid) and CMC (element-wise)
backends based on the computation mode.

Architecture (Nov 2025):
------------------------
The physics backends are intentionally separate:
- **physics_nlsq.py**: Meshgrid computations for 2D correlation matrices (NLSQ)
- **physics_cmc.py**: Element-wise computations for paired arrays (CMC MCMC)
- **physics_utils.py**: Shared utility functions used by both

This facade provides a clean import interface for users who don't need
to know about the underlying implementation details.

Usage::

    # For NLSQ optimization (meshgrid mode)
    from homodyne.core.backend_api import compute_g2_nlsq, apply_diagonal_correction

    # For CMC MCMC (element-wise mode)
    from homodyne.core.backend_api import compute_g1_cmc

    # For shared utilities
    from homodyne.core.backend_api import (
        calculate_diffusion_coefficient,
        calculate_shear_rate,
        safe_sinc,
        safe_exp,
    )

    # Check available backends
    from homodyne.core.backend_api import get_available_backends
    print(get_available_backends())  # {'nlsq': True, 'cmc': True, 'jax': True}

Part of Architecture Refactoring v2.9.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Import CMC backend (element-wise computations)
from homodyne.core.physics_cmc import (
    compute_g1_diffusion as compute_g1_diffusion_cmc,
)
from homodyne.core.physics_cmc import (
    compute_g1_total as compute_g1_cmc,
)

# Import NLSQ backend (meshgrid computations)
from homodyne.core.physics_nlsq import (
    compute_g2_scaled as compute_g2_nlsq,
)
from homodyne.core.physics_nlsq import (
    compute_g2_scaled_with_factors as compute_g2_nlsq_with_factors,
)

# Import from physics_utils (shared base)
from homodyne.core.physics_utils import (
    EPS,
    PI,
    apply_diagonal_correction,
    apply_diagonal_correction_batch,
    calculate_diffusion_coefficient,
    calculate_shear_rate,
    calculate_shear_rate_cmc,
    create_time_integral_matrix,
    safe_exp,
    safe_len,
    safe_sinc,
    trapezoid_cumsum,
)

if TYPE_CHECKING:
    pass


def get_available_backends() -> dict[str, bool]:
    """Check which physics backends are available.

    Returns
    -------
    dict[str, bool]
        Dictionary with backend availability:
        - 'nlsq': NLSQ meshgrid backend available
        - 'cmc': CMC element-wise backend available
        - 'jax': JAX library available
        - 'utils': Shared utilities available

    Example
    -------
    >>> backends = get_available_backends()
    >>> if backends['nlsq']:
    ...     from homodyne.core.backend_api import compute_g2_nlsq
    """
    backends = {
        "nlsq": False,
        "cmc": False,
        "jax": False,
        "utils": False,
    }

    # Check JAX availability
    try:
        import jax

        backends["jax"] = True
    except ImportError:
        pass

    # Check NLSQ backend
    try:
        from homodyne.core.physics_nlsq import compute_g2_scaled

        backends["nlsq"] = True
    except ImportError:
        pass

    # Check CMC backend
    try:
        from homodyne.core.physics_cmc import compute_g1_total

        backends["cmc"] = True
    except ImportError:
        pass

    # Check utilities
    try:
        from homodyne.core.physics_utils import safe_sinc

        backends["utils"] = True
    except ImportError:
        pass

    return backends


def get_backend_info() -> dict[str, str]:
    """Get detailed information about physics backends.

    Returns
    -------
    dict[str, str]
        Dictionary with backend descriptions and status.

    Example
    -------
    >>> info = get_backend_info()
    >>> print(info['nlsq_description'])
    'Meshgrid computations for 2D correlation matrices'
    """
    backends = get_available_backends()

    return {
        "nlsq_available": "Yes" if backends["nlsq"] else "No",
        "nlsq_description": "Meshgrid computations for 2D correlation matrices",
        "nlsq_use_case": "NLSQ (Nonlinear Least Squares) optimization",
        "cmc_available": "Yes" if backends["cmc"] else "No",
        "cmc_description": "Element-wise computations for paired arrays",
        "cmc_use_case": "CMC (Consensus Monte Carlo) MCMC sampling",
        "jax_available": "Yes" if backends["jax"] else "No",
        "utils_available": "Yes" if backends["utils"] else "No",
        "architecture_note": (
            "NLSQ and CMC backends are intentionally separate (Nov 2025) "
            "to prevent 80GB OOM during MCMC. Use appropriate backend for your mode."
        ),
    }


# Re-export all public functions for convenience
__all__ = [
    # Constants
    "PI",
    "EPS",
    # Shared utilities
    "safe_len",
    "safe_exp",
    "safe_sinc",
    "calculate_diffusion_coefficient",
    "calculate_shear_rate",
    "calculate_shear_rate_cmc",
    "create_time_integral_matrix",
    "trapezoid_cumsum",
    "apply_diagonal_correction",
    "apply_diagonal_correction_batch",
    # NLSQ backend (meshgrid)
    "compute_g2_nlsq",
    "compute_g2_nlsq_with_factors",
    # CMC backend (element-wise)
    "compute_g1_diffusion_cmc",
    "compute_g1_cmc",
    # Introspection
    "get_available_backends",
    "get_backend_info",
]
