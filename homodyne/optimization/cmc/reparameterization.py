# homodyne/optimization/cmc/reparameterization.py
"""Internal reparameterization for CMC sampling.

Transforms correlated parameters to orthogonal sampling space,
then converts back to physics parameters for output.

Reparameterizations:
- D0, D_offset → D_total, D_offset_frac (breaks linear degeneracy)
- gamma_dot_t0 → log_gamma_dot_t0 (improves conditioning)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReparamConfig:
    """Configuration for internal reparameterization.

    Attributes
    ----------
    enable_d_total : bool
        Enable D_total reparameterization (D0 + D_offset → D_total).
    enable_log_gamma : bool
        Sample log(gamma_dot_t0) instead of gamma_dot_t0.
    t_ref : float
        Reference time for shear scaling (not currently used).
    """

    enable_d_total: bool = True
    enable_log_gamma: bool = True
    t_ref: float = 1.0


def transform_to_sampling_space(
    params: dict[str, float],
    config: ReparamConfig,
) -> dict[str, float]:
    """Transform physics params → sampling params.

    Parameters
    ----------
    params : dict[str, float]
        Physics parameters (D0, D_offset, gamma_dot_t0, etc.).
    config : ReparamConfig
        Reparameterization configuration.

    Returns
    -------
    dict[str, float]
        Transformed parameters for sampling.
    """
    result = dict(params)

    if config.enable_d_total and "D0" in params and "D_offset" in params:
        D0 = params["D0"]
        D_offset = params["D_offset"]
        D_total = D0 + D_offset
        D_offset_frac = D_offset / D_total if D_total != 0 else 0.0
        result["D_total"] = D_total
        result["D_offset_frac"] = D_offset_frac
        del result["D0"]
        del result["D_offset"]

    if config.enable_log_gamma and "gamma_dot_t0" in params:
        result["log_gamma_dot_t0"] = np.log(params["gamma_dot_t0"])
        del result["gamma_dot_t0"]

    return result


def transform_to_physics_space(
    samples: dict[str, np.ndarray],
    config: ReparamConfig,
) -> dict[str, np.ndarray]:
    """Transform sampling params → physics params (vectorized).

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Sampled parameters from MCMC.
    config : ReparamConfig
        Reparameterization configuration.

    Returns
    -------
    dict[str, np.ndarray]
        Physics parameters.
    """
    result = dict(samples)

    if config.enable_d_total and "D_total" in samples:
        D_total = samples["D_total"]
        D_offset_frac = samples["D_offset_frac"]
        result["D0"] = D_total * (1 - D_offset_frac)
        result["D_offset"] = D_total * D_offset_frac
        del result["D_total"]
        del result["D_offset_frac"]

    if config.enable_log_gamma and "log_gamma_dot_t0" in samples:
        result["gamma_dot_t0"] = np.exp(samples["log_gamma_dot_t0"])
        del result["log_gamma_dot_t0"]

    return result
