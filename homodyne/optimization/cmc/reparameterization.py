# homodyne/optimization/cmc/reparameterization.py
"""Internal reparameterization for CMC sampling.

Transforms correlated parameters to orthogonal sampling space,
then converts back to physics parameters for output.

Reparameterizations (v2.23.0+, reference-time):
- D0, D_offset → log_D_ref, D_offset_frac (decorrelates D0/alpha via observable)
- gamma_dot_t0 → log_gamma_ref (decorrelates gamma_dot_t0/beta via observable)

Theory:
  For f(t) = A × t^α, the value at a reference time f_ref = A × t_ref^α is
  well-constrained by data. Sampling (f_ref, α) instead of (A, α) decorrelates
  them because f_ref relates to observables while α controls the shape.

  t_ref = √(dt × t_max) is the geometric mean of the time range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def compute_t_ref(
    dt: float, t_max: float, *, fallback_value: float | None = None
) -> float:
    """Compute reference time as geometric mean of time range.

    Parameters
    ----------
    dt : float
        Time step (minimum time scale).
    t_max : float
        Maximum time in the dataset.
    fallback_value : float | None
        If provided and inputs are invalid, return this value with a warning
        instead of raising ValueError. If None (default), raises on invalid inputs.

    Returns
    -------
    float
        Reference time t_ref = sqrt(dt * t_max), or fallback_value if inputs invalid.

    Raises
    ------
    ValueError
        If dt or t_max are non-positive or non-finite and fallback_value is None.
    """
    if dt <= 0 or t_max <= 0 or not math.isfinite(dt) or not math.isfinite(t_max):
        if fallback_value is not None:
            logger.warning(
                f"Invalid inputs for t_ref computation: dt={dt}, t_max={t_max}. "
                f"Using fallback t_ref={fallback_value}."
            )
            return fallback_value
        raise ValueError(
            f"Invalid inputs for t_ref computation: dt={dt}, t_max={t_max}. "
            "Both must be positive and finite."
        )
    return math.sqrt(dt * t_max)


def transform_nlsq_to_reparam_space(
    nlsq_values: dict[str, float],
    nlsq_uncertainties: dict[str, float] | None,
    t_ref: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Transform NLSQ estimates to reference-time reparameterized space.

    Computes D_ref, log_D_ref, gamma_ref, log_gamma_ref from NLSQ physics
    parameters, with delta-method uncertainty propagation.

    Parameters
    ----------
    nlsq_values : dict[str, float]
        NLSQ parameter estimates (D0, alpha, D_offset, gamma_dot_t0, beta, ...).
    nlsq_uncertainties : dict[str, float] | None
        NLSQ standard errors for each parameter.
    t_ref : float
        Reference time for reparameterization.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        (reparam_values, reparam_uncertainties) in the reparameterized space.
        Keys: "log_D_ref", "D_offset_frac", "log_gamma_ref" (when available).
    """
    reparam_values: dict[str, float] = {}
    reparam_uncertainties: dict[str, float] = {}

    # --- Diffusion reparameterization ---
    D0 = nlsq_values.get("D0")
    alpha = nlsq_values.get("alpha")
    D_offset = nlsq_values.get("D_offset")

    if D0 is not None and alpha is not None:
        # D_ref = D0 * t_ref^alpha
        D_ref = D0 * (t_ref**alpha)
        D_ref = max(D_ref, 1e-10)  # Prevent log(0)
        log_D_ref = math.log(D_ref)
        reparam_values["log_D_ref"] = log_D_ref

        # D_offset_frac = D_offset / (D_ref + D_offset)
        if D_offset is not None:
            denom = D_ref + D_offset
            if denom > 0:
                reparam_values["D_offset_frac"] = D_offset / denom
            else:
                reparam_values["D_offset_frac"] = 0.05  # Safe default

        # Delta-method uncertainty propagation for log_D_ref
        # log_D_ref = log(D0) + alpha * log(t_ref)
        # Var(log_D_ref) ≈ (1/D0)^2 * Var(D0) + log(t_ref)^2 * Var(alpha)
        #                  + 2 * (1/D0) * log(t_ref) * Cov(D0, alpha)
        # We ignore covariance (conservative, slightly overestimates uncertainty)
        if nlsq_uncertainties:
            D0_std = nlsq_uncertainties.get("D0", 0.0)
            alpha_std = nlsq_uncertainties.get("alpha", 0.0)
            D_offset_std = nlsq_uncertainties.get("D_offset", 0.0)

            # Var(log_D_ref) via delta method (ignoring covariance)
            log_t_ref = math.log(max(t_ref, 1e-10))
            var_log_D_ref = 0.0
            if D0 > 0 and D0_std > 0:
                var_log_D_ref += (D0_std / D0) ** 2
            if alpha_std > 0:
                var_log_D_ref += (log_t_ref * alpha_std) ** 2
            reparam_uncertainties["log_D_ref"] = math.sqrt(max(var_log_D_ref, 1e-20))

            # Uncertainty for D_offset_frac (simplified)
            if D_offset is not None and denom > 0 and D_offset_std > 0:
                # d(frac)/d(D_offset) = D_ref / (D_ref + D_offset)^2
                dfrac_doffset = D_ref / (denom**2)
                reparam_uncertainties["D_offset_frac"] = abs(
                    dfrac_doffset * D_offset_std
                )

    # --- Shear reparameterization ---
    gamma_dot_t0 = nlsq_values.get("gamma_dot_t0")
    beta = nlsq_values.get("beta")

    if gamma_dot_t0 is not None and beta is not None:
        # gamma_ref = gamma_dot_t0 * t_ref^beta
        gamma_ref = gamma_dot_t0 * (t_ref**beta)
        gamma_ref = max(gamma_ref, 1e-20)  # Prevent log(0)
        log_gamma_ref = math.log(gamma_ref)
        reparam_values["log_gamma_ref"] = log_gamma_ref

        # Delta-method uncertainty for log_gamma_ref
        if nlsq_uncertainties:
            gamma_std = nlsq_uncertainties.get("gamma_dot_t0", 0.0)
            beta_std = nlsq_uncertainties.get("beta", 0.0)

            log_t_ref = math.log(max(t_ref, 1e-10))
            var_log_gamma_ref = 0.0
            if gamma_dot_t0 > 0 and gamma_std > 0:
                var_log_gamma_ref += (gamma_std / gamma_dot_t0) ** 2
            if beta_std > 0:
                var_log_gamma_ref += (log_t_ref * beta_std) ** 2
            reparam_uncertainties["log_gamma_ref"] = math.sqrt(
                max(var_log_gamma_ref, 1e-20)
            )

    return reparam_values, reparam_uncertainties


@dataclass
class ReparamConfig:
    """Configuration for internal reparameterization.

    Attributes
    ----------
    enable_d_ref : bool
        Enable D_ref reparameterization (D0, alpha → log_D_ref, alpha).
    enable_gamma_ref : bool
        Sample log(gamma_ref) where gamma_ref = gamma_dot_t0 * t_ref^beta.
    t_ref : float
        Reference time for reparameterization (geometric mean of time range).
    """

    enable_d_ref: bool = True
    enable_gamma_ref: bool = True
    t_ref: float = 1.0

    # Backward compatibility properties
    @property
    def enable_d_total(self) -> bool:
        """Backward-compatible alias for enable_d_ref."""
        return self.enable_d_ref

    @property
    def enable_log_gamma(self) -> bool:
        """Backward-compatible alias for enable_gamma_ref."""
        return self.enable_gamma_ref


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
    t_ref = config.t_ref

    if config.enable_d_ref and "D0" in params and "D_offset" in params:
        D0 = params["D0"]
        alpha = params.get("alpha", 0.0)
        D_offset = params["D_offset"]

        # D_ref = D0 * t_ref^alpha
        D_ref = D0 * (t_ref**alpha)
        D_ref = max(D_ref, 1e-10)
        log_D_ref = np.log(D_ref)

        # D_offset_frac = D_offset / (D_ref + D_offset)
        denom = D_ref + D_offset
        D_offset_frac = D_offset / denom if denom > 0 else 0.0

        result["log_D_ref"] = float(log_D_ref)
        result["D_offset_frac"] = float(D_offset_frac)
        del result["D0"]
        del result["D_offset"]

    if config.enable_gamma_ref and "gamma_dot_t0" in params:
        gamma_dot_t0 = params["gamma_dot_t0"]
        beta = params.get("beta", 0.0)

        # gamma_ref = gamma_dot_t0 * t_ref^beta
        gamma_ref = gamma_dot_t0 * (t_ref**beta)
        gamma_ref = max(gamma_ref, 1e-20)
        result["log_gamma_ref"] = float(np.log(gamma_ref))
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
    t_ref = config.t_ref

    if config.enable_d_ref and "log_D_ref" in samples:
        log_D_ref = samples["log_D_ref"]
        D_offset_frac = samples["D_offset_frac"]
        alpha = samples.get("alpha", np.zeros_like(log_D_ref))

        D_ref = np.exp(log_D_ref)
        # D0 = D_ref * t_ref^(-alpha)
        result["D0"] = D_ref * (t_ref ** (-alpha))
        # D_offset = D_ref * frac / (1 - frac)
        # P1-2: Clip D_offset_frac to prevent D_offset → ∞ when frac → 1.0.
        # The sampling prior caps at 0.5 via TruncatedNormal, but posterior
        # samples may exceed this during post-processing back-transform.
        D_offset_frac_safe = np.clip(D_offset_frac, 0.0, 1.0 - 1e-6)
        result["D_offset"] = (
            D_ref * D_offset_frac_safe / (1 - D_offset_frac_safe)
        )
        del result["log_D_ref"]
        del result["D_offset_frac"]
    elif config.enable_d_ref and "D_total" in samples:
        # Legacy D_total path (backward compatibility)
        D_total = samples["D_total"]
        D_offset_frac = samples["D_offset_frac"]
        result["D0"] = D_total * (1 - D_offset_frac)
        result["D_offset"] = D_total * D_offset_frac
        del result["D_total"]
        del result["D_offset_frac"]

    if config.enable_gamma_ref and "log_gamma_ref" in samples:
        log_gamma_ref = samples["log_gamma_ref"]
        beta = samples.get("beta", np.zeros_like(log_gamma_ref))

        # gamma_dot_t0 = exp(log_gamma_ref) * t_ref^(-beta)
        result["gamma_dot_t0"] = np.exp(log_gamma_ref) * (t_ref ** (-beta))
        del result["log_gamma_ref"]
    elif config.enable_gamma_ref and "log_gamma_dot_t0" in samples:
        # Legacy log_gamma_dot_t0 path (backward compatibility)
        result["gamma_dot_t0"] = np.exp(samples["log_gamma_dot_t0"])
        del result["log_gamma_dot_t0"]

    return result
