"""NUTS sampler wrapper for CMC analysis.

This module provides utilities for running NumPyro NUTS sampling
with proper initialization and progress tracking.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median, init_to_value

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.priors import (
    build_init_values_dict,
    get_param_names_in_order,
)
from homodyne.optimization.cmc.scaling import (
    compute_scaling_factors,
    transform_initial_values_to_z,
)
from homodyne.utils.logging import get_logger, with_context

# Divergence rate thresholds for NUTS convergence diagnostics.
# P2-2: Centralized constants to ensure consistency across sampler paths.
DIVERGENCE_RATE_TARGET = 0.05  # Below this: acceptable
DIVERGENCE_RATE_HIGH = 0.10  # Above this: posterior may be biased
DIVERGENCE_RATE_CRITICAL = 0.30  # Above this: posterior likely unreliable


def _subset_model_kwargs_for_preflight(
    model_kwargs: dict[str, Any],
    *,
    max_points: int = 512,
) -> dict[str, Any]:
    """Build a reduced model_kwargs dict for fast preflight diagnostics."""
    data = model_kwargs.get("data")
    if data is None:
        return dict(model_kwargs)

    if not hasattr(data, "shape"):
        return dict(model_kwargs)

    n = int(data.shape[0])
    if n <= max_points:
        return dict(model_kwargs)

    # Pick evenly spaced points to cover the full time range.
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    reduced = dict(model_kwargs)
    for key in ("data", "t1", "t2", "phi_indices"):
        arr = model_kwargs.get(key)
        if arr is not None:
            reduced[key] = arr[idx]
    # Drop pre-computed shard_grid: its idx1/idx2 are aligned with the full
    # shard (length N), not the preflight subset (length max_points).  The
    # model functions fall back to the legacy compute_g1_total path when
    # shard_grid is absent.
    reduced.pop("shard_grid", None)
    return reduced


def _preflight_log_density(
    *,
    model: Callable,
    model_kwargs: dict[str, Any],
    params: dict[str, Any],
    run_logger,
    max_points: int = 512,
) -> None:
    """Compute initial log density and basic finiteness diagnostics.

    This is intended to catch the common failure modes behind near-zero
    acceptance (NaN/-inf log prob, non-finite deterministics) before spending
    hours running many CMC shards.
    """
    try:
        from numpyro import handlers
        from numpyro.infer.util import log_density

        subset_kwargs = _subset_model_kwargs_for_preflight(
            model_kwargs, max_points=max_points
        )

        seeded = handlers.seed(model, jax.random.PRNGKey(0))
        log_joint, trace = log_density(seeded, (), subset_kwargs, params)
        log_joint_val = float(np.asarray(log_joint))

        n_nonfinite_log_prob = 0
        n_total_log_prob = 0
        for site in trace.values():
            if site.get("type") != "sample":
                continue

            fn = site.get("fn")
            value = site.get("value")
            if fn is None or value is None:
                continue

            try:
                log_prob = fn.log_prob(value)
            except Exception:  # nosec B112
                continue

            log_prob_np = np.asarray(log_prob)
            n_total_log_prob += log_prob_np.size
            n_nonfinite_log_prob += int(np.sum(~np.isfinite(log_prob_np)))

        n_issues = None
        if "n_numerical_issues" in trace:
            try:
                n_issues = int(np.asarray(trace["n_numerical_issues"]["value"]))
            except Exception:
                n_issues = None

        run_logger.info(
            "Preflight log_density: "
            f"log_joint={log_joint_val:.4g}, "
            f"nonfinite_log_prob={n_nonfinite_log_prob}/{n_total_log_prob}, "
            f"n_numerical_issues={n_issues}"
        )

        if not np.isfinite(log_joint_val) or n_nonfinite_log_prob > 0:
            raise RuntimeError(
                "Preflight detected non-finite log density/log_prob at initialization. "
                "This typically leads to 0% NUTS acceptance; check bounds, initial values, "
                "and numerical stability of the physics model."
            )

    except RuntimeError:
        raise
    except Exception as e:
        # If the preflight itself fails, keep going but make it loud.
        run_logger.warning(f"Preflight diagnostics failed: {e}")


def _compute_mcmc_safe_d0(
    initial_values: dict[str, float] | None,
    q: float,
    dt: float,
    time_grid: np.ndarray | None,
    logger_inst,
    *,
    target_g1: float = 0.5,
    g1_threshold: float = 0.1,
) -> dict[str, float] | None:
    """Check if initial D0 causes g1→0 and compute MCMC-safe adjustment.

    When D0 is very large (or alpha very negative), the diffusion integral
    can become enormous, causing g1 = exp(-integral) → 0. This creates
    a flat likelihood surface with vanishing gradients, causing NUTS to
    reject all proposals (0% acceptance rate).

    This function detects this condition and computes a scaled D0 that
    produces g1 ≈ target_g1 at a typical time lag, ensuring gradients
    are alive for MCMC exploration.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial parameter values containing D0, alpha, D_offset.
    q : float
        Wavevector magnitude.
    dt : float
        Time step.
    time_grid : np.ndarray | None
        Time grid for integration.
    logger_inst
        Logger instance for warnings.
    target_g1 : float
        Target g1 value for adjusted D0 (default 0.5).
    g1_threshold : float
        Threshold below which D0 is adjusted (default 0.1).

    Returns
    -------
    dict[str, float] | None
        Adjusted initial values if D0 was scaled, None otherwise.
    """
    if initial_values is None:
        return None

    # Get diffusion parameters
    d0 = initial_values.get("D0")
    alpha = initial_values.get("alpha")
    d_offset = initial_values.get("D_offset")

    if d0 is None or alpha is None or d_offset is None:
        return None

    if time_grid is None or len(time_grid) < 2:
        return None

    # Safety checks
    if not np.isfinite(d0) or not np.isfinite(alpha) or not np.isfinite(d_offset):
        return None

    try:
        # Compute D(t) on the time grid
        # D(t) = D0 * t^alpha + D_offset
        epsilon = 1e-10
        time_safe = np.asarray(time_grid) + epsilon
        D_grid = d0 * (time_safe**alpha) + d_offset
        D_grid = np.maximum(D_grid, 1e-10)

        # Compute trapezoidal cumsum (without dt scaling)
        if len(D_grid) > 1:
            trap_avg = 0.5 * (D_grid[:-1] + D_grid[1:])
            cumsum = np.concatenate([[0.0], np.cumsum(trap_avg)])
        else:
            cumsum = np.cumsum(D_grid)

        # Estimate integral at typical time lag (1/4 to 3/4 of range)
        n = len(cumsum)
        idx_low = n // 4
        idx_high = 3 * n // 4
        integral_estimate = abs(cumsum[idx_high] - cumsum[idx_low])

        # Compute g1 = exp(-q^2 * dt * integral)
        # Physics: g1 = exp(-q² ∫D(t)dt). The homodyne 0.5 factor appears on g1²,
        # not in the log of g1 directly. Using the full q² gives a correct estimate
        # of the ISF decay rate for the safety guard.
        prefactor = q**2 * dt
        log_g1 = -prefactor * integral_estimate
        log_g1_clipped = max(log_g1, -700.0)  # Prevent underflow
        g1_estimate = np.exp(log_g1_clipped)

        logger_inst.debug(
            f"[MCMC-SAFE] g1 estimate: D0={d0:.4g}, alpha={alpha:.4g}, "
            f"integral={integral_estimate:.4g}, log_g1={log_g1:.4g}, g1={g1_estimate:.4g}"
        )

        # If g1 is too small, compute scaled D0
        if g1_estimate < g1_threshold:
            # For g1 = target_g1:
            # log(target_g1) = -prefactor * target_integral
            # target_integral = -log(target_g1) / prefactor
            target_log_g1 = np.log(target_g1)
            target_integral = -target_log_g1 / prefactor

            # Scale factor: how much smaller should the integral be?
            if integral_estimate > 0:
                scale_factor = target_integral / integral_estimate
            else:
                scale_factor = 0.01  # Fallback

            # Apply scaling to D0 (approximately linear for moderate adjustments)
            # Also adjust D_offset proportionally for consistency
            new_d0 = d0 * scale_factor
            new_d_offset = d_offset * scale_factor

            # Ensure new values are within reasonable range
            new_d0 = max(new_d0, 1.0)  # Minimum D0
            new_d_offset = max(new_d_offset, -1e6)  # Allow negative but bound

            logger_inst.warning(
                f"MCMC-SAFE ADJUSTMENT: Initial D0={d0:.4g} causes g1≈{g1_estimate:.2e} (vanishing gradients). "
                f"Scaling D0 to {new_d0:.4g} (×{scale_factor:.4f}) for MCMC exploration stability. "
                f"The sampler can still converge to optimal values if supported by likelihood."
            )

            # Return adjusted values
            adjusted = dict(initial_values)
            adjusted["D0"] = new_d0
            adjusted["D_offset"] = new_d_offset
            return adjusted

        return None  # No adjustment needed

    except Exception as e:
        logger_inst.debug(f"[MCMC-SAFE] Check failed: {e}")
        return None


if TYPE_CHECKING:
    from homodyne.config.parameter_space import ParameterSpace

logger = get_logger(__name__)


def _summarize_inverse_mass_matrix(inv_mass: Any) -> str:
    """Return a compact summary of the adapted inverse mass matrix."""

    def _one(mat: Any) -> str:
        if isinstance(mat, dict):
            keys = list(mat.keys())
            if not keys:
                return "dict(empty)"
            first = mat[keys[0]]
            return f"dict(keys={len(keys)}) first[{keys[0]}]: {_one(first)}"

        try:
            arr = np.asarray(mat)
        except Exception:
            return f"type={type(mat).__name__}"

        if arr.ndim == 0:
            # Could be an object scalar (e.g., dict) depending on upstream types.
            try:
                return f"scalar={float(arr):.3g}"
            except Exception:
                return f"scalar(type={type(arr.item()).__name__})"

        if arr.ndim == 1:
            diag = arr
            diag = diag[np.isfinite(diag)]
            if diag.size == 0:
                return f"diag(dim={arr.size}) all-nonfinite"
            dmin = float(np.min(diag))
            dmax = float(np.max(diag))
            cond = float(dmax / dmin) if dmin > 0 else float("inf")
            return f"diag(dim={arr.size}) min={dmin:.3g} max={dmax:.3g} cond≈{cond:.3g}"

        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            diag = np.diag(arr)
            diag = diag[np.isfinite(diag)]
            if diag.size == 0:
                return f"dense(dim={arr.shape[0]}) diag all-nonfinite"
            dmin = float(np.min(diag))
            dmax = float(np.max(diag))
            try:
                cond = float(np.linalg.cond(arr))
            except Exception:
                cond = float("nan")
            return (
                f"dense(dim={arr.shape[0]}) diag[min={dmin:.3g}, max={dmax:.3g}] "
                f"cond={cond:.3g}"
            )

        # Per-chain dense matrices: (n_chains, dim, dim)
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2]:
            n_chains = arr.shape[0]
            dim = arr.shape[1]
            # summarize first two chains
            parts = []
            for i in range(min(n_chains, 2)):
                parts.append(_one(arr[i]))
            more = "" if n_chains <= 2 else f" (+{n_chains - 2} more)"
            return f"per-chain dense(dim={dim})[{', '.join(parts)}]{more}"

        return f"array(shape={arr.shape}, ndim={arr.ndim})"

    if isinstance(inv_mass, (list, tuple)):
        parts = [_one(m) for m in inv_mass[:2]]
        more = "" if len(inv_mass) <= 2 else f" (+{len(inv_mass) - 2} more)"
        return f"per-chain[{', '.join(parts)}]{more}"

    return _one(inv_mass)


def _extract_adapt_states(last_state: Any) -> list[Any]:
    """Extract per-chain adapt_state objects from a NumPyro MCMC last_state."""
    if last_state is None:
        return []

    if hasattr(last_state, "adapt_state"):
        return [last_state.adapt_state]

    if isinstance(last_state, (list, tuple)):
        out: list[Any] = []
        for item in last_state:
            if hasattr(item, "adapt_state"):
                out.append(item.adapt_state)
        return out

    # NumPyro may omit adapt_state (e.g., API differences or failed adaptation).
    return []


def _log_array_stats(
    run_logger,
    *,
    name: str,
    arr: Any,
) -> None:
    try:
        a = np.asarray(arr)
    except Exception:
        return

    if a.size == 0:
        return

    finite = np.isfinite(a)
    if not np.any(finite):
        run_logger.info(f"{name} stats: all non-finite, shape={a.shape}")
        return

    run_logger.info(
        f"{name} stats: "
        f"min={float(np.min(a[finite])):.3g}, "
        f"median={float(np.median(a[finite])):.3g}, "
        f"max={float(np.max(a[finite])):.3g}, "
        f"mean={float(np.mean(a[finite])):.3g}, "
        f"std={float(np.std(a[finite])):.3g}, "
        f"finite={float(np.mean(finite)):.1%}, shape={a.shape}"
    )


def _extract_step_sizes(adapt_states: list[Any]) -> list[float]:
    """Extract step_size values from NumPyro adapt_state objects."""
    step_sizes: list[float] = []
    for adapt_state in adapt_states:
        if adapt_state is None:
            continue

        if hasattr(adapt_state, "step_size"):
            try:
                step_sizes.append(float(adapt_state.step_size))
                continue
            except Exception:  # noqa: S110 - Fallback for adapt_state variants
                pass

        if isinstance(adapt_state, dict) and "step_size" in adapt_state:
            try:
                step_sizes.append(float(adapt_state["step_size"]))
            except Exception:  # noqa: S110 - Fallback for adapt_state variants
                pass

    return step_sizes


@dataclass
class SamplingStats:
    """Statistics from MCMC sampling.

    Attributes
    ----------
    warmup_time : float
        Time spent in warmup phase (seconds).
    sampling_time : float
        Time spent in sampling phase (seconds).
    total_time : float
        Total sampling time (seconds).
    num_divergent : int
        Number of divergent transitions.
    accept_prob : float
        Mean acceptance probability.
    step_size : float
        Final step size.
    step_size_min : float
        Minimum adapted step size across chains (if available).
    step_size_max : float
        Maximum adapted step size across chains (if available).
    inverse_mass_matrix_summary : str | None
        Compact summary of the adapted inverse mass matrix (if available).
    tree_depth : float
        Mean tree depth.
    """

    warmup_time: float = 0.0
    sampling_time: float = 0.0
    total_time: float = 0.0
    num_divergent: int = 0
    accept_prob: float = 0.0
    step_size: float = 0.0
    step_size_min: float | None = None
    step_size_max: float | None = None
    inverse_mass_matrix_summary: str | None = None
    tree_depth: float = 0.0
    plan: SamplingPlan | None = None


@dataclass(frozen=True)
class SamplingPlan:
    """Adapted MCMC sampling counts for a single shard.

    Captures the actual warmup/sample counts after adaptive scaling,
    which may differ from CMCConfig defaults for small shards.

    Use SamplingPlan.from_config() instead of accessing
    config.num_warmup / config.num_samples in hot paths.
    """

    n_warmup: int
    n_samples: int
    n_chains: int
    shard_size: int
    n_params: int
    was_adapted: bool

    @classmethod
    def from_config(
        cls, config: CMCConfig, shard_size: int, n_params: int
    ) -> SamplingPlan:
        n_warmup, n_samples = config.get_adaptive_sample_counts(
            shard_size=shard_size, n_params=n_params
        )
        return cls(
            n_warmup=n_warmup,
            n_samples=n_samples,
            n_chains=config.num_chains,
            shard_size=shard_size,
            n_params=n_params,
            was_adapted=(
                n_warmup != config.num_warmup or n_samples != config.num_samples
            ),
        )

    @property
    def total_samples(self) -> int:
        return self.n_samples * self.n_chains


@dataclass
class MCMCSamples:
    """Container for MCMC samples.

    Attributes
    ----------
    samples : dict[str, np.ndarray]
        Parameter samples, shape (n_chains, n_samples) per parameter.
    param_names : list[str]
        Parameter names in sampling order.
    n_chains : int
        Number of chains.
    n_samples : int
        Number of samples per chain.
    extra_fields : dict[str, Any]
        Additional MCMC info (divergences, energy, etc.).
    num_shards : int
        Number of shards combined (1 for single shard, >1 for CMC).
        Used for correct divergence rate calculation in CMC.
    """

    samples: dict[str, np.ndarray]
    param_names: list[str]
    n_chains: int
    n_samples: int
    extra_fields: dict[str, Any] = field(default_factory=dict)
    num_shards: int = 1
    shard_adapted_n_warmup: int | None = None
    bimodal_consensus: Any = (
        None  # BimodalConsensusResult when mode-aware consensus used
    )


def create_init_strategy(
    initial_values: dict[str, float] | None,
    param_names: list[str],
    use_init_to_value: bool = True,
    z_space_values: dict[str, float] | None = None,
) -> Callable:
    """Create initialization strategy for NUTS.

    Parameters
    ----------
    initial_values : dict[str, float] | None
        Initial values from config (original space).
    param_names : list[str]
        Expected parameter names in order.
    use_init_to_value : bool
        If True, use init_to_value when values provided.
    z_space_values : dict[str, float] | None
        Initial values in z-space (for scaled model). If provided,
        these are used directly as {name}_z values.

    Returns
    -------
    Callable
        NumPyro initialization function.
    """
    # For scaled model, use z-space values
    if z_space_values is not None and use_init_to_value:
        if z_space_values:
            logger.debug(
                f"Using init_to_value (z-space) for {len(z_space_values)} params: "
                f"{list(z_space_values.keys())[:5]}..."
            )
            return init_to_value(values=z_space_values)

    # For unscaled model, use original values
    if initial_values is not None and use_init_to_value:
        # Filter to only parameters we're sampling (exclude deterministics)
        init_dict = {}
        for name in param_names:
            if name in initial_values:
                init_dict[name] = initial_values[name]

        if init_dict:
            logger.debug(
                f"Using init_to_value for {len(init_dict)} params: {list(init_dict.keys())}"
            )
            return init_to_value(values=init_dict)

    # Fallback to median initialization
    logger.debug("Using init_to_median (no initial values)")
    return init_to_median()


def run_nuts_sampling(
    model: Callable,
    model_kwargs: dict[str, Any],
    config: CMCConfig,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
    n_phi: int,
    analysis_mode: str,
    rng_key: jax.random.PRNGKey | None = None,
    progress_bar: bool = True,
    per_angle_mode: str = "individual",
) -> tuple[MCMCSamples, SamplingStats]:
    """Run NUTS sampling with configuration.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    model_kwargs : dict[str, Any]
        Keyword arguments to pass to model.
    config : CMCConfig
        CMC configuration.
    initial_values : dict[str, float] | None
        Initial parameter values from config.
    parameter_space : ParameterSpace
        Parameter space for building init values.
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    rng_key : jax.random.PRNGKey | None
        Random key. If None, creates from seed.
    progress_bar : bool
        Whether to show progress bar.
    per_angle_mode : str
        Per-angle scaling mode: "individual", "auto", "constant", or
        "constant_averaged". Controls which parameters are sampled vs fixed.

    Returns
    -------
    tuple[MCMCSamples, SamplingStats]
        Samples and timing statistics.
    """
    run_logger = with_context(logger, run=getattr(config, "run_id", None))

    # Get parameter names in correct order
    param_names = get_param_names_in_order(n_phi, analysis_mode, per_angle_mode)
    # Add sigma (noise parameter)
    param_names_with_sigma = param_names + ["sigma"]

    # Build full init values dict if needed
    # Extract data arrays from model_kwargs for data-driven estimation
    c2_data = model_kwargs.get("data")
    t1_data = model_kwargs.get("t1")
    t2_data = model_kwargs.get("t2")
    phi_indices = model_kwargs.get("phi_indices")

    # Convert JAX arrays to numpy for estimation (if needed)
    if c2_data is not None and hasattr(c2_data, "__array__"):
        c2_data = np.asarray(c2_data)
    if t1_data is not None and hasattr(t1_data, "__array__"):
        t1_data = np.asarray(t1_data)
    if t2_data is not None and hasattr(t2_data, "__array__"):
        t2_data = np.asarray(t2_data)
    if phi_indices is not None and hasattr(phi_indices, "__array__"):
        phi_indices = np.asarray(phi_indices)

    # =========================================================================
    # MCMC-SAFE D0 CHECK: Detect and fix vanishing gradient regime
    # =========================================================================
    # When D0 is very large (or alpha very negative), g1 → 0 everywhere,
    # causing vanishing gradients and 0% NUTS acceptance rate.
    # This check detects that condition and scales D0 to ensure gradients are alive.
    q = model_kwargs.get("q", 0.01)
    dt = model_kwargs.get("dt", 0.1)
    time_grid = model_kwargs.get("time_grid")
    if time_grid is not None and hasattr(time_grid, "__array__"):
        time_grid_np = np.asarray(time_grid)
    else:
        time_grid_np = time_grid

    adjusted_init = _compute_mcmc_safe_d0(
        initial_values=initial_values,
        q=q,
        dt=dt,
        time_grid=time_grid_np,
        logger_inst=run_logger,
    )

    # Use adjusted values if D0 was scaled
    effective_init_values = (
        adjusted_init if adjusted_init is not None else initial_values
    )

    full_init = build_init_values_dict(
        n_phi=n_phi,
        analysis_mode=analysis_mode,
        initial_values=effective_init_values,
        parameter_space=parameter_space,
        c2_data=c2_data,
        t1=t1_data,
        t2=t2_data,
        phi_indices=phi_indices,
        per_angle_mode=per_angle_mode,
    )

    # =========================================================================
    # GRADIENT BALANCING: Transform initial values to z-space for scaled model
    # =========================================================================
    # The scaled model (xpcs_model_scaled) samples in normalized z-space where
    # z ~ Normal(0, 1). We need to transform our initial values to this space
    # for proper initialization with init_to_value.
    scalings = compute_scaling_factors(parameter_space, n_phi, analysis_mode)
    z_space_init = transform_initial_values_to_z(full_init, scalings)

    # Log scaling transformation info
    run_logger.info(
        f"Gradient balancing: {len(scalings)} params transformed to unit scale. "
        f"Sample scale range: {min(s.scale for s in scalings.values()):.2e} to "
        f"{max(s.scale for s in scalings.values()):.2e}"
    )

    # =========================================================================
    # REPARAMETERIZED INIT VALUES: Add log_D_ref, D_offset_frac, log_gamma_ref
    # =========================================================================
    # When using the reparameterized model, the sampled parameters are different
    # from the z-space params. Add init values for the reparam parameters.
    reparam_config = model_kwargs.get("reparam_config")
    if reparam_config is not None:
        t_ref_init = model_kwargs.get("t_ref", getattr(reparam_config, "t_ref", 1.0))

        # Compute reparameterized init values from physics init values
        D0_init = full_init.get("D0", 1e4)
        alpha_init = full_init.get("alpha", -0.5)
        D_offset_init = full_init.get("D_offset", 1e3)

        if getattr(reparam_config, "enable_d_ref", False):
            D_ref_init = D0_init * (t_ref_init**alpha_init)
            D_ref_init = max(D_ref_init, 1e-10)
            z_space_init["log_D_ref"] = float(np.log(D_ref_init))
            denom = D_ref_init + D_offset_init
            z_space_init["D_offset_frac"] = (
                float(np.clip(D_offset_init / denom, 0.0, 0.5)) if denom > 0 else 0.05
            )

        if (
            getattr(reparam_config, "enable_gamma_ref", False)
            and analysis_mode == "laminar_flow"
        ):
            gamma_dot_t0_init = full_init.get("gamma_dot_t0", 1e-3)
            beta_init = full_init.get("beta", -0.3)
            gamma_ref_init = gamma_dot_t0_init * (t_ref_init**beta_init)
            gamma_ref_init = max(gamma_ref_init, 1e-20)
            z_space_init["log_gamma_ref"] = float(np.log(gamma_ref_init))

        run_logger.info(
            f"Reparameterized init: t_ref={t_ref_init:.4g}, "
            + ", ".join(
                f"{k}={z_space_init[k]:.4g}"
                for k in ["log_D_ref", "D_offset_frac", "log_gamma_ref"]
                if k in z_space_init
            )
        )

    # Create init strategy for NUTS kernel.
    # P1-R5-01: NLSQ-informed models ("auto"/"constant_averaged" per_angle_mode)
    # sample in original parameter space — their NumPyro sites are named "D0",
    # "alpha", etc., NOT "D0_z", "alpha_z". Passing z_space_values with z-space
    # site names causes init_to_value to silently ignore all entries (no site name
    # match), falling back to init_to_median and discarding the NLSQ warm-start.
    # For these modes, use the original-space full_init directly.
    nlsq_prior_config = model_kwargs.get("nlsq_prior_config")
    if nlsq_prior_config is not None and per_angle_mode in (
        "auto",
        "constant_averaged",
    ):
        # Original-space initialization for NLSQ-informed models
        init_strategy = create_init_strategy(full_init, param_names_with_sigma)
    else:
        # Z-space initialization for scaled models
        init_strategy = create_init_strategy(
            full_init, param_names_with_sigma, z_space_values=z_space_init
        )

    # =========================================================================
    # PREFLIGHT: Validate initial log density and finiteness
    # =========================================================================
    # This catches common causes of 0% acceptance (NaNs/-inf log prob) before
    # spending wall-clock hours running many CMC shards.
    sigma_init = float(max(model_kwargs.get("noise_scale", 0.1), 1e-6))

    # P1-5 / P1-R5-01: NLSQ-informed models (averaged, constant_averaged) sample
    # parameters with original-space names ("D0", "alpha", ...) NOT z-space names
    # ("D0_z"). Both the preflight and init_strategy must use matching names.
    # nlsq_prior_config was read above when building init_strategy.
    if nlsq_prior_config is not None and per_angle_mode in (
        "auto",
        "constant_averaged",
    ):
        # Use original-space names for NLSQ-informed models
        preflight_params = dict(full_init)
    else:
        preflight_params = dict(z_space_init)
    preflight_params["sigma"] = sigma_init
    _preflight_log_density(
        model=model,
        model_kwargs=model_kwargs,
        params=preflight_params,
        run_logger=run_logger,
    )

    # Create NUTS kernel
    # GRADIENT BALANCING FIX (Dec 2025): Use dense_mass=True to learn
    # cross-correlations between parameters with vastly different scales.
    # Without this, the 10^6:1 gradient imbalance between D0 (~10^4) and
    # gamma_dot_t0 (~10^-3) causes 0% acceptance rate because no single
    # step size ε works for all dimensions. Dense mass matrix allows NUTS
    # to adapt per-dimension and learn covariance structure during warmup.
    #
    # CONVERGENCE FIX (Jan 2026): Elevate target_accept_prob for laminar_flow.
    # The 28.4% divergence rate in 3-angle laminar_flow was caused by step size
    # adaptation settling on values too large for the complex posterior geometry.
    # Higher target_accept_prob forces smaller steps, reducing divergences.
    effective_target_accept = config.target_accept_prob
    if analysis_mode == "laminar_flow" and config.target_accept_prob < 0.9:
        effective_target_accept = 0.9
        run_logger.info(
            f"Elevating target_accept_prob from {config.target_accept_prob} to {effective_target_accept} "
            f"for laminar_flow mode (reduces divergences in complex posterior)"
        )
    kernel = NUTS(
        model,
        init_strategy=init_strategy,
        target_accept_prob=effective_target_accept,
        dense_mass=True,
        max_tree_depth=config.max_tree_depth,
    )

    # Get shard size for adaptive sampling
    data = model_kwargs.get("data")
    shard_size = len(data) if data is not None else 10000

    # Build SamplingPlan: captures adapted warmup/samples for this shard.
    # Profiling showed 1310s for 50 points with 500/1500 defaults.
    # Adaptive scaling reduces overhead for small datasets.
    plan = SamplingPlan.from_config(
        config, shard_size=shard_size, n_params=len(param_names_with_sigma)
    )
    num_warmup, num_samples = plan.n_warmup, plan.n_samples

    # Create MCMC runner with adaptive sample counts
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=config.num_chains,
        progress_bar=progress_bar,
    )

    # Create RNG key from config seed (defaults to 42 for reproducibility)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(config.seed)

    # Run sampling with timing
    adaptive_note = ""
    if config.adaptive_sampling and (
        num_warmup != config.num_warmup or num_samples != config.num_samples
    ):
        adaptive_note = f" (adaptive: {shard_size:,} pts)"
    run_logger.info(
        f"Starting NUTS sampling: {config.num_chains} chains, "
        f"{num_warmup} warmup, {num_samples} samples{adaptive_note}"
    )

    start_time = time.perf_counter()
    run_logger.info("NUTS phase: JIT compile + sampling started (may take minutes)...")

    # JAX profiler setup (Feb 2026): Capture XLA-level performance data
    # py-spy can only profile Python code; XLA runs native code invisible to py-spy.
    # When enabled, trace XLA operations to identify HLO graph bottlenecks.
    profile_context = None
    if config.enable_jax_profiling:
        import os

        os.makedirs(config.jax_profile_dir, exist_ok=True)
        run_logger.info(f"JAX profiling enabled, output: {config.jax_profile_dir}")
        profile_context = jax.profiler.trace(config.jax_profile_dir)
        profile_context.__enter__()

    try:
        # Request only essential extra fields to minimize extraction overhead.
        # Previously we requested 7 fields which caused 25-45 minute extraction times
        # due to JAX lazy evaluation materializing large intermediate arrays.
        # NOTE: "adapt_state.inverse_mass_matrix" is intentionally omitted here.
        # Requesting it stores the full mass matrix at every warmup+sample step,
        # producing a (n_chains, n_warmup+n_samples, n_params, n_params) tensor that
        # wastes 5-180 MB per shard (scales with n_params²). The final adapted mass
        # matrix is already extracted from mcmc.last_state via _extract_adapt_states.
        mcmc.run(
            rng_key,
            extra_fields=(
                "accept_prob",
                "diverging",
                "num_steps",
                "potential_energy",
                "adapt_state.step_size",
            ),
            **model_kwargs,
        )
    except Exception as e:
        run_logger.error(f"MCMC sampling failed: {e}")
        raise RuntimeError(f"MCMC sampling failed: {e}") from e
    finally:
        if profile_context is not None:
            profile_context.__exit__(None, None, None)
            run_logger.info(f"JAX profile saved to {config.jax_profile_dir}")

    # Force JAX to complete all pending computations before timing extraction.
    # JAX uses lazy evaluation, so without this the actual computation happens
    # during device_get(), causing misleading timing and 25-45 min "extraction" times.
    last_state = getattr(mcmc, "last_state", None)
    if last_state is not None:
        jax.block_until_ready(last_state)

    total_time = time.perf_counter() - start_time
    run_logger.info(f"NUTS finished in {total_time:.1f}s")

    # Extract samples - should be fast now that computation is complete
    t_extract = time.perf_counter()
    run_logger.info("Extracting samples + extra_fields...")
    samples = mcmc.get_samples(group_by_chain=True)

    # Use block_until_ready before device_get to ensure computation is complete
    jax.block_until_ready(samples)
    samples = jax.device_get(samples)

    # Convert to numpy and proper format
    samples_np: dict[str, np.ndarray] = {}
    for name, arr in samples.items():
        samples_np[name] = np.asarray(arr)

    # Get extra fields (divergences, etc.)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    jax.block_until_ready(extra)
    extra = jax.device_get(extra)
    extra_fields = {k: np.asarray(v) for k, v in extra.items()}
    run_logger.info(
        f"Extraction complete in {time.perf_counter() - t_extract:.2f}s "
        f"(samples={len(samples_np)}, extra_fields={len(extra_fields)})"
    )

    # Compute statistics
    num_divergent = 0
    if "diverging" in extra_fields:
        num_divergent = int(np.sum(extra_fields["diverging"]))

    # CONVERGENCE CHECK (Jan 2026): Early divergence rate detection
    # High divergence rates indicate NUTS is struggling with the posterior geometry.
    # The 28.4% divergence rate in the 3-angle failure case signals unreliable posteriors.
    # Use actual chain count from samples array (may differ from config if init fails).
    _first_sample = next(iter(samples_np.values()), None)
    _actual_chains_for_div = (
        _first_sample.shape[0] if _first_sample is not None else config.num_chains
    )
    total_samples = num_samples * _actual_chains_for_div
    if total_samples > 0:
        divergence_rate = num_divergent / total_samples
        if divergence_rate > DIVERGENCE_RATE_CRITICAL:
            run_logger.error(
                f"CRITICAL: Divergence rate {divergence_rate:.1%} ({num_divergent}/{total_samples}) "
                f"exceeds {DIVERGENCE_RATE_CRITICAL:.0%} threshold. Posterior is likely unreliable. Consider:\n"
                "  1. Reducing shard size (smaller max_points_per_shard)\n"
                "  2. Using NLSQ warm-start for better initial values\n"
                "  3. Widening priors or fixing problematic parameters"
            )
        elif divergence_rate > DIVERGENCE_RATE_HIGH:
            run_logger.warning(
                f"High divergence rate: {divergence_rate:.1%} ({num_divergent}/{total_samples}). "
                f"Posterior may be biased. Target: <{DIVERGENCE_RATE_TARGET:.0%}"
            )
        elif divergence_rate > DIVERGENCE_RATE_TARGET:
            run_logger.info(
                f"Elevated divergence rate: {divergence_rate:.1%} ({num_divergent}/{total_samples}). "
                f"Acceptable but monitor closely."
            )

    accept_prob = float("nan")
    accept_prob_arr = None
    if "accept_prob" in extra_fields:
        accept_prob_arr = np.asarray(extra_fields["accept_prob"])
        if accept_prob_arr.size:
            accept_prob = float(np.nanmean(accept_prob_arr))

    # Adaptation diagnostics (step size, mass matrix)
    step_size = 0.0
    inv_mass_summary = None
    last_state = getattr(mcmc, "last_state", None)
    adapt_states = _extract_adapt_states(last_state)
    step_sizes = _extract_step_sizes(adapt_states)
    if step_sizes:
        step_size = float(np.nanmedian(step_sizes))
        step_size_min = float(np.nanmin(step_sizes))
        step_size_max = float(np.nanmax(step_sizes))
    else:
        step_size_min = None
        step_size_max = None

    # If we couldn't extract step_size from last_state, try extra_fields
    # (available across NumPyro versions when explicitly requested).
    if step_size == 0.0 and "adapt_state.step_size" in extra_fields:
        try:
            ss = np.asarray(extra_fields["adapt_state.step_size"]).reshape(-1)
            ss = ss[np.isfinite(ss)]
            if ss.size:
                step_size = float(np.median(ss))
        except Exception:  # noqa: S110 - Fallback for step_size extraction
            pass

    inv_mass = None
    if adapt_states:
        a0 = adapt_states[0]
        inv_mass = getattr(a0, "inverse_mass_matrix", None)
        if inv_mass is None and isinstance(a0, dict):
            inv_mass = a0.get("inverse_mass_matrix")
    if inv_mass is not None:
        inv_mass_summary = _summarize_inverse_mass_matrix(inv_mass)

    if step_sizes:
        run_logger.info(
            "Adapted step_size stats: "
            f"min={float(np.nanmin(step_sizes)):.3g}, "
            f"median={float(np.nanmedian(step_sizes)):.3g}, "
            f"max={float(np.nanmax(step_sizes)):.3g}"
        )
    elif step_size > 0:
        run_logger.info(f"Adapted step_size~={step_size:.3g} (from extra_fields)")
    if inv_mass_summary is not None:
        run_logger.info(f"Adapted inverse_mass_matrix: {inv_mass_summary}")

    # Always log basic accept/energy diagnostics when available.
    if accept_prob_arr is not None and accept_prob_arr.size:
        run_logger.info(
            "accept_prob stats: "
            f"mean={float(np.nanmean(accept_prob_arr)):.3g}, "
            f"min={float(np.nanmin(accept_prob_arr)):.3g}, "
            f"median={float(np.nanmedian(accept_prob_arr)):.3g}, "
            f"max={float(np.nanmax(accept_prob_arr)):.3g}, "
            f"frac<1e-12={float(np.nanmean(accept_prob_arr < 1e-12)):.1%}, "
            f"shape={accept_prob_arr.shape}"
        )

        # Per-chain stats are often the easiest way to spot a single stuck chain.
        if accept_prob_arr.ndim >= 2:
            for i in range(min(accept_prob_arr.shape[0], 8)):
                a = np.asarray(accept_prob_arr[i]).reshape(-1)
                if a.size:
                    run_logger.info(
                        f"accept_prob chain[{i}] mean={float(np.nanmean(a)):.3g} "
                        f"min={float(np.nanmin(a)):.3g} median={float(np.nanmedian(a)):.3g} "
                        f"max={float(np.nanmax(a)):.3g}"
                    )

    if "diverging" in extra_fields:
        div = np.asarray(extra_fields["diverging"])
        run_logger.info(f"diverging total={int(np.sum(div))} shape={div.shape}")
        if div.ndim >= 2:
            for i in range(min(div.shape[0], 8)):
                run_logger.info(f"diverging chain[{i}]={int(np.sum(div[i]))}")

    # Step count stats help identify stiffness/underflow that forces tiny step sizes.
    if "num_steps" in extra_fields:
        _log_array_stats(run_logger, name="num_steps", arr=extra_fields["num_steps"])

    if "mean_accept_prob" in extra_fields:
        _log_array_stats(
            run_logger, name="mean_accept_prob", arr=extra_fields["mean_accept_prob"]
        )

    for energy_key in ("potential_energy", "energy"):
        if energy_key in extra_fields:
            _log_array_stats(run_logger, name=energy_key, arr=extra_fields[energy_key])

    # Critical warning for zero acceptance rate
    if np.isfinite(accept_prob) and accept_prob < 0.001:
        run_logger.warning(
            "CRITICAL: Acceptance rate is essentially 0% - all proposals rejected! "
            "This indicates severe sampling problems. Possible causes:\n"
            "  1. Initial values are outside prior support or at boundaries\n"
            "  2. Likelihood returns -inf due to numerical issues (NaN/overflow)\n"
            "  3. Prior is too narrow for the data\n"
            "  4. Step size adaptation failed during warmup\n"
            "Consider: checking initial values, widening priors, or running NLSQ first."
        )

        if accept_prob_arr is not None and accept_prob_arr.size:
            finite = np.isfinite(accept_prob_arr)
            run_logger.warning(
                "accept_prob stats: "
                f"min={float(np.nanmin(accept_prob_arr)):.3g}, "
                f"median={float(np.nanmedian(accept_prob_arr)):.3g}, "
                f"max={float(np.nanmax(accept_prob_arr)):.3g}, "
                f"frac<1e-12={float(np.mean(accept_prob_arr < 1e-12)):.1%}, "
                f"finite={float(np.mean(finite)):.1%}, shape={accept_prob_arr.shape}"
            )
        else:
            run_logger.warning(
                f"accept_prob array missing/empty; extra_fields keys={sorted(extra_fields.keys())}"
            )

        for energy_key in ("potential_energy", "energy"):
            if energy_key in extra_fields:
                e = np.asarray(extra_fields[energy_key])
                finite = np.isfinite(e)
                if np.any(finite):
                    run_logger.warning(
                        f"{energy_key} stats: "
                        f"min={float(np.min(e[finite])):.3g}, "
                        f"median={float(np.median(e[finite])):.3g}, "
                        f"max={float(np.max(e[finite])):.3g}, "
                        f"finite={float(np.mean(finite)):.1%}"
                    )
                else:
                    run_logger.warning(f"{energy_key} all non-finite")
            else:
                run_logger.warning(
                    f"{energy_key} not present in extra_fields (keys={sorted(extra_fields.keys())})"
                )

    # Check for numerical issues exposed by the model.
    # NOTE: numpyro stores deterministics in `get_samples`, not `get_extra_fields`.
    if "n_numerical_issues" in samples_np:
        try:
            n_issues_total = float(np.sum(samples_np["n_numerical_issues"]))
            if n_issues_total > 0:
                total_evals = num_samples * config.num_chains
                issue_rate = n_issues_total / max(total_evals, 1)
                run_logger.warning(
                    f"Numerical issues detected: {n_issues_total:.0f} NaN/Inf occurrences "
                    f"({issue_rate:.1%} of evaluations). "
                    "This may indicate parameter combinations causing overflow in physics model."
                )
        except Exception:  # noqa: S110 - Fallback for step_size extraction
            pass

    # Estimate warmup vs sampling time (rough estimate)
    # Use actual num_warmup/num_samples (may differ from config if adaptive sampling)
    warmup_ratio = num_warmup / (num_warmup + num_samples)
    warmup_time = total_time * warmup_ratio
    sampling_time = total_time * (1 - warmup_ratio)

    # Compute mean tree depth from NUTS num_steps (tree_depth = log2(num_steps))
    mean_tree_depth = 0.0
    if "num_steps" in extra_fields:
        num_steps_arr = np.asarray(extra_fields["num_steps"])
        finite_steps = num_steps_arr[np.isfinite(num_steps_arr) & (num_steps_arr > 0)]
        if finite_steps.size > 0:
            mean_tree_depth = float(np.mean(np.log2(finite_steps)))

    stats = SamplingStats(
        warmup_time=warmup_time,
        sampling_time=sampling_time,
        total_time=total_time,
        num_divergent=num_divergent,
        accept_prob=accept_prob,
        step_size=step_size,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        inverse_mass_matrix_summary=inv_mass_summary,
        plan=plan,
        tree_depth=mean_tree_depth,
    )

    run_logger.info(
        f"Sampling complete in {total_time:.1f}s, "
        f"{num_divergent} divergences, "
        f"accept_prob={accept_prob:.3f}"
    )

    # Create MCMCSamples object
    # Derive actual chain count from first sample array (not from config)
    first_param_samples = next(iter(samples_np.values()), None)
    actual_n_chains = (
        first_param_samples.shape[0]
        if first_param_samples is not None
        else config.num_chains
    )
    mcmc_samples = MCMCSamples(
        samples=samples_np,
        param_names=[k for k in samples_np.keys() if k != "obs"],
        n_chains=actual_n_chains,
        n_samples=num_samples,  # Use actual samples count (may be adaptive)
        extra_fields=extra_fields,
    )

    return mcmc_samples, stats


def run_nuts_with_retry(
    model: Callable,
    model_kwargs: dict[str, Any],
    config: CMCConfig,
    initial_values: dict[str, float] | None,
    parameter_space: ParameterSpace,
    n_phi: int,
    analysis_mode: str,
    max_retries: int = 3,
    rng_key: jax.random.PRNGKey | None = None,
    per_angle_mode: str = "individual",
) -> tuple[MCMCSamples, SamplingStats]:
    """Run NUTS sampling with automatic retry on failure.

    Parameters
    ----------
    model : Callable
        NumPyro model function.
    model_kwargs : dict[str, Any]
        Model arguments.
    config : CMCConfig
        Configuration.
    initial_values : dict[str, float] | None
        Initial values.
    parameter_space : ParameterSpace
        Parameter space.
    n_phi : int
        Number of phi angles.
    analysis_mode : str
        Analysis mode.
    max_retries : int
        Maximum number of retry attempts.
    rng_key : jax.random.PRNGKey | None
        Random key.

    Returns
    -------
    tuple[MCMCSamples, SamplingStats]
        Samples and statistics.

    Raises
    ------
    RuntimeError
        If all retries fail.
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(config.seed)

    last_error = None
    run_logger = with_context(logger, run=getattr(config, "run_id", None))

    # Track current target_accept_prob for adaptive escalation
    current_target_accept_prob = config.target_accept_prob

    for attempt in range(max_retries):
        attempt_num = attempt + 1
        attempt_logger = with_context(run_logger, attempt=attempt_num)
        attempt_start = time.perf_counter()

        # Adaptive divergence threshold: stricter on first attempt
        divergence_threshold = (
            DIVERGENCE_RATE_TARGET if attempt == 0 else DIVERGENCE_RATE_HIGH
        )

        attempt_logger.info(
            f"Attempt {attempt_num}/{max_retries}: starting NUTS "
            f"(chains={config.num_chains}, samples={config.num_samples}, "
            f"target_accept={current_target_accept_prob:.2f}, div_threshold={divergence_threshold:.0%})"
        )
        try:
            # Use different RNG key for each attempt
            attempt_key = jax.random.fold_in(rng_key, attempt)

            # Create config with potentially escalated target_accept_prob
            attempt_config = replace(
                config, target_accept_prob=current_target_accept_prob
            )

            samples, stats = run_nuts_sampling(
                model=model,
                model_kwargs=model_kwargs,
                config=attempt_config,
                initial_values=initial_values,
                parameter_space=parameter_space,
                n_phi=n_phi,
                analysis_mode=analysis_mode,
                rng_key=attempt_key,
                progress_bar=attempt == 0,  # Only show progress on first attempt
                per_angle_mode=per_angle_mode,
            )

            # Check for excessive divergences (adaptive threshold)
            # Use samples.n_samples (actual adapted count) instead of
            # config.num_samples which ignores adaptive sampling reduction.
            total_retry_samples = samples.n_samples * samples.n_chains
            divergence_rate = (
                stats.num_divergent / total_retry_samples
                if total_retry_samples > 0
                else 1.0
            )
            duration = time.perf_counter() - attempt_start
            if divergence_rate > divergence_threshold:
                attempt_logger.warning(
                    f"Attempt {attempt_num}/{max_retries}: divergence_rate={divergence_rate:.1%} "
                    f"> threshold {divergence_threshold:.0%}, retrying with smaller step sizes..."
                )
                # Escalate target_accept_prob for next retry (smaller step sizes)
                current_target_accept_prob = min(
                    0.95, current_target_accept_prob + 0.05
                )
                last_error = RuntimeError(
                    f"High divergence rate: {divergence_rate:.1%}"
                )
                continue

            attempt_logger.info(
                f"Attempt {attempt_num}/{max_retries} succeeded in {duration:.2f}s "
                f"(divergences={stats.num_divergent}, accept_prob={stats.accept_prob:.3f})"
            )
            return samples, stats

        except Exception as e:
            duration = time.perf_counter() - attempt_start
            attempt_logger.warning(
                f"Attempt {attempt_num}/{max_retries} failed after {duration:.2f}s: {e}"
            )
            last_error = e

    run_logger.error(f"MCMC sampling failed after {max_retries} attempts: {last_error}")
    # All retries failed
    raise RuntimeError(
        f"MCMC sampling failed after {max_retries} attempts: {last_error}"
    )
