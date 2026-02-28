"""CMC result dataclass and ArviZ integration.

This module provides the CMCResult dataclass that encapsulates MCMC
posterior samples and diagnostics in a format compatible with ArviZ
and the existing CLI save functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import arviz as az

    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False
    az = None  # type: ignore

import numpy as np

from homodyne.optimization.cmc.diagnostics import DEFAULT_MIN_ESS
from homodyne.optimization.cmc.sampler import MCMCSamples, SamplingStats
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class ParameterStats(dict):
    """Hybrid mapping/sequence for posterior summaries.

    Supports dict-style access by name (for tests/back-compat) and
    list/array-style access by index (for plotting utilities).
    """

    def __init__(self, ordered_names: list[str], values: list[float]) -> None:
        super().__init__(zip(ordered_names, values, strict=True))
        self._ordered_names = list(ordered_names)
        self._ordered_values = list(values)

    def __getitem__(self, key: int | str) -> float:
        if isinstance(key, int):
            return self._ordered_values[key]
        return super().__getitem__(key)

    def __len__(self) -> int:  # sequence semantics
        return len(self._ordered_values)

    def __array__(self, dtype=None) -> np.ndarray:  # numpy friendliness
        return np.asarray(self._ordered_values, dtype=dtype)

    @property
    def as_array(self) -> np.ndarray:
        """Return ordered values as numpy array."""
        return np.asarray(self._ordered_values, dtype=float)

    def tolist(self) -> list[float]:
        """Return ordered values as list (numpy compatibility)."""
        return list(self._ordered_values)


@dataclass
class CMCResult:
    """CMC analysis result with posterior samples and diagnostics.

    This dataclass is compatible with save_mcmc_results() in cli/commands.py.

    Attributes
    ----------
    parameters : np.ndarray
        Posterior mean values, shape (n_params,).
    uncertainties : np.ndarray
        Posterior standard deviations, shape (n_params,).
    param_names : list[str]
        Parameter names in sampling order.
    samples : dict[str, np.ndarray]
        Raw samples, {name: (n_chains, n_samples)}.
    convergence_status : str
        "converged" | "divergences" | "not_converged".
    r_hat : dict[str, float]
        Per-parameter R-hat values.
    ess_bulk : dict[str, float]
        Per-parameter bulk ESS.
    ess_tail : dict[str, float]
        Per-parameter tail ESS.
    divergences : int
        Total number of divergent transitions.
    inference_data : az.InferenceData
        ArviZ InferenceData for plotting.
    execution_time : float
        Total sampling time in seconds.
    warmup_time : float
        Warmup time in seconds.
    n_chains : int
        Number of MCMC chains.
    n_samples : int
        Samples per chain.
    n_warmup : int
        Warmup samples.
    analysis_mode : str
        Analysis mode used.
    covariance : np.ndarray
        Parameter covariance matrix (from samples).
    chi_squared : float
        Placeholder for compatibility (not directly computed in MCMC).
    reduced_chi_squared : float
        Placeholder for compatibility.
    device_info : dict[str, Any]
        Device used for computation.
    """

    # Core results
    parameters: np.ndarray
    uncertainties: np.ndarray
    param_names: list[str]

    # MCMC-specific
    samples: dict[str, np.ndarray]
    convergence_status: str
    r_hat: dict[str, float]
    ess_bulk: dict[str, float]
    ess_tail: dict[str, float]
    divergences: int

    # ArviZ
    inference_data: az.InferenceData

    # Timing
    execution_time: float
    warmup_time: float

    # Config
    n_chains: int = 4
    n_samples: int = 2000
    n_warmup: int = 500
    analysis_mode: str = "static"
    per_angle_mode: str = (
        "auto"  # Per-angle scaling mode (auto/constant/constant_averaged/individual)
    )
    num_shards: int = 1  # Number of shards combined (for correct divergence rate)

    # Compatibility fields
    covariance: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_squared: float = 0.0
    reduced_chi_squared: float = 0.0
    device_info: dict[str, Any] = field(default_factory=dict)
    recovery_actions: list[str] = field(default_factory=list)
    quality_flag: str = "good"
    # Legacy/CLI plot compatibility
    mean_params: ParameterStats = field(default_factory=lambda: ParameterStats([], []))
    std_params: ParameterStats = field(default_factory=lambda: ParameterStats([], []))
    mean_contrast: float | None = None
    std_contrast: float | None = None
    mean_offset: float | None = None
    std_offset: float | None = None

    def is_cmc_result(self) -> bool:
        """Return True - required by CLI for diagnostic generation."""
        return True

    @property
    def success(self) -> bool:
        """Return True if sampling converged (backward compatibility)."""
        return self.convergence_status == "converged"

    @property
    def message(self) -> str:
        """Return descriptive message about result."""
        if self.convergence_status == "converged":
            return f"CMC sampling converged. {self.divergences} divergences."
        elif self.convergence_status == "divergences":
            # Account for num_shards in divergence rate calculation
            total_transitions = self.num_shards * self.n_chains * self.n_samples
            rate = self.divergences / total_transitions if total_transitions > 0 else 0
            return f"CMC completed with {rate:.1%} divergence rate."
        else:
            return f"CMC did not converge: {self.convergence_status}"

    @classmethod
    def from_mcmc_samples(
        cls,
        mcmc_samples: MCMCSamples,
        stats: SamplingStats,
        analysis_mode: str,
        n_warmup: int = 500,
        min_ess: float | None = None,
    ) -> CMCResult:
        """Create CMCResult from MCMC samples.

        Parameters
        ----------
        mcmc_samples : MCMCSamples
            Raw MCMC samples.
        stats : SamplingStats
            Sampling statistics.
        analysis_mode : str
            Analysis mode used.
        n_warmup : int
            Number of warmup samples.
        min_ess : float | None
            Minimum effective sample size for convergence checks.
            If None, uses ``DEFAULT_MIN_ESS`` from diagnostics module.

        Returns
        -------
        CMCResult
            Complete result object.
        """
        from homodyne.optimization.cmc.diagnostics import (
            DEFAULT_MIN_ESS,
            check_convergence,
            compute_ess,
            compute_r_hat,
        )

        if min_ess is None:
            min_ess = DEFAULT_MIN_ESS

        # Compute diagnostics
        r_hat = compute_r_hat(mcmc_samples.samples)
        ess_bulk, ess_tail = compute_ess(mcmc_samples.samples)

        # Check convergence
        # Pass num_shards for correct divergence rate calculation in CMC
        convergence_status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=stats.num_divergent,
            n_samples=mcmc_samples.n_samples,
            n_chains=mcmc_samples.n_chains,
            min_ess=min_ess,
            num_shards=getattr(mcmc_samples, "num_shards", 1),
        )

        if warnings:
            for warning in warnings:
                logger.warning(f"Convergence warning: {warning}")

        # Compute posterior statistics
        param_names = mcmc_samples.param_names
        parameters = np.zeros(len(param_names))
        uncertainties = np.zeros(len(param_names))

        # Aggregate convenience stats for legacy consumers (CLI plots, writers)
        contrast_values: list[float] = []
        contrast_stds: list[float] = []
        offset_values: list[float] = []
        offset_stds: list[float] = []
        physical_param_names: list[str] = []
        mean_params_physical: list[float] = []
        std_params_physical: list[float] = []

        for i, name in enumerate(param_names):
            if name in mcmc_samples.samples:
                samples_flat = mcmc_samples.samples[name].flatten()
                parameters[i] = np.nanmean(samples_flat)
                uncertainties[i] = np.nanstd(samples_flat)

                # CRITICAL FIX (Dec 2025): Exclude _z (z-space) parameters from legacy stats
                # The scaled model samples contrast_0_z ~ N(0,1) and registers contrast_0 as
                # deterministic. Only use original-space values (without _z suffix).
                if name.startswith("contrast_") and not name.endswith("_z"):
                    contrast_values.append(float(parameters[i]))
                    contrast_stds.append(float(uncertainties[i]))
                elif name.startswith("offset_") and not name.endswith("_z"):
                    offset_values.append(float(parameters[i]))
                    offset_stds.append(float(uncertainties[i]))
                elif not name.endswith("_z"):
                    # Physical parameters (D0, alpha, etc.) - exclude _z variants
                    physical_param_names.append(name)
                    mean_params_physical.append(float(parameters[i]))
                    std_params_physical.append(float(uncertainties[i]))
                # Skip _z parameters - they are z-space samples, not original-space values

        # Compute covariance (requires at least 2 samples)
        # P2-R6-03: Guard against param_names entries absent from samples
        # (e.g. deterministic sites not returned by get_samples, failed shards).
        present_names = [n for n in param_names if n in mcmc_samples.samples]
        all_samples = (
            np.column_stack(
                [mcmc_samples.samples[name].flatten() for name in present_names]
            )
            if present_names
            else np.zeros((0, 0))
        )
        if all_samples.shape[0] < 2:
            # Not enough samples for covariance - return zeros
            covariance = np.zeros((all_samples.shape[1], all_samples.shape[1]))
        else:
            # Filter rows with any NaN before computing covariance.
            # NaN samples arise from failed shards or NUTS divergences that
            # produce non-finite values; np.cov propagates them to the full matrix.
            finite_mask = np.all(np.isfinite(all_samples), axis=1)
            all_samples_finite = all_samples[finite_mask]
            if all_samples_finite.shape[0] < 2:
                covariance = np.zeros((all_samples.shape[1], all_samples.shape[1]))
            else:
                # Q4: Subsample to at most 10K rows before computing covariance.
                # np.cov is O(N*P^2); for N=600K, P=9 this takes ~1 s and uses ~170 MB.
                # 10K rows give a statistically equivalent 9x9 result in ~50 ms.
                _max_cov_samples = 10_000
                if all_samples_finite.shape[0] > _max_cov_samples:
                    # Fixed seed for reproducible covariance subsampling.
                    rng = np.random.default_rng(seed=0)
                    idx = rng.choice(
                        all_samples_finite.shape[0], size=_max_cov_samples, replace=False
                    )
                    covariance = np.cov(all_samples_finite[idx], rowvar=False)
                else:
                    covariance = np.cov(all_samples_finite, rowvar=False)

        # Create ArviZ InferenceData
        inference_data = create_inference_data(mcmc_samples)

        mean_params_stats = ParameterStats(physical_param_names, mean_params_physical)
        std_params_stats = ParameterStats(physical_param_names, std_params_physical)

        if contrast_values:
            mean_params_stats["contrast"] = float(np.nanmean(contrast_values))
            std_params_stats["contrast"] = float(np.nanmean(contrast_stds))
        if offset_values:
            mean_params_stats["offset"] = float(np.nanmean(offset_values))
            std_params_stats["offset"] = float(np.nanmean(offset_stds))

        return cls(
            parameters=parameters,
            uncertainties=uncertainties,
            param_names=param_names,
            samples=mcmc_samples.samples,
            convergence_status=convergence_status,
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            ess_tail=ess_tail,
            divergences=stats.num_divergent,
            inference_data=inference_data,
            execution_time=stats.total_time,
            warmup_time=stats.warmup_time,
            n_chains=mcmc_samples.n_chains,
            n_samples=mcmc_samples.n_samples,
            n_warmup=n_warmup,
            analysis_mode=analysis_mode,
            num_shards=getattr(mcmc_samples, "num_shards", 1),
            covariance=covariance,
            device_info={"platform": "cpu", "device": "CPU"},
            # Legacy/compat fields expected by CLI writers/plots
            mean_params=mean_params_stats,
            std_params=std_params_stats,
            mean_contrast=mean_params_stats.get("contrast"),
            std_contrast=std_params_stats.get("contrast"),
            mean_offset=mean_params_stats.get("offset"),
            std_offset=std_params_stats.get("offset"),
        )

    def get_posterior_stats(self) -> dict[str, dict[str, float]]:
        """Get posterior statistics for each parameter.

        Returns
        -------
        dict[str, dict[str, float]]
            Statistics per parameter: mean, std, median, hdi_5%, hdi_95%.
        """
        stats: dict[str, dict[str, float]] = {}

        for name in self.param_names:
            if name not in self.samples:
                continue

            samples_flat = self.samples[name].flatten()

            stats[name] = {
                "mean": float(np.nanmean(samples_flat)),
                "std": float(np.nanstd(samples_flat)),
                "median": float(np.nanmedian(samples_flat)),
                "hdi_5%": float(np.nanpercentile(samples_flat, 5)),
                "hdi_95%": float(np.nanpercentile(samples_flat, 95)),
                "r_hat": self.r_hat.get(name, np.nan),
                "ess_bulk": self.ess_bulk.get(name, np.nan),
                "ess_tail": self.ess_tail.get(name, np.nan),
            }

        return stats

    def get_samples_array(self) -> np.ndarray:
        """Get samples as 3D array.

        Returns
        -------
        np.ndarray
            Shape (n_chains, n_samples, n_params).
        """
        n_params = len(self.param_names)
        samples_3d = np.zeros((self.n_chains, self.n_samples, n_params))

        for i, name in enumerate(self.param_names):
            if name in self.samples:
                samples_3d[:, :, i] = self.samples[name]

        return samples_3d

    def validate_parameters(self, n_phi: int | None = None) -> list[str]:
        """Validate that result contains expected parameters.

        Parameters
        ----------
        n_phi : int | None
            Number of phi angles expected. If None, infers from samples.

        Returns
        -------
        list[str]
            List of validation warnings (empty if all valid).
        """
        warnings: list[str] = []

        # Check required physical parameters for analysis mode
        if self.analysis_mode == "laminar_flow":
            required_physical = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
        else:
            required_physical = ["D0", "alpha", "D_offset"]

        for param in required_physical:
            if param not in self.samples:
                warnings.append(f"Missing required parameter: {param}")
            elif not np.all(np.isfinite(self.samples[param])):
                warnings.append(f"Non-finite values in parameter: {param}")

        # Check for contrast/offset parameters — only in individual mode.
        # auto mode uses "contrast"/"offset" sites (not indexed), and
        # constant/constant_averaged modes have no sampled contrast/offset sites.
        # Checking for contrast_0..N in those modes produces spurious warnings.
        if n_phi is None:
            # Infer from samples (only count indexed per-angle sites)
            contrast_params = [p for p in self.param_names if p.startswith("contrast_")]
            n_phi = len(contrast_params) if contrast_params else 0

        if n_phi > 0:
            # Individual mode: verify all indexed per-angle sites are present
            for i in range(n_phi):
                contrast_name = f"contrast_{i}"
                offset_name = f"offset_{i}"
                if contrast_name not in self.samples:
                    warnings.append(f"Missing per-angle parameter: {contrast_name}")
                if offset_name not in self.samples:
                    warnings.append(f"Missing per-angle parameter: {offset_name}")
        elif "contrast" not in self.samples and "contrast_0" not in self.samples:
            # Neither auto-mode site nor individual-mode site found;
            # only warn if the mode actually expects sampled contrast/offset.
            # analysis_mode vocabulary is "static"/"laminar_flow" (not "constant"),
            # so check per_angle_mode (which tracks the scaling strategy) instead.
            if self.analysis_mode not in ("constant", "constant_averaged") and getattr(
                self, "per_angle_mode", None
            ) not in ("constant", "constant_averaged"):
                warnings.append(
                    "No contrast/offset sites found in posterior samples. "
                    "Expected 'contrast'/'offset' (auto mode) or 'contrast_0'/'offset_0' "
                    "(individual mode). Use constant/constant_averaged mode if scaling is fixed."
                )

        # Check diagnostic values
        _r_hat_finite = [v for v in self.r_hat.values() if np.isfinite(v)]
        _ess_finite = [v for v in self.ess_bulk.values() if np.isfinite(v)]
        max_r_hat = max(_r_hat_finite) if _r_hat_finite else float("nan")
        min_ess = min(_ess_finite) if _ess_finite else 0.0

        if max_r_hat > 1.1:
            warnings.append(f"High R-hat detected: {max_r_hat:.3f} > 1.1")
        if min_ess < DEFAULT_MIN_ESS:
            warnings.append(f"Low ESS detected: {min_ess:.0f} < {DEFAULT_MIN_ESS}")

        # Check for divergences
        if self.divergences > 0:
            total_transitions = self.num_shards * self.n_chains * self.n_samples
            div_rate = (
                self.divergences / total_transitions if total_transitions > 0 else 0
            )
            if div_rate > 0.01:
                warnings.append(f"High divergence rate: {div_rate:.1%}")

        return warnings


def create_inference_data(mcmc_samples: MCMCSamples) -> az.InferenceData:
    """Create ArviZ InferenceData from MCMC samples.

    Parameters
    ----------
    mcmc_samples : MCMCSamples
        Raw MCMC samples.

    Returns
    -------
    az.InferenceData
        ArviZ-compatible data structure.
    """
    if not HAS_ARVIZ:
        raise ImportError("ArviZ is required to create InferenceData")

    # Build posterior dictionary
    posterior_dict: dict[str, np.ndarray] = {}

    for name in mcmc_samples.param_names:
        if name in mcmc_samples.samples:
            # ArviZ expects (n_chains, n_samples)
            posterior_dict[name] = mcmc_samples.samples[name]

    # Map NumPyro extra_fields to ArviZ sample_stats conventions
    stats: dict[str, np.ndarray] | None = None
    if mcmc_samples.extra_fields:
        stats = {}
        for key, val in mcmc_samples.extra_fields.items():
            if key == "potential_energy":
                # ArviZ plot_energy expects "energy"
                stats["energy"] = val
            elif "." in key:
                # xarray doesn't allow dots in variable names (e.g. adapt_state.step_size)
                stats[key.replace(".", "_")] = val
            else:
                stats[key] = val

    # Create InferenceData
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=stats,
    )

    return idata


def samples_dict_from_array(
    samples_array: np.ndarray,
    param_names: list[str],
) -> dict[str, np.ndarray]:
    """Convert samples array to dictionary.

    Parameters
    ----------
    samples_array : np.ndarray
        Shape (n_chains, n_samples, n_params).
    param_names : list[str]
        Parameter names.

    Returns
    -------
    dict[str, np.ndarray]
        Samples dictionary.
    """
    samples_dict: dict[str, np.ndarray] = {}

    for i, name in enumerate(param_names):
        samples_dict[name] = samples_array[:, :, i]

    return samples_dict


def compute_fitted_c2(
    result: CMCResult,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    q: float,
    L: float,
    dt: float,
    analysis_mode: str,
    fixed_contrasts: np.ndarray | None = None,
    fixed_offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute fitted C2 values from posterior mean.

    Parameters
    ----------
    result : CMCResult
        CMC result with posterior samples.
    t1, t2, phi : np.ndarray
        Coordinates (pooled 1D).
    q, L, dt : float
        Physics parameters.
    analysis_mode : str
        Analysis mode.
    fixed_contrasts : np.ndarray | None
        Per-angle contrast array of shape (n_phi,) for ``constant`` and
        ``constant_averaged`` modes where contrast is not sampled.
        Required when neither ``contrast_0`` nor ``contrast`` appears
        in posterior samples.
    fixed_offsets : np.ndarray | None
        Per-angle offset array of shape (n_phi,) paired with
        ``fixed_contrasts``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (c2_fitted_mean, c2_fitted_std) from posterior.
    """
    import jax
    import jax.numpy as jnp

    from homodyne.core.physics_cmc import compute_g1_total
    from homodyne.optimization.cmc.priors import LAMINAR_PARAMS, STATIC_PARAMS

    # Get posterior mean parameters
    stats = result.get_posterior_stats()

    # Extract physical parameters
    if analysis_mode == "laminar_flow":
        param_names = LAMINAR_PARAMS
    else:
        param_names = STATIC_PARAMS

    params = np.array([stats[name]["mean"] for name in param_names])

    # Prepare unique phi for physics call (compute_g1_total expects unique phi)
    phi_unique = np.unique(phi)
    # Compute g1 with posterior mean
    g1 = compute_g1_total(
        jnp.array(params),
        jnp.array(t1),
        jnp.array(t2),
        jnp.array(phi_unique),
        q,
        L,
        dt,
    )

    # Get per-angle contrast/offset
    n_phi = len(phi_unique)

    # Handle all per-angle modes:
    #   individual: contrast_0, contrast_1, ... are sampled
    #   auto:       contrast (single) is sampled and broadcast
    #   constant/constant_averaged: not sampled; caller supplies fixed_contrasts
    if "contrast_0" in stats:
        contrasts = np.array([stats[f"contrast_{i}"]["mean"] for i in range(n_phi)])
        offsets = np.array([stats[f"offset_{i}"]["mean"] for i in range(n_phi)])
    elif "contrast" in stats:
        # Auto mode: single sampled contrast/offset broadcast to all angles
        contrasts = np.full(n_phi, stats["contrast"]["mean"])
        offsets = np.full(n_phi, stats["offset"]["mean"])
    elif fixed_contrasts is not None and fixed_offsets is not None:
        # Constant/constant_averaged mode: contrast/offset are fixed, not sampled.
        # Caller must supply the pre-computed fixed arrays.
        contrasts = np.asarray(fixed_contrasts, dtype=float)
        offsets = np.asarray(fixed_offsets, dtype=float)
        if contrasts.shape != (n_phi,) or offsets.shape != (n_phi,):
            raise ValueError(
                f"fixed_contrasts/fixed_offsets must have shape ({n_phi},), "
                f"got {contrasts.shape} and {offsets.shape}"
            )
    else:
        raise KeyError(
            f"Cannot find contrast parameters in posterior stats "
            f"(available keys: {sorted(stats.keys())}). "
            f"For constant/constant_averaged mode, pass fixed_contrasts and "
            f"fixed_offsets arrays (shape ({n_phi},)) from the original model_kwargs."
        )

    # Map phi to indices using nearest-neighbor matching (consistent with
    # data_prep.extract_phi_info). Raw searchsorted silently assigns points to the
    # wrong angle when phi values have float precision differences.
    if n_phi <= 256:
        phi_indices = np.argmin(
            np.abs(phi[:, None] - phi_unique[None, :]), axis=1
        ).astype(np.int32)
    else:
        idx = np.searchsorted(phi_unique, phi)
        idx = np.clip(idx, 0, n_phi - 1)
        left = np.clip(idx - 1, 0, n_phi - 1)
        use_left = np.abs(phi - phi_unique[left]) < np.abs(phi - phi_unique[idx])
        phi_indices = np.where(use_left, left, idx).astype(np.int32)

    # Apply scaling: gather the right phi row from g1 per data point.
    # g1 shape is (n_phi, n_points); phi_indices maps each point to its phi row.
    contrast_per_point = contrasts[phi_indices]
    offset_per_point = offsets[phi_indices]

    g1_arr = np.array(g1)  # (n_phi, n_points)
    g1_at_phi = g1_arr[phi_indices, np.arange(len(phi_indices))]  # (n_points,)

    c2_fitted = contrast_per_point * g1_at_phi**2 + offset_per_point

    # D4: Compute uncertainty by batched vmap instead of a Python loop.
    # Previously: 100 sequential compute_g1_total calls (100 JAX dispatches).
    # Now: build (n_posterior_samples, n_params) batch array and call vmap once.
    n_posterior_samples = min(100, result.n_samples)

    # Build batch arrays: each row is one posterior draw.
    # Index draws round-robin across chains to match the original ordering.
    chain_indices = np.arange(n_posterior_samples) % result.n_chains
    within_chain_indices = np.arange(n_posterior_samples) // result.n_chains

    batched_params = np.stack(
        [
            np.array(
                [
                    result.samples[name][chain_indices[i], within_chain_indices[i]]
                    for name in param_names
                ]
            )
            for i in range(n_posterior_samples)
        ]
    )  # shape: (n_posterior_samples, n_physical_params)

    # Handle all per-angle modes for posterior draws
    if "contrast_0" in result.samples:
        # Individual mode: per-angle contrast/offset sampled independently
        batched_contrasts = np.stack(
            [
                np.array(
                    [
                        result.samples[f"contrast_{j}"][
                            chain_indices[i], within_chain_indices[i]
                        ]
                        for j in range(n_phi)
                    ]
                )
                for i in range(n_posterior_samples)
            ]
        )  # shape: (n_posterior_samples, n_phi)

        batched_offsets = np.stack(
            [
                np.array(
                    [
                        result.samples[f"offset_{j}"][
                            chain_indices[i], within_chain_indices[i]
                        ]
                        for j in range(n_phi)
                    ]
                )
                for i in range(n_posterior_samples)
            ]
        )  # shape: (n_posterior_samples, n_phi)
    elif "contrast" in result.samples:
        # Auto mode: single sampled contrast/offset broadcast to all angles
        batched_contrasts = np.stack(
            [
                np.full(
                    n_phi,
                    result.samples["contrast"][
                        chain_indices[i], within_chain_indices[i]
                    ],
                )
                for i in range(n_posterior_samples)
            ]
        )  # shape: (n_posterior_samples, n_phi)

        batched_offsets = np.stack(
            [
                np.full(
                    n_phi,
                    result.samples["offset"][chain_indices[i], within_chain_indices[i]],
                )
                for i in range(n_posterior_samples)
            ]
        )  # shape: (n_posterior_samples, n_phi)
    else:
        # Constant/constant_averaged mode: fixed values, no uncertainty over contrast.
        # Use the fixed arrays from the mean-computation step (already validated above).
        batched_contrasts = np.tile(contrasts, (n_posterior_samples, 1))
        batched_offsets = np.tile(offsets, (n_posterior_samples, 1))

    # vmap over the first axis (sample index); all other args are fixed.
    _t1_jnp = jnp.array(t1)
    _t2_jnp = jnp.array(t2)
    _phi_jnp = jnp.array(phi_unique)

    def _g1_single(single_params: jnp.ndarray) -> jnp.ndarray:
        return compute_g1_total(single_params, _t1_jnp, _t2_jnp, _phi_jnp, q, L, dt)

    batched_g1 = jax.vmap(_g1_single)(jnp.array(batched_params))
    # batched_g1 shape: (n_posterior_samples, n_phi, n_points)

    # Apply per-angle contrast/offset scaling for each sample.
    # batched_contrasts[:, phi_indices] -> (n_posterior_samples, n_points)
    sample_contrasts_mapped = batched_contrasts[:, phi_indices]  # (S, N)
    sample_offsets_mapped = batched_offsets[:, phi_indices]  # (S, N)

    # Gather the right phi row per data point.
    # phi_indices is (n_points,); combine with a point index to select from dim-2.
    n_points = len(phi_indices)
    batched_g1_at_phi = batched_g1[:, phi_indices, np.arange(n_points)]
    # shape: (n_posterior_samples, n_points)

    c2_samples_arr = np.array(
        sample_contrasts_mapped * np.array(batched_g1_at_phi) ** 2
        + sample_offsets_mapped
    )  # (n_posterior_samples, n_points)
    c2_fitted_std = np.nanstd(c2_samples_arr, axis=0)

    return c2_fitted, c2_fitted_std
