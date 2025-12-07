"""CMC result dataclass and ArviZ integration.

This module provides the CMCResult dataclass that encapsulates MCMC
posterior samples and diagnostics in a format compatible with ArviZ
and the existing CLI save functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import arviz as az
import numpy as np

from homodyne.optimization.cmc.sampler import MCMCSamples, SamplingStats
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class ParameterStats(dict):
    """Hybrid mapping/sequence for posterior summaries.

    Supports dict-style access by name (for tests/back-compat) and
    list/array-style access by index (for plotting utilities).
    """

    def __init__(self, ordered_names: list[str], values: list[float]) -> None:
        super().__init__(zip(ordered_names, values))
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
            rate = self.divergences / (self.n_chains * self.n_samples)
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

        Returns
        -------
        CMCResult
            Complete result object.
        """
        from homodyne.optimization.cmc.diagnostics import (
            check_convergence,
            compute_ess,
            compute_r_hat,
        )

        # Compute diagnostics
        r_hat = compute_r_hat(mcmc_samples.samples)
        ess_bulk, ess_tail = compute_ess(mcmc_samples.samples)

        # Check convergence
        convergence_status, warnings = check_convergence(
            r_hat=r_hat,
            ess_bulk=ess_bulk,
            divergences=stats.num_divergent,
            n_samples=mcmc_samples.n_samples,
            n_chains=mcmc_samples.n_chains,
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
                parameters[i] = np.mean(samples_flat)
                uncertainties[i] = np.std(samples_flat)

                if name.startswith("contrast_"):
                    contrast_values.append(float(parameters[i]))
                    contrast_stds.append(float(uncertainties[i]))
                elif name.startswith("offset_"):
                    offset_values.append(float(parameters[i]))
                    offset_stds.append(float(uncertainties[i]))
                else:
                    physical_param_names.append(name)
                    mean_params_physical.append(float(parameters[i]))
                    std_params_physical.append(float(uncertainties[i]))

        # Compute covariance
        all_samples = np.column_stack(
            [mcmc_samples.samples[name].flatten() for name in param_names]
        )
        covariance = np.cov(all_samples, rowvar=False)

        # Create ArviZ InferenceData
        inference_data = create_inference_data(mcmc_samples)

        mean_params_stats = ParameterStats(physical_param_names, mean_params_physical)
        std_params_stats = ParameterStats(physical_param_names, std_params_physical)

        if contrast_values:
            mean_params_stats["contrast"] = float(np.mean(contrast_values))
            std_params_stats["contrast"] = float(np.mean(contrast_stds))
        if offset_values:
            mean_params_stats["offset"] = float(np.mean(offset_values))
            std_params_stats["offset"] = float(np.mean(offset_stds))

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
                "mean": float(np.mean(samples_flat)),
                "std": float(np.std(samples_flat)),
                "median": float(np.median(samples_flat)),
                "hdi_5%": float(np.percentile(samples_flat, 5)),
                "hdi_95%": float(np.percentile(samples_flat, 95)),
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
    # Build posterior dictionary
    posterior_dict: dict[str, np.ndarray] = {}

    for name in mcmc_samples.param_names:
        if name in mcmc_samples.samples:
            # ArviZ expects (n_chains, n_samples)
            posterior_dict[name] = mcmc_samples.samples[name]

    # Create InferenceData
    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=mcmc_samples.extra_fields if mcmc_samples.extra_fields else None,
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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (c2_fitted_mean, c2_fitted_std) from posterior.
    """
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

    # Compute g1 with posterior mean
    g1 = compute_g1_total(
        jnp.array(params),
        jnp.array(t1),
        jnp.array(t2),
        jnp.array(phi),
        q,
        L,
        dt,
        analysis_mode,
    )

    # Get per-angle contrast/offset
    phi_unique = np.unique(phi)
    n_phi = len(phi_unique)

    contrasts = np.array([stats[f"contrast_{i}"]["mean"] for i in range(n_phi)])
    offsets = np.array([stats[f"offset_{i}"]["mean"] for i in range(n_phi)])

    # Map phi to indices
    phi_indices = np.searchsorted(phi_unique, phi)

    # Apply scaling
    contrast_per_point = contrasts[phi_indices]
    offset_per_point = offsets[phi_indices]

    c2_fitted = contrast_per_point * np.array(g1) ** 2 + offset_per_point

    # Compute uncertainty by sampling
    n_posterior_samples = min(100, result.n_samples)
    c2_samples = []

    for sample_idx in range(n_posterior_samples):
        # Get parameters from this sample
        chain_idx = sample_idx % result.n_chains
        within_chain_idx = sample_idx // result.n_chains

        sample_params = np.array(
            [result.samples[name][chain_idx, within_chain_idx] for name in param_names]
        )

        sample_g1 = compute_g1_total(
            jnp.array(sample_params),
            jnp.array(t1),
            jnp.array(t2),
            jnp.array(phi),
            q,
            L,
            dt,
            analysis_mode,
        )

        sample_contrasts = np.array(
            [
                result.samples[f"contrast_{i}"][chain_idx, within_chain_idx]
                for i in range(n_phi)
            ]
        )
        sample_offsets = np.array(
            [
                result.samples[f"offset_{i}"][chain_idx, within_chain_idx]
                for i in range(n_phi)
            ]
        )

        sample_c2 = (
            sample_contrasts[phi_indices] * np.array(sample_g1) ** 2
            + sample_offsets[phi_indices]
        )
        c2_samples.append(sample_c2)

    c2_samples_arr = np.array(c2_samples)
    c2_fitted_std = np.std(c2_samples_arr, axis=0)

    return c2_fitted, c2_fitted_std
