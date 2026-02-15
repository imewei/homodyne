"""Base class for CMC execution backends.

This module defines the abstract interface for CMC backends
and provides a factory function for selecting backends.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.diagnostics import BimodalConsensusResult
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)


class CMCBackend(ABC):
    """Abstract base class for CMC execution backends.

    Backends handle the parallel execution of MCMC sampling across
    data shards and the combination of results.
    """

    @abstractmethod
    def run(
        self,
        model: Callable,
        model_kwargs: dict[str, Any],
        config: CMCConfig,
        shards: list[PreparedData] | None = None,
    ) -> MCMCSamples:
        """Run MCMC sampling (potentially across shards).

        Parameters
        ----------
        model : Callable
            NumPyro model function.
        model_kwargs : dict[str, Any]
            Common model arguments.
        config : CMCConfig
            CMC configuration.
        shards : list[PreparedData] | None
            Data shards for parallel execution.
            If None, runs single-threaded on full data.

        Returns
        -------
        MCMCSamples
            Combined samples from all shards.
        """

    @abstractmethod
    def get_name(self) -> str:
        """Get backend name."""

    def is_available(self) -> bool:
        """Check if backend is available.

        Returns
        -------
        bool
            True if backend can be used.
        """
        return True


def select_backend(
    config: CMCConfig,
) -> CMCBackend:
    """Select appropriate backend based on configuration.

    Parameters
    ----------
    config : CMCConfig
        CMC configuration.

    Returns
    -------
    CMCBackend
        Selected backend instance.

    Raises
    ------
    ValueError
        If requested backend is not available.
    """
    backend_name = config.backend_name

    if backend_name == "auto":
        # Default to multiprocessing for CPU
        backend_name = "multiprocessing"

    # Backward compatibility: allow legacy "jax" alias
    # NOTE: Map to multiprocessing, not pjit, because pjit backend is sequential
    # (it processes shards one at a time in a for loop, not in parallel)
    if backend_name == "jax":
        logger.warning(
            "CMC backend 'jax' is deprecated; mapping to 'multiprocessing' "
            "for parallel execution. Set backend_config.name to "
            "'multiprocessing' or 'auto' instead."
        )
        backend_name = "multiprocessing"

    if backend_name == "multiprocessing":
        from homodyne.optimization.cmc.backends.multiprocessing import (
            MultiprocessingBackend,
        )

        return MultiprocessingBackend()

    elif backend_name == "pjit":
        try:
            from homodyne.optimization.cmc.backends.pjit import PjitBackend

            return PjitBackend()
        except ImportError:
            logger.warning(
                "pjit backend not available, falling back to multiprocessing"
            )
            from homodyne.optimization.cmc.backends.multiprocessing import (
                MultiprocessingBackend,
            )

            return MultiprocessingBackend()

    elif backend_name == "pbs":
        try:
            from homodyne.optimization.cmc.backends.pbs import PBSBackend

            return PBSBackend()
        except ImportError:
            logger.warning("PBS backend not available, falling back to multiprocessing")
            from homodyne.optimization.cmc.backends.multiprocessing import (
                MultiprocessingBackend,
            )

            return MultiprocessingBackend()

    else:
        raise ValueError(f"Unknown backend: {backend_name}")


def combine_shard_samples(
    shard_samples: list[MCMCSamples],
    method: str = "weighted_gaussian",
    chunk_size: int = 500,
) -> MCMCSamples:
    """Combine samples from multiple shards.

    Uses hierarchical combination for large shard counts to limit peak memory.
    For K > chunk_size shards, combines in chunks of chunk_size, then combines
    the intermediate results. This reduces peak memory from O(K) to O(chunk_size).

    Memory scaling:
    - Each shard result: ~100KB (13 params × 2 chains × 1500 samples × 8 bytes)
    - Direct combination: K × 100KB peak memory
    - Hierarchical (chunk=500): max(500 × 100KB, K/500 × 100KB) ≈ 50MB peak

    Parameters
    ----------
    shard_samples : list[MCMCSamples]
        Samples from each shard.
    method : str
        Combination method: "consensus_mc" (recommended),
        "weighted_gaussian", "simple_average", or "auto".
    chunk_size : int
        Number of shards to combine at once for hierarchical combination.
        Default 500 keeps peak memory under ~50MB per combination step.

    Returns
    -------
    MCMCSamples
        Combined samples.
    """

    if len(shard_samples) == 1:
        return shard_samples[0]

    # For large shard counts, use hierarchical combination to limit memory
    if len(shard_samples) > chunk_size:
        import gc

        logger.info(
            f"Hierarchical combination: {len(shard_samples)} shards "
            f"in chunks of {chunk_size}"
        )
        intermediate_results = []
        n_chunks = (len(shard_samples) + chunk_size - 1) // chunk_size
        for i in range(0, len(shard_samples), chunk_size):
            chunk = shard_samples[i : i + chunk_size]
            chunk_result = _combine_shard_chunk(chunk, method)
            intermediate_results.append(chunk_result)

            # Clear chunk references and force GC to reduce peak memory
            # Each shard is ~100KB, so freeing chunk_size shards saves ~50MB
            del chunk
            gc.collect()

            logger.debug(f"Combined chunk {i // chunk_size + 1}/{n_chunks}")

        # Recursively combine intermediate results
        return combine_shard_samples(intermediate_results, method, chunk_size)

    return _combine_shard_chunk(shard_samples, method)


def _combine_shard_chunk(
    shard_samples: list[MCMCSamples],
    method: str,
) -> MCMCSamples:
    """Combine a chunk of shard samples (internal helper).

    Parameters
    ----------
    shard_samples : list[MCMCSamples]
        Samples from each shard in the chunk.
    method : str
        Combination method:
        - "consensus_mc": Correct Consensus Monte Carlo (precision-weighted means)
        - "robust_consensus_mc": Robust CMC with trimmed statistics (Jan 2026)
        - "weighted_gaussian": Legacy element-wise weighted averaging (deprecated)
        - "simple_average": Simple element-wise averaging (deprecated)

    Returns
    -------
    MCMCSamples
        Combined samples for this chunk.

    Notes
    -----
    The "consensus_mc" method implements the correct Consensus Monte Carlo
    algorithm (Scott et al., 2016):

    1. For each shard s, compute posterior mean μ_s and variance σ²_s
    2. Combined precision: 1/σ² = Σ_s (1/σ²_s)
    3. Combined mean: μ = σ² × Σ_s (μ_s / σ²_s)
    4. Generate new samples from N(μ, σ²)

    The "robust_consensus_mc" method (Jan 2026) extends this with:
    - Trimmed statistics to exclude outlier shards
    - Winsorization of extreme variances
    - Automatic outlier detection based on median absolute deviation
    """
    import numpy as np

    from homodyne.optimization.cmc.sampler import MCMCSamples

    if len(shard_samples) == 1:
        return shard_samples[0]

    # Get parameter names from first shard
    param_names = shard_samples[0].param_names
    n_chains = shard_samples[0].n_chains
    n_samples = shard_samples[0].n_samples

    if method == "robust_consensus_mc":
        # ROBUST Consensus Monte Carlo (Jan 2026):
        # Uses trimmed statistics to handle heterogeneous shards
        combined_samples: dict[str, np.ndarray] = {}
        rng = np.random.default_rng(42)

        for name in param_names:
            # Compute per-shard posterior mean and variance
            shard_means = []
            shard_variances = []
            for s in shard_samples:
                samples = s.samples[name].flatten()
                shard_means.append(np.mean(samples))
                shard_variances.append(np.var(samples))

            means_arr = np.array(shard_means)
            vars_arr = np.array(shard_variances)

            # Detect outliers using median absolute deviation (MAD)
            # More robust than using standard deviation
            median_mean = np.median(means_arr)
            mad = np.median(np.abs(means_arr - median_mean))
            # Modified Z-score threshold (commonly used: 3.5)
            threshold = 3.5
            if mad > 0:
                modified_z = 0.6745 * np.abs(means_arr - median_mean) / mad
                inlier_mask = modified_z < threshold
            else:
                # If MAD is 0 (all means identical), keep all
                inlier_mask = np.ones(len(means_arr), dtype=bool)

            # Require at least 3 shards for robust statistics
            n_inliers = np.sum(inlier_mask)
            if n_inliers < 3:
                # Fall back to standard CMC if too few inliers
                logger.warning(
                    f"Robust CMC: Only {n_inliers} inliers for {name}, "
                    "falling back to standard combination"
                )
                inlier_mask = np.ones(len(means_arr), dtype=bool)

            # Use only inlier shards for combination
            filtered_means = means_arr[inlier_mask]
            filtered_vars = vars_arr[inlier_mask]

            # Winsorize extreme variances (cap at 5th and 95th percentiles)
            if len(filtered_vars) >= 5:
                var_low, var_high = np.percentile(filtered_vars, [5, 95])
                filtered_vars = np.clip(filtered_vars, var_low, var_high)

            # Precision-weighted combination on filtered data
            precisions = [1.0 / max(v, 1e-10) for v in filtered_vars]
            combined_precision = sum(precisions)
            combined_variance = 1.0 / combined_precision

            weighted_mean_sum = sum(
                p * m for p, m in zip(precisions, filtered_means, strict=False)
            )
            combined_mean = combined_variance * weighted_mean_sum

            # Generate new samples from the combined Gaussian
            combined_std = np.sqrt(combined_variance)
            new_samples = rng.normal(
                loc=combined_mean,
                scale=combined_std,
                size=(n_chains, n_samples),
            )
            combined_samples[name] = new_samples

            # Log if outliers were excluded
            n_excluded = len(means_arr) - n_inliers
            if n_excluded > 0:
                logger.debug(
                    f"Robust CMC: {name} excluded {n_excluded}/{len(means_arr)} "
                    f"outlier shards (MAD-based detection)"
                )

    elif method == "consensus_mc":
        # CORRECT Consensus Monte Carlo (Scott et al., 2016):
        # Combine posterior moments, then generate new samples
        combined_samples = {}
        rng = np.random.default_rng(42)  # Deterministic for reproducibility

        for name in param_names:
            # Compute per-shard posterior mean and variance
            shard_means = []
            shard_variances = []
            for s in shard_samples:
                samples = s.samples[name].flatten()
                shard_means.append(np.mean(samples))
                shard_variances.append(np.var(samples))

            # Precision-weighted combination
            # Combined precision = sum of precisions
            precisions = [1.0 / max(v, 1e-10) for v in shard_variances]
            combined_precision = sum(precisions)
            combined_variance = 1.0 / combined_precision

            # Combined mean = (combined_variance) * sum(precision_s * mean_s)
            weighted_mean_sum = sum(
                p * m for p, m in zip(precisions, shard_means, strict=False)
            )
            combined_mean = combined_variance * weighted_mean_sum

            # Generate new samples from the combined Gaussian
            # Shape: (n_chains, n_samples)
            combined_std = np.sqrt(combined_variance)
            new_samples = rng.normal(
                loc=combined_mean,
                scale=combined_std,
                size=(n_chains, n_samples),
            )
            combined_samples[name] = new_samples

    elif method == "simple_average":
        warnings.warn(
            "combination_method='simple_average' is deprecated since v2.12.0 "
            "and will be removed in v3.0. Use 'consensus_mc' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy: Simple element-wise average across shards (deprecated)
        combined_samples = {}
        for name in param_names:
            all_shard_samples = [s.samples[name] for s in shard_samples]
            combined_samples[name] = np.mean(all_shard_samples, axis=0)

    else:  # weighted_gaussian (legacy default)
        warnings.warn(
            "combination_method='weighted_gaussian' is deprecated since v2.12.0 "
            "and will be removed in v3.0. Use 'consensus_mc' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Legacy: Element-wise weighted averaging (deprecated)
        # WARNING: This is mathematically incorrect but kept for backward compatibility
        combined_samples = {}
        for name in param_names:
            all_shard_samples = [s.samples[name] for s in shard_samples]
            variances = [np.var(s) for s in all_shard_samples]
            precisions = [1.0 / max(v, 1e-10) for v in variances]
            total_precision = sum(precisions)
            weights = [p / total_precision for p in precisions]
            weighted_sum = sum(
                w * s for w, s in zip(weights, all_shard_samples, strict=False)
            )
            combined_samples[name] = weighted_sum

    # Combine extra fields
    combined_extra: dict[str, Any] = {}
    for key in shard_samples[0].extra_fields.keys():
        all_extra = [
            s.extra_fields.get(key) for s in shard_samples if key in s.extra_fields
        ]
        if all_extra:
            try:
                # Handle scalar fields (zero-dimensional arrays)
                if all_extra[0].ndim == 0:
                    # Stack scalars into a 1D array
                    combined_extra[key] = np.stack(all_extra, axis=0)
                else:
                    # Concatenate arrays along the chain dimension (axis=0)
                    combined_extra[key] = np.concatenate(all_extra, axis=0)
            except Exception as e:
                # Fallback for incompatible shapes
                logger.warning(f"Failed to combine extra field '{key}': {e}")
                combined_extra[key] = all_extra[0]

    # Track total shards for correct divergence rate calculation
    total_shards = sum(getattr(s, "num_shards", 1) for s in shard_samples)

    return MCMCSamples(
        samples=combined_samples,
        param_names=param_names,
        n_chains=n_chains,
        n_samples=n_samples,
        extra_fields=combined_extra,
        num_shards=total_shards,
    )


def combine_shard_samples_bimodal(
    shard_samples: list[MCMCSamples],
    cluster_assignments: tuple[list[int], list[int]],
    bimodal_detections: list[dict[str, Any]],
    modal_params: list[str],
    co_occurrence: dict[str, Any],
    method: str = "consensus_mc",
) -> tuple[MCMCSamples, BimodalConsensusResult]:
    """Combine shard samples using mode-aware consensus.

    For bimodal shards, uses per-component GMM statistics instead of
    full-posterior statistics to avoid density-trough corruption.

    Parameters
    ----------
    shard_samples : list[MCMCSamples]
        All successful shard samples.
    cluster_assignments : tuple[list[int], list[int]]
        (lower_cluster_shards, upper_cluster_shards) from cluster_shard_modes().
        Bimodal shards may appear in both lists.
    bimodal_detections : list[dict[str, Any]]
        Per-detection records with "shard", "param", "mode1", "mode2",
        "std1", "std2", "weights".
    modal_params : list[str]
        Parameters that triggered bimodal detection.
    co_occurrence : dict[str, Any]
        Cross-parameter co-occurrence info.
    method : str
        Base combination method for non-modal params.

    Returns
    -------
    tuple[MCMCSamples, BimodalConsensusResult]
        (combined_samples, bimodal_result) where combined_samples has
        mixture-drawn primary samples and bimodal_result has per-mode details.
    """
    import numpy as np

    from homodyne.optimization.cmc.diagnostics import (
        BimodalConsensusResult,
        ModeCluster,
    )
    from homodyne.optimization.cmc.sampler import MCMCSamples

    cluster_lower, cluster_upper = cluster_assignments
    n_total = len(shard_samples)
    param_names = shard_samples[0].param_names
    n_chains = shard_samples[0].n_chains
    n_samples = shard_samples[0].n_samples
    rng = np.random.default_rng(42)

    # Index bimodal detections by (shard, param) for fast lookup
    bimodal_index: dict[tuple[int, str], dict[str, Any]] = {}
    for det in bimodal_detections:
        bimodal_index[(det["shard"], det["param"])] = det

    modal_set = set(modal_params)

    def _consensus_for_cluster(
        cluster_shards: list[int],
        is_lower: bool,
    ) -> dict[str, tuple[float, float]]:
        """Compute consensus (mean, std) for each param in a cluster.

        Returns dict of {param: (combined_mean, combined_std)}.
        """
        result: dict[str, tuple[float, float]] = {}

        for name in param_names:
            shard_means: list[float] = []
            shard_variances: list[float] = []

            for shard_idx in cluster_shards:
                key = (shard_idx, name)
                if name in modal_set and key in bimodal_index:
                    # Bimodal shard + modal param: use component-level stats
                    det = bimodal_index[key]
                    m1, m2 = det["mode1"], det["mode2"]
                    s1, s2 = det["std1"], det["std2"]
                    lo, hi = sorted([(m1, s1), (m2, s2)], key=lambda x: x[0])
                    if is_lower:
                        shard_means.append(lo[0])
                        shard_variances.append(lo[1] ** 2)
                    else:
                        shard_means.append(hi[0])
                        shard_variances.append(hi[1] ** 2)
                else:
                    # Unimodal shard or non-modal param: use full posterior
                    samples = shard_samples[shard_idx].samples[name].flatten()
                    shard_means.append(float(np.mean(samples)))
                    shard_variances.append(float(np.var(samples)))

            if len(shard_means) < 3:
                # Too few shards: use simple mean
                combined_mean = float(np.mean(shard_means)) if shard_means else 0.0
                combined_std = (
                    float(np.std(shard_means)) if len(shard_means) > 1 else 1e-6
                )
            else:
                # Precision-weighted consensus
                precisions = [1.0 / max(v, 1e-10) for v in shard_variances]
                combined_precision = sum(precisions)
                combined_variance = 1.0 / combined_precision
                weighted_mean_sum = sum(
                    p * m for p, m in zip(precisions, shard_means, strict=False)
                )
                combined_mean = combined_variance * weighted_mean_sum
                combined_std = float(np.sqrt(combined_variance))

            result[name] = (float(combined_mean), float(combined_std))

        return result

    # Run per-mode consensus
    lower_stats = _consensus_for_cluster(cluster_lower, is_lower=True)
    upper_stats = _consensus_for_cluster(cluster_upper, is_lower=False)

    # Build mode weights
    # For bimodal shards that appear in both clusters, count them once total
    unique_shards = set(cluster_lower) | set(cluster_upper)
    w_lower = len(cluster_lower) / max(len(unique_shards), 1)
    w_upper = len(cluster_upper) / max(len(unique_shards), 1)
    # Normalize (bimodal shards counted in both lists inflate the sum)
    total_w = w_lower + w_upper
    w_lower /= total_w
    w_upper /= total_w

    # Generate per-mode samples
    n_lower_samples = int(round(w_lower * n_samples))
    n_upper_samples = n_samples - n_lower_samples

    lower_samples: dict[str, np.ndarray] = {}
    upper_samples: dict[str, np.ndarray] = {}
    combined_samples: dict[str, np.ndarray] = {}

    for name in param_names:
        lo_mean, lo_std = lower_stats[name]
        up_mean, up_std = upper_stats[name]

        lower_samples[name] = rng.normal(
            loc=lo_mean,
            scale=max(lo_std, 1e-10),
            size=(n_chains, n_lower_samples),
        )
        upper_samples[name] = rng.normal(
            loc=up_mean,
            scale=max(up_std, 1e-10),
            size=(n_chains, n_upper_samples),
        )
        # Mixture-draw: concatenate and shuffle within each chain
        mixed = np.concatenate(
            [lower_samples[name], upper_samples[name]], axis=1
        )
        for c in range(n_chains):
            rng.shuffle(mixed[c])
        combined_samples[name] = mixed

    # Build ModeCluster objects with independent draws from each mode's consensus
    # Gaussian. These are separate from the mixture-drawn primary samples above;
    # they provide full per-mode sample sets for downstream analysis.
    mode_lower = ModeCluster(
        mean={n: lower_stats[n][0] for n in param_names},
        std={n: lower_stats[n][1] for n in param_names},
        weight=w_lower,
        n_shards=len(cluster_lower),
        samples={
            n: rng.normal(
                loc=lower_stats[n][0],
                scale=max(lower_stats[n][1], 1e-10),
                size=(n_chains, n_samples),
            )
            for n in param_names
        },
    )
    mode_upper = ModeCluster(
        mean={n: upper_stats[n][0] for n in param_names},
        std={n: upper_stats[n][1] for n in param_names},
        weight=w_upper,
        n_shards=len(cluster_upper),
        samples={
            n: rng.normal(
                loc=upper_stats[n][0],
                scale=max(upper_stats[n][1], 1e-10),
                size=(n_chains, n_samples),
            )
            for n in param_names
        },
    )

    bimodal_result = BimodalConsensusResult(
        modes=[mode_lower, mode_upper],
        modal_params=modal_params,
        co_occurrence=co_occurrence,
    )

    # Combine extra fields from all shards
    combined_extra: dict[str, Any] = {}
    for key in shard_samples[0].extra_fields.keys():
        all_extra = [
            s.extra_fields.get(key)
            for s in shard_samples
            if key in s.extra_fields
        ]
        if all_extra:
            try:
                if all_extra[0].ndim == 0:
                    combined_extra[key] = np.stack(all_extra, axis=0)
                else:
                    combined_extra[key] = np.concatenate(all_extra, axis=0)
            except Exception as e:
                logger.warning(f"Failed to combine extra field '{key}': {e}")
                combined_extra[key] = all_extra[0]

    combined = MCMCSamples(
        samples=combined_samples,
        param_names=param_names,
        n_chains=n_chains,
        n_samples=n_samples,
        extra_fields=combined_extra,
        num_shards=n_total,
    )

    return combined, bimodal_result
