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
            "CMC backend 'jax' is deprecated; mapping to 'multiprocessing' for parallel execution. "
            "Set backend_config.name to 'multiprocessing' or 'auto' instead."
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
        Combination method: "consensus_mc" (recommended), "weighted_gaussian", "simple_average", or "auto".
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
            f"Hierarchical combination: {len(shard_samples)} shards in chunks of {chunk_size}"
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

    The legacy methods do element-wise averaging of sample arrays, which is
    mathematically incorrect as sample indices have no correspondence across
    shards.
    """
    import numpy as np

    from homodyne.optimization.cmc.sampler import MCMCSamples

    if len(shard_samples) == 1:
        return shard_samples[0]

    # Get parameter names from first shard
    param_names = shard_samples[0].param_names
    n_chains = shard_samples[0].n_chains
    n_samples = shard_samples[0].n_samples

    if method == "consensus_mc":
        # CORRECT Consensus Monte Carlo (Scott et al., 2016):
        # Combine posterior moments, then generate new samples
        combined_samples: dict[str, np.ndarray] = {}
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
            combined_extra[key] = np.concatenate(all_extra, axis=0)

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
