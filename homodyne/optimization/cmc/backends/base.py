"""Base class for CMC execution backends.

This module defines the abstract interface for CMC backends
and provides a factory function for selecting backends.
"""

from __future__ import annotations

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
) -> MCMCSamples:
    """Combine samples from multiple shards.

    Parameters
    ----------
    shard_samples : list[MCMCSamples]
        Samples from each shard.
    method : str
        Combination method: "weighted_gaussian" or "simple_average".

    Returns
    -------
    MCMCSamples
        Combined samples.
    """
    import numpy as np

    from homodyne.optimization.cmc.sampler import MCMCSamples

    if len(shard_samples) == 1:
        return shard_samples[0]

    # Get parameter names from first shard
    param_names = shard_samples[0].param_names
    n_chains = shard_samples[0].n_chains
    n_samples = shard_samples[0].n_samples

    if method == "simple_average":
        # Simple average across shards
        combined_samples: dict[str, np.ndarray] = {}
        for name in param_names:
            all_shard_samples = [s.samples[name] for s in shard_samples]
            # Average the posterior means
            combined_samples[name] = np.mean(all_shard_samples, axis=0)

    else:  # weighted_gaussian
        # Weighted combination based on precision (1/variance)
        combined_samples = {}
        for name in param_names:
            # Get samples from all shards
            all_shard_samples = [s.samples[name] for s in shard_samples]

            # Compute weights based on inverse variance
            variances = [np.var(s) for s in all_shard_samples]
            precisions = [1.0 / max(v, 1e-10) for v in variances]
            total_precision = sum(precisions)
            weights = [p / total_precision for p in precisions]

            # Weighted average
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

    return MCMCSamples(
        samples=combined_samples,
        param_names=param_names,
        n_chains=n_chains,
        n_samples=n_samples,
        extra_fields=combined_extra,
    )
