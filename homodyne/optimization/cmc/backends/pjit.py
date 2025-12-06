"""JAX pjit backend for CMC distributed execution.

This module provides distributed MCMC execution using JAX's pjit
for sharded computation across CPU devices.

Note: This is a CPU-only implementation per v2.3.0 architecture decision.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from homodyne.optimization.cmc.backends.base import CMCBackend, combine_shard_samples
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)


class PjitBackend(CMCBackend):
    """JAX pjit backend for distributed MCMC execution.

    Uses JAX's pjit for parallel execution across CPU devices.
    This backend is suitable for multi-core CPU systems where
    JAX can leverage multiple devices.

    Note
    ----
    CPU-only per homodyne v2.3.0 architecture decision.
    For GPU support, use homodyne v2.2.1 or earlier.
    """

    def __init__(self) -> None:
        """Initialize pjit backend."""
        self._validate_jax_devices()

    def _validate_jax_devices(self) -> None:
        """Validate JAX device configuration."""
        devices = jax.devices()
        n_devices = len(devices)
        logger.info(f"PjitBackend: Found {n_devices} JAX devices")

        if n_devices < 2:
            logger.warning(
                "PjitBackend: Only 1 device available. "
                "Consider using multiprocessing backend for better parallelism."
            )

    def get_name(self) -> str:
        """Get backend name.

        Returns
        -------
        str
            Backend identifier.
        """
        return "pjit"

    def is_available(self) -> bool:
        """Check if pjit backend is available.

        Returns
        -------
        bool
            True if JAX pjit can be used.
        """
        try:
            # Check that JAX is properly configured
            _ = jax.devices()
            return True
        except Exception:
            return False

    def run(
        self,
        model: Callable,
        model_kwargs: dict[str, Any],
        config: CMCConfig,
        shards: list[PreparedData] | None = None,
    ) -> MCMCSamples:
        """Run MCMC sampling using pjit for parallelism.

        Parameters
        ----------
        model : Callable
            NumPyro model function.
        model_kwargs : dict[str, Any]
            Common model arguments (q, L, dt, etc.).
        config : CMCConfig
            CMC configuration.
        shards : list[PreparedData] | None
            Data shards for parallel execution.
            If None, runs on full data without sharding.

        Returns
        -------
        MCMCSamples
            Combined samples from all shards.
        """
        from homodyne.optimization.cmc.sampler import run_nuts_sampling

        start_time = time.time()

        if shards is None or len(shards) <= 1:
            # No sharding - run single MCMC
            logger.info("PjitBackend: Running single MCMC (no sharding)")
            prepared_data = shards[0] if shards else model_kwargs.get("prepared_data")

            if prepared_data is None:
                raise ValueError("No data provided for MCMC sampling")

            # Run NUTS sampling
            rng_key = jax.random.PRNGKey(config.seed if hasattr(config, "seed") else 0)
            samples, stats = run_nuts_sampling(
                model=model,
                model_kwargs={
                    **model_kwargs,
                    "data": jnp.array(prepared_data.data),
                    "t1": jnp.array(prepared_data.t1),
                    "t2": jnp.array(prepared_data.t2),
                    "phi": jnp.array(prepared_data.phi),
                    "phi_indices": jnp.array(prepared_data.phi_indices),
                },
                config=config,
                initial_values=model_kwargs.get("initial_values"),
                rng_key=rng_key,
            )

            return samples

        # Multiple shards - run in parallel using pjit
        logger.info(f"PjitBackend: Running on {len(shards)} shards")

        shard_results: list[MCMCSamples] = []
        devices = jax.devices()
        n_devices = len(devices)

        for i, shard in enumerate(shards):
            device_idx = i % n_devices
            logger.debug(f"Processing shard {i + 1}/{len(shards)} on device {device_idx}")

            # Place data on specific device
            with jax.default_device(devices[device_idx]):
                rng_key = jax.random.PRNGKey(
                    (config.seed if hasattr(config, "seed") else 0) + i
                )

                samples, stats = run_nuts_sampling(
                    model=model,
                    model_kwargs={
                        **model_kwargs,
                        "data": jnp.array(shard.data),
                        "t1": jnp.array(shard.t1),
                        "t2": jnp.array(shard.t2),
                        "phi": jnp.array(shard.phi),
                        "phi_indices": jnp.array(shard.phi_indices),
                    },
                    config=config,
                    initial_values=model_kwargs.get("initial_values"),
                    rng_key=rng_key,
                )

                shard_results.append(samples)

        # Combine results from all shards
        combined = combine_shard_samples(
            shard_results,
            method=config.combination_method
            if hasattr(config, "combination_method")
            else "weighted_gaussian",
        )

        elapsed = time.time() - start_time
        logger.info(f"PjitBackend: Completed in {elapsed:.1f}s")

        return combined


def create_sharded_model(
    model: Callable,
    n_shards: int,
) -> Callable:
    """Create a sharded version of the model for pjit.

    Parameters
    ----------
    model : Callable
        Original NumPyro model.
    n_shards : int
        Number of shards.

    Returns
    -------
    Callable
        Sharded model function.

    Note
    ----
    This is a placeholder for future pjit optimization.
    Currently, shards are processed sequentially with device placement.
    """
    # For now, return the original model
    # Future: implement true pjit sharding with jax.experimental.pjit
    return model
