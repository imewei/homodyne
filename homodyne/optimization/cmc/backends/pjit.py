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
        except (RuntimeError, OSError):
            return False

    def run(
        self,
        model: Callable,
        model_kwargs: dict[str, Any],
        config: CMCConfig,
        shards: list[PreparedData] | None = None,
        *,
        initial_values: dict[str, float] | None = None,
        parameter_space: Any | None = None,
        analysis_mode: str | None = None,
        progress_bar: bool = True,
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

        Notes
        -----
        Additional keyword arguments are accepted for signature compatibility
        with other backends (multiprocessing). They are currently unused but
        harmless, ensuring legacy calls with initial_values/parameter_space
        do not fail.

        Returns
        -------
        MCMCSamples
            Combined samples from all shards.
        """
        from homodyne.optimization.cmc.sampler import run_nuts_sampling

        # P1-R5-03: Fail explicitly when analysis_mode is None rather than
        # silently defaulting to "laminar_flow" which uses the wrong physics model
        # for static datasets (7 physical params instead of 3).
        if analysis_mode is None:
            raise ValueError(
                "analysis_mode must be explicitly set ('static' or 'laminar_flow'); "
                "got None. Pass analysis_mode to PjitBackend.run()."
            )

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
                    "phi_unique": jnp.array(prepared_data.phi_unique),
                    "phi_indices": jnp.array(prepared_data.phi_indices),
                    "q": model_kwargs.get("q"),
                    "L": model_kwargs.get("L"),
                    "dt": model_kwargs.get("dt"),
                    "time_grid": (
                        jnp.array(model_kwargs.get("time_grid"))
                        if model_kwargs.get("time_grid") is not None
                        else None
                    ),
                    "analysis_mode": analysis_mode,
                    "parameter_space": parameter_space,
                    "n_phi": prepared_data.n_phi,
                    "noise_scale": model_kwargs.get("noise_scale", 0.1),
                },
                config=config,
                initial_values=initial_values,
                parameter_space=parameter_space,
                n_phi=prepared_data.n_phi,
                analysis_mode=analysis_mode,
                rng_key=rng_key,
                progress_bar=progress_bar,
            )

            return samples

        # Multiple shards - run in parallel using pjit
        logger.info(f"PjitBackend: Running on {len(shards)} shards")

        shard_results: list[MCMCSamples] = []
        devices = jax.devices()
        n_devices = len(devices)

        for i, shard in enumerate(shards):
            device_idx = i % n_devices
            logger.debug(
                f"Processing shard {i + 1}/{len(shards)} on device {device_idx}"
            )

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
                        "phi_unique": jnp.array(shard.phi_unique),
                        "phi_indices": jnp.array(shard.phi_indices),
                        "q": model_kwargs.get("q"),
                        "L": model_kwargs.get("L"),
                        "dt": model_kwargs.get("dt"),
                        "time_grid": (
                            jnp.array(model_kwargs.get("time_grid"))
                            if model_kwargs.get("time_grid") is not None
                            else None
                        ),
                        "analysis_mode": analysis_mode,
                        "parameter_space": parameter_space,
                        "n_phi": shard.n_phi,
                        "noise_scale": model_kwargs.get("noise_scale", 0.1),
                    },
                    config=config,
                    initial_values=initial_values,
                    parameter_space=parameter_space,
                    n_phi=shard.n_phi,
                    analysis_mode=analysis_mode,
                    rng_key=rng_key,
                    progress_bar=progress_bar,
                )

                shard_results.append(samples)

        # Combine results from all shards
        # P2-R6-01: Use config.combination_method directly; CMCConfig always
        # has this field (defaults to "robust_consensus_mc"). The stale
        # "weighted_gaussian" fallback was misleading and incorrect.
        combined = combine_shard_samples(
            shard_results,
            method=config.combination_method,
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
