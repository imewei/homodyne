"""Consensus Monte Carlo (CMC) for Large-Scale XPCS Datasets

This module implements Consensus Monte Carlo for scalable Bayesian uncertainty
quantification on large XPCS datasets (4M-200M+ points).

Key Components:
- Data sharding: Splits large datasets into manageable chunks (stratified/random/contiguous)
- Parallel MCMC execution: Runs independent MCMC on each shard
- Subposterior combination: Combines shard posteriors into final result

References:
    Scott et al. (2016): "Bayes and big data: the consensus Monte Carlo algorithm"
    https://arxiv.org/abs/1411.7435
"""

from homodyne.optimization.mcmc.cmc.combination import combine_subposteriors
from homodyne.optimization.mcmc.cmc.coordinator import CMCCoordinator
from homodyne.optimization.mcmc.cmc.diagnostics import (
    compute_between_shard_kl_divergence,
    compute_combined_posterior_diagnostics,
    compute_per_shard_diagnostics,
    validate_cmc_results,
)
from homodyne.optimization.mcmc.cmc.result import MCMCResult
from homodyne.optimization.mcmc.cmc.sharding import (
    calculate_optimal_num_shards,
    shard_data_contiguous,
    shard_data_random,
    shard_data_stratified,
    validate_shards,
)

__all__ = [
    # CMC Coordinator (Main Orchestrator)
    "CMCCoordinator",
    # Extended MCMCResult
    "MCMCResult",
    # Data sharding
    "calculate_optimal_num_shards",
    "shard_data_random",
    "shard_data_stratified",
    "shard_data_contiguous",
    "validate_shards",
    # Subposterior combination
    "combine_subposteriors",
    # Diagnostics and validation
    "validate_cmc_results",
    "compute_per_shard_diagnostics",
    "compute_between_shard_kl_divergence",
    "compute_combined_posterior_diagnostics",
]
