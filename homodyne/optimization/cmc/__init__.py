"""CMC (Consensus Monte Carlo) module for XPCS analysis.

This module provides Bayesian MCMC inference for homodyne XPCS
two-time correlation function fitting using NumPyro/NUTS.

Public API:
    fit_mcmc_jax: Main entry point for CMC analysis
    CMCResult: Result dataclass with posterior samples and diagnostics
    CMCConfig: Configuration dataclass for CMC settings

Example:
    from homodyne.optimization.cmc import fit_mcmc_jax

    result = fit_mcmc_jax(
        data=c2_pooled,
        t1=t1_pooled,
        t2=t2_pooled,
        phi=phi_pooled,
        q=wavevector,
        L=gap_length,
        analysis_mode="laminar_flow",
        method="mcmc",
        cmc_config=config.get_cmc_config(),
        initial_values=config.get_initial_parameters(),
        parameter_space=parameter_space,
    )
"""

from homodyne.optimization.cmc.config import CMCConfig
from homodyne.optimization.cmc.core import fit_mcmc_jax
from homodyne.optimization.cmc.results import CMCResult

__all__ = [
    "fit_mcmc_jax",
    "CMCResult",
    "CMCConfig",
]
