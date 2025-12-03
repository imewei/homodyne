"""I/O operations for homodyne XPCS analysis.

This module provides functions for saving and loading optimization results,
experimental data, and analysis outputs.
"""

from homodyne.io.json_utils import json_safe, json_serializer
from homodyne.io.mcmc_writers import (
    create_mcmc_analysis_dict,
    create_mcmc_diagnostics_dict,
    create_mcmc_parameters_dict,
)
from homodyne.io.nlsq_writers import (
    save_nlsq_json_files,
    save_nlsq_npz_file,
)

__all__ = [
    # NLSQ result writers
    "save_nlsq_json_files",
    "save_nlsq_npz_file",
    # MCMC result writers
    "create_mcmc_parameters_dict",
    "create_mcmc_analysis_dict",
    "create_mcmc_diagnostics_dict",
    # JSON utilities
    "json_safe",
    "json_serializer",
]
