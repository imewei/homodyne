"""
Python API for Homodyne v2
==========================

High-level Python API providing convenient programmatic access to
all homodyne analysis capabilities with simplified interfaces.

Main Components:
- high_level.py: High-level analysis functions (run_analysis, fit_data)
- convenience.py: Convenience utilities and shortcuts
- examples.py: Complete usage examples and tutorials

Key Features:
- One-line analysis execution
- Automatic configuration management
- Simplified parameter specification
- Result processing and export
- Integration with Jupyter notebooks
"""

from homodyne.api.convenience import (export_results, load_data, plot_results,
                                      quick_hybrid_fit, quick_mcmc_fit,
                                      quick_vi_fit)
from homodyne.api.high_level import (AnalysisSession, compare_methods,
                                     fit_data, load_and_analyze, run_analysis)

__all__ = [
    # High-level functions
    "run_analysis",
    "fit_data",
    "load_and_analyze",
    "compare_methods",
    "AnalysisSession",
    # Convenience functions
    "quick_vi_fit",
    "quick_mcmc_fit",
    "quick_hybrid_fit",
    "load_data",
    "export_results",
    "plot_results",
]
