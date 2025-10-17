"""Visualization backends for homodyne XPCS analysis.

This module provides high-performance visualization backends for C2 heatmap plotting.
"""

from homodyne.viz.datashader_backend import (
    DatashaderRenderer,
    plot_c2_heatmap_fast,
    plot_c2_comparison_fast,
)

__all__ = [
    "DatashaderRenderer",
    "plot_c2_heatmap_fast",
    "plot_c2_comparison_fast",
]
