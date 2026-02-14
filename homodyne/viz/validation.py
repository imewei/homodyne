"""Pre-plot data validation utilities.

Lightweight checks for NaN/Inf in arrays before rendering.
Logs warnings instead of raising, so plots degrade gracefully.
"""

from __future__ import annotations

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def validate_plot_arrays(
    *arrays: np.ndarray,
    names: list[str] | None = None,
) -> bool:
    """Check arrays for NaN/Inf values before plotting.

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to validate.
    names : list[str] | None
        Optional names for each array (for log messages).

    Returns
    -------
    bool
        True if all arrays are clean, False if any contain NaN/Inf.
    """
    all_clean = True
    for i, arr in enumerate(arrays):
        label = names[i] if names and i < len(names) else f"array[{i}]"
        arr_np = np.asarray(arr)
        if arr_np.size == 0:
            continue
        nan_count = int(np.sum(np.isnan(arr_np)))
        inf_count = int(np.sum(np.isinf(arr_np)))
        if nan_count > 0 or inf_count > 0:
            logger.warning(
                f"Plot data '{label}' contains "
                f"{nan_count} NaN and {inf_count} Inf values "
                f"(shape={arr_np.shape}). Plot may be incomplete."
            )
            all_clean = False
    return all_clean
