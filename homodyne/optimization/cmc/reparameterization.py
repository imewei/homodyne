# homodyne/optimization/cmc/reparameterization.py
"""Internal reparameterization for CMC sampling.

Transforms correlated parameters to orthogonal sampling space,
then converts back to physics parameters for output.

Reparameterizations:
- D0, D_offset → D_total, D_offset_frac (breaks linear degeneracy)
- gamma_dot_t0 → log_gamma_dot_t0 (improves conditioning)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReparamConfig:
    """Configuration for internal reparameterization.

    Attributes
    ----------
    enable_d_total : bool
        Enable D_total reparameterization (D0 + D_offset → D_total).
    enable_log_gamma : bool
        Sample log(gamma_dot_t0) instead of gamma_dot_t0.
    t_ref : float
        Reference time for shear scaling (not currently used).
    """

    enable_d_total: bool = True
    enable_log_gamma: bool = True
    t_ref: float = 1.0
