from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PerPhiInitConfig:
    percentiles: tuple[float, float, float] = (5.0, 50.0, 95.0)
    contrast_bounds: tuple[float, float] = (0.01, 0.8)
    offset_bounds: tuple[float, float] = (0.8, 1.2)
    atol: float = 1e-6
    seed_base: int = 0


def _finite_percentile(arr: np.ndarray, pct: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.percentile(finite, pct))


def _angle_hash(angle: float) -> int:
    # Stable hash to derive deterministic seeds per phi
    return int(np.uint64(np.round(angle * 1e6)) % np.uint64(2**31 - 1))


def build_per_phi_initial_values(
    phi: np.ndarray,
    g2: np.ndarray | None,
    config_per_phi: dict[str, dict[str, Any]] | None,
    cfg: PerPhiInitConfig,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Create per-phi initial values with config-first then percentile fallback.

    Parameters
    ----------
    phi : np.ndarray
        Flattened phi array aligned with g2 data.
    g2 : np.ndarray | None
        Correlation data used for percentile-based fallback (finite values only).
    config_per_phi : dict | None
        Optional mapping of phi angle string → {contrast, offset} provided by config.
    cfg : PerPhiInitConfig
        Bounds, percentiles, and seed base settings.

    Returns
    -------
    (init_values, metadata)
        init_values: dict with contrast_i/offset_i keys ordered by sorted unique phi.
        metadata: diagnostics including source per phi and deterministic seeds.
    """

    phi = np.asarray(phi, dtype=float).ravel()
    phi_unique = np.unique(phi)
    provided = config_per_phi or {}
    init_values: dict[str, float] = {}
    metadata: dict[str, Any] = {
        "phi_unique": phi_unique.tolist(),
        "seed_base": cfg.seed_base,
        "source": {},
        "missing_phi": [],
        "seeds": {},
    }

    for idx, angle in enumerate(phi_unique):
        provided_match = None
        for raw_angle_str, vals in provided.items():
            try:
                raw_angle = float(raw_angle_str)
            except (TypeError, ValueError):
                continue
            if np.isclose(raw_angle, angle, atol=cfg.atol):
                provided_match = vals
                break

        if provided_match:
            contrast = float(provided_match.get("contrast", np.nan))
            offset = float(provided_match.get("offset", np.nan))
            source = "config"
        else:
            if g2 is None:
                contrast = np.nan
                offset = np.nan
            else:
                mask = np.isclose(phi, angle, atol=cfg.atol)
                g2_slice = np.asarray(g2)[mask]
                p_low, p_mid, p_high = cfg.percentiles
                p1 = _finite_percentile(g2_slice, p_low)
                p50 = _finite_percentile(g2_slice, p_mid)
                p99 = _finite_percentile(g2_slice, p_high)
                span = np.nan if np.isnan(p1) or np.isnan(p99) else p99 - p1
                contrast = span if span == span else 0.1
                offset = p50 if p50 == p50 else 1.0
            source = "percentile"

        contrast = np.clip(contrast, *cfg.contrast_bounds)
        offset = np.clip(offset, *cfg.offset_bounds)
        init_values[f"contrast_{idx}"] = float(contrast)
        init_values[f"offset_{idx}"] = float(offset)
        metadata["source"][f"phi_{idx}"] = source
        metadata["seeds"][f"phi_{idx}"] = int(cfg.seed_base + _angle_hash(angle))

    if provided:
        missing = []
        for raw_angle_str in provided:
            try:
                raw_angle = float(raw_angle_str)
            except (TypeError, ValueError):
                continue
            if not np.any(np.isclose(phi_unique, raw_angle, atol=cfg.atol)):
                missing.append(raw_angle_str)
        if missing:
            metadata["missing_phi"] = missing
            logger.warning(
                "Per-phi config provided for angles not in data: %s", missing
            )

    return init_values, metadata
