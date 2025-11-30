"""CMC bypass heuristics for low-parallelism scenarios.

This module centralizes the decision logic that determines when Consensus
Monte Carlo should be bypassed in favour of the hardened single-chain NUTS
implementation.  It is primarily aimed at regimes where running CMC adds
overhead without delivering additional convergence guarantees – e.g.,
single-angle datasets, tiny shards, or explicit user overrides.

The public CLI does not expose new flags for this behaviour.  Instead, the
decision surface reuses the existing `optimization.cmc` configuration tree and
can be overridden through two optional keys:

```
optimization:
  cmc:
    bypass_mode: auto        # "auto" | "force_nuts" | "force_cmc"
    bypass_thresholds:
      min_phi_angles: 2
      min_samples_for_cmc: 8
      min_dataset_size: 100000
      min_expected_shards: 2
```

When `bypass_mode` is left as `auto` (default), the heuristics below evaluate
observed dataset characteristics and trigger a bypass if any guard is hit.
Strict overrides (`force_nuts`, `force_cmc`) are respected regardless of the
heuristics.  The calling site is responsible for executing the fallback NUTS
path once a bypass is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

BYPASS_MODES = {"auto", "force_nuts", "force_cmc"}

DEFAULT_THRESHOLDS = {
    "min_phi_angles": 2,
    "min_samples_for_cmc": 8,
    "min_dataset_size": 100_000,
    "min_expected_shards": 2,
}


@dataclass
class CMCBypassDecision:
    """Represents the result of the bypass heuristic evaluation."""

    should_bypass: bool
    reason: Optional[str] = None
    triggered_by: Optional[str] = None
    mode: str = "auto"


def evaluate_cmc_bypass(
    cmc_config: Dict[str, Any],
    *,
    num_samples: int,
    dataset_size: int,
    phi: Optional[Sequence[float]] = None,
) -> CMCBypassDecision:
    """Determine whether the current dataset should bypass CMC.

    Parameters
    ----------
    cmc_config : dict
        Validated CMC configuration dictionary (from ConfigManager).
    num_samples : int
        Number of independent samples (usually unique phi angles).
    dataset_size : int
        Total number of observed data points.
    phi : sequence, optional
        Raw phi measurements.  Used to sanity-check single-angle datasets when
        available.
    """

    cfg_mode = cmc_config.get("bypass_mode") or cmc_config.get("bypass", {}).get("mode")
    mode = (cfg_mode or "auto").lower()
    if mode not in BYPASS_MODES:
        mode = "auto"

    thresholds = DEFAULT_THRESHOLDS.copy()
    user_thresholds = cmc_config.get("bypass_thresholds") or cmc_config.get("bypass", {}).get(
        "thresholds"
    )
    if isinstance(user_thresholds, dict):
        for key, value in user_thresholds.items():
            if key in thresholds and isinstance(value, (int, float)) and value > 0:
                thresholds[key] = int(value)

    phi_count = _count_unique_phi(phi, fallback=num_samples)
    expected_shards = _estimate_expected_shards(cmc_config, dataset_size)

    if mode == "force_nuts":
        return CMCBypassDecision(
            should_bypass=True,
            reason="CMC forced off via configuration (bypass_mode=force_nuts)",
            triggered_by="config_override",
            mode=mode,
        )

    if mode == "force_cmc":
        return CMCBypassDecision(should_bypass=False, reason=None, mode=mode)

    # Automatic heuristics (default)
    if phi_count < thresholds["min_phi_angles"]:
        return CMCBypassDecision(
            should_bypass=True,
            reason=(
                f"Only {phi_count} unique phi angle(s) present (< {thresholds['min_phi_angles']}), "
                "no parallelism benefit from CMC"
            ),
            triggered_by="single_angle",
            mode=mode,
        )

    if num_samples < thresholds["min_samples_for_cmc"]:
        return CMCBypassDecision(
            should_bypass=True,
            reason=(
                f"Only {num_samples} independent samples (< {thresholds['min_samples_for_cmc']}), "
                "jittered NUTS chains are more reliable"
            ),
            triggered_by="insufficient_samples",
            mode=mode,
        )

    if dataset_size < thresholds["min_dataset_size"]:
        return CMCBypassDecision(
            should_bypass=True,
            reason=(
                f"Dataset has {dataset_size:,} points (< {thresholds['min_dataset_size']:,}), "
                "CMC sharding would create under-filled shards"
            ),
            triggered_by="small_dataset",
            mode=mode,
        )

    if expected_shards < thresholds["min_expected_shards"]:
        return CMCBypassDecision(
            should_bypass=True,
            reason=(
                f"Expected shard count {expected_shards} < {thresholds['min_expected_shards']}, "
                "insufficient parallelism for CMC"
            ),
            triggered_by="low_parallelism",
            mode=mode,
        )

    return CMCBypassDecision(should_bypass=False, mode=mode)


def _count_unique_phi(phi: Optional[Sequence[float]], fallback: int) -> int:
    if phi is None:
        return fallback

    try:
        phi_array = np.asarray(phi)
        if phi_array.size == 0:
            return fallback
        return int(np.unique(phi_array).size)
    except Exception:
        return fallback


def _estimate_expected_shards(cmc_config: Dict[str, Any], dataset_size: int) -> int:
    sharding_cfg = cmc_config.get("sharding", {})
    num_shards_cfg = sharding_cfg.get("num_shards")
    if isinstance(num_shards_cfg, int) and num_shards_cfg > 0:
        return num_shards_cfg

    max_points = sharding_cfg.get("max_points_per_shard")
    if isinstance(max_points, int) and max_points > 0:
        return max(1, int(np.ceil(dataset_size / max_points)))

    # Fall back to a coarse heuristic: aim for ~200k points per shard
    heuristic_shard_size = 200_000
    estimated = int(np.ceil(dataset_size / heuristic_shard_size)) if dataset_size else 1
    return max(1, estimated)
