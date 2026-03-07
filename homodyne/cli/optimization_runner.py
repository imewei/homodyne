"""Optimization runner for Homodyne CLI.

Handles NLSQ optimization dispatch and NLSQ warm-start resolution.

Note: _run_optimization and _generate_cmc_diagnostic_plots remain in
commands.py because tests mock-patch names in the commands module namespace
(e.g. @patch("homodyne.cli.commands.fit_mcmc_jax")).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from homodyne.config.types import (
    LAMINAR_FLOW_PARAM_NAMES,
    STATIC_PARAM_NAMES,
)
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Import core modules with fallback
try:
    from homodyne.config.manager import ConfigManager
    from homodyne.optimization import fit_nlsq_jax

    HAS_CORE_MODULES = True
except ImportError:
    HAS_CORE_MODULES = False

# Reduced chi-squared threshold for accepting NLSQ results as CMC warm-start.
# Generous threshold (100.0) catches catastrophic failures (e.g. CMA-ES not
# converging, chi-squared = 1e8+) while accepting reasonable fits (e.g. chi-squared ~ 33).
_CMC_WARMSTART_CHI2_THRESHOLD = 100.0


def load_nlsq_result_from_file(nlsq_result_path: Path) -> dict[str, Any] | None:
    """Load pre-computed NLSQ results from a previous hm-nlsq run.

    This allows CMC to use warm-start values from a previous NLSQ analysis
    without re-running NLSQ inline. The recommended workflow is:

    1. Run: homodyne --method nlsq --config config.yaml --output-dir results/
    2. Run: homodyne --method cmc --config config.yaml --nlsq-result results/

    Parameters
    ----------
    nlsq_result_path : Path
        Path to NLSQ results directory (should contain nlsq/parameters.json)
        or directly to a parameters.json file.

    Returns
    -------
    dict or None
        Dict with 'params' and 'uncertainties' keys containing parameter values,
        or None if loading failed.
    """
    # Resolve path to parameters.json
    if nlsq_result_path.is_dir():
        # Check for nlsq/parameters.json (standard output structure)
        params_file = nlsq_result_path / "nlsq" / "parameters.json"
        if not params_file.exists():
            # Fall back to parameters.json directly in the directory
            params_file = nlsq_result_path / "parameters.json"
    else:
        params_file = nlsq_result_path

    if not params_file.exists():
        logger.warning(
            f"NLSQ result file not found: {params_file}. "
            "Expected nlsq/parameters.json in the specified directory."
        )
        return None

    try:
        with open(params_file, encoding="utf-8") as f:
            data = json.load(f)

        # Extract parameter values and uncertainties from the nested structure
        # parameters.json format: {"parameters": {"D0": {"value": ..., "uncertainty": ...}, ...}}
        raw_params = data.get("parameters", {})

        params = {}
        uncertainties = {}

        for name, param_data in raw_params.items():
            if isinstance(param_data, dict) and "value" in param_data:
                params[name] = float(param_data["value"])
                if "uncertainty" in param_data:
                    uncertainties[name] = float(param_data["uncertainty"])
            elif isinstance(param_data, (int, float)):
                # Handle flat structure if present
                params[name] = float(param_data)

        if not params:
            logger.warning(f"No parameters found in {params_file}")
            return None

        # Build result dict compatible with extract_nlsq_values_for_cmc
        result = {
            "params": params,
            "uncertainties": uncertainties if uncertainties else None,
            "chi_squared": data.get("chi_squared"),
            "reduced_chi_squared": data.get("reduced_chi_squared"),
            "convergence_status": data.get("convergence_status"),
            "analysis_mode": data.get("analysis_mode"),
            "source_file": str(params_file),
        }

        logger.info(f"Loaded NLSQ results from {params_file}")
        _rchi2 = result["reduced_chi_squared"]
        _rchi2_str = f"{_rchi2:.2f}" if _rchi2 is not None else "N/A"
        logger.info(
            f"  Convergence: {result['convergence_status']}, reduced chi2 = {_rchi2_str}"
        )

        # Log physical parameters for diagnostics
        _log_warmstart_physical_params(params)

        return result

    except (OSError, json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Failed to load NLSQ results from {params_file}: {e}")
        return None


def _get_warmstart_reduced_chi2(nlsq_result: Any) -> float:
    """Extract reduced chi-squared from an NLSQ result (dict or object)."""
    if isinstance(nlsq_result, dict):
        return float(nlsq_result.get("reduced_chi_squared", float("inf")))
    return float(getattr(nlsq_result, "reduced_chi_squared", float("inf")))


def _log_warmstart_physical_params(params: Any) -> None:
    """Log physical parameter values from an NLSQ warm-start result.

    Accepts either a dict of {name: value} or a parameter array (ndarray).
    """
    if isinstance(params, dict):
        # Dict from load_nlsq_result_from_file
        all_physical = list(LAMINAR_FLOW_PARAM_NAMES)
        physical_vals = [
            f"{name}={params[name]:.4g}" for name in all_physical if name in params
        ]
        if physical_vals:
            logger.info(f"  Physical params: {', '.join(physical_vals)}")
    elif hasattr(params, "__len__"):
        # Array from inline NLSQ OptimizationResult
        n_params = len(params)
        n_laminar = len(LAMINAR_FLOW_PARAM_NAMES)
        n_static = len(STATIC_PARAM_NAMES)

        if n_params >= n_laminar + 2:
            physical_names = list(LAMINAR_FLOW_PARAM_NAMES)
            physical_start = n_params - n_laminar
        else:
            physical_names = list(STATIC_PARAM_NAMES)
            physical_start = n_params - n_static

        physical_vals = params[physical_start:]
        param_str = ", ".join(
            f"{name}={val:.4g}"
            for name, val in zip(physical_names, physical_vals, strict=False)
        )
        logger.info(f"  Physical params: {param_str}")


def _validate_warmstart_quality(nlsq_result: Any, source: str) -> bool:
    """Validate NLSQ result quality for CMC warm-start.

    Parameters
    ----------
    nlsq_result : dict or OptimizationResult
        The NLSQ result to validate.
    source : str
        Human-readable label for log messages (e.g. "pre-computed", "inline").

    Returns
    -------
    bool
        True if result passes quality threshold, False otherwise.
    """
    reduced_chi2 = _get_warmstart_reduced_chi2(nlsq_result)

    if reduced_chi2 > _CMC_WARMSTART_CHI2_THRESHOLD:
        logger.warning(
            f"NLSQ warm-start ({source}) has poor fit quality "
            f"(reduced chi2 = {reduced_chi2:.2f} > threshold {_CMC_WARMSTART_CHI2_THRESHOLD}). "
            "Falling back to config initial values."
        )
        return False

    logger.info(
        f"NLSQ warm-start ({source}) accepted (reduced chi2 = {reduced_chi2:.2f}). "
        "Using NLSQ estimates as CMC initial values."
    )
    return True


def _resolve_nlsq_warmstart(
    args: argparse.Namespace,
    filtered_data: dict[str, Any],
    config: ConfigManager,
) -> Any | None:
    """Resolve NLSQ warm-start for CMC from all possible sources.

    Priority order:
    1. --nlsq-result <path>: Load from pre-computed hm-nlsq pipeline output
    2. Inline NLSQ: Run local trust-region optimization
    3. None: Fall back to config initial values

    Returns
    -------
    dict, OptimizationResult, or None
        The NLSQ result for warm-start, or None if unavailable.
    """
    skip_warmstart = getattr(args, "no_nlsq_warmstart", False)
    nlsq_result_path = getattr(args, "nlsq_result", None)

    if skip_warmstart:
        logger.warning(
            "NLSQ warm-start disabled (--no-nlsq-warmstart). "
            "CMC may have higher divergence rates without warm-start."
        )
        return None

    # Priority 1: Load from pre-computed NLSQ results (RECOMMENDED)
    if nlsq_result_path is not None:
        logger.info(f"Loading NLSQ warm-start from: {nlsq_result_path}")
        nlsq_result = load_nlsq_result_from_file(nlsq_result_path)

        if nlsq_result is not None:
            if _validate_warmstart_quality(nlsq_result, "pre-computed"):
                return nlsq_result
            return None  # Quality check failed

        logger.warning(
            f"Failed to load NLSQ results from {nlsq_result_path}. "
            "Falling back to inline NLSQ optimization."
        )

    # Priority 2: Run inline NLSQ
    logger.info("Running NLSQ optimization for CMC warm-start...")
    try:
        nlsq_result = _run_nlsq_optimization(
            filtered_data, config, args, force_local=True
        )
        if _validate_warmstart_quality(nlsq_result, "inline"):
            # Log physical parameters from inline result
            if (
                hasattr(nlsq_result, "parameters")
                and nlsq_result.parameters is not None
            ):
                _log_warmstart_physical_params(nlsq_result.parameters)
            return nlsq_result
        return None
    except Exception as e:
        logger.warning(
            f"NLSQ warm-start failed: {e}. Proceeding with CMC without warm-start."
        )
        return None


def _run_nlsq_optimization(
    filtered_data: dict[str, Any],
    config: ConfigManager,
    args: argparse.Namespace,
    force_local: bool = False,
) -> Any:
    """Run NLSQ optimization via unified entry point.

    This function always calls fit_nlsq_jax, which handles global optimization
    selection internally with the following priority:
      1. CMA-ES (if enabled and available) - for multi-scale problems
      2. Multi-start (if enabled) - for exploring parameter space
      3. Local optimization - standard trust-region method

    Args:
        filtered_data: Preprocessed experimental data
        config: Configuration manager
        args: CLI arguments
        force_local: If True, bypass CMA-ES/multi-start and use local optimization.
            This is used for CMC warm-start where a reliable point estimate is
            needed quickly, rather than global exploration.

    Returns:
        Optimization result
    """
    # Always use fit_nlsq_jax as the unified entry point
    # It handles global optimization selection: CMA-ES -> Multi-start -> Local
    # For CMC warm-start, we bypass global optimization to get a reliable
    # local optimum quickly (force_local=True skips CMA-ES/multi-start)
    result = fit_nlsq_jax(filtered_data, config, _skip_global_selection=force_local)

    return result
