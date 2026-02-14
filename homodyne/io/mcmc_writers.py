"""MCMC result saving functions for homodyne XPCS analysis.

This module provides functions for creating dictionaries to save MCMC/CMC
optimization results to disk.
Extracted from cli/commands.py for better modularity.
"""

from datetime import datetime
from typing import Any, Literal

import numpy as np

from homodyne.config.parameter_names import get_physical_param_names
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def _get_parameter_names(analysis_mode: str) -> list[str]:
    """Get physical parameter names for given analysis mode.

    This is a thin wrapper around get_physical_param_names() that handles
    unknown modes gracefully with a warning instead of raising an exception.

    Parameters
    ----------
    analysis_mode : str
        Analysis mode ("static" or "laminar_flow")

    Returns
    -------
    list[str]
        List of physical parameter names (without contrast/offset)
    """
    mode: Literal["static_isotropic", "laminar_flow"]
    if analysis_mode == "static":
        mode = "static_isotropic"
    elif analysis_mode == "laminar_flow":
        mode = "laminar_flow"
    else:
        logger.warning(f"Unknown analysis mode: {analysis_mode}, assuming static")
        mode = "static_isotropic"

    return get_physical_param_names(mode)


def create_mcmc_parameters_dict(result: Any) -> dict:
    """Create parameters dictionary with posterior statistics.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with posterior samples and statistics

    Returns
    -------
    dict
        Structured parameter dictionary with posterior mean ± std
    """
    diag_summary = getattr(result, "diagnostic_summary", {}) or {}
    deterministic_params = set(diag_summary.get("deterministic_params") or [])

    param_dict = {
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": getattr(result, "analysis_mode", "unknown"),
        "method": (
            "cmc"
            if (hasattr(result, "is_cmc_result") and result.is_cmc_result())
            else "mcmc"
        ),
        "sampling_summary": {
            "n_samples": getattr(result, "n_samples", 0),
            "n_warmup": getattr(result, "n_warmup", 0),
            "n_chains": getattr(result, "n_chains", 1),
            "total_samples": getattr(result, "n_samples", 0)
            * getattr(result, "n_chains", 1),
            "computation_time": getattr(result, "computation_time", 0.0),
        },
        "convergence": {},
        "parameters": {},
    }

    # Add convergence diagnostics if available
    if hasattr(result, "r_hat") and result.r_hat is not None:
        if isinstance(result.r_hat, dict):
            r_hat_values = [
                v
                for name, v in result.r_hat.items()
                if v is not None and name not in deterministic_params
            ]
            if r_hat_values:
                convergence_dict = param_dict["convergence"]
                assert isinstance(convergence_dict, dict)
                convergence_dict["all_chains_converged"] = bool(
                    all(v < 1.1 for v in r_hat_values)
                )
                convergence_dict["min_r_hat"] = float(min(r_hat_values))
                convergence_dict["max_r_hat"] = float(max(r_hat_values))
        else:
            r_hat = np.asarray(result.r_hat)
            convergence_dict = param_dict["convergence"]
            assert isinstance(convergence_dict, dict)
            convergence_dict["all_chains_converged"] = bool(np.all(r_hat < 1.1))
            convergence_dict["min_r_hat"] = float(np.min(r_hat))
            convergence_dict["max_r_hat"] = float(np.max(r_hat))

    if (
        hasattr(result, "effective_sample_size")
        and result.effective_sample_size is not None
    ):
        if isinstance(result.effective_sample_size, dict):
            ess_values = [
                v for v in result.effective_sample_size.values() if v is not None
            ]
            if ess_values:
                convergence_dict = param_dict["convergence"]
                assert isinstance(convergence_dict, dict)
                convergence_dict["min_ess"] = float(min(ess_values))
        else:
            ess = np.asarray(result.effective_sample_size)
            convergence_dict = param_dict["convergence"]
            assert isinstance(convergence_dict, dict)
            convergence_dict["min_ess"] = float(np.min(ess))

    if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
        convergence_dict = param_dict["convergence"]
        assert isinstance(convergence_dict, dict)
        convergence_dict["acceptance_rate"] = float(result.acceptance_rate)

    # Add scaling parameters (contrast, offset)
    if hasattr(result, "mean_contrast"):
        parameters_dict = param_dict["parameters"]
        assert isinstance(parameters_dict, dict)
        parameters_dict["contrast"] = {
            "mean": float(result.mean_contrast),
            "std": float(getattr(result, "std_contrast", 0.0)),
        }

    if hasattr(result, "mean_offset"):
        parameters_dict = param_dict["parameters"]
        assert isinstance(parameters_dict, dict)
        parameters_dict["offset"] = {
            "mean": float(result.mean_offset),
            "std": float(getattr(result, "std_offset", 0.0)),
        }

    # Add physical parameters
    if hasattr(result, "mean_params") and result.mean_params is not None:
        analysis_mode = getattr(result, "analysis_mode", "static")
        param_names = _get_parameter_names(analysis_mode)

        mean_params_obj = result.mean_params
        std_params_obj = getattr(result, "std_params", None)

        # CRITICAL FIX (Dec 2025): Check dict FIRST, before as_array.
        # ParameterStats inherits from dict AND has as_array property.
        # The as_array returns values in build order (from from_mcmc_samples),
        # which may not match canonical param_names order from get_physical_param_names().
        # Using dict access ensures correct name-to-value mapping regardless of order.
        if isinstance(mean_params_obj, dict):
            mean_params_arr = np.array(
                [mean_params_obj.get(name, np.nan) for name in param_names]
            )
        elif hasattr(mean_params_obj, "as_array"):
            mean_params_arr = np.asarray(mean_params_obj.as_array)
        else:
            mean_params_arr = np.asarray(mean_params_obj)

        # Same fix for std_params_obj - check dict first
        if isinstance(std_params_obj, dict):
            std_params_arr = np.array(
                [std_params_obj.get(name, 0.0) for name in param_names]
            )
        elif std_params_obj is not None and hasattr(std_params_obj, "as_array"):
            std_params_arr = np.asarray(std_params_obj.as_array)
        else:
            std_params_arr = (
                np.asarray(std_params_obj)
                if std_params_obj is not None
                else np.zeros_like(mean_params_arr)
            )

        parameters_dict = param_dict["parameters"]
        assert isinstance(parameters_dict, dict)
        for i, name in enumerate(param_names):
            if i < len(mean_params_arr):
                parameters_dict[name] = {
                    "mean": float(mean_params_arr[i]),
                    "std": float(std_params_arr[i]) if i < len(std_params_arr) else 0.0,
                }

    return param_dict


def create_mcmc_analysis_dict(
    result: Any,
    data: dict[str, Any],
    method_name: str,
) -> dict:
    """Create analysis results dictionary for MCMC/CMC.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with diagnostics
    data : dict
        Experimental data dictionary
    method_name : str
        "mcmc" or "cmc"

    Returns
    -------
    dict
        Analysis summary dictionary
    """
    # Get dataset dimensions
    c2_exp = data.get("c2_exp", [])
    n_angles = len(data.get("phi_angles_list", []))
    n_time_points = (
        c2_exp.shape[1] * c2_exp.shape[2]
        if hasattr(c2_exp, "shape") and len(c2_exp.shape) >= 3
        else 0
    )
    total_data_points = c2_exp.size if hasattr(c2_exp, "size") else 0

    # Determine sampling quality
    quality_flag = "unknown"
    warnings = []
    recommendations = []

    if hasattr(result, "r_hat") and result.r_hat is not None:
        if isinstance(result.r_hat, dict):
            r_hat_values = [v for v in result.r_hat.values() if v is not None]
            max_r_hat = max(r_hat_values) if r_hat_values else None
        else:
            r_hat = np.asarray(result.r_hat)
            max_r_hat = np.max(r_hat)

        if max_r_hat is not None:
            if max_r_hat < 1.05:
                quality_flag = "good"
            elif max_r_hat < 1.1:
                quality_flag = "acceptable"
                warnings.append(
                    f"Some parameters have R-hat between 1.05-1.1 (max={max_r_hat:.3f})"
                )
            else:
                quality_flag = "poor"
                warnings.append(
                    f"Convergence issues detected (max R-hat={max_r_hat:.3f})"
                )
                recommendations.append("Consider increasing n_warmup or n_samples")

    if (
        hasattr(result, "effective_sample_size")
        and result.effective_sample_size is not None
    ):
        if isinstance(result.effective_sample_size, dict):
            ess_values = [
                v for v in result.effective_sample_size.values() if v is not None
            ]
            min_ess = min(ess_values) if ess_values else None
        else:
            ess = np.asarray(result.effective_sample_size)
            min_ess = np.min(ess)

        if min_ess is not None and min_ess < 400:
            warnings.append(f"Low effective sample size (min ESS={min_ess:.0f})")
            recommendations.append(
                "Consider increasing n_samples for better posterior estimates"
            )

    analysis_dict = {
        "method": method_name,
        "timestamp": datetime.now().isoformat(),
        "analysis_mode": getattr(result, "analysis_mode", "unknown"),
        "sampling_quality": {
            "convergence_status": (
                "converged"
                if quality_flag in ["good", "acceptable"]
                else "not_converged"
            ),
            "quality_flag": quality_flag,
            "warnings": warnings,
            "recommendations": recommendations,
        },
        "dataset_info": {
            "n_angles": n_angles,
            "n_time_points": n_time_points,
            "total_data_points": total_data_points,
            "q_value": (
                float(data.get("wavevector_q_list", [0.0])[0])
                if data.get("wavevector_q_list") is not None
                else 0.0
            ),
        },
        "sampling_summary": {
            "n_samples": getattr(result, "n_samples", 0),
            "n_warmup": getattr(result, "n_warmup", 0),
            "n_chains": getattr(result, "n_chains", 1),
            "execution_time": float(getattr(result, "computation_time", 0.0)),
        },
    }

    # v2.1.0: Add config-driven metadata if available
    if (
        hasattr(result, "parameter_space_metadata")
        and result.parameter_space_metadata is not None
    ):
        analysis_dict["parameter_space"] = result.parameter_space_metadata

    if (
        hasattr(result, "initial_values_metadata")
        and result.initial_values_metadata is not None
    ):
        analysis_dict["initial_values"] = result.initial_values_metadata

    if (
        hasattr(result, "selection_decision_metadata")
        and result.selection_decision_metadata is not None
    ):
        analysis_dict["selection_decision"] = result.selection_decision_metadata

    return analysis_dict


def create_mcmc_diagnostics_dict(result: Any) -> dict:
    """Create diagnostics dictionary for MCMC/CMC.

    Parameters
    ----------
    result : MCMCResult
        MCMC result with convergence diagnostics

    Returns
    -------
    dict
        Diagnostics dictionary with convergence metrics
    """
    diagnostics_dict: dict[str, Any] = {
        "convergence": {},
        "sampling_efficiency": {},
        "posterior_checks": {},
    }

    diag_summary = getattr(result, "diagnostic_summary", {}) or {}
    deterministic_params = set(diag_summary.get("deterministic_params") or [])
    per_param_stats = diag_summary.get("per_param_stats") or {}

    # Convergence diagnostics
    if hasattr(result, "r_hat") and result.r_hat is not None:
        if isinstance(result.r_hat, dict):
            r_hat_values = [v for v in result.r_hat.values() if v is not None]
            if r_hat_values:
                diagnostics_dict["convergence"]["all_chains_converged"] = bool(
                    all(v < 1.1 for v in r_hat_values)
                )
                diagnostics_dict["convergence"]["r_hat_threshold"] = 1.1

            # Add per-parameter diagnostics
            per_param = []
            for param_name, r_hat_val in result.r_hat.items():
                ess_val = None
                if hasattr(result, "effective_sample_size") and isinstance(
                    result.effective_sample_size, dict
                ):
                    ess_val = result.effective_sample_size.get(param_name, None)

                per_param.append(
                    {
                        "name": param_name,
                        "r_hat": float(r_hat_val) if r_hat_val is not None else None,
                        "ess": float(ess_val) if ess_val is not None else None,
                        "converged": bool(r_hat_val is not None and r_hat_val < 1.1),
                        "deterministic": param_name in deterministic_params,
                    }
                )
            if per_param:
                diagnostics_dict["convergence"]["per_parameter_diagnostics"] = per_param
        else:
            r_hat = np.asarray(result.r_hat)
            diagnostics_dict["convergence"]["all_chains_converged"] = bool(
                np.all(r_hat < 1.1)
            )
            diagnostics_dict["convergence"]["r_hat_threshold"] = 1.1

            analysis_mode = getattr(result, "analysis_mode", "static")
            param_names = _get_parameter_names(analysis_mode)

            per_param = []
            ess_array = (
                np.asarray(result.effective_sample_size)
                if (
                    hasattr(result, "effective_sample_size")
                    and result.effective_sample_size is not None
                    and not isinstance(result.effective_sample_size, dict)
                )
                else None
            )

            for i, name in enumerate(param_names):
                if i < len(r_hat):
                    ess_val = (
                        ess_array[i]
                        if (ess_array is not None and i < len(ess_array))
                        else 0.0
                    )
                    per_param.append(
                        {
                            "name": name,
                            "r_hat": float(r_hat[i]),
                            "ess": float(ess_val),
                            "converged": bool(r_hat[i] < 1.1),
                            "deterministic": name in deterministic_params,
                        }
                    )

            diagnostics_dict["convergence"]["per_parameter_diagnostics"] = per_param

    if (
        hasattr(result, "effective_sample_size")
        and result.effective_sample_size is not None
    ):
        diagnostics_dict["convergence"]["ess_threshold"] = 400

    # Sampling efficiency
    if hasattr(result, "acceptance_rate") and result.acceptance_rate is not None:
        diagnostics_dict["sampling_efficiency"]["acceptance_rate"] = float(
            result.acceptance_rate
        )
        diagnostics_dict["sampling_efficiency"]["target_acceptance"] = 0.80

    if hasattr(result, "divergences"):
        diagnostics_dict["sampling_efficiency"]["divergences"] = int(result.divergences)

    if hasattr(result, "tree_depth_warnings"):
        diagnostics_dict["sampling_efficiency"]["tree_depth_warnings"] = int(
            result.tree_depth_warnings
        )

    # Posterior checks
    if hasattr(result, "ess") and hasattr(result, "n_samples"):
        ess = np.asarray(result.ess)
        total_samples = result.n_samples * getattr(result, "n_chains", 1)
        if total_samples > 0:
            ess_ratio = float(np.mean(ess) / total_samples)
            diagnostics_dict["posterior_checks"]["effective_sample_size_ratio"] = (
                ess_ratio
            )

    # Fallback per-parameter diagnostics
    if "per_parameter_diagnostics" not in diagnostics_dict["convergence"]:
        param_keys = set(per_param_stats.keys())
        if isinstance(result.r_hat, dict):
            param_keys.update(result.r_hat.keys())
        if isinstance(result.effective_sample_size, dict):
            param_keys.update(result.effective_sample_size.keys())
        if param_keys:
            fallback_entries = []
            for name in sorted(param_keys):
                stats = per_param_stats.get(name, {})
                r_hat_val = None
                if isinstance(result.r_hat, dict):
                    r_hat_val = result.r_hat.get(name)
                elif "r_hat" in stats:
                    r_hat_val = stats.get("r_hat")
                ess_val = None
                if isinstance(result.effective_sample_size, dict):
                    ess_val = result.effective_sample_size.get(name)
                elif "ess" in stats:
                    ess_val = stats.get("ess")

                fallback_entries.append(
                    {
                        "name": name,
                        "r_hat": float(r_hat_val) if r_hat_val is not None else None,
                        "ess": float(ess_val) if ess_val is not None else None,
                        "converged": bool(
                            r_hat_val is not None
                            and r_hat_val
                            < diagnostics_dict["convergence"].get(
                                "r_hat_threshold", 1.1
                            )
                        ),
                        "deterministic": name in deterministic_params
                        or stats.get("deterministic", False),
                    }
                )
            diagnostics_dict["convergence"]["per_parameter_diagnostics"] = (
                fallback_entries
            )

    # CMC-specific diagnostics
    if hasattr(result, "is_cmc_result") and result.is_cmc_result():
        diagnostics_dict["cmc_specific"] = {}

        if hasattr(result, "per_shard_diagnostics") and result.per_shard_diagnostics:
            per_shard = result.per_shard_diagnostics

            acceptance_rates = []
            converged_shards = 0

            for shard in per_shard:
                if isinstance(shard, dict):
                    if shard.get("acceptance_rate") is not None:
                        acceptance_rates.append(float(shard["acceptance_rate"]))
                    if shard.get("converged", False):
                        converged_shards += 1

            shard_summary: dict[str, Any] = {
                "num_shards": len(per_shard),
                "shards_converged": converged_shards,
                "convergence_rate": (
                    float(converged_shards / len(per_shard))
                    if len(per_shard) > 0
                    else 0.0
                ),
            }

            if acceptance_rates:
                shard_summary["acceptance_rate_stats"] = {
                    "mean": float(np.mean(acceptance_rates)),
                    "min": float(np.min(acceptance_rates)),
                    "max": float(np.max(acceptance_rates)),
                    "std": float(np.std(acceptance_rates)),
                }

            diagnostics_dict["cmc_specific"]["shard_summary"] = shard_summary

        if hasattr(result, "cmc_diagnostics") and result.cmc_diagnostics:
            cmc_diag = result.cmc_diagnostics

            overall_metrics: dict[str, Any] = {}

            if isinstance(cmc_diag, dict):
                if "combination_success" in cmc_diag:
                    overall_metrics["combination_success"] = bool(
                        cmc_diag["combination_success"]
                    )
                if "n_shards_converged" in cmc_diag:
                    overall_metrics["n_shards_converged"] = int(
                        cmc_diag["n_shards_converged"]
                    )
                if "n_shards_total" in cmc_diag:
                    overall_metrics["n_shards_total"] = int(cmc_diag["n_shards_total"])
                if "weighted_product_std" in cmc_diag:
                    overall_metrics["weighted_product_std"] = float(
                        cmc_diag["weighted_product_std"]
                    )
                if "combination_time" in cmc_diag:
                    overall_metrics["combination_time"] = float(
                        cmc_diag["combination_time"]
                    )
                if "success_rate" in cmc_diag:
                    overall_metrics["success_rate"] = float(cmc_diag["success_rate"])

                diagnostics_dict["cmc_specific"]["overall_diagnostics"] = (
                    overall_metrics
                )

        if hasattr(result, "combination_method") and result.combination_method:
            diagnostics_dict["cmc_specific"]["combination_method"] = str(
                result.combination_method
            )

        if hasattr(result, "num_shards") and result.num_shards:
            diagnostics_dict["cmc_specific"]["num_shards"] = int(result.num_shards)

    return diagnostics_dict
