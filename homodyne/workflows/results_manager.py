"""
Results Manager for Homodyne v2
===============================

Comprehensive results processing, validation, and export system.
Handles multiple output formats and generates analysis summaries.

Key Features:
- Multi-format export (YAML, JSON, NPZ, HDF5)
- Results validation and quality checks
- Comprehensive analysis summaries
- Performance metrics tracking
- Error handling and recovery
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from homodyne.optimization.hybrid import HybridResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.variational import VIResult
from homodyne.optimization.lsq_wrapper import LSQResult
from homodyne.utils.logging import get_logger, log_performance
from homodyne.core.jax_backend import safe_len

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult, LSQResult]


class ResultsManager:
    """
    Comprehensive results processing and export manager.

    Coordinates results validation, export to multiple formats,
    and generation of analysis summaries and reports.
    """

    def __init__(self, output_dir: Path, config: Dict[str, Any]):
        """
        Initialize results manager.

        Args:
            output_dir: Output directory for results
            config: Analysis configuration
        """
        self.output_dir = Path(output_dir)
        self.config = config
        self.results_dir = self.output_dir / "results"

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results manager initialized: {self.results_dir}")

    @log_performance()
    def process_results(self, result: ResultType) -> None:
        """
        Process analysis results with comprehensive export.

        Args:
            result: Analysis result object
        """
        try:
            logger.info("📊 Processing analysis results")

            # Validate results
            self._validate_results(result)

            # Export to multiple formats
            self._export_results(result)

            # Generate analysis summary
            self._generate_analysis_summary(result)

            # Create method-specific outputs
            self._create_method_specific_outputs(result)

            logger.info("✓ Results processing completed")

        except Exception as e:
            logger.error(f"❌ Results processing failed: {e}")
            raise

    def save_fitted_data(self, result: ResultType, data_dict: Dict[str, Any]) -> None:
        """
        Save fitted data in consolidated NPZ format.

        Args:
            result: Analysis result
            data_dict: Original experimental data
        """
        try:
            logger.info("💾 Saving fitted data")

            # Compute fitted correlation function
            fitted_data = self._compute_fitted_correlation(result, data_dict)

            # Create comprehensive data package
            fitted_package = {
                "c2_experimental": data_dict["c2_exp"],
                "c2_fitted": fitted_data["c2_fitted"],
                "residuals": fitted_data["residuals"],
                "parameters": self._extract_parameters(result),
                "uncertainties": self._extract_uncertainties(result),
                "chi_squared": getattr(result, "chi_squared", np.nan),
                "reduced_chi_squared": getattr(result, "reduced_chi_squared", np.nan),
                "phi_angles": data_dict["phi_angles"],
                "t1": data_dict["t1"],
                "t2": data_dict["t2"],
                "q": data_dict["q"],
                "wavevector_q": data_dict["q"],  # Add wavevector_q as requested
                "L": data_dict["L"],
                "analysis_metadata": self._create_analysis_metadata(result),
            }

            # Save as NPZ
            output_file = self.results_dir / "fitted_data.npz"
            np.savez_compressed(output_file, **fitted_package)

            logger.info(f"✓ Fitted data saved: {output_file}")

        except Exception as e:
            logger.error(f"❌ Fitted data saving failed: {e}")

    def _validate_results(self, result: ResultType) -> None:
        """
        Validate analysis results for consistency and quality.

        Args:
            result: Results to validate
        """
        logger.debug("🔍 Validating analysis results")

        # Check for required attributes
        required_attrs = ["computation_time", "dataset_size", "analysis_mode"]
        for attr in required_attrs:
            if not hasattr(result, attr):
                logger.warning(f"Missing required attribute: {attr}")

        # Method-specific validation
        if isinstance(result, VIResult):
            self._validate_vi_results(result)
        elif isinstance(result, MCMCResult):
            self._validate_mcmc_results(result)
        elif isinstance(result, HybridResult):
            self._validate_hybrid_results(result)
        elif isinstance(result, LSQResult):
            self._validate_lsq_results(result)

        logger.debug("✓ Results validation completed")

    def _validate_vi_results(self, result: VIResult) -> None:
        """Validate VI-specific results."""
        if not result.converged:
            logger.warning("VI optimization did not converge")

        if result.final_elbo > 0:
            logger.warning(
                f"Positive ELBO may indicate optimization issues: {result.final_elbo}"
            )

        if result.kl_divergence < 0:
            logger.warning(
                f"Negative KL divergence is unexpected: {result.kl_divergence}"
            )

    def _validate_mcmc_results(self, result: MCMCResult) -> None:
        """Validate MCMC-specific results."""
        # Check R-hat convergence diagnostic
        max_rhat = np.max(result.r_hat) if hasattr(result, "r_hat") else np.nan
        if max_rhat > 1.1:
            logger.warning(
                f"MCMC chains may not have converged (R-hat = {max_rhat:.3f} > 1.1)"
            )

        # Check effective sample size
        min_ess = np.min(result.ess) if hasattr(result, "ess") else np.nan
        if min_ess < 100:
            logger.warning(f"Low effective sample size (ESS = {min_ess:.0f} < 100)")

    def _validate_hybrid_results(self, result: HybridResult) -> None:
        """Validate Hybrid-specific results."""
        self._validate_vi_results(result.vi_result)
        self._validate_mcmc_results(result.mcmc_result)

        if not hasattr(result, "recommended_method"):
            logger.warning("Missing method recommendation in hybrid results")

    def _validate_lsq_results(self, result: LSQResult) -> None:
        """Validate LSQ-specific results."""
        # Check chi-squared values
        if hasattr(result, "chi_squared") and result.chi_squared < 0:
            logger.warning(f"Negative chi-squared value: {result.chi_squared}")

        if hasattr(result, "reduced_chi_squared"):
            if result.reduced_chi_squared < 0:
                logger.warning(f"Negative reduced chi-squared: {result.reduced_chi_squared}")
            elif result.reduced_chi_squared > 10:
                logger.warning(f"High reduced chi-squared may indicate poor fit: {result.reduced_chi_squared:.3f}")

        # Check residual quality
        if hasattr(result, "residual_std") and result.residual_std <= 0:
            logger.warning(f"Invalid residual standard deviation: {result.residual_std}")

        # Validate noise estimation if enabled
        if result.noise_estimated:
            if not result.noise_model:
                logger.warning("Noise estimation enabled but no noise model specified")
            if result.estimated_sigma is None:
                logger.warning("Noise estimation enabled but no sigma estimates available")

        # LSQ should always converge (direct solution)
        if not result.converged:
            logger.warning("LSQ should always converge - check solver implementation")

    def _export_results(self, result: ResultType) -> None:
        """
        Export results to multiple formats.

        Args:
            result: Results to export
        """
        # Export to JSON
        self._export_json(result)

        # Export to YAML
        self._export_yaml(result)

        # Export parameters to CSV
        self._export_parameters_csv(result)

        # Export method-specific formats
        if isinstance(result, MCMCResult):
            self._export_mcmc_traces(result)

    def _export_json(self, result: ResultType) -> None:
        """Export results to JSON format."""
        try:
            # Convert result to JSON-serializable format
            json_data = self._convert_result_to_json_serializable(result)

            output_file = self.results_dir / "analysis_results.json"
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2)

            logger.debug(f"✓ JSON export: {output_file}")

        except Exception as e:
            logger.warning(f"JSON export failed: {e}")
            
    def _convert_result_to_json_serializable(self, result: ResultType) -> Dict[str, Any]:
        """Convert analysis result to JSON-serializable format."""
        import numpy as np
        
        def convert_value(value):
            """Convert numpy/JAX arrays and other non-serializable types to Python types."""
            # Check for large arrays first to avoid memory issues
            def is_large_array(val):
                """Check if array is too large for JSON serialization (>100k elements)."""
                try:
                    if hasattr(val, 'size'):
                        return val.size > 100000
                    elif hasattr(val, 'shape'):
                        import numpy as np
                        return np.prod(val.shape) > 100000
                    return False
                except:
                    return False

            # Handle specific JAX types like ArrayImpl
            value_type_str = str(type(value))
            if 'ArrayImpl' in value_type_str or 'DeviceArray' in value_type_str:
                try:
                    # Check size before conversion to avoid hanging
                    if is_large_array(value):
                        return {
                            "type": "large_array",
                            "shape": list(value.shape) if hasattr(value, 'shape') else "unknown",
                            "dtype": str(value.dtype) if hasattr(value, 'dtype') else "unknown",
                            "size": int(value.size) if hasattr(value, 'size') else "unknown",
                            "note": "Array too large for JSON serialization"
                        }
                    return np.asarray(value).tolist()
                except:
                    try:
                        return float(value) if hasattr(value, '__float__') else str(value)
                    except:
                        return str(value)

            # Handle JAX arrays explicitly by converting to numpy first
            if hasattr(value, '__module__') and value.__module__ and 'jax' in value.__module__:
                try:
                    # Check size before conversion to avoid hanging
                    if is_large_array(value):
                        return {
                            "type": "large_jax_array",
                            "shape": list(value.shape) if hasattr(value, 'shape') else "unknown",
                            "dtype": str(value.dtype) if hasattr(value, 'dtype') else "unknown",
                            "size": int(value.size) if hasattr(value, 'size') else "unknown",
                            "note": "JAX array too large for JSON serialization"
                        }
                    # Convert JAX array to numpy, then to list
                    return np.asarray(value).tolist()
                except:
                    try:
                        return float(value) if hasattr(value, '__float__') else str(value)
                    except:
                        return str(value)
            
            # Handle numpy arrays and other array-like objects
            if isinstance(value, np.ndarray):
                # Check size before conversion to avoid hanging
                if is_large_array(value):
                    return {
                        "type": "large_numpy_array",
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "size": int(value.size),
                        "note": "NumPy array too large for JSON serialization"
                    }
                return value.tolist()
            elif hasattr(value, 'tolist') and hasattr(value, 'shape'):  # Array-like with tolist method
                try:
                    # Check size before conversion to avoid hanging
                    if is_large_array(value):
                        return {
                            "type": "large_array_like",
                            "shape": list(value.shape) if hasattr(value, 'shape') else "unknown",
                            "dtype": str(value.dtype) if hasattr(value, 'dtype') else "unknown",
                            "size": int(value.size) if hasattr(value, 'size') else "unknown",
                            "note": "Array-like object too large for JSON serialization"
                        }
                    return value.tolist()
                except:
                    # Fallback: convert to numpy first
                    try:
                        return np.asarray(value).tolist()
                    except:
                        return str(value)
            elif hasattr(value, '__array__'):  # Array-like objects
                try:
                    # Create numpy array first to check size
                    np_array = np.asarray(value)
                    if is_large_array(np_array):
                        return {
                            "type": "large_array_convertible",
                            "shape": list(np_array.shape) if hasattr(np_array, 'shape') else "unknown",
                            "dtype": str(np_array.dtype) if hasattr(np_array, 'dtype') else "unknown",
                            "size": int(np_array.size) if hasattr(np_array, 'size') else "unknown",
                            "note": "Array-convertible object too large for JSON serialization"
                        }
                    return np_array.tolist()
                except:
                    return str(value)
            elif isinstance(value, (np.integer, np.floating)):
                return value.item()
            elif isinstance(value, (list, tuple)):
                return [convert_value(item) for item in value]
            elif isinstance(value, dict):
                return {key: convert_value(val) for key, val in value.items()}
            elif hasattr(value, '__dict__'):  # Complex objects
                return {key: convert_value(val) for key, val in value.__dict__.items()}
            else:
                return value
        
        # Convert all result attributes to JSON-serializable format
        result_dict = {}
        for attr in dir(result):
            if not attr.startswith('_'):  # Skip private attributes
                try:
                    value = getattr(result, attr)
                    if not callable(value):  # Skip methods
                        result_dict[attr] = convert_value(value)
                except:
                    continue  # Skip attributes that can't be accessed
                    
        return result_dict
        
    def _create_simple_summary(self, result: ResultType) -> Dict[str, Any]:
        """Create a simple analysis summary."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "analysis_type": type(result).__name__,
            "converged": getattr(result, 'converged', False),
            "computation_time": getattr(result, 'computation_time', 0.0),
            "n_iterations": getattr(result, 'n_iterations', 0),
        }
        
        # Add method-specific fields
        if hasattr(result, 'final_elbo'):
            summary["final_elbo"] = float(getattr(result, 'final_elbo', float('-inf')))
        if hasattr(result, 'chi_squared'):
            summary["chi_squared"] = float(getattr(result, 'chi_squared', float('inf')))
        if hasattr(result, 'reduced_chi_squared'):
            summary["reduced_chi_squared"] = float(getattr(result, 'reduced_chi_squared', float('inf')))
        if hasattr(result, 'mean_params'):
            try:
                mean_params = getattr(result, 'mean_params')
                if hasattr(mean_params, 'tolist'):
                    summary["mean_parameters"] = mean_params.tolist()
                else:
                    summary["mean_parameters"] = list(mean_params)
            except:
                pass
                
        return summary

    def _export_yaml(self, result: ResultType) -> None:
        """Export results to YAML format."""
        try:
            import yaml

            # Convert result to YAML-serializable format (same as JSON)
            yaml_data = self._convert_result_to_json_serializable(result)

            output_file = self.results_dir / "analysis_results.yaml"
            with open(output_file, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, indent=2)

            logger.debug(f"✓ YAML export: {output_file}")

        except Exception as e:
            logger.warning(f"YAML export failed: {e}")

    def _export_parameters_csv(self, result: ResultType) -> None:
        """Export parameters to CSV format."""
        try:
            import pandas as pd

            # Extract parameters and uncertainties
            params = self._extract_parameters(result)
            uncertainties = self._extract_uncertainties(result)

            # Create parameter DataFrame
            param_names = self._get_parameter_names(result)

            # Handle uncertainties properly - LSQ doesn't provide them
            params_len = safe_len(params)
            if uncertainties is not None and hasattr(uncertainties, '__len__'):
                uncertainty_values = list(uncertainties[: params_len])
            else:
                uncertainty_values = [np.nan] * params_len

            df = pd.DataFrame(
                {
                    "parameter": param_names[: params_len],
                    "value": params,
                    "uncertainty": uncertainty_values,
                }
            )

            output_file = self.results_dir / "parameters.csv"
            df.to_csv(output_file, index=False)

            logger.debug(f"✓ CSV export: {output_file}")

        except Exception as e:
            logger.warning(f"CSV export failed: {e}")

    def _export_mcmc_traces(self, result: MCMCResult) -> None:
        """Export MCMC traces for analysis."""
        try:
            if hasattr(result, "posterior_samples"):
                output_file = self.results_dir / "mcmc_traces.npz"
                np.savez_compressed(
                    output_file,
                    samples=result.posterior_samples,
                    r_hat=getattr(result, "r_hat", None),
                    ess=getattr(result, "ess", None),
                )
                logger.debug(f"✓ MCMC traces export: {output_file}")

        except Exception as e:
            logger.warning(f"MCMC traces export failed: {e}")

    def _generate_analysis_summary(self, result: ResultType) -> None:
        """
        Generate comprehensive analysis summary.

        Args:
            result: Analysis result
        """
        try:
            # Create a simple summary directly since SummaryGenerator might not exist
            summary = self._create_simple_summary(result)

            # Save summary
            output_file = self.results_dir / "analysis_summary.json"
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2)

            # Also save human-readable summary
            readable_summary = self._create_readable_summary(result)
            output_file = self.results_dir / "summary.txt"
            with open(output_file, "w") as f:
                f.write(readable_summary)

            logger.debug("✓ Analysis summary generated")

        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")

    def _create_method_specific_outputs(self, result: ResultType) -> None:
        """
        Create method-specific output files.

        Args:
            result: Analysis result
        """
        method_name = type(result).__name__.lower().replace("result", "")
        method_dir = self.results_dir / method_name
        method_dir.mkdir(exist_ok=True)

        # Create method-specific summary
        if isinstance(result, VIResult):
            self._create_vi_outputs(result, method_dir)
        elif isinstance(result, MCMCResult):
            self._create_mcmc_outputs(result, method_dir)
        elif isinstance(result, HybridResult):
            self._create_hybrid_outputs(result, method_dir)
        elif isinstance(result, LSQResult):
            self._create_lsq_outputs(result, method_dir)

    def _create_vi_outputs(self, result: VIResult, output_dir: Path) -> None:
        """Create VI-specific outputs."""
        import numpy as np
        
        # Safe conversion for JAX/numpy arrays
        def safe_convert(value):
            value_type_str = str(type(value))
            if 'ArrayImpl' in value_type_str or 'DeviceArray' in value_type_str:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, '__module__') and value.__module__ and 'jax' in value.__module__:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, 'tolist'):
                try:
                    return value.tolist()
                except:
                    return str(value)
            else:
                return value
        
        vi_summary = {
            "optimization": {
                "final_elbo": result.final_elbo,
                "kl_divergence": result.kl_divergence,
                "likelihood": result.likelihood,
                "converged": result.converged,
                "iterations": result.n_iterations,
            },
            "parameters": {
                "means": safe_convert(result.mean_params),
                "std_devs": safe_convert(result.std_params),
                "contrast": result.mean_contrast,
                "offset": result.mean_offset,
            },
        }

        with open(output_dir / "vi_summary.json", "w") as f:
            json.dump(vi_summary, f, indent=2)

    def _create_mcmc_outputs(self, result: MCMCResult, output_dir: Path) -> None:
        """Create MCMC-specific outputs."""
        import numpy as np
        
        # Safe conversion for JAX/numpy arrays
        def safe_convert(value):
            value_type_str = str(type(value))
            if 'ArrayImpl' in value_type_str or 'DeviceArray' in value_type_str:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, '__module__') and value.__module__ and 'jax' in value.__module__:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, 'tolist'):
                try:
                    return value.tolist()
                except:
                    return str(value)
            else:
                return value
        
        mcmc_summary = {
            "sampling": {
                "n_samples": result.n_samples,
                "n_chains": getattr(result, "n_chains", 1),
                "r_hat": safe_convert(result.r_hat) if hasattr(result, "r_hat") else [],
                "ess": safe_convert(result.ess) if hasattr(result, "ess") else [],
                "computation_time": result.computation_time,
            },
            "parameters": {
                "means": safe_convert(result.mean_params),
                "std_devs": safe_convert(result.std_params),
                "contrast": result.mean_contrast,
                "offset": result.mean_offset,
            },
        }

        with open(output_dir / "mcmc_summary.json", "w") as f:
            json.dump(mcmc_summary, f, indent=2)

    def _create_hybrid_outputs(self, result: HybridResult, output_dir: Path) -> None:
        """Create Hybrid-specific outputs."""
        hybrid_summary = {
            "recommended_method": result.recommended_method,
            "vi_phase": {
                "elbo": result.vi_result.final_elbo,
                "converged": result.vi_result.converged,
            },
            "mcmc_phase": {
                "r_hat_max": (
                    np.max(result.mcmc_result.r_hat)
                    if hasattr(result.mcmc_result, "r_hat")
                    else np.nan
                ),
                "ess_min": (
                    np.min(result.mcmc_result.ess)
                    if hasattr(result.mcmc_result, "ess")
                    else np.nan
                ),
            },
        }

        with open(output_dir / "hybrid_summary.json", "w") as f:
            json.dump(hybrid_summary, f, indent=2)

    def _create_lsq_outputs(self, result: LSQResult, output_dir: Path) -> None:
        """Create LSQ-specific outputs."""
        import numpy as np

        # Safe conversion for JAX/numpy arrays
        def safe_convert(value):
            value_type_str = str(type(value))
            if 'ArrayImpl' in value_type_str or 'DeviceArray' in value_type_str:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, '__module__') and value.__module__ and 'jax' in value.__module__:
                try:
                    return np.asarray(value).tolist()
                except:
                    return str(value)
            elif hasattr(value, 'tolist'):
                try:
                    return value.tolist()
                except:
                    return str(value)
            else:
                return value

        lsq_summary = {
            "fit_quality": {
                "chi_squared": result.chi_squared,
                "reduced_chi_squared": result.reduced_chi_squared,
                "residual_std": result.residual_std,
                "max_residual": result.max_residual,
                "degrees_of_freedom": result.degrees_of_freedom,
            },
            "optimization": {
                "converged": result.converged,
                "iterations": result.n_iterations,
                "computation_time": result.computation_time,
                "backend": result.backend,
                "dataset_size": result.dataset_size,
                "analysis_mode": result.analysis_mode,
            },
            "parameters": {
                "means": safe_convert(result.mean_params),
                "std_devs": None,  # LSQ doesn't provide uncertainties
                "contrast": result.mean_contrast,
                "offset": result.mean_offset,
            },
            "noise_estimation": {
                "estimated": result.noise_estimated,
                "model": result.noise_model,
                "estimated_sigma": safe_convert(result.estimated_sigma) if result.estimated_sigma is not None else None,
                "parameters": result.noise_params,
            } if result.noise_estimated else None,
        }

        with open(output_dir / "lsq_summary.json", "w") as f:
            json.dump(lsq_summary, f, indent=2)

    def _compute_fitted_correlation(
        self, result: ResultType, data_dict: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Compute fitted correlation function from analysis results.

        Args:
            result: Analysis result
            data_dict: Original experimental data

        Returns:
            Dictionary with fitted data and residuals
        """
        try:
            from homodyne.core.theory import TheoryEngine

            # Extract parameters
            params = self._extract_parameters(result)
            contrast = getattr(result, "mean_contrast", 1.0)
            offset = getattr(result, "mean_offset", 0.0)

            # Initialize theory engine
            analysis_mode = getattr(result, "analysis_mode", "laminar_flow")
            theory_engine = TheoryEngine(analysis_mode)

            # Compute theoretical correlation for each phi angle
            phi_angles_raw = data_dict.get("phi_angles_list", data_dict.get("phi_angles", [0.0]))

            # Ensure we have individual phi angles, not nested arrays
            if isinstance(phi_angles_raw, np.ndarray):
                phi_angles_list = phi_angles_raw.tolist() if phi_angles_raw.ndim == 1 else phi_angles_raw.flatten().tolist()
            elif isinstance(phi_angles_raw, list):
                # Check if list contains arrays instead of individual values
                if len(phi_angles_raw) > 0 and isinstance(phi_angles_raw[0], np.ndarray):
                    phi_angles_list = phi_angles_raw[0].tolist() if phi_angles_raw[0].ndim == 1 else phi_angles_raw[0].flatten().tolist()
                else:
                    phi_angles_list = phi_angles_raw
            else:
                phi_angles_list = [phi_angles_raw]

            c2_fitted_list = []
            residuals_list = []
            g1_theory_list = []

            # Prepare time arrays once, outside the loop
            # Extract 1D arrays from meshgrids if needed
            t1_raw = data_dict["t1"]
            t2_raw = data_dict["t2"]

            # If t1 and t2 are 2D meshgrids, extract the 1D arrays
            if t1_raw.ndim == 2:
                # In a meshgrid, t1 varies along axis 1 (columns)
                t1_1d = t1_raw[0, :]  # Get first row for unique t1 values
                t2_1d = t2_raw[:, 0]  # Get first column for unique t2 values
            else:
                t1_1d = t1_raw
                t2_1d = t2_raw

            # Create meshgrids for theory computation
            # Theory engine needs 2D meshgrids for proper correlation calculation
            t2_grid, t1_grid = np.meshgrid(t2_1d, t1_1d, indexing='ij')

            # Extract dt from data_dict if available
            dt = data_dict.get("dt", None)

            # Handle q array - use mean or first value if it's an array
            q_value = data_dict["q"]
            if isinstance(q_value, np.ndarray):
                if q_value.size > 1:
                    # For post-processing, we typically use the mean q value
                    q_value = float(np.mean(q_value))
                    logger.debug(f"Using mean q value: {q_value} from array with {data_dict['q'].size} elements")
                else:
                    q_value = float(q_value.item()) if hasattr(q_value, 'item') else float(q_value)
            else:
                q_value = float(q_value)

            for i, phi in enumerate(phi_angles_list):
                # Compute g1 for this specific phi angle
                # Debug logging
                logger.debug(f"compute_g1 inputs: params shape={params.shape}, t1_grid shape={t1_grid.shape}, t2_grid shape={t2_grid.shape}, phi={phi}, q={q_value}, L={data_dict['L']}")

                # Get experimental data dimensions to ensure compatibility
                exp_shape = data_dict["c2_exp"][i].shape if data_dict["c2_exp"].ndim == 3 else data_dict["c2_exp"].shape
                expected_size = exp_shape[0]  # Should be 1001 for time correlation matrix

                try:
                    g1_theory_single = theory_engine.compute_g1(
                        params=params,
                        t1=t1_grid,
                        t2=t2_grid,
                        phi=np.array([phi]),  # Single phi angle as array
                        q=q_value,  # Already converted to float scalar
                        L=data_dict["L"],
                        dt=dt,
                    )

                    logger.debug(f"Theory engine returned shape: {g1_theory_single.shape}")

                    # Handle different output shapes robustly
                    if g1_theory_single.ndim == 4:
                        # Shape like (1, n_t2, n_t1, 1) or similar - extract the correlation matrix
                        # Find which dimensions correspond to time
                        time_dims = [d for d in range(4) if g1_theory_single.shape[d] > 1]
                        if len(time_dims) >= 2:
                            # Extract the 2D time correlation matrix
                            # Squeeze out singleton dimensions
                            g1_matrix = np.squeeze(g1_theory_single)
                            if g1_matrix.ndim > 2:
                                # Still has extra dims, extract first phi angle's data
                                if g1_matrix.shape[0] == 1:
                                    g1_matrix = g1_matrix[0]
                                elif g1_matrix.ndim == 3 and g1_matrix.shape[-1] == 1:
                                    g1_matrix = g1_matrix[:, :, 0]
                        else:
                            # Unexpected shape - try to extract meaningful data
                            g1_matrix = np.squeeze(g1_theory_single)
                            if g1_matrix.ndim < 2:
                                # Theory returned essentially a scalar - this indicates a problem
                                logger.warning(f"Theory engine returned scalar-like result, creating diagonal decay matrix")
                                # Create a diagonal decay matrix as fallback
                                time_len = len(t1) if t1.ndim == 1 else t1.shape[1]
                                g1_matrix = np.eye(time_len)
                    elif g1_theory_single.ndim == 3:
                        # Standard 3D case: (n_phi, n_t1, n_t2)
                        if g1_theory_single.shape[0] == 1:
                            g1_matrix = g1_theory_single[0]  # Shape: (n_t1, n_t2)
                        else:
                            g1_matrix = g1_theory_single[min(i, g1_theory_single.shape[0]-1)]
                    elif g1_theory_single.ndim == 2:
                        # Already 2D matrix
                        g1_matrix = g1_theory_single
                    elif g1_theory_single.ndim == 1:
                        # 1D array - reshape to square matrix
                        matrix_size = int(np.sqrt(len(g1_theory_single)))
                        if matrix_size * matrix_size == len(g1_theory_single):
                            g1_matrix = g1_theory_single.reshape(matrix_size, matrix_size)
                        else:
                            # Can't reshape - use constant matrix
                            g1_matrix = np.full((expected_size, expected_size), g1_theory_single[0])
                    else:
                        # Scalar or unexpected shape - create constant matrix
                        scalar_val = float(g1_theory_single.flatten()[0])
                        g1_matrix = np.full((expected_size, expected_size), scalar_val)

                    # Ensure the matrix has the expected shape
                    if g1_matrix.shape != (expected_size, expected_size):
                        logger.warning(f"Resizing g1_matrix from {g1_matrix.shape} to {(expected_size, expected_size)}")
                        # Resize or pad/crop to match expected dimensions
                        if g1_matrix.size == 1:
                            # Single value - broadcast to full matrix
                            g1_matrix = np.full((expected_size, expected_size), float(g1_matrix.flatten()[0]))
                        else:
                            # Use mean value for safety
                            g1_matrix = np.full((expected_size, expected_size), float(np.mean(g1_matrix)))

                except Exception as theory_error:
                    logger.warning(f"Theory computation failed for phi={phi}: {theory_error}")
                    # Fallback: use a default matrix with reasonable values
                    g1_matrix = np.ones((expected_size, expected_size)) * 0.5

                # Apply scaling: c2_fitted = contrast * g1² + offset
                c2_fitted_single = contrast * (g1_matrix**2) + offset

                # Get experimental data for this angle
                if data_dict["c2_exp"].ndim == 3:
                    c2_exp_single = data_dict["c2_exp"][i]  # Shape: (601, 601)
                else:
                    c2_exp_single = data_dict["c2_exp"]

                # Compute residuals
                residuals_single = c2_exp_single - c2_fitted_single

                c2_fitted_list.append(c2_fitted_single)
                residuals_list.append(residuals_single)
                g1_theory_list.append(g1_matrix)

            # Convert to numpy arrays with proper shapes
            c2_fitted = np.array(c2_fitted_list)  # Shape: (n_phi, 601, 601)
            residuals = np.array(residuals_list)  # Shape: (n_phi, 601, 601)
            g1_theory = np.array(g1_theory_list)  # Shape: (n_phi, 601, 601)

            return {
                "c2_fitted": c2_fitted,
                "residuals": residuals,
                "g1_theory": g1_theory,
            }

        except Exception as e:
            logger.warning(f"Could not compute fitted correlation: {e}")
            return {
                "c2_fitted": np.zeros_like(data_dict["c2_exp"]),
                "residuals": np.zeros_like(data_dict["c2_exp"]),
                "g1_theory": np.zeros_like(data_dict["c2_exp"]),
            }

    def _extract_parameters(self, result: ResultType) -> np.ndarray:
        """Extract parameter values from result."""
        if hasattr(result, "mean_params"):
            params = np.array(result.mean_params)
            # Ensure we have a 1D array, not a 0-dimensional array
            if params.ndim == 0:
                params = np.array([params.item()])
            return params
        return np.array([])

    def _extract_uncertainties(self, result: ResultType) -> Optional[np.ndarray]:
        """Extract parameter uncertainties from result."""
        if hasattr(result, "std_params"):
            uncertainties = np.array(result.std_params)
            # Ensure we have a 1D array, not a 0-dimensional array
            if uncertainties.ndim == 0:
                uncertainties = np.array([uncertainties.item()])
            return uncertainties
        return None

    def _get_parameter_names(self, result: ResultType) -> list:
        """Get parameter names based on analysis mode."""
        analysis_mode = getattr(result, "analysis_mode", "laminar_flow")

        if analysis_mode in ["static_isotropic", "static_anisotropic"]:
            return ["D0", "alpha", "D_offset", "contrast", "offset"]
        else:  # laminar_flow
            return [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_0",
                "beta",
                "gamma_dot_offset",
                "phi_0",
                "contrast",
                "offset",
            ]

    def _create_analysis_metadata(self, result: ResultType) -> Dict[str, Any]:
        """Create analysis metadata."""
        return {
            "timestamp": datetime.now().isoformat(),
            "method": type(result).__name__.lower().replace("result", ""),
            "analysis_mode": getattr(result, "analysis_mode", "unknown"),
            "dataset_size": getattr(result, "dataset_size", "unknown"),
            "computation_time": getattr(result, "computation_time", 0.0),
            "homodyne_version": getattr(
                __import__("homodyne"), "__version__", "unknown"
            ),
        }

    def _create_readable_summary(self, result: ResultType) -> str:
        """Create human-readable summary."""
        lines = []
        lines.append("HOMODYNE v2 ANALYSIS SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Method: {type(result).__name__.replace('Result', '')}")
        lines.append(f"Analysis Mode: {getattr(result, 'analysis_mode', 'unknown')}")
        lines.append(
            f"Computation Time: {getattr(result, 'computation_time', 0.0):.2f} seconds"
        )
        lines.append("")

        # Parameters
        lines.append("PARAMETERS:")
        params = self._extract_parameters(result)
        uncertainties = self._extract_uncertainties(result)
        param_names = self._get_parameter_names(result)

        for i, (name, value) in enumerate(zip(param_names, params)):
            uncertainties_len = safe_len(uncertainties) if uncertainties is not None else 0
            if uncertainties is not None and hasattr(uncertainties, '__len__') and i < uncertainties_len:
                lines.append(f"  {name}: {value:.6f} ± {uncertainties[i]:.6f}")
            else:
                lines.append(f"  {name}: {value:.6f}")

        # Quality metrics
        lines.append("")
        lines.append("QUALITY METRICS:")
        if hasattr(result, "chi_squared"):
            lines.append(f"  Chi-squared: {result.chi_squared:.4f}")
        if hasattr(result, "reduced_chi_squared"):
            lines.append(f"  Reduced χ²: {result.reduced_chi_squared:.4f}")
        if hasattr(result, "final_elbo"):
            lines.append(f"  Final ELBO: {result.final_elbo:.4f}")

        return "\n".join(lines)
