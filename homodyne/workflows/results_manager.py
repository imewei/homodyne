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
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import numpy as np

from homodyne.utils.logging import get_logger, log_performance
from homodyne.optimization.variational import VIResult
from homodyne.optimization.mcmc import MCMCResult  
from homodyne.optimization.hybrid import HybridResult

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


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
        self.results_dir = self.output_dir / 'results'
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Results manager initialized: {self.results_dir}")
    
    @log_performance
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
                'c2_experimental': data_dict['c2_exp'],
                'c2_fitted': fitted_data['c2_fitted'],
                'residuals': fitted_data['residuals'],
                'parameters': self._extract_parameters(result),
                'uncertainties': self._extract_uncertainties(result),
                'chi_squared': getattr(result, 'chi_squared', np.nan),
                'phi_angles': data_dict['phi_angles'],
                't1': data_dict['t1'],
                't2': data_dict['t2'],
                'q': data_dict['q'],
                'L': data_dict['L'],
                'analysis_metadata': self._create_analysis_metadata(result)
            }
            
            # Save as NPZ
            output_file = self.results_dir / 'fitted_data.npz'
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
        required_attrs = ['computation_time', 'dataset_size', 'analysis_mode']
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
        
        logger.debug("✓ Results validation completed")
    
    def _validate_vi_results(self, result: VIResult) -> None:
        """Validate VI-specific results."""
        if not result.converged:
            logger.warning("VI optimization did not converge")
        
        if result.final_elbo > 0:
            logger.warning(f"Positive ELBO may indicate optimization issues: {result.final_elbo}")
        
        if result.kl_divergence < 0:
            logger.warning(f"Negative KL divergence is unexpected: {result.kl_divergence}")
    
    def _validate_mcmc_results(self, result: MCMCResult) -> None:
        """Validate MCMC-specific results."""
        # Check R-hat convergence diagnostic
        max_rhat = np.max(result.r_hat) if hasattr(result, 'r_hat') else np.nan
        if max_rhat > 1.1:
            logger.warning(f"MCMC chains may not have converged (R-hat = {max_rhat:.3f} > 1.1)")
        
        # Check effective sample size
        min_ess = np.min(result.ess) if hasattr(result, 'ess') else np.nan
        if min_ess < 100:
            logger.warning(f"Low effective sample size (ESS = {min_ess:.0f} < 100)")
    
    def _validate_hybrid_results(self, result: HybridResult) -> None:
        """Validate Hybrid-specific results."""
        self._validate_vi_results(result.vi_result)
        self._validate_mcmc_results(result.mcmc_result)
        
        if not hasattr(result, 'recommended_method'):
            logger.warning("Missing method recommendation in hybrid results")
    
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
            from homodyne.output.formatters import ResultFormatter
            
            formatter = ResultFormatter()
            json_data = formatter.format_for_json(result)
            
            output_file = self.results_dir / 'analysis_results.json'
            with open(output_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.debug(f"✓ JSON export: {output_file}")
            
        except Exception as e:
            logger.warning(f"JSON export failed: {e}")
    
    def _export_yaml(self, result: ResultType) -> None:
        """Export results to YAML format."""
        try:
            import yaml
            from homodyne.output.formatters import ResultFormatter
            
            formatter = ResultFormatter()
            yaml_data = formatter.format_for_json(result)  # Same format works for YAML
            
            output_file = self.results_dir / 'analysis_results.yaml'
            with open(output_file, 'w') as f:
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
            
            df = pd.DataFrame({
                'parameter': param_names[:len(params)],
                'value': params,
                'uncertainty': uncertainties[:len(params)] if uncertainties is not None else [np.nan] * len(params)
            })
            
            output_file = self.results_dir / 'parameters.csv'
            df.to_csv(output_file, index=False)
            
            logger.debug(f"✓ CSV export: {output_file}")
            
        except Exception as e:
            logger.warning(f"CSV export failed: {e}")
    
    def _export_mcmc_traces(self, result: MCMCResult) -> None:
        """Export MCMC traces for analysis."""
        try:
            if hasattr(result, 'posterior_samples'):
                output_file = self.results_dir / 'mcmc_traces.npz'
                np.savez_compressed(
                    output_file,
                    samples=result.posterior_samples,
                    r_hat=getattr(result, 'r_hat', None),
                    ess=getattr(result, 'ess', None)
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
            from homodyne.output.summarizers import SummaryGenerator
            
            generator = SummaryGenerator()
            summary = generator.create_analysis_summary(result, self.config)
            
            # Save summary
            output_file = self.results_dir / 'analysis_summary.json'
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            # Also save human-readable summary
            readable_summary = self._create_readable_summary(result)
            output_file = self.results_dir / 'summary.txt'
            with open(output_file, 'w') as f:
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
        method_name = type(result).__name__.lower().replace('result', '')
        method_dir = self.results_dir / method_name
        method_dir.mkdir(exist_ok=True)
        
        # Create method-specific summary
        if isinstance(result, VIResult):
            self._create_vi_outputs(result, method_dir)
        elif isinstance(result, MCMCResult):
            self._create_mcmc_outputs(result, method_dir)
        elif isinstance(result, HybridResult):
            self._create_hybrid_outputs(result, method_dir)
    
    def _create_vi_outputs(self, result: VIResult, output_dir: Path) -> None:
        """Create VI-specific outputs."""
        vi_summary = {
            'optimization': {
                'final_elbo': result.final_elbo,
                'kl_divergence': result.kl_divergence,
                'likelihood': result.likelihood,
                'converged': result.converged,
                'iterations': result.n_iterations
            },
            'parameters': {
                'means': result.mean_params.tolist(),
                'std_devs': result.std_params.tolist(),
                'contrast': result.mean_contrast,
                'offset': result.mean_offset
            }
        }
        
        with open(output_dir / 'vi_summary.json', 'w') as f:
            json.dump(vi_summary, f, indent=2)
    
    def _create_mcmc_outputs(self, result: MCMCResult, output_dir: Path) -> None:
        """Create MCMC-specific outputs."""
        mcmc_summary = {
            'sampling': {
                'n_samples': result.n_samples,
                'n_chains': getattr(result, 'n_chains', 1),
                'r_hat': result.r_hat.tolist() if hasattr(result, 'r_hat') else [],
                'ess': result.ess.tolist() if hasattr(result, 'ess') else [],
                'computation_time': result.computation_time
            },
            'parameters': {
                'means': result.mean_params.tolist(),
                'std_devs': result.std_params.tolist(),
                'contrast': result.mean_contrast,
                'offset': result.mean_offset
            }
        }
        
        with open(output_dir / 'mcmc_summary.json', 'w') as f:
            json.dump(mcmc_summary, f, indent=2)
    
    def _create_hybrid_outputs(self, result: HybridResult, output_dir: Path) -> None:
        """Create Hybrid-specific outputs."""
        hybrid_summary = {
            'recommended_method': result.recommended_method,
            'vi_phase': {
                'elbo': result.vi_result.final_elbo,
                'converged': result.vi_result.converged
            },
            'mcmc_phase': {
                'r_hat_max': np.max(result.mcmc_result.r_hat) if hasattr(result.mcmc_result, 'r_hat') else np.nan,
                'ess_min': np.min(result.mcmc_result.ess) if hasattr(result.mcmc_result, 'ess') else np.nan
            }
        }
        
        with open(output_dir / 'hybrid_summary.json', 'w') as f:
            json.dump(hybrid_summary, f, indent=2)
    
    def _compute_fitted_correlation(self, result: ResultType, data_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
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
            contrast = getattr(result, 'mean_contrast', 1.0)
            offset = getattr(result, 'mean_offset', 0.0)
            
            # Initialize theory engine
            analysis_mode = getattr(result, 'analysis_mode', 'laminar_flow')
            theory_engine = TheoryEngine(analysis_mode)
            
            # Compute theoretical correlation
            g1_theory = theory_engine.compute_g1(
                params=params,
                t1=data_dict['t1'],
                t2=data_dict['t2'],
                phi=data_dict['phi_angles'],
                q=data_dict['q'],
                L=data_dict['L']
            )
            
            # Apply scaling: c2_fitted = contrast * g1² + offset
            c2_fitted = contrast * (g1_theory ** 2) + offset
            
            # Compute residuals
            residuals = data_dict['c2_exp'] - c2_fitted
            
            return {
                'c2_fitted': c2_fitted,
                'residuals': residuals,
                'g1_theory': g1_theory
            }
            
        except Exception as e:
            logger.warning(f"Could not compute fitted correlation: {e}")
            return {
                'c2_fitted': np.zeros_like(data_dict['c2_exp']),
                'residuals': np.zeros_like(data_dict['c2_exp']),
                'g1_theory': np.zeros_like(data_dict['c2_exp'])
            }
    
    def _extract_parameters(self, result: ResultType) -> np.ndarray:
        """Extract parameter values from result."""
        if hasattr(result, 'mean_params'):
            return np.array(result.mean_params)
        return np.array([])
    
    def _extract_uncertainties(self, result: ResultType) -> Optional[np.ndarray]:
        """Extract parameter uncertainties from result.""" 
        if hasattr(result, 'std_params'):
            return np.array(result.std_params)
        return None
    
    def _get_parameter_names(self, result: ResultType) -> list:
        """Get parameter names based on analysis mode."""
        analysis_mode = getattr(result, 'analysis_mode', 'laminar_flow')
        
        if analysis_mode in ['static_isotropic', 'static_anisotropic']:
            return ['D0', 'alpha', 'D_offset', 'contrast', 'offset']
        else:  # laminar_flow
            return ['D0', 'alpha', 'D_offset', 'gamma_dot_0', 'beta', 'gamma_dot_offset', 'phi_0', 'contrast', 'offset']
    
    def _create_analysis_metadata(self, result: ResultType) -> Dict[str, Any]:
        """Create analysis metadata."""
        return {
            'timestamp': datetime.now().isoformat(),
            'method': type(result).__name__.lower().replace('result', ''),
            'analysis_mode': getattr(result, 'analysis_mode', 'unknown'),
            'dataset_size': getattr(result, 'dataset_size', 'unknown'),
            'computation_time': getattr(result, 'computation_time', 0.0),
            'homodyne_version': getattr(__import__('homodyne'), '__version__', 'unknown')
        }
    
    def _create_readable_summary(self, result: ResultType) -> str:
        """Create human-readable summary."""
        lines = []
        lines.append("HOMODYNE v2 ANALYSIS SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Method: {type(result).__name__.replace('Result', '')}")
        lines.append(f"Analysis Mode: {getattr(result, 'analysis_mode', 'unknown')}")
        lines.append(f"Computation Time: {getattr(result, 'computation_time', 0.0):.2f} seconds")
        lines.append("")
        
        # Parameters
        lines.append("PARAMETERS:")
        params = self._extract_parameters(result)
        uncertainties = self._extract_uncertainties(result)
        param_names = self._get_parameter_names(result)
        
        for i, (name, value) in enumerate(zip(param_names, params)):
            if uncertainties is not None and i < len(uncertainties):
                lines.append(f"  {name}: {value:.6f} ± {uncertainties[i]:.6f}")
            else:
                lines.append(f"  {name}: {value:.6f}")
        
        # Quality metrics
        lines.append("")
        lines.append("QUALITY METRICS:")
        if hasattr(result, 'chi_squared'):
            lines.append(f"  Chi-squared: {result.chi_squared:.4f}")
        if hasattr(result, 'final_elbo'):
            lines.append(f"  Final ELBO: {result.final_elbo:.4f}")
        
        return "\n".join(lines)