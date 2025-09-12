"""
Plotting Controller for Homodyne v2
===================================

Comprehensive visualization system for XPCS analysis results.
Coordinates plot generation for data validation, fit results, and diagnostics.

Key Features:
- Experimental data validation plots
- Simulated data visualization with custom scaling
- Fit comparison plots (experimental vs fitted)
- Method-specific diagnostic plots
- Publication-quality figure generation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from homodyne.utils.logging import get_logger, log_performance
from homodyne.optimization.variational import VIResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.hybrid import HybridResult

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


class PlottingController:
    """
    Comprehensive plotting system for XPCS analysis visualization.
    
    Generates publication-quality plots for data validation,
    analysis results, and method-specific diagnostics.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize plotting controller.
        
        Args:
            output_dir: Output directory for plots
        """
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup matplotlib for publication-quality plots
        self._setup_matplotlib()
        
        logger.info(f"Plotting controller initialized: {self.plots_dir}")
    
    def _setup_matplotlib(self) -> None:
        """Setup matplotlib for high-quality plots."""
        try:
            # Use non-interactive backend for headless environments
            import matplotlib
            matplotlib.use('Agg')
            
            # Set publication-quality defaults
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'font.family': 'serif',
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.figsize': (10, 8),
                'figure.dpi': 150,
                'savefig.dpi': 300,
                'savefig.format': 'png',
                'savefig.bbox': 'tight'
            })
            
        except Exception as e:
            logger.warning(f"Matplotlib setup warning: {e}")
    
    def plot_experimental_data(self, data: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Generate experimental data validation plots.
        
        Args:
            data: Experimental data dictionary
            config: Analysis configuration
        """
        try:
            logger.info("🎨 Generating experimental data plots")
            
            # C2 correlation heatmap
            self._plot_c2_heatmap(
                data['c2_exp'], 
                data['t1'], 
                data['t2'],
                data['phi_angles'],
                title="Experimental C₂ Correlation Function",
                filename="c2_experimental_heatmap.png"
            )
            
            # Time-averaged correlation
            self._plot_time_averaged_correlation(data)
            
            # Angular dependence plots
            self._plot_angular_dependence(data)
            
            # Data quality diagnostics
            self._plot_data_quality_diagnostics(data)
            
            logger.info("✓ Experimental data plots generated")
            
        except Exception as e:
            logger.error(f"❌ Experimental data plotting failed: {e}")
    
    def plot_simulated_data(self, 
                          config: Dict[str, Any],
                          contrast: float = 1.0,
                          offset: float = 0.0,
                          phi_angles_str: Optional[str] = None) -> None:
        """
        Generate simulated data plots using theoretical model.
        
        Args:
            config: Analysis configuration
            contrast: Scaling contrast parameter
            offset: Scaling offset parameter
            phi_angles_str: Custom phi angles (comma-separated)
        """
        try:
            logger.info("🎨 Generating simulated data plots")
            
            # Parse custom phi angles or use defaults
            phi_angles = self._parse_phi_angles(phi_angles_str, config)
            
            # Generate theoretical data
            simulated_data = self._generate_simulated_data(config, phi_angles, contrast, offset)
            
            # C2 theoretical heatmap
            self._plot_c2_heatmap(
                simulated_data['c2_theory'],
                simulated_data['t1'],
                simulated_data['t2'], 
                simulated_data['phi_angles'],
                title=f"Theoretical C₂ (contrast={contrast:.2f}, offset={offset:.2f})",
                filename="c2_simulated_heatmap.png"
            )
            
            # Parameter sensitivity plots
            self._plot_parameter_sensitivity(config, phi_angles)
            
            logger.info("✓ Simulated data plots generated")
            
        except Exception as e:
            logger.error(f"❌ Simulated data plotting failed: {e}")
    
    @log_performance
    def plot_fit_results(self, result: ResultType, data: Dict[str, Any]) -> None:
        """
        Generate fit comparison and diagnostic plots.
        
        Args:
            result: Analysis result
            data: Experimental data
        """
        try:
            logger.info("🎨 Generating fit result plots")
            
            # Fit comparison heatmaps
            self._plot_fit_comparison(result, data)
            
            # Residual analysis
            self._plot_residual_analysis(result, data)
            
            # Parameter plots
            self._plot_parameter_results(result)
            
            # Method-specific plots
            if isinstance(result, VIResult):
                self._plot_vi_diagnostics(result)
            elif isinstance(result, MCMCResult):
                self._plot_mcmc_diagnostics(result)
            elif isinstance(result, HybridResult):
                self._plot_hybrid_diagnostics(result)
            
            logger.info("✓ Fit result plots generated")
            
        except Exception as e:
            logger.error(f"❌ Fit result plotting failed: {e}")
    
    def _plot_c2_heatmap(self, 
                        c2_data: np.ndarray,
                        t1: np.ndarray,
                        t2: np.ndarray,
                        phi_angles: np.ndarray,
                        title: str,
                        filename: str) -> None:
        """Plot C2 correlation function as heatmap."""
        try:
            # Create figure with subplots for different angles
            n_angles = len(phi_angles)
            cols = min(4, n_angles)
            rows = (n_angles + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
            if n_angles == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, phi in enumerate(phi_angles):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Plot heatmap for this angle
                im = ax.imshow(
                    c2_data[i], 
                    extent=[t1.min(), t1.max(), t2.min(), t2.max()],
                    origin='lower',
                    aspect='auto',
                    cmap='viridis'
                )
                
                ax.set_xlabel('t₁ (s)')
                ax.set_ylabel('t₂ (s)')
                ax.set_title(f'φ = {phi:.1f}°')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
            
            # Hide unused subplots
            for i in range(n_angles, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            # Save plot
            output_file = self.plots_dir / filename
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ C2 heatmap saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"C2 heatmap plotting failed: {e}")
    
    def _plot_time_averaged_correlation(self, data: Dict[str, Any]) -> None:
        """Plot time-averaged correlation function."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Average over time dimension
            c2_avg = np.mean(data['c2_exp'], axis=(1, 2))  # Average over t1, t2
            
            ax.plot(data['phi_angles'], c2_avg, 'o-', linewidth=2, markersize=6)
            ax.set_xlabel('Phi Angle (degrees)')
            ax.set_ylabel('⟨C₂⟩ (time-averaged)')
            ax.set_title('Time-Averaged Correlation vs Angle')
            ax.grid(True, alpha=0.3)
            
            output_file = self.plots_dir / 'time_averaged_correlation.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Time-averaged correlation saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Time-averaged correlation plotting failed: {e}")
    
    def _plot_angular_dependence(self, data: Dict[str, Any]) -> None:
        """Plot angular dependence of correlation function."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Select representative time points
            n_times = len(data['t1'])
            time_indices = [n_times//4, n_times//2, 3*n_times//4, n_times-1]
            time_labels = ['Early', 'Mid-early', 'Mid-late', 'Late']
            
            for i, (t_idx, label) in enumerate(zip(time_indices, time_labels)):
                ax = axes[i//2, i%2]
                
                # Extract correlation at this time
                c2_at_time = data['c2_exp'][:, t_idx, t_idx]  # Diagonal elements
                
                ax.plot(data['phi_angles'], c2_at_time, 'o-', label=f't = {data["t1"][t_idx]:.3f}s')
                ax.set_xlabel('Phi Angle (degrees)')
                ax.set_ylabel('C₂')
                ax.set_title(f'{label} Time Correlation')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            plt.suptitle('Angular Dependence at Different Times')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'angular_dependence.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Angular dependence saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Angular dependence plotting failed: {e}")
    
    def _plot_data_quality_diagnostics(self, data: Dict[str, Any]) -> None:
        """Plot data quality diagnostic information."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Data statistics
            c2_mean = np.mean(data['c2_exp'])
            c2_std = np.std(data['c2_exp'])
            c2_min = np.min(data['c2_exp'])
            c2_max = np.max(data['c2_exp'])
            
            # Histogram of correlation values
            axes[0, 0].hist(data['c2_exp'].flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('C₂ Value')
            axes[0, 0].set_ylabel('Frequency') 
            axes[0, 0].set_title('Distribution of C₂ Values')
            axes[0, 0].axvline(c2_mean, color='red', linestyle='--', label=f'Mean: {c2_mean:.3f}')
            axes[0, 0].legend()
            
            # Time evolution of mean correlation
            time_evolution = np.mean(data['c2_exp'], axis=(0, 2))  # Average over angles and t2
            axes[0, 1].plot(data['t1'], time_evolution, 'b-', linewidth=2)
            axes[0, 1].set_xlabel('Time t₁ (s)')
            axes[0, 1].set_ylabel('⟨C₂⟩')
            axes[0, 1].set_title('Time Evolution of Mean Correlation')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Data quality metrics
            quality_metrics = [
                f'Mean: {c2_mean:.4f}',
                f'Std Dev: {c2_std:.4f}',
                f'Min: {c2_min:.4f}',
                f'Max: {c2_max:.4f}',
                f'Dynamic Range: {c2_max - c2_min:.4f}',
                f'SNR Estimate: {c2_mean/c2_std:.2f}'
            ]
            
            axes[1, 0].text(0.1, 0.9, '\n'.join(quality_metrics), 
                           transform=axes[1, 0].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Data Quality Metrics')
            axes[1, 0].axis('off')
            
            # Experimental parameters
            exp_params = [
                f'q: {data["q"]:.4f} Å⁻¹',
                f'L: {data["L"]:.2f} mm',
                f'Angles: {len(data["phi_angles"])} points',
                f'Time points: {len(data["t1"])}',
                f'Data shape: {data["c2_exp"].shape}',
                f'Total points: {data["c2_exp"].size:,}'
            ]
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(exp_params),
                           transform=axes[1, 1].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Experimental Parameters')
            axes[1, 1].axis('off')
            
            plt.suptitle('Data Quality Diagnostics')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'data_quality_diagnostics.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Data quality diagnostics saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Data quality diagnostics plotting failed: {e}")
    
    def _plot_fit_comparison(self, result: ResultType, data: Dict[str, Any]) -> None:
        """Plot comparison between experimental and fitted data.""" 
        try:
            # Compute fitted data
            fitted_data = self._compute_fitted_data(result, data)
            
            # Create comparison heatmaps
            n_angles = min(4, len(data['phi_angles']))  # Limit to 4 for visibility
            
            fig, axes = plt.subplots(3, n_angles, figsize=(4*n_angles, 12))
            if n_angles == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(n_angles):
                phi = data['phi_angles'][i]
                
                # Experimental data
                im1 = axes[0, i].imshow(data['c2_exp'][i], cmap='viridis', aspect='auto')
                axes[0, i].set_title(f'Experimental (φ={phi:.1f}°)')
                plt.colorbar(im1, ax=axes[0, i])
                
                # Fitted data
                im2 = axes[1, i].imshow(fitted_data['c2_fitted'][i], cmap='viridis', aspect='auto')
                axes[1, i].set_title(f'Fitted (φ={phi:.1f}°)')
                plt.colorbar(im2, ax=axes[1, i])
                
                # Residuals
                im3 = axes[2, i].imshow(fitted_data['residuals'][i], cmap='RdBu_r', aspect='auto')
                axes[2, i].set_title(f'Residuals (φ={phi:.1f}°)')
                axes[2, i].set_xlabel('t₁ index')
                plt.colorbar(im3, ax=axes[2, i])
                
                if i == 0:
                    axes[0, i].set_ylabel('t₂ index')
                    axes[1, i].set_ylabel('t₂ index') 
                    axes[2, i].set_ylabel('t₂ index')
            
            plt.suptitle('Experimental vs Fitted Data Comparison')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'fit_comparison.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Fit comparison saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Fit comparison plotting failed: {e}")
    
    def _plot_residual_analysis(self, result: ResultType, data: Dict[str, Any]) -> None:
        """Plot residual analysis diagnostics."""
        try:
            fitted_data = self._compute_fitted_data(result, data)
            residuals = fitted_data['residuals']
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Residuals histogram
            axes[0, 0].hist(residuals.flatten(), bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Residual')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Residuals Distribution')
            
            # Q-Q plot for normality check
            from scipy import stats
            stats.probplot(residuals.flatten(), dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot (Normality Check)')
            
            # Residuals vs fitted values
            fitted_values = fitted_data['c2_fitted'].flatten()
            axes[1, 0].scatter(fitted_values, residuals.flatten(), alpha=0.5, s=1)
            axes[1, 0].axhline(0, color='red', linestyle='--')
            axes[1, 0].set_xlabel('Fitted Values')
            axes[1, 0].set_ylabel('Residuals')
            axes[1, 0].set_title('Residuals vs Fitted Values')
            
            # Residuals statistics
            residual_stats = [
                f'Mean: {np.mean(residuals):.6f}',
                f'Std Dev: {np.std(residuals):.6f}',
                f'Min: {np.min(residuals):.6f}', 
                f'Max: {np.max(residuals):.6f}',
                f'RMS: {np.sqrt(np.mean(residuals**2)):.6f}',
                f'Chi-squared: {getattr(result, "chi_squared", "N/A")}'
            ]
            
            axes[1, 1].text(0.1, 0.9, '\n'.join(residual_stats),
                           transform=axes[1, 1].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray'))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].set_title('Residual Statistics')
            axes[1, 1].axis('off')
            
            plt.suptitle('Residual Analysis')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'residual_analysis.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Residual analysis saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Residual analysis plotting failed: {e}")
    
    def _plot_parameter_results(self, result: ResultType) -> None:
        """Plot parameter estimation results."""
        try:
            # Extract parameters and uncertainties
            params = getattr(result, 'mean_params', [])
            uncertainties = getattr(result, 'std_params', None)
            
            if len(params) == 0:
                return
            
            # Get parameter names
            analysis_mode = getattr(result, 'analysis_mode', 'laminar_flow')
            param_names = self._get_parameter_names(analysis_mode)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            x_pos = np.arange(len(params))
            
            if uncertainties is not None:
                ax.errorbar(x_pos, params, yerr=uncertainties, 
                           fmt='o', capsize=5, capthick=2, markersize=8)
            else:
                ax.plot(x_pos, params, 'o', markersize=8)
            
            ax.set_xlabel('Parameter')
            ax.set_ylabel('Value')
            ax.set_title('Parameter Estimation Results')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(param_names[:len(params)], rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add parameter values as text
            for i, (param, value) in enumerate(zip(param_names[:len(params)], params)):
                if uncertainties is not None and i < len(uncertainties):
                    label = f'{value:.4f}±{uncertainties[i]:.4f}'
                else:
                    label = f'{value:.4f}'
                ax.text(i, value, label, ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            output_file = self.plots_dir / 'parameter_results.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Parameter results saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Parameter results plotting failed: {e}")
    
    def _plot_vi_diagnostics(self, result: VIResult) -> None:
        """Plot VI-specific diagnostic plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Convergence info
            convergence_info = [
                f'Final ELBO: {result.final_elbo:.4f}',
                f'KL Divergence: {result.kl_divergence:.4f}',
                f'Likelihood: {result.likelihood:.4f}',
                f'Converged: {result.converged}',
                f'Iterations: {result.n_iterations}',
                f'Computation Time: {result.computation_time:.2f}s'
            ]
            
            axes[0, 0].text(0.1, 0.9, '\n'.join(convergence_info),
                           transform=axes[0, 0].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_title('VI Optimization Info')
            axes[0, 0].axis('off')
            
            # Hide other subplots for now (can add ELBO trace if available)
            for ax in axes.flat[1:]:
                ax.axis('off')
            
            plt.suptitle('Variational Inference Diagnostics')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'vi_diagnostics.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ VI diagnostics saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"VI diagnostics plotting failed: {e}")
    
    def _plot_mcmc_diagnostics(self, result: MCMCResult) -> None:
        """Plot MCMC-specific diagnostic plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Convergence diagnostics
            r_hat = getattr(result, 'r_hat', [])
            ess = getattr(result, 'ess', [])
            
            if len(r_hat) > 0:
                axes[0, 0].bar(range(len(r_hat)), r_hat)
                axes[0, 0].axhline(1.1, color='red', linestyle='--', label='Target < 1.1')
                axes[0, 0].set_xlabel('Parameter')
                axes[0, 0].set_ylabel('R̂')
                axes[0, 0].set_title('Gelman-Rubin Diagnostic')
                axes[0, 0].legend()
            
            if len(ess) > 0:
                axes[0, 1].bar(range(len(ess)), ess)
                axes[0, 1].axhline(100, color='red', linestyle='--', label='Target > 100')
                axes[0, 1].set_xlabel('Parameter')
                axes[0, 1].set_ylabel('ESS')
                axes[0, 1].set_title('Effective Sample Size')
                axes[0, 1].legend()
            
            # Sampling info
            sampling_info = [
                f'Samples: {result.n_samples}',
                f'Chains: {getattr(result, "n_chains", "N/A")}',
                f'Max R̂: {np.max(r_hat) if len(r_hat) > 0 else "N/A"}',
                f'Min ESS: {np.min(ess) if len(ess) > 0 else "N/A"}',
                f'Computation Time: {result.computation_time:.2f}s'
            ]
            
            axes[1, 0].text(0.1, 0.9, '\n'.join(sampling_info),
                           transform=axes[1, 0].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Sampling Information')
            axes[1, 0].axis('off')
            
            axes[1, 1].axis('off')  # Reserved for trace plots if available
            
            plt.suptitle('MCMC Diagnostics')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'mcmc_diagnostics.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ MCMC diagnostics saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"MCMC diagnostics plotting failed: {e}")
    
    def _plot_hybrid_diagnostics(self, result: HybridResult) -> None:
        """Plot Hybrid-specific diagnostic plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # VI phase info
            vi_info = [
                f'VI ELBO: {result.vi_result.final_elbo:.4f}',
                f'VI Converged: {result.vi_result.converged}',
                f'VI Time: {result.vi_result.computation_time:.2f}s'
            ]
            
            axes[0, 0].text(0.1, 0.9, '\n'.join(vi_info),
                           transform=axes[0, 0].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen'))
            axes[0, 0].set_xlim(0, 1)
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].set_title('VI Phase Results')
            axes[0, 0].axis('off')
            
            # MCMC phase info
            r_hat = getattr(result.mcmc_result, 'r_hat', [])
            ess = getattr(result.mcmc_result, 'ess', [])
            
            mcmc_info = [
                f'MCMC Samples: {result.mcmc_result.n_samples}',
                f'Max R̂: {np.max(r_hat) if len(r_hat) > 0 else "N/A"}',
                f'Min ESS: {np.min(ess) if len(ess) > 0 else "N/A"}',
                f'MCMC Time: {result.mcmc_result.computation_time:.2f}s'
            ]
            
            axes[0, 1].text(0.1, 0.9, '\n'.join(mcmc_info),
                           transform=axes[0, 1].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_title('MCMC Phase Results')
            axes[0, 1].axis('off')
            
            # Recommendation
            recommendation = [
                f'Recommended Method: {result.recommended_method}',
                f'Total Time: {result.total_computation_time:.2f}s',
                f'Quality Score: {getattr(result, "quality_score", "N/A")}'
            ]
            
            axes[1, 0].text(0.1, 0.9, '\n'.join(recommendation),
                           transform=axes[1, 0].transAxes, fontsize=12,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow'))
            axes[1, 0].set_xlim(0, 1)
            axes[1, 0].set_ylim(0, 1)
            axes[1, 0].set_title('Hybrid Recommendation')
            axes[1, 0].axis('off')
            
            axes[1, 1].axis('off')
            
            plt.suptitle('Hybrid VI→MCMC Diagnostics')
            plt.tight_layout()
            
            output_file = self.plots_dir / 'hybrid_diagnostics.png'
            plt.savefig(output_file)
            plt.close()
            
            logger.debug(f"✓ Hybrid diagnostics saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Hybrid diagnostics plotting failed: {e}")
    
    def _parse_phi_angles(self, phi_angles_str: Optional[str], config: Dict[str, Any]) -> np.ndarray:
        """Parse phi angles from string or config."""
        if phi_angles_str:
            from homodyne.cli.validators import validate_phi_angles
            return np.array(validate_phi_angles(phi_angles_str))
        else:
            # Use default angles from config or standard set
            default_angles = config.get('experimental_data', {}).get('phi_angles', [0, 36, 72, 108, 144])
            return np.array(default_angles)
    
    def _generate_simulated_data(self, config: Dict[str, Any], phi_angles: np.ndarray, 
                               contrast: float, offset: float) -> Dict[str, Any]:
        """Generate simulated data using theoretical model."""
        # This is a simplified implementation - in practice would use TheoryEngine
        
        # Default time grid
        t_max = config.get('experimental_data', {}).get('t_max', 1.0)
        n_times = config.get('experimental_data', {}).get('n_times', 50)
        
        t1 = np.linspace(0.001, t_max, n_times)
        t2 = np.linspace(0.001, t_max, n_times)
        
        # Generate simple theoretical correlation (placeholder)
        T1, T2 = np.meshgrid(t1, t2)
        PHI = phi_angles[:, np.newaxis, np.newaxis]
        
        # Simple exponential decay with angular dependence (placeholder)
        g1_theory = np.exp(-0.1 * (T1 + T2)) * np.cos(PHI * np.pi / 180)
        c2_theory = contrast * (g1_theory ** 2) + offset
        
        return {
            'c2_theory': c2_theory,
            't1': t1,
            't2': t2,
            'phi_angles': phi_angles
        }
    
    def _compute_fitted_data(self, result: ResultType, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Compute fitted correlation function."""
        # Simplified implementation - would use results_manager in practice
        try:
            from homodyne.workflows.results_manager import ResultsManager
            
            # Create a temporary results manager to compute fitted data
            temp_manager = ResultsManager(Path.cwd(), {})
            return temp_manager._compute_fitted_correlation(result, data)
            
        except Exception as e:
            logger.warning(f"Could not compute fitted data for plotting: {e}")
            # Return placeholder data
            return {
                'c2_fitted': np.zeros_like(data['c2_exp']),
                'residuals': np.zeros_like(data['c2_exp'])
            }
    
    def _get_parameter_names(self, analysis_mode: str) -> List[str]:
        """Get parameter names for the analysis mode."""
        if analysis_mode in ['static_isotropic', 'static_anisotropic']:
            return ['D₀', 'α', 'D_offset', 'contrast', 'offset']
        else:  # laminar_flow
            return ['D₀', 'α', 'D_offset', 'γ̇₀', 'β', 'γ̇_offset', 'φ₀', 'contrast', 'offset']
    
    def _plot_parameter_sensitivity(self, config: Dict[str, Any], phi_angles: np.ndarray) -> None:
        """Plot parameter sensitivity analysis (placeholder)."""
        # This would implement sensitivity analysis plots
        # For now, just create a placeholder
        logger.debug("Parameter sensitivity plotting not yet implemented")