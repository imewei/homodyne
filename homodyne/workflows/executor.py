"""
Method Execution Controller for Homodyne v2
===========================================

Coordinates execution of different optimization methods (VI, MCMC, Hybrid)
with unified error handling, dataset optimization, and hardware management.

Key Features:
- Method-agnostic execution interface
- Automatic dataset size optimization
- GPU/CPU hardware management
- Comprehensive error handling and recovery
- Performance monitoring and logging
"""

import time
from typing import Any, Dict, Optional, Union

import numpy as np

from homodyne.data.optimization import optimize_for_method
from homodyne.optimization.hybrid import HybridResult, optimize_hybrid
from homodyne.optimization.mcmc import MCMCResult, fit_mcmc_jax
from homodyne.optimization.variational import VIResult, fit_vi_jax
from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


class MethodExecutor:
    """
    Method execution controller with unified interface.

    Handles execution of all optimization methods with consistent
    error handling, dataset optimization, and performance monitoring.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        force_cpu: bool = False,
        gpu_memory_fraction: float = 0.8,
        disable_dataset_optimization: bool = False,
    ):
        """
        Initialize method executor.

        Args:
            config: Analysis configuration
            force_cpu: Force CPU-only processing
            gpu_memory_fraction: GPU memory fraction to use
            disable_dataset_optimization: Disable automatic optimization
        """
        self.config = config
        self.force_cpu = force_cpu
        self.gpu_memory_fraction = gpu_memory_fraction
        self.disable_dataset_optimization = disable_dataset_optimization

        # Setup hardware environment
        self._setup_hardware_environment()

        logger.info(f"Method executor initialized:")
        logger.info(f"  Hardware: {'CPU-only' if force_cpu else 'GPU-enabled'}")
        logger.info(
            f"  Dataset optimization: {'disabled' if disable_dataset_optimization else 'enabled'}"
        )

    def _setup_hardware_environment(self) -> None:
        """Setup hardware environment for computations."""
        try:
            if not self.force_cpu:
                from homodyne.runtime.gpu.wrapper import setup_gpu_environment

                gpu_setup = setup_gpu_environment(quiet=False)

                if gpu_setup:
                    logger.info("✓ GPU environment configured")
                else:
                    logger.info("GPU not available, using CPU")
            else:
                logger.info("✓ CPU-only mode enabled")

        except Exception as e:
            logger.warning(f"Hardware setup warning: {e}")
            logger.info("Continuing with default hardware configuration")

    @log_performance()
    def execute_vi(
        self,
        data: np.ndarray,
        sigma: Optional[np.ndarray],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        estimate_noise: bool = False,
        noise_model: str = "hierarchical",
    ) -> Optional[VIResult]:
        """
        Execute Variational Inference optimization.

        Args:
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q, L: Experimental parameters
            estimate_noise: Enable hybrid NumPyro noise estimation
            noise_model: Noise model type for estimation

        Returns:
            VI optimization result or None if failed
        """
        try:
            logger.info("🎲 Starting VI+JAX optimization")
            start_time = time.time()

            # Get analysis mode from config
            analysis_mode = self._get_analysis_mode()

            # Setup method parameters
            vi_params = self._get_vi_parameters()

            # Apply dataset optimization if enabled
            if not self.disable_dataset_optimization:
                optimization_config = optimize_for_method(
                    data, sigma, t1, t2, phi, method="vi"
                )
                dataset_category = optimization_config["dataset_info"].category
                logger.info(f"Dataset optimization: {dataset_category} dataset")

                # Apply optimized parameters
                vi_params.update(
                    {
                        "enable_dataset_optimization": True,
                        "dataset_info": optimization_config["dataset_info"],
                    }
                )

            # Add noise estimation parameters if noise estimation is enabled
            if estimate_noise:
                noise_params = self._get_noise_estimation_parameters()
                
                # Apply CLI/config precedence logic and log parameter sources
                final_noise_model = self._resolve_noise_model_precedence(
                    cli_model=noise_model, 
                    config_model=noise_params.get("config_noise_model", "hierarchical")
                )
                
                # Log parameter sources for user clarity
                self._log_noise_parameter_sources(noise_params, estimate_noise, final_noise_model)
                
                # Remove config_noise_model from params to avoid conflicts
                noise_params.pop("config_noise_model", None)
                
                # Update vi_params but don't include noise_model (passed separately)
                vi_params.update(noise_params)
                
                # Use resolved noise model
                noise_model = final_noise_model

            # Execute VI optimization
            result = fit_vi_jax(
                data=data,
                sigma=sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                analysis_mode=analysis_mode,
                estimate_noise=estimate_noise,
                noise_model=noise_model,
                **vi_params,
            )

            # Log results summary
            execution_time = time.time() - start_time
            logger.info(f"✓ VI optimization completed in {execution_time:.2f}s")
            logger.info(f"  Final ELBO: {result.final_elbo:.4f}")
            logger.info(f"  Chi-squared: {result.chi_squared:.4f}")
            logger.info(f"  Converged: {result.converged}")

            return result

        except Exception as e:
            logger.error(f"❌ VI execution failed: {e}")
            self._log_execution_error("VI", e)
            return None

    @log_performance()
    def execute_mcmc(
        self,
        data: np.ndarray,
        sigma: Optional[np.ndarray],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        estimate_noise: bool = False,
        noise_model: str = "hierarchical",
    ) -> Optional[MCMCResult]:
        """
        Execute MCMC sampling optimization.

        Args:
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q, L: Experimental parameters
            estimate_noise: Enable hybrid NumPyro noise estimation
            noise_model: Noise model type for estimation

        Returns:
            MCMC optimization result or None if failed
        """
        try:
            logger.info("🎰 Starting MCMC+JAX sampling")
            start_time = time.time()

            # Get analysis mode from config
            analysis_mode = self._get_analysis_mode()

            # Setup method parameters
            mcmc_params = self._get_mcmc_parameters()

            # Apply dataset optimization if enabled
            if not self.disable_dataset_optimization:
                optimization_config = optimize_for_method(
                    data, sigma, t1, t2, phi, method="mcmc"
                )
                dataset_category = optimization_config["dataset_info"].category
                logger.info(f"Dataset optimization: {dataset_category} dataset")

                # Apply optimized parameters
                mcmc_params.update(
                    {
                        "enable_dataset_optimization": True,
                        "dataset_info": optimization_config["dataset_info"],
                    }
                )

            # Add noise estimation parameters if noise estimation is enabled
            if estimate_noise:
                noise_params = self._get_noise_estimation_parameters()
                
                # Apply CLI/config precedence logic and log parameter sources
                final_noise_model = self._resolve_noise_model_precedence(
                    cli_model=noise_model, 
                    config_model=noise_params.get("config_noise_model", "hierarchical")
                )
                
                # Log parameter sources for user clarity
                self._log_noise_parameter_sources(noise_params, estimate_noise, final_noise_model)
                
                # Remove config_noise_model from params to avoid conflicts
                noise_params.pop("config_noise_model", None)
                
                # Update mcmc_params but don't include noise_model (passed separately)
                mcmc_params.update(noise_params)
                
                # Use resolved noise model
                noise_model = final_noise_model

            # Execute MCMC sampling
            result = fit_mcmc_jax(
                data=data,
                sigma=sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                analysis_mode=analysis_mode,
                estimate_noise=estimate_noise,
                noise_model=noise_model,
                **mcmc_params,
            )

            # Log results summary
            execution_time = time.time() - start_time
            logger.info(f"✓ MCMC sampling completed in {execution_time:.2f}s")
            logger.info(f"  Posterior samples: {result.n_samples}")
            logger.info(f"  R-hat: {np.mean(result.r_hat):.3f} (target: <1.1)")
            logger.info(f"  ESS: {np.mean(result.ess):.0f}")

            return result

        except Exception as e:
            logger.error(f"❌ MCMC execution failed: {e}")
            self._log_execution_error("MCMC", e)
            return None

    @log_performance()
    def execute_hybrid(
        self,
        data: np.ndarray,
        sigma: Optional[np.ndarray],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        estimate_noise: bool = False,
        noise_model: str = "hierarchical",
    ) -> Optional[HybridResult]:
        """
        Execute Hybrid VI→MCMC optimization pipeline.

        Args:
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2: Time grids
            phi: Angle grid
            q, L: Experimental parameters
            estimate_noise: Enable hybrid NumPyro noise estimation
            noise_model: Noise model type for estimation

        Returns:
            Hybrid optimization result or None if failed
        """
        try:
            logger.info("🔄 Starting Hybrid VI→MCMC pipeline")
            start_time = time.time()

            # Get analysis mode from config
            analysis_mode = self._get_analysis_mode()

            # Setup hybrid parameters (combines VI and MCMC settings)
            hybrid_params = self._get_hybrid_parameters()

            # Apply dataset optimization if enabled
            if not self.disable_dataset_optimization:
                optimization_config = optimize_for_method(
                    data,
                    sigma,
                    t1,
                    t2,
                    phi,
                    method="vi",  # Use VI settings for initial phase
                )
                dataset_category = optimization_config["dataset_info"].category
                logger.info(f"Dataset optimization: {dataset_category} dataset")

                # Apply optimized parameters
                hybrid_params.update(
                    {
                        "enable_dataset_optimization": True,
                        "dataset_info": optimization_config["dataset_info"],
                    }
                )

            # Add noise estimation parameters if noise estimation is enabled
            if estimate_noise:
                noise_params = self._get_noise_estimation_parameters()
                
                # Apply CLI/config precedence logic and log parameter sources
                final_noise_model = self._resolve_noise_model_precedence(
                    cli_model=noise_model, 
                    config_model=noise_params.get("config_noise_model", "hierarchical")
                )
                
                # Log parameter sources for user clarity
                self._log_noise_parameter_sources(noise_params, estimate_noise, final_noise_model)
                
                # Remove config_noise_model from params to avoid conflicts
                noise_params.pop("config_noise_model", None)
                
                # Update hybrid_params but don't include noise_model (passed separately)
                hybrid_params.update(noise_params)
                
                # Use resolved noise model
                noise_model = final_noise_model

            # Execute hybrid optimization
            result = optimize_hybrid(
                data=data,
                sigma=sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                analysis_mode=analysis_mode,
                estimate_noise=estimate_noise,
                noise_model=noise_model,
                **hybrid_params,
            )

            # Log results summary
            execution_time = time.time() - start_time
            logger.info(f"✓ Hybrid optimization completed in {execution_time:.2f}s")
            logger.info(f"  VI Phase: ELBO={result.vi_result.final_elbo:.4f}")
            logger.info(f"  MCMC Phase: R-hat={np.mean(result.mcmc_result.r_hat):.3f}")
            logger.info(f"  Recommended method: {result.recommended_method}")

            return result

        except Exception as e:
            logger.error(f"❌ Hybrid execution failed: {e}")
            self._log_execution_error("Hybrid", e)
            return None

    def _get_analysis_mode(self) -> str:
        """
        Get analysis mode from configuration.

        Returns:
            Analysis mode string
        """
        return self.config.get("analysis_mode", "laminar_flow")

    def _get_vi_parameters(self) -> Dict[str, Any]:
        """
        Get VI-specific parameters from configuration.

        Returns:
            VI parameter dictionary
        """
        vi_config = self.config.get("optimization", {}).get("vi", {})

        return {
            "n_iterations": vi_config.get("n_iterations", 2000),
            "learning_rate": vi_config.get("learning_rate", 0.01),
            "convergence_tol": vi_config.get("convergence_tol", 1e-6),
            "n_elbo_samples": vi_config.get("n_elbo_samples", 1),
        }

    def _get_mcmc_parameters(self) -> Dict[str, Any]:
        """
        Get MCMC-specific parameters from configuration.

        Returns:
            MCMC parameter dictionary
        """
        mcmc_config = self.config.get("optimization", {}).get("mcmc", {})

        return {
            "n_samples": mcmc_config.get("n_samples", 1000),
            "n_warmup": mcmc_config.get("n_warmup", 1000),
            "n_chains": mcmc_config.get("n_chains", 4),
            "target_accept_prob": mcmc_config.get("target_accept_prob", 0.8),
        }

    def _get_hybrid_parameters(self) -> Dict[str, Any]:
        """
        Get Hybrid-specific parameters from configuration.

        Returns:
            Hybrid parameter dictionary
        """
        hybrid_config = self.config.get("optimization", {}).get("hybrid", {})
        vi_params = self._get_vi_parameters()
        mcmc_params = self._get_mcmc_parameters()

        # Combine VI and MCMC parameters for hybrid approach
        return {
            # VI phase parameters
            "vi_iterations": vi_params["n_iterations"],
            "vi_learning_rate": vi_params["learning_rate"],
            # MCMC phase parameters
            "mcmc_samples": mcmc_params["n_samples"],
            "mcmc_warmup": mcmc_params["n_warmup"],
            "mcmc_chains": mcmc_params["n_chains"],
            # Hybrid-specific parameters
            "use_vi_init": hybrid_config.get("use_vi_init", True),
            "convergence_threshold": hybrid_config.get("convergence_threshold", 0.1),
        }

    def _get_noise_estimation_parameters(self) -> Dict[str, Any]:
        """
        Get comprehensive noise estimation parameters from configuration.
        
        Supports all template parameters including Adam optimization,
        priors, posterior sampling, and validation settings.

        Returns:
            Complete noise estimation parameter dictionary
        """
        noise_config = self.config.get("noise_estimation", {})
        adam_config = noise_config.get("adam_config", {})
        priors_config = noise_config.get("priors", {})
        validation_config = noise_config.get("validation", {})
        
        # Get model-specific defaults based on noise model type
        # This will be overridden by CLI --noise-model if provided
        config_model = noise_config.get("model", "hierarchical")
        
        return {
            # Core noise model configuration
            "config_noise_model": config_model,
            
            # Adam optimization parameters
            "learning_rate": adam_config.get("learning_rate", 0.01),
            "adam_steps": adam_config.get("max_epochs", 300),
            "convergence_threshold": adam_config.get("convergence_threshold", 1e-6),
            "early_stopping": adam_config.get("early_stopping", True),
            "random_seed": adam_config.get("random_seed", 0),
            
            # Prior distribution parameters
            "priors": {
                "sigma_concentration": priors_config.get("sigma_concentration", 2.0),
                "sigma_rate": priors_config.get("sigma_rate", 20.0),
                "sigma_base_range": priors_config.get("sigma_base_range", [0.001, 1.0]),
                "sigma_scale_range": priors_config.get("sigma_scale_range", [0.0, 0.5]),
            },
            
            # Posterior sampling parameters
            "posterior_samples": noise_config.get("posterior_samples", 1000),
            
            # Validation and quality control
            "validation": {
                "check_convergence": validation_config.get("check_convergence", True),
                "reasonable_range": validation_config.get("reasonable_range", [1e-4, 1.0]),
                "warn_outliers": validation_config.get("warn_outliers", True),
            },
            
            # Method integration behavior
            "method_integration": noise_config.get("method_integration", {}),
        }

    def _resolve_noise_model_precedence(self, cli_model: str, config_model: str) -> str:
        """
        Resolve noise model precedence: CLI overrides config file.
        
        Args:
            cli_model: Noise model from CLI argument
            config_model: Noise model from config file
            
        Returns:
            Final noise model to use
        """
        # CLI always takes precedence over config
        # Default CLI value is "hierarchical", so if different from config, CLI was explicitly set
        if cli_model != config_model:
            logger.info(f"Noise model: {cli_model} (CLI override, config had: {config_model})")
            return cli_model
        else:
            logger.info(f"Noise model: {cli_model} (from config)")
            return cli_model

    def _log_noise_parameter_sources(self, noise_params: Dict[str, Any], 
                                   estimate_noise: bool, noise_model: str) -> None:
        """
        Log the sources of noise estimation parameters for user clarity.
        
        Args:
            noise_params: Complete noise parameter dictionary
            estimate_noise: CLI noise estimation flag
            noise_model: Final noise model selection
        """
        logger.info("🔧 Noise estimation parameter sources:")
        logger.info(f"  --estimate-noise: {estimate_noise} (CLI)")
        logger.info(f"  --noise-model: {noise_model} (CLI)")
        logger.info(f"  Adam learning rate: {noise_params['learning_rate']} (config)")
        logger.info(f"  Adam max epochs: {noise_params['adam_steps']} (config)")
        logger.info(f"  Early stopping: {noise_params['early_stopping']} (config)")
        logger.info(f"  Posterior samples: {noise_params['posterior_samples']} (config)")
        logger.info(f"  Validation enabled: {noise_params['validation']['check_convergence']} (config)")

    def _log_execution_error(self, method: str, error: Exception) -> None:
        """
        Log detailed execution error information.

        Args:
            method: Method name
            error: Exception that occurred
        """
        logger.error(f"Detailed {method} execution error:")
        logger.error(f"  Error type: {type(error).__name__}")
        logger.error(f"  Error message: {str(error)}")

        # Check for common error types and provide suggestions
        if "JAX" in str(error) or "jax" in str(error):
            logger.error("  Suggestion: Check JAX installation and GPU drivers")
        elif "memory" in str(error).lower():
            logger.error(
                "  Suggestion: Reduce dataset size or enable dataset optimization"
            )
        elif "convergence" in str(error).lower():
            logger.error(
                "  Suggestion: Adjust optimization parameters or try different method"
            )

        # Log system information for debugging
        try:
            import jax

            logger.debug(f"JAX devices: {jax.devices()}")
        except:
            logger.debug("JAX not available")


def estimate_method_performance(data_size: int, method: str) -> Dict[str, float]:
    """
    Estimate method performance characteristics.

    Args:
        data_size: Size of dataset
        method: Method name (vi, mcmc, hybrid)

    Returns:
        Performance estimates dictionary
    """
    # Base rates (points per second) from benchmarking
    base_rates = {
        "vi": 50000,  # Very fast
        "mcmc": 5000,  # Moderate
        "hybrid": 10000,  # Between VI and MCMC
    }

    # Size scaling factors
    if data_size < 1_000_000:
        scale = 1.0  # No scaling penalty
    elif data_size < 10_000_000:
        scale = 0.8  # Some overhead
    else:
        scale = 0.6  # More overhead

    effective_rate = base_rates.get(method, 1000) * scale
    estimated_time = data_size / effective_rate

    return {
        "estimated_seconds": estimated_time,
        "estimated_minutes": estimated_time / 60,
        "effective_rate": effective_rate,
        "scaling_factor": scale,
    }
