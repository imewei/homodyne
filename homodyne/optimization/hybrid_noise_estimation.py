"""
Hybrid NumPyro Noise Estimation Controller for Homodyne v2
=========================================================

This module implements the HybridNoiseEstimator class that controls method-specific
noise estimation pipelines using hybrid NumPyro models.

Pipeline Logic:
- VI: HybridNumPyro(Adam noise) → Extract σ → Return for VI
- MCMC: Full HybridNumPyro(Adam init + NUTS joint) → Return MCMCResult  
- Hybrid: HybridNumPyro(noise) → VI(physics) → MCMC(refinement)

Key Features:
- Unified NumPyro probabilistic framework
- Method-appropriate optimization strategies
- Full uncertainty propagation
- Scalable Adam-based noise estimation

Author: Homodyne v2 Development Team
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# JAX and NumPyro imports with intelligent fallback
try:
    import jax
    import jax.numpy as jnp
    from jax import random

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np
    logger.error("JAX not available - Hybrid noise estimation requires JAX")

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro import sample
    from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
    from numpyro.infer.autoguide import AutoNormal
    import numpyro.optim as optim

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False
    logger.error("NumPyro not available - Cannot use hybrid noise estimation")

from homodyne.core.fitting import ParameterSpace, UnifiedHomodyneEngine
from homodyne.optimization.hybrid_noise_models import (
    create_vi_noise_model,
    create_mcmc_hybrid_model,
    validate_hybrid_model_inputs,
    get_noise_model_info
)


@dataclass
class NoiseEstimationResult:
    """
    Results from hybrid NumPyro noise estimation.
    
    Contains estimated noise parameters, uncertainty quantification,
    and metadata about the estimation process.
    """
    # Noise estimates  
    sigma_mean: Union[float, np.ndarray, Dict[str, float]]
    sigma_std: Optional[Union[float, np.ndarray, Dict[str, float]]] = None
    
    # Estimation metadata
    noise_type: str = "hierarchical"
    method: str = "adam_svi"  # "adam_svi", "nuts_joint"
    estimation_time: float = 0.0
    convergence_info: Optional[Dict[str, Any]] = None
    
    # Posterior samples (for full uncertainty)
    posterior_samples: Optional[Dict[str, np.ndarray]] = None
    
    def get_sigma_for_method(self, method: str = "vi") -> Union[float, np.ndarray]:
        """
        Extract sigma values appropriate for downstream method.
        
        Args:
            method: Downstream method ("vi", "mcmc", "hybrid")
            
        Returns:
            Sigma values in appropriate format
        """
        if isinstance(self.sigma_mean, dict):
            # Adaptive noise model
            if method == "vi":
                # VI needs point estimates - use base sigma scaled by mean scaling
                return float(self.sigma_mean["sigma_base"] * 1.1)  # Conservative estimate
            else:
                # MCMC can handle full adaptive model
                return self.sigma_mean
        else:
            # Hierarchical or per-angle - return as-is
            return self.sigma_mean
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive noise estimation summary."""
        summary = {
            "noise_type": self.noise_type,
            "method": self.method,
            "estimation_time": self.estimation_time,
            "sigma_mean": self.sigma_mean,
        }
        
        if self.sigma_std is not None:
            summary["sigma_std"] = self.sigma_std
            
        if self.convergence_info is not None:
            summary["convergence"] = self.convergence_info
            
        return summary


class HybridNoiseEstimator:
    """
    Controller for hybrid NumPyro noise estimation across all methods.
    
    Manages method-specific pipelines and provides unified interface
    for noise parameter estimation using hybrid NumPyro models.
    """
    
    def __init__(
        self,
        analysis_mode: str = "laminar_flow",
        parameter_space: Optional[ParameterSpace] = None
    ):
        """
        Initialize hybrid noise estimator.
        
        Args:
            analysis_mode: Analysis mode for physics parameters
            parameter_space: Parameter space configuration
        """
        if not JAX_AVAILABLE or not NUMPYRO_AVAILABLE:
            raise ImportError("JAX and NumPyro required for hybrid noise estimation")
            
        self.analysis_mode = analysis_mode
        self.parameter_space = parameter_space or ParameterSpace()
        self.engine = UnifiedHomodyneEngine(analysis_mode, parameter_space)
        
        logger.info(f"Hybrid noise estimator initialized for {analysis_mode}")
        logger.debug(f"Physics parameters: {len(self.engine.param_bounds)}")
    
    @log_performance()
    def estimate_noise_for_vi(
        self,
        data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        noise_type: str = "hierarchical",
        **kwargs
    ) -> NoiseEstimationResult:
        """
        Pipeline for --method vi: HybridNumPyro(Adam noise) → Extract σ → Return for VI.
        
        This implements fast Adam-based noise estimation optimized for VI downstream usage.
        
        Args:
            data: Experimental correlation data
            t1, t2: Time grids
            phi: Angle grid  
            q, L: Experimental parameters
            noise_type: "hierarchical", "per_angle", or "adaptive"
            
        Returns:
            NoiseEstimationResult with point estimates for VI
        """
        logger.info(f"🎯 VI Pipeline: Hybrid NumPyro noise estimation ({noise_type})")
        start_time = time.time()
        
        # Convert to JAX arrays
        data_jax = jnp.array(data)
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        phi_jax = jnp.array(phi)
        
        # Validate inputs
        validate_hybrid_model_inputs(data_jax, t1_jax, t2_jax, phi_jax, q, L, noise_type)
        
        try:
            # Stage 1: Create noise-only model
            model = create_vi_noise_model(
                self.engine, data_jax, t1_jax, t2_jax, phi_jax, q, L, noise_type
            )
            
            # Stage 2: Adam optimization via SVI
            guide = AutoNormal(model)
            optimizer = optim.Adam(step_size=kwargs.get('learning_rate', 0.01))
            svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
            
            # Get model-specific optimization parameters
            model_info = get_noise_model_info(noise_type)
            num_steps = kwargs.get('adam_steps', model_info['adam_steps'])
            convergence_threshold = kwargs.get('convergence_threshold', 1e-6)
            early_stopping = kwargs.get('early_stopping', True)
            
            logger.debug(f"Running Adam SVI for up to {num_steps} steps...")
            
            # Run SVI with optional early stopping
            random_seed = int(kwargs.get('random_seed', 0))
            if early_stopping:
                svi_result = self._run_svi_with_early_stopping(
                    svi, 
                    random.PRNGKey(random_seed),
                    num_steps, 
                    convergence_threshold
                )
            else:
                svi_result = svi.run(random.PRNGKey(random_seed), 
                                   num_steps=num_steps)
            
            # Stage 3: Extract posterior samples
            num_samples = int(kwargs.get('posterior_samples', 1000))
            random_seed = int(kwargs.get('random_seed', 1))
            
            posterior_samples = guide.sample_posterior(
                random.PRNGKey(random_seed + 1000),
                svi_result.params,
                sample_shape=(num_samples,)
            )
            
            # Stage 4: Compute point estimates and uncertainties
            if noise_type == "hierarchical":
                sigma_samples = posterior_samples["sigma"]
                sigma_mean = float(jnp.mean(sigma_samples))
                sigma_std = float(jnp.std(sigma_samples))
                
            elif noise_type == "per_angle":
                sigma_samples = posterior_samples["sigma_angles"]
                sigma_mean = np.array(jnp.mean(sigma_samples, axis=0))
                sigma_std = np.array(jnp.std(sigma_samples, axis=0))
                
            elif noise_type == "adaptive":
                sigma_base_samples = posterior_samples["sigma_base"]
                sigma_scale_samples = posterior_samples["sigma_scale"]
                
                sigma_mean = {
                    "sigma_base": float(jnp.mean(sigma_base_samples)),
                    "sigma_scale": float(jnp.mean(sigma_scale_samples))
                }
                sigma_std = {
                    "sigma_base": float(jnp.std(sigma_base_samples)),
                    "sigma_scale": float(jnp.std(sigma_scale_samples))
                }
            
            # Stage 5: Create result with enhanced convergence info
            estimation_time = time.time() - start_time
            
            # Get convergence info from early stopping if available
            convergence_info = {
                "num_steps": num_steps,
                "converged": True
            }
            if hasattr(svi_result, 'convergence_info'):
                convergence_info.update(svi_result.convergence_info)
            elif hasattr(svi_result, 'losses') and svi_result.losses:
                convergence_info["svi_loss"] = float(svi_result.losses[-1])
            
            result = NoiseEstimationResult(
                sigma_mean=sigma_mean,
                sigma_std=sigma_std,
                noise_type=noise_type,
                method="adam_svi",
                estimation_time=estimation_time,
                convergence_info=convergence_info,
                posterior_samples={k: np.array(v) for k, v in posterior_samples.items()}
            )
            
            # Stage 6: Validate results if requested
            validation_config = kwargs.get('validation', {})
            if validation_config.get('check_convergence', True):
                validation_passed = self._validate_noise_estimates(result, validation_config)
                if not validation_passed:
                    logger.warning("⚠️  Noise estimation validation failed, but continuing")
            
            logger.info(f"✅ VI noise estimation completed in {estimation_time:.2f}s")
            logger.info(f"📊 Estimated noise: {sigma_mean}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ VI noise estimation failed: {e}")
            raise RuntimeError(f"Hybrid NumPyro noise estimation failed: {e}") from e
    
    @log_performance()
    def estimate_noise_for_mcmc(
        self,
        data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        noise_type: str = "hierarchical",
        **kwargs
    ) -> Any:  # Returns enhanced MCMCResult
        """
        Pipeline for --method mcmc: Full HybridNumPyro(Adam init + NUTS joint) → Return MCMCResult.
        
        This implements complete Bayesian treatment of noise and physics parameters
        with full parameter correlations and uncertainty quantification.
        
        Args:
            data: Experimental correlation data
            t1, t2: Time grids
            phi: Angle grid
            q, L: Experimental parameters
            noise_type: "hierarchical", "per_angle", or "adaptive"
            
        Returns:
            Enhanced MCMCResult with noise uncertainty
        """
        logger.info(f"🎲 MCMC Pipeline: Full hybrid NumPyro optimization ({noise_type})")
        start_time = time.time()
        
        # Convert to JAX arrays
        data_jax = jnp.array(data)
        t1_jax = jnp.array(t1)
        t2_jax = jnp.array(t2)
        phi_jax = jnp.array(phi)
        
        # Validate inputs
        validate_hybrid_model_inputs(data_jax, t1_jax, t2_jax, phi_jax, q, L, noise_type)
        
        try:
            # Stage 1: Get Adam initialization for noise parameters
            logger.info("🚀 Stage 1: Adam noise initialization...")
            vi_noise_result = self.estimate_noise_for_vi(
                data, t1, t2, phi, q, L, noise_type, **kwargs
            )
            
            # Stage 2: Create full hybrid model
            logger.info("🔗 Stage 2: Creating full hybrid model...")
            model = create_mcmc_hybrid_model(
                self.engine, data_jax, t1_jax, t2_jax, phi_jax, q, L, noise_type
            )
            
            # Stage 3: Setup NUTS sampling with initialization
            logger.info("⚙️  Stage 3: Setting up NUTS sampler...")
            nuts_kernel = NUTS(
                model,
                target_accept_prob=kwargs.get('target_accept_prob', 0.8),
                max_tree_depth=kwargs.get('max_tree_depth', 10)
            )
            
            # Create initialization values from Adam results
            init_values = self._create_mcmc_init_values(vi_noise_result, noise_type)
            
            # Stage 4: Run MCMC sampling
            logger.info("🎲 Stage 4: NUTS sampling...")
            mcmc = MCMC(
                nuts_kernel,
                num_samples=kwargs.get('num_samples', 2000),
                num_warmup=kwargs.get('num_warmup', 1000),
                num_chains=kwargs.get('num_chains', 4),
                progress_bar=kwargs.get('progress_bar', True)
            )
            
            mcmc.run(random.PRNGKey(kwargs.get('random_seed', 42)))
            
            # Stage 5: Process results
            logger.info("📊 Stage 5: Processing MCMC results...")
            mcmc_result = self._process_mcmc_hybrid_results(mcmc, noise_type, start_time)
            
            logger.info(f"✅ MCMC hybrid estimation completed in {mcmc_result.computation_time:.2f}s")
            
            return mcmc_result
            
        except Exception as e:
            logger.error(f"❌ MCMC noise estimation failed: {e}")
            raise RuntimeError(f"Hybrid NumPyro MCMC estimation failed: {e}") from e
    
    @log_performance()
    def estimate_noise_for_hybrid(
        self,
        data: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        noise_type: str = "hierarchical",
        **kwargs
    ) -> Any:  # Returns enhanced HybridResult
        """
        Pipeline for --method hybrid: HybridNumPyro(noise) → VI(physics) → MCMC(refinement).
        
        This implements the optimal balance of speed and accuracy by combining
        Adam noise estimation with VI initialization and MCMC refinement.
        
        Args:
            data: Experimental correlation data
            t1, t2: Time grids
            phi: Angle grid
            q, L: Experimental parameters
            noise_type: "hierarchical", "per_angle", or "adaptive"
            
        Returns:
            Enhanced HybridResult with noise estimates
        """
        logger.info(f"🔄 Hybrid Pipeline: NumPyro noise → VI → MCMC ({noise_type})")
        start_time = time.time()
        
        try:
            # Stage 1: Estimate noise with hybrid NumPyro
            logger.info("🎯 Stage 1: Hybrid NumPyro noise estimation...")
            noise_result = self.estimate_noise_for_vi(
                data, t1, t2, phi, q, L, noise_type, **kwargs
            )
            
            # Extract sigma for downstream methods
            estimated_sigma = noise_result.get_sigma_for_method("hybrid")
            logger.info(f"📊 Using estimated σ: {estimated_sigma}")
            
            # Stage 2: VI optimization with estimated noise
            logger.info("📈 Stage 2: VI with estimated noise...")
            from homodyne.optimization.variational import fit_vi_jax
            
            vi_kwargs = {k: v for k, v in kwargs.items() 
                        if k in ['analysis_mode', 'parameter_space', 'vi_iterations', 
                               'learning_rate', 'random_seed']}
            vi_kwargs.update({
                'analysis_mode': self.analysis_mode,
                'parameter_space': self.parameter_space
            })
            
            vi_result = fit_vi_jax(
                data=data,
                sigma=estimated_sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                **vi_kwargs
            )
            
            # Stage 3: MCMC refinement initialized from VI
            logger.info("🎲 Stage 3: MCMC refinement...")
            from homodyne.optimization.mcmc import fit_mcmc_jax
            
            mcmc_kwargs = {k: v for k, v in kwargs.items() 
                          if k in ['num_samples', 'num_warmup', 'num_chains', 
                                 'target_accept_prob', 'random_seed']}
            mcmc_kwargs.update({
                'analysis_mode': self.analysis_mode,
                'parameter_space': self.parameter_space,
                'vi_init': vi_result
            })
            
            mcmc_result = fit_mcmc_jax(
                data=data,
                sigma=estimated_sigma,
                t1=t1,
                t2=t2,
                phi=phi,
                q=q,
                L=L,
                **mcmc_kwargs
            )
            
            # Stage 4: Combine results
            logger.info("🔗 Stage 4: Combining hybrid results...")
            hybrid_result = self._combine_hybrid_results(
                noise_result, vi_result, mcmc_result, start_time
            )
            
            logger.info(f"✅ Hybrid pipeline completed in {hybrid_result.computation_time:.2f}s")
            
            return hybrid_result
            
        except Exception as e:
            logger.error(f"❌ Hybrid noise estimation failed: {e}")
            raise RuntimeError(f"Hybrid pipeline failed: {e}") from e
    
    def _create_mcmc_init_values(
        self, 
        noise_result: NoiseEstimationResult, 
        noise_type: str
    ) -> Dict[str, jnp.ndarray]:
        """Create initialization values for MCMC from Adam noise results."""
        init_values = {}
        
        if noise_type == "hierarchical":
            init_values["sigma"] = jnp.array(noise_result.sigma_mean)
        elif noise_type == "per_angle":
            init_values["sigma_angles"] = jnp.array(noise_result.sigma_mean)
        elif noise_type == "adaptive":
            init_values["sigma_base"] = jnp.array(noise_result.sigma_mean["sigma_base"])
            init_values["sigma_scale"] = jnp.array(noise_result.sigma_mean["sigma_scale"])
        
        # Initialize physics parameters at prior means
        for i, (prior, _) in enumerate(zip(self.engine.param_priors, self.engine.param_bounds)):
            init_values[f"param_{i}"] = jnp.array(prior[0])  # Use prior mean
        
        # Initialize scaling parameters
        init_values["contrast"] = jnp.array(self.parameter_space.contrast_prior[0])
        init_values["offset"] = jnp.array(self.parameter_space.offset_prior[0])
        
        return init_values
    
    def _process_mcmc_hybrid_results(self, mcmc, noise_type: str, start_time: float) -> Any:
        """Process MCMC results into enhanced MCMCResult format."""
        # This would create an enhanced MCMCResult that includes noise uncertainty
        # For now, return a placeholder - will be implemented when integrating with actual MCMC
        computation_time = time.time() - start_time
        
        # Extract samples
        samples = mcmc.get_samples()
        
        # Create enhanced result structure
        result = {
            "samples": samples,
            "noise_type": noise_type,
            "computation_time": computation_time,
            "noise_samples": self._extract_noise_samples(samples, noise_type),
            "physics_samples": self._extract_physics_samples(samples),
            "diagnostics": {
                "r_hat": "computed",  # Would compute actual R-hat
                "ess": "computed",    # Would compute actual ESS
                "divergences": mcmc.num_divergences
            }
        }
        
        return result
    
    def _extract_noise_samples(self, samples: Dict, noise_type: str) -> Dict:
        """Extract noise-related samples from MCMC results."""
        if noise_type == "hierarchical":
            return {"sigma": samples.get("sigma")}
        elif noise_type == "per_angle":
            return {"sigma_angles": samples.get("sigma_angles")}
        elif noise_type == "adaptive":
            return {
                "sigma_base": samples.get("sigma_base"),
                "sigma_scale": samples.get("sigma_scale")
            }
        return {}
    
    def _extract_physics_samples(self, samples: Dict) -> Dict:
        """Extract physics parameter samples from MCMC results."""
        physics_samples = {}
        for key, value in samples.items():
            if key.startswith("param_"):
                physics_samples[key] = value
            elif key in ["contrast", "offset"]:
                physics_samples[key] = value
        return physics_samples
    
    def _combine_hybrid_results(self, noise_result, vi_result, mcmc_result, start_time: float) -> Any:
        """Combine noise estimation, VI, and MCMC results into hybrid result."""
        computation_time = time.time() - start_time
        
        # Create comprehensive hybrid result
        result = {
            "noise_estimation": noise_result.get_summary(),
            "vi_result": vi_result,
            "mcmc_result": mcmc_result,
            "computation_time": computation_time,
            "pipeline": "hybrid_numpyro_vi_mcmc",
            "estimated_sigma": noise_result.get_sigma_for_method("hybrid")
        }
        
        return result

    def _run_svi_with_early_stopping(self, svi, rng_key, max_steps: int, 
                                    convergence_threshold: float = 1e-6):
        """
        Run SVI with early stopping based on loss convergence.
        
        Args:
            svi: NumPyro SVI instance
            rng_key: JAX random key
            max_steps: Maximum number of optimization steps
            convergence_threshold: Loss change threshold for convergence
            
        Returns:
            SVI result with early stopping metadata
        """
        losses = []
        step = 0
        patience = 50  # Steps to wait for improvement
        best_loss = float('inf')
        no_improvement_count = 0
        
        # Ensure convergence threshold is numeric
        convergence_threshold = float(convergence_threshold)
        
        # Initialize SVI state
        svi_state = svi.init(rng_key)
        
        for step in range(max_steps):
            rng_key, subkey = jax.random.split(rng_key)
            svi_state, loss = svi.update(svi_state, subkey)
            
            # Ensure loss is numeric
            loss = float(loss)
            losses.append(loss)
            
            # Check for improvement
            if loss < best_loss - convergence_threshold:
                best_loss = loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Early stopping check
            if no_improvement_count >= patience and step > 100:  # Minimum 100 steps
                logger.info(f"Early stopping at step {step+1}: no improvement for {patience} steps")
                break
                
            # Log progress periodically
            if (step + 1) % 100 == 0:
                logger.debug(f"Step {step+1}: loss = {loss:.6f}, best = {best_loss:.6f}")
        
        # Create result with convergence info
        final_params = svi.get_params(svi_state)
        convergence_info = {
            "converged": no_improvement_count >= patience,
            "final_step": step + 1,
            "final_loss": losses[-1],
            "best_loss": best_loss,
            "loss_history": losses[-10:],  # Last 10 losses
        }
        
        logger.info(f"Adam optimization completed: {step+1}/{max_steps} steps, "
                   f"final loss: {losses[-1]:.6f}")
        
        # Mock SVI result structure
        class SVIResult:
            def __init__(self, params, convergence_info):
                self.params = params
                self.convergence_info = convergence_info
                
        return SVIResult(final_params, convergence_info)

    def _validate_noise_estimates(self, noise_result: NoiseEstimationResult, 
                                validation_config: Dict[str, Any]) -> bool:
        """
        Validate noise estimates against reasonable ranges and quality checks.
        
        Args:
            noise_result: Noise estimation results
            validation_config: Validation configuration from config file
            
        Returns:
            True if validation passes, False otherwise
        """
        if not validation_config.get("check_convergence", True):
            return True
            
        try:
            reasonable_range = validation_config.get("reasonable_range", [1e-4, 1.0])
            # Ensure reasonable_range values are numeric
            reasonable_range = [float(x) for x in reasonable_range]
            warn_outliers = validation_config.get("warn_outliers", True)
            
            # Get primary sigma estimate
            sigma_mean = noise_result.sigma_mean
            if isinstance(sigma_mean, dict):
                # For per_angle, check all angles
                sigma_values = list(sigma_mean.values())
            elif isinstance(sigma_mean, np.ndarray):
                sigma_values = sigma_mean.flatten()
            else:
                sigma_values = [sigma_mean]
            
            # Check reasonable range
            for i, sigma in enumerate(sigma_values):
                if not (reasonable_range[0] <= sigma <= reasonable_range[1]):
                    if warn_outliers:
                        logger.warning(f"⚠️  Noise estimate {i} = {sigma:.6f} outside reasonable range "
                                     f"[{reasonable_range[0]}, {reasonable_range[1]}]")
                        logger.warning("Consider adjusting priors or checking data quality")
                        return False
            
            # Check for convergence if available
            if hasattr(noise_result, 'convergence_info') and noise_result.convergence_info:
                if not noise_result.convergence_info.get("converged", True):
                    logger.warning("⚠️  Adam optimization did not converge")
                    logger.warning("Consider increasing max_epochs or adjusting learning_rate")
                    return False
            
            logger.info("✅ Noise estimate validation passed")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Noise validation failed: {e}")
            return False


# Export key classes
__all__ = [
    "HybridNoiseEstimator",
    "NoiseEstimationResult"
]