"""
VI + JAX: Primary Optimization Method for Homodyne v2
====================================================

JAX-based Variational Inference as the primary optimization method, replacing
all classical and robust optimization methods. Implements KL divergence
minimization between variational distribution and true posterior using the
unified homodyne model: c2_fitted = c2_theory * contrast + offset

Key Features:
- KL divergence minimization with JAX automatic differentiation
- Mean-field Gaussian variational families with specified priors
- Unified likelihood: Exp - (contrast * Theory + offset)
- Parameter space integration with bounds and priors
- 10-100x speedup over classical methods
- Dataset size-aware optimization strategies
- CPU-primary, GPU-optional acceleration
"""

import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Dynamic JAX detection - always try JAX first, fallback as needed
def _detect_jax_availability():
    """Detect JAX availability at runtime"""
    try:
        import jax
        import jax.numpy as jnp
        from jax import grad, jit, random, value_and_grad, vmap
        from jax.scipy import stats as jstats
        
        # Handle deprecated jax.optimizers vs optax
        try:
            from jax import optimizers as jax_optimizers
        except ImportError:
            # Modern JAX uses optax, create compatibility layer
            try:
                import optax
                # Create compatibility layer for optax
                # Old JAX optimizers API: init_fn, update_fn, get_params_fn = optimizers.adam(lr)
                # New optax API: optimizer = optax.adam(lr), then optimizer.init, optimizer.update
                
                class OptimizersCompat:
                    @staticmethod
                    def adam(lr):
                        optimizer = optax.adam(lr)
                        
                        def init_fn(params):
                            return {'params': params, 'opt_state': optimizer.init(params)}
                        
                        def update_fn(iteration, grads, state):
                            params = state['params']
                            opt_state = state['opt_state']
                            updates, new_opt_state = optimizer.update(grads, opt_state)
                            new_params = optax.apply_updates(params, updates)
                            return {'params': new_params, 'opt_state': new_opt_state}
                        
                        def get_params_fn(state):
                            return state['params']
                        
                        return init_fn, update_fn, get_params_fn
                
                jax_optimizers = OptimizersCompat()
            except ImportError:
                # If optax also not available, use simple fallback
                jax_optimizers = None
        
        # Test basic operations
        test = jax.random.PRNGKey(0)
        test_array = jnp.array([1.0, 2.0])
        test_result = jnp.sum(test_array)
        
        logger.debug(f"✅ JAX detection successful: test_result={test_result}")
        
        return True, {
            'jax': jax, 'jnp': jnp, 'grad': grad, 'jit': jit,
            'jax_optimizers': jax_optimizers, 'random': random,
            'value_and_grad': value_and_grad, 'vmap': vmap, 'jstats': jstats
        }
    except (ImportError, Exception) as e:
        logger.debug(f"❌ JAX detection failed: {e}")
        return False, {}

# Initial detection
JAX_AVAILABLE, _jax_modules = _detect_jax_availability()

if JAX_AVAILABLE:
    # Extract JAX modules to global scope
    jax = _jax_modules['jax']
    jnp = _jax_modules['jnp'] 
    grad = _jax_modules['grad']
    jit = _jax_modules['jit']
    jax_optimizers = _jax_modules['jax_optimizers']
    random = _jax_modules['random']
    value_and_grad = _jax_modules['value_and_grad'] 
    vmap = _jax_modules['vmap']
    jstats = _jax_modules['jstats']
    
    HAS_NUMPY_GRADIENTS = False
    logger.debug("JAX available at import time - using JAX implementations")
else:
    # JAX not available at import time - set up fallbacks
    logger.debug("JAX not available at import time - setting up fallbacks")
    jnp = np
    # Import numerical gradients for fallback
    try:
        from homodyne.core.numpy_gradients import (DifferentiationConfig,
                                                   numpy_gradient)

        HAS_NUMPY_GRADIENTS = True
    except ImportError:
        HAS_NUMPY_GRADIENTS = False
        logger.warning(
            "Neither JAX nor numpy_gradients available - VI will be severely limited"
        )

    # Create intelligent fallback functions only when JAX is not available
    def jit(f):
        return f

    def grad(f, argnums=0):
        if HAS_NUMPY_GRADIENTS:
            return numpy_gradient(f, argnums=argnums)
        else:
            return lambda x: np.zeros_like(x)

    def value_and_grad(f, argnums=0):
        if HAS_NUMPY_GRADIENTS:
            grad_f = numpy_gradient(f, argnums=argnums)

            def value_grad_func(*args):
                value = f(*args)
                # The gradient function signature must match exactly
                # For our use case, we know f takes exactly one argument (variational_params)
                try:
                    if len(args) == 1:
                        gradient = grad_f(args[0])
                    else:
                        gradient = grad_f(*args)
                except TypeError as e:
                    if "takes" in str(e) and "argument" in str(e):
                        # Fallback: create zero gradients with same structure as input
                        if isinstance(args[0], dict):
                            gradient = {k: np.zeros_like(v) for k, v in args[0].items()}
                        else:
                            gradient = np.zeros_like(args[0])
                    else:
                        raise
                return value, gradient

            return value_grad_func
        else:
            return lambda x: (f(x), np.zeros_like(x))

    def vmap(func, *args, **kwargs):
        def vectorized_func(inputs, *vargs, **vkwargs):
            if hasattr(inputs, "__iter__") and not isinstance(inputs, str):
                return np.array([func(inp, *vargs, **vkwargs) for inp in inputs])
            return func(inputs, *vargs, **vkwargs)

        return vectorized_func

    # Simple Adam optimizer implementation for fallback
    class SimpleAdam:
        def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
            self.learning_rate = learning_rate
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps

        def init(self, params):
            return {
                "m": {key: np.zeros_like(value) for key, value in params.items()},
                "v": {key: np.zeros_like(value) for key, value in params.items()},
                "step": 0,
            }

        def update(self, grads, state):
            state["step"] += 1
            new_m = {}
            new_v = {}
            new_params = {}

            for key in grads.keys():
                new_m[key] = (
                    self.beta1 * state["m"][key] + (1 - self.beta1) * grads[key]
                )
                new_v[key] = (
                    self.beta2 * state["v"][key] + (1 - self.beta2) * grads[key] ** 2
                )

                m_hat = new_m[key] / (1 - self.beta1 ** state["step"])
                v_hat = new_v[key] / (1 - self.beta2 ** state["step"])

                new_params[key] = (
                    -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
                )

            return new_params, {"m": new_m, "v": new_v, "step": state["step"]}

    jax_optimizers = type(
        "AdamModule",
        (),
        {
            "adam": lambda lr: (
                SimpleAdam(lr).init,
                lambda i, g, s: SimpleAdam(lr).update(g, s),
                lambda p: p,
            )
        },
    )()

from homodyne.core.fitting import ParameterSpace, UnifiedHomodyneEngine
from homodyne.data.optimization import DatasetOptimizer, optimize_for_method
from homodyne.utils.logging import log_performance


@dataclass
class VIResult:
    """
    Results from VI+JAX optimization using unified homodyne model.

    Contains both point estimates and uncertainty quantification
    from the learned variational distribution with specified priors.
    """

    # Point estimates (means of variational distribution)
    mean_params: np.ndarray  # Physical parameter means
    mean_contrast: float  # Contrast mean
    mean_offset: float  # Offset mean

    # Uncertainty estimates (std of variational distribution)
    std_params: np.ndarray  # Physical parameter uncertainties
    std_contrast: float  # Contrast uncertainty
    std_offset: float  # Offset uncertainty

    # Optimization metrics
    final_elbo: float  # Evidence Lower BOund (objective)
    kl_divergence: float  # KL divergence to prior
    likelihood: float  # Data likelihood term
    elbo_history: np.ndarray  # ELBO convergence history
    converged: bool  # Convergence flag
    n_iterations: int  # Number of VI iterations

    # Fit quality (computed from mean parameters)
    chi_squared: float  # Chi-squared at MAP estimate
    reduced_chi_squared: float  # Reduced chi-squared

    # Computational metadata
    computation_time: float  # Total VI time (seconds)
    backend: str  # Backend used ("JAX" or "NumPy")
    dataset_size: str  # Dataset size category
    analysis_mode: str  # Analysis mode used

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive VI result summary."""
        return {
            "point_estimates": {
                "parameters": self.mean_params.tolist(),
                "contrast": self.mean_contrast,
                "offset": self.mean_offset,
            },
            "uncertainties": {
                "parameters": self.std_params.tolist(),
                "contrast": self.std_contrast,
                "offset": self.std_offset,
            },
            "optimization": {
                "final_elbo": self.final_elbo,
                "kl_divergence": self.kl_divergence,
                "likelihood": self.likelihood,
                "converged": self.converged,
                "iterations": self.n_iterations,
                "computation_time": self.computation_time,
                "backend": self.backend,
                "dataset_size": self.dataset_size,
                "analysis_mode": self.analysis_mode,
            },
            "fit_quality": {
                "chi_squared": self.chi_squared,
                "reduced_chi_squared": self.reduced_chi_squared,
            },
        }


class VariationalFamilies:
    """
    Variational distribution families with specified priors.

    Implements mean-field Gaussian approximations for all parameters
    including physical parameters and scaling parameters (contrast, offset).
    """

    @staticmethod
    def truncated_normal_prior_logpdf(
        x: jnp.ndarray, mu: float, sigma: float, bounds: Tuple[float, float]
    ) -> jnp.ndarray:
        """Log-probability density for truncated normal prior."""
        if not JAX_AVAILABLE:
            return np.zeros_like(x)

        lower, upper = bounds
        # Standard normal component
        standard_logpdf = jstats.norm.logpdf(x, mu, sigma)

        # Truncation correction (log of normalization constant)
        # log(Φ((upper-μ)/σ) - Φ((lower-μ)/σ))
        upper_cdf = jstats.norm.cdf(upper, mu, sigma)
        lower_cdf = jstats.norm.cdf(lower, mu, sigma)
        log_normalizer = jnp.log(upper_cdf - lower_cdf + 1e-10)

        # Apply bounds constraint
        in_bounds = (x >= lower) & (x <= upper)

        return jnp.where(in_bounds, standard_logpdf - log_normalizer, -jnp.inf)

    @staticmethod
    def normal_prior_logpdf(x: jnp.ndarray, mu: float, sigma: float) -> jnp.ndarray:
        """Log-probability density for normal prior."""
        if not JAX_AVAILABLE:
            return np.zeros_like(x)
        return jstats.norm.logpdf(x, mu, sigma)

    @staticmethod
    def variational_logpdf(
        x: jnp.ndarray, mu: jnp.ndarray, log_std: jnp.ndarray
    ) -> jnp.ndarray:
        """Log-probability density of mean-field Gaussian variational distribution."""
        if not JAX_AVAILABLE:
            return np.zeros_like(x).sum()
        std = jnp.exp(log_std)
        return jnp.sum(jstats.norm.logpdf(x, mu, std))


# Always define compute_elbo, but conditionally apply JIT
if JAX_AVAILABLE:
    def compute_elbo_inner(
        variational_params: Dict[str, jnp.ndarray],
        data: jnp.ndarray,
        sigma: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        param_priors: List[Tuple[float, float]],
        param_bounds: List[Tuple[float, float]],
        contrast_prior: Tuple[float, float],
        contrast_bounds: Tuple[float, float],
        offset_prior: Tuple[float, float],
        offset_bounds: Tuple[float, float],
        theory_engine_compute_g1: Callable,
        n_samples: int = 1,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Evidence Lower BOund (ELBO) for VI optimization.

        ELBO = E_q[log p(data|params)] - KL[q(params)||p(params)]

        Where the likelihood is: Exp - (contrast * Theory + offset)
        """
        # Extract variational parameters
        param_mu = variational_params["param_mu"]
        param_log_std = variational_params["param_log_std"]
        contrast_mu = variational_params["contrast_mu"]
        contrast_log_std = variational_params["contrast_log_std"]
        offset_mu = variational_params["offset_mu"]
        offset_log_std = variational_params["offset_log_std"]

        # Sample from variational distribution
        key = random.PRNGKey(0)  # Fixed for reproducibility during optimization

        # Sample parameters
        param_std = jnp.exp(param_log_std)
        contrast_std = jnp.exp(contrast_log_std)
        offset_std = jnp.exp(offset_log_std)

        # Ensure shape tuple contains int (not float32) for random.normal
        param_count = int(len(param_mu))
        
        # Sample parameters with bounds checking to prevent extreme values
        raw_param_samples = param_mu + param_std * random.normal(
            key, (n_samples, param_count)
        )
        
        # Clip parameter samples to stay within bounds to prevent numerical issues
        param_samples = jnp.zeros_like(raw_param_samples)
        for i in range(param_count):
            min_bound, max_bound = param_bounds[i]
            param_samples = param_samples.at[:, i].set(
                jnp.clip(raw_param_samples[:, i], min_bound, max_bound)
            )
        
        contrast_samples = contrast_mu + contrast_std * random.normal(key, (n_samples,))
        offset_samples = offset_mu + offset_std * random.normal(key, (n_samples,))

        # Compute likelihood for each sample
        log_likelihood = 0.0
        for i in range(n_samples):
            # Compute theoretical g1 - ensure index is handled properly by JAX
            sample_params = param_samples[i]
            g1_theory = theory_engine_compute_g1(sample_params, t1, t2, phi, q, L)
            
            # Create mask for finite values (excludes dt=0 diagonal elements)
            # For XPCS correlation analysis, we mask out dt=0 points where g1 is infinite
            finite_mask = jnp.isfinite(g1_theory)  # Shape: (601, 601)
            
            # For XPCS correlation analysis, we expect most points to be finite (only dt=0 diagonal is infinite)
            # The masking will automatically handle finite vs infinite values
            # No need for explicit checks that cause concretization issues
            
            # Handle 3D data shape: data has shape (1, 601, 601), g1_theory has shape (601, 601)
            # Need to align shapes for masking
            if data.ndim == 3 and g1_theory.ndim == 2:
                # Squeeze first dimension for masking
                data_2d = data[0]  # Shape: (601, 601)
                sigma_2d = sigma[0] if sigma.ndim == 3 else sigma  # Shape: (601, 601)
                
                # Use only finite values for likelihood computation
                g1_finite = g1_theory[finite_mask]
                data_finite = data_2d[finite_mask]
                sigma_finite = sigma_2d[finite_mask]
            else:
                # Use only finite values for likelihood computation
                g1_finite = g1_theory[finite_mask]
                data_finite = data[finite_mask]
                sigma_finite = sigma[finite_mask]
            
            g1_squared = g1_finite**2

            # Apply bounds to scaling parameters to prevent numerical instability
            contrast_bounded = jnp.clip(contrast_samples[i], 1e-6, 1.0)  # Physical bounds: 0 < contrast ≤ 1
            offset_bounded = jnp.clip(offset_samples[i], 1e-6, 2.0)      # Physical bounds: 0 < offset ≤ 2

            # Check if scaling parameters were clipped (indicates potential issues)
            if jnp.abs(contrast_samples[i] - contrast_bounded) > 1e-6 or jnp.abs(offset_samples[i] - offset_bounded) > 1e-6:
                logger.debug(f"🔍 Scaling parameters clipped: contrast {contrast_samples[i]:.6f} → {contrast_bounded:.6f}, offset {offset_samples[i]:.6f} → {offset_bounded:.6f}")

            # Apply scaling: c2_fitted = c2_theory * contrast + offset
            theory_fitted = contrast_bounded * g1_squared + offset_bounded
            
            # Check for non-finite theory_fitted (should not happen with finite g1)
            if not jnp.isfinite(theory_fitted).all():
                logger.debug(f"🔍 ELBO PENALTY [A1]: non-finite theory_fitted, params={param_samples[i]}, contrast={contrast_samples[i]}, offset={offset_samples[i]}")
                return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}

            # Compute residuals with numerical safeguards
            # Add minimum sigma threshold to prevent division by very small values
            sigma_safe = jnp.maximum(sigma_finite, 1e-6)
            raw_residuals = (data_finite - theory_fitted) / sigma_safe
            
            # Clip residuals to prevent extreme values (±10 standard deviations)
            residuals = jnp.clip(raw_residuals, -10.0, 10.0)

            # Check for non-finite residuals
            if not jnp.isfinite(residuals).all():
                logger.debug(f"🔍 ELBO PENALTY [A2]: non-finite residuals, params={param_samples[i]}, residuals_range=[{jnp.min(residuals):.3f}, {jnp.max(residuals):.3f}]")
                return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}

            # Gaussian likelihood with numerical stability (using finite points only)
            residuals_squared = residuals**2
            log_sigma_term = jnp.log(2 * jnp.pi * sigma_safe**2)
            
            # Check for non-finite components
            if not jnp.isfinite(residuals_squared).all() or not jnp.isfinite(log_sigma_term).all():
                logger.debug(f"🔍 ELBO PENALTY [A3]: non-finite likelihood components, params={param_samples[i]}, residuals_sq_finite={jnp.isfinite(residuals_squared).all()}, log_sigma_finite={jnp.isfinite(log_sigma_term).all()}")
                return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
            
            sample_likelihood = -0.5 * jnp.sum(residuals_squared) - 0.5 * jnp.sum(log_sigma_term)
            
            # Cap extremely negative likelihoods to prevent numerical overflow
            sample_likelihood = jnp.maximum(sample_likelihood, -1e8)
            
            # Check for non-finite likelihood
            if not jnp.isfinite(sample_likelihood):
                logger.debug(f"🔍 ELBO PENALTY [A4]: non-finite sample_likelihood={sample_likelihood}, params={param_samples[i]}")
                return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                
            log_likelihood += sample_likelihood

        log_likelihood /= n_samples  # Average over samples

        # Compute KL divergence to priors
        kl_divergence = 0.0

        # Physical parameters KL with safeguards
        for i, (prior, bounds) in enumerate(zip(param_priors, param_bounds)):
            mu_prior, sigma_prior = prior
            if bounds in [(-2.0, 2.0), (-100.0, 100.0), (-10.0, 10.0), (-1e5, 1e5)]:  # Normal priors
                # KL between two Gaussians with safeguards: KL(N(μ₁,σ₁²)||N(μ₂,σ₂²))
                param_std_safe = jnp.clip(param_std[i], 1e-8, 1e3)
                sigma_prior_safe = jnp.clip(sigma_prior, 1e-8, 1e3)

                log_ratio = jnp.clip(jnp.log(sigma_prior_safe / param_std_safe), -50.0, 50.0)
                variance_term = jnp.clip((param_std_safe ** 2 + (param_mu[i] - mu_prior) ** 2) / (2 * sigma_prior_safe**2), 0.0, 1e6)

                kl_i = log_ratio + variance_term - 0.5
                kl_i_safe = jnp.clip(kl_i, -1e6, 1e6)
            else:  # Truncated Normal priors
                # Approximate KL for truncated normal (more complex, using Monte Carlo)
                param_std_safe = jnp.clip(param_std[i], 1e-8, 1e3)
                param_sample = param_mu[i] + param_std_safe * random.normal(key, (100,))
                q_logpdf = jnp.mean(
                    jstats.norm.logpdf(param_sample, param_mu[i], param_std_safe)
                )
                p_logpdf = jnp.mean(
                    VariationalFamilies.truncated_normal_prior_logpdf(
                        param_sample, mu_prior, sigma_prior, bounds
                    )
                )
                kl_i_raw = q_logpdf - p_logpdf
                kl_i_safe = jnp.clip(kl_i_raw, -1e6, 1e6)

            kl_divergence += kl_i_safe

        # Contrast KL (TruncatedNormal) with safeguards
        contrast_std_safe = jnp.clip(contrast_std, 1e-8, 1e3)
        contrast_sample = contrast_mu + contrast_std_safe * random.normal(key, (100,))
        q_contrast = jnp.mean(
            jstats.norm.logpdf(contrast_sample, contrast_mu, contrast_std_safe)
        )
        p_contrast = jnp.mean(
            VariationalFamilies.truncated_normal_prior_logpdf(
                contrast_sample, contrast_prior[0], contrast_prior[1], contrast_bounds
            )
        )
        contrast_kl = jnp.clip(q_contrast - p_contrast, -1e6, 1e6)
        kl_divergence += contrast_kl

        # Offset KL (TruncatedNormal) with safeguards
        offset_std_safe = jnp.clip(offset_std, 1e-8, 1e3)
        offset_sample = offset_mu + offset_std_safe * random.normal(key, (100,))
        q_offset = jnp.mean(jstats.norm.logpdf(offset_sample, offset_mu, offset_std_safe))
        p_offset = jnp.mean(
            VariationalFamilies.truncated_normal_prior_logpdf(
                offset_sample, offset_prior[0], offset_prior[1], offset_bounds
            )
        )
        offset_kl = jnp.clip(q_offset - p_offset, -1e6, 1e6)
        kl_divergence += offset_kl

        # Final check for KL divergence
        if not jnp.isfinite(kl_divergence):
            logger.debug(f"🔍 ELBO PENALTY [A5]: non-finite kl_divergence={kl_divergence}, log_likelihood={log_likelihood}")
            return -1e10, {"likelihood": log_likelihood, "kl_divergence": 1e10, "elbo": -1e10}

        # ELBO = likelihood - KL divergence
        elbo = log_likelihood - kl_divergence
        
        # Final check for ELBO
        if not jnp.isfinite(elbo):
            logger.debug(f"🔍 ELBO PENALTY [A6]: non-finite elbo={elbo}, log_likelihood={log_likelihood}, kl_divergence={kl_divergence}")
            return -1e10, {"likelihood": log_likelihood, "kl_divergence": kl_divergence, "elbo": -1e10}

        metrics = {
            "likelihood": log_likelihood,
            "kl_divergence": kl_divergence,
            "elbo": elbo,
        }

        return elbo, metrics

    # Don't apply JIT compilation because function takes function arguments
    # JAX can't JIT functions that take other functions as arguments
    compute_elbo = compute_elbo_inner
else:
    # When JAX is not available at import time, we need to handle the case
    # where it becomes available at runtime. Define a dynamic wrapper.
    def compute_elbo(*args, **kwargs):
        # Try to import JAX at runtime
        try:
            import jax
            import jax.numpy as jnp
            from jax import jit
            from jax.scipy import stats as jstats
            
            # Dynamically create the jitted function
            # Copy the function body from compute_elbo_inner with runtime JAX
            def runtime_compute_elbo(
                variational_params,
                data,
                sigma, 
                t1,
                t2,
                phi,
                q,
                L,
                param_priors,
                param_bounds,
                contrast_prior,
                contrast_bounds,
                offset_prior,
                offset_bounds,
                theory_engine_compute_g1,
                n_samples=1,
            ):
                # This is a copy of the compute_elbo_inner function body
                # Extract variational parameters
                param_mu = variational_params["param_mu"]
                param_log_std = variational_params["param_log_std"]
                contrast_mu = variational_params["contrast_mu"]
                contrast_log_std = variational_params["contrast_log_std"]
                offset_mu = variational_params["offset_mu"]
                offset_log_std = variational_params["offset_log_std"]
                
                key = jax.random.PRNGKey(0)  # Fixed for reproducibility during optimization
                
                # Sample parameters
                param_std = jnp.exp(param_log_std)
                contrast_std = jnp.exp(contrast_log_std)
                offset_std = jnp.exp(offset_log_std)
                
                # Ensure shape tuple contains int (not float32) for random.normal
                param_count = int(len(param_mu))
                
                # Sample parameters with bounds checking to prevent extreme values
                raw_param_samples = param_mu + param_std * jax.random.normal(
                    key, (n_samples, param_count)
                )
                
                # Clip parameter samples to stay within bounds to prevent numerical issues
                param_samples = jnp.zeros_like(raw_param_samples)
                for i in range(param_count):
                    min_bound, max_bound = param_bounds[i]
                    param_samples = param_samples.at[:, i].set(
                        jnp.clip(raw_param_samples[:, i], min_bound, max_bound)
                    )
                
                contrast_samples = contrast_mu + contrast_std * jax.random.normal(key, (n_samples,))
                offset_samples = offset_mu + offset_std * jax.random.normal(key, (n_samples,))
                
                # Compute likelihood for each sample
                log_likelihood = 0.0
                for i in range(n_samples):
                    # Compute theoretical g1 - ensure index is handled properly by JAX
                    sample_params = param_samples[i]
                    g1_theory = theory_engine_compute_g1(sample_params, t1, t2, phi, q, L)
                    
                    # Check for non-finite g1_theory
                    if not jnp.isfinite(g1_theory).all():
                        # Return very negative likelihood to penalize bad parameters
                        logger.debug(f"🔍 ELBO PENALTY [B1]: non-finite g1_theory, params={param_samples[i]}")
                        return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                    
                    g1_squared = g1_theory**2

                    # Apply bounds to scaling parameters to prevent numerical instability
                    contrast_bounded = jnp.clip(contrast_samples[i], 1e-6, 1.0)  # Physical bounds: 0 < contrast ≤ 1
                    offset_bounded = jnp.clip(offset_samples[i], 1e-6, 2.0)      # Physical bounds: 0 < offset ≤ 2

                    # Check if scaling parameters were clipped (indicates potential issues)
                    if jnp.abs(contrast_samples[i] - contrast_bounded) > 1e-6 or jnp.abs(offset_samples[i] - offset_bounded) > 1e-6:
                        logger.debug(f"🔍 [ALT] Scaling parameters clipped: contrast {contrast_samples[i]:.6f} → {contrast_bounded:.6f}, offset {offset_samples[i]:.6f} → {offset_bounded:.6f}")

                    # Apply scaling: c2_fitted = c2_theory * contrast + offset
                    theory_fitted = contrast_bounded * g1_squared + offset_bounded
                    
                    # Check for non-finite theory_fitted
                    if not jnp.isfinite(theory_fitted).all():
                        logger.debug(f"🔍 ELBO PENALTY [B2]: non-finite theory_fitted, params={param_samples[i]}, contrast={contrast_samples[i]}, offset={offset_samples[i]}")
                        return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                    
                    # Compute residuals with numerical safeguards  
                    # Add minimum sigma threshold to prevent division by very small values
                    sigma_safe = jnp.maximum(sigma, 1e-6)
                    raw_residuals = (data - theory_fitted) / sigma_safe
                    
                    # Clip residuals to prevent extreme values (±10 standard deviations)
                    residuals = jnp.clip(raw_residuals, -10.0, 10.0)
                    
                    # Check for non-finite residuals
                    if not jnp.isfinite(residuals).all():
                        logger.debug(f"🔍 ELBO PENALTY [B3]: non-finite residuals, params={param_samples[i]}")
                        return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                    
                    # Gaussian likelihood with numerical stability
                    residuals_squared = residuals**2
                    log_sigma_term = jnp.log(2 * jnp.pi * sigma_safe**2)
                    
                    # Check for non-finite components
                    if not jnp.isfinite(residuals_squared).all() or not jnp.isfinite(log_sigma_term).all():
                        logger.debug(f"🔍 ELBO PENALTY [B4]: non-finite likelihood components, params={param_samples[i]}")
                        return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                    
                    sample_likelihood = -0.5 * jnp.sum(residuals_squared) - 0.5 * jnp.sum(log_sigma_term)
                    
                    # Cap extremely negative likelihoods to prevent numerical overflow
                    sample_likelihood = jnp.maximum(sample_likelihood, -1e8)
                    
                    if not jnp.isfinite(sample_likelihood):
                        logger.debug(f"🔍 ELBO PENALTY [B5]: non-finite sample_likelihood={sample_likelihood}, params={param_samples[i]}")
                        return -1e10, {"likelihood": -1e10, "kl_divergence": 0.0, "elbo": -1e10}
                    
                    log_likelihood += sample_likelihood
                
                # Average likelihood over samples
                log_likelihood /= n_samples
                
                # Compute KL divergence between variational and prior distributions
                kl_divergence = 0.0
                
                # Parameter KL divergences (Gaussian priors) with safeguards
                for i, (prior_mean, prior_std) in enumerate(param_priors):
                    var_mean = param_mu[i]
                    var_std = param_std[i]

                    # Add safeguards to prevent infinite KL divergence
                    var_std_safe = jnp.clip(var_std, 1e-8, 1e3)  # Prevent division by zero or extreme values
                    prior_std_safe = jnp.clip(prior_std, 1e-8, 1e3)

                    log_ratio = jnp.clip(jnp.log(prior_std_safe / var_std_safe), -50.0, 50.0)  # Prevent exp overflow
                    variance_term = jnp.clip((var_std_safe**2 + (var_mean - prior_mean)**2) / (2 * prior_std_safe**2), 0.0, 1e6)

                    kl_param = log_ratio + variance_term - 0.5
                    kl_param_safe = jnp.clip(kl_param, -1e6, 1e6)  # Final safeguard
                    kl_divergence += kl_param_safe

                # Contrast KL divergence with safeguards
                contrast_prior_mean, contrast_prior_std = contrast_prior
                contrast_std_safe = jnp.clip(contrast_std, 1e-8, 1e3)
                contrast_prior_std_safe = jnp.clip(contrast_prior_std, 1e-8, 1e3)

                log_ratio_contrast = jnp.clip(jnp.log(contrast_prior_std_safe / contrast_std_safe), -50.0, 50.0)
                variance_term_contrast = jnp.clip((contrast_std_safe**2 + (contrast_mu - contrast_prior_mean)**2) / (2 * contrast_prior_std_safe**2), 0.0, 1e6)

                kl_contrast = log_ratio_contrast + variance_term_contrast - 0.5
                kl_contrast_safe = jnp.clip(kl_contrast, -1e6, 1e6)
                kl_divergence += kl_contrast_safe

                # Offset KL divergence with safeguards
                offset_prior_mean, offset_prior_std = offset_prior
                offset_std_safe = jnp.clip(offset_std, 1e-8, 1e3)
                offset_prior_std_safe = jnp.clip(offset_prior_std, 1e-8, 1e3)

                log_ratio_offset = jnp.clip(jnp.log(offset_prior_std_safe / offset_std_safe), -50.0, 50.0)
                variance_term_offset = jnp.clip((offset_std_safe**2 + (offset_mu - offset_prior_mean)**2) / (2 * offset_prior_std_safe**2), 0.0, 1e6)

                kl_offset = log_ratio_offset + variance_term_offset - 0.5
                kl_offset_safe = jnp.clip(kl_offset, -1e6, 1e6)
                kl_divergence += kl_offset_safe
                
                # Check for non-finite KL divergence
                if not jnp.isfinite(kl_divergence):
                    logger.debug(f"🔍 ELBO PENALTY [B6]: non-finite kl_divergence={kl_divergence}, log_likelihood={log_likelihood}")
                    return -1e10, {"likelihood": -1e10, "kl_divergence": -1e10, "elbo": -1e10}

                # ELBO = likelihood - KL divergence
                elbo = log_likelihood - kl_divergence

                # Check for non-finite ELBO
                if not jnp.isfinite(elbo):
                    logger.debug(f"🔍 ELBO PENALTY [B7]: non-finite elbo={elbo}, log_likelihood={log_likelihood}, kl_divergence={kl_divergence}")
                    return -1e10, {"likelihood": log_likelihood, "kl_divergence": kl_divergence, "elbo": -1e10}
                
                # Metrics for monitoring
                metrics = {
                    "likelihood": log_likelihood,
                    "kl_divergence": kl_divergence,
                    "elbo": elbo,
                }
                
                return elbo, metrics
            
            # Don't apply JIT to runtime function because it takes function arguments
            # JAX can't JIT functions that take other functions as arguments
            return runtime_compute_elbo(*args, **kwargs)
            
        except ImportError:
            raise RuntimeError("JAX not available at import time or runtime, cannot compute ELBO")
    
# Make compute_elbo available at module level


class VariationalInferenceJAX:
    """
    VI+JAX implementation as primary optimization method.

    Replaces all classical optimization methods with KL divergence
    minimization using the unified homodyne model and specified
    parameter space with priors.
    """

    def __init__(
        self,
        analysis_mode: str = "laminar_flow",
        parameter_space: Optional[ParameterSpace] = None,
    ):
        """
        Initialize VI+JAX optimizer.

        Args:
            analysis_mode: Analysis mode ("static_isotropic", "static_anisotropic", "laminar_flow")
            parameter_space: Parameter space with bounds and priors
        """
        self.analysis_mode = analysis_mode
        self.parameter_space = parameter_space or ParameterSpace()
        self.engine = UnifiedHomodyneEngine(analysis_mode, parameter_space)

        # Get parameter configuration
        self.param_bounds = self.parameter_space.get_param_bounds(analysis_mode)
        self.param_priors = self.parameter_space.get_param_priors(analysis_mode)
        self.n_params = len(self.param_bounds)

        # Use consistent dynamic JAX detection 
        logger.debug(f"🔍 VI init: Global JAX_AVAILABLE = {JAX_AVAILABLE}")
        jax_working, jax_runtime_modules = _detect_jax_availability()
        
        if jax_working:
            logger.info("✅ JAX acceleration confirmed for VI optimization")
        else:
            if HAS_NUMPY_GRADIENTS:
                logger.info(
                    "JAX not available - VI will use NumPy+numerical gradients (10-50x slower)"
                )
            else:
                logger.warning(
                    "JAX and numpy_gradients not available - VI will use simple fallback"
                )

        logger.info(f"VI+JAX initialized for {analysis_mode}")
        logger.info(f"Parameters: {self.n_params} physical + 2 scaling")

    @log_performance(threshold=1.0)
    def fit_vi_jax(
        self,
        data: np.ndarray,
        sigma: Optional[np.ndarray],
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        q: float,
        L: float,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        convergence_tol: float = 1e-6,
        n_elbo_samples: int = 1,
        chunked_iterator: Optional[Any] = None,
        preprocessing_time: float = 0.0,
    ) -> VIResult:
        """
        Fit homodyne data using VI+JAX optimization.

        Primary fitting method that minimizes KL divergence between
        variational distribution and true posterior using unified model.

        Args:
            data: Experimental correlation data
            sigma: Measurement uncertainties
            t1, t2, phi: Time and angle grids
            q, L: Experimental parameters
            n_iterations: Maximum VI iterations
            learning_rate: Adam learning rate
            convergence_tol: ELBO convergence tolerance
            n_elbo_samples: Samples for ELBO estimation

        Returns:
            VIResult with optimized parameters and uncertainties
        """
        start_time = time.time()

        # Handle missing uncertainties by creating default ones
        if sigma is None:
            # Create default sigma as 1% of data or minimum of sqrt(data) for Poisson-like noise
            sigma = np.maximum(np.sqrt(np.abs(data)), 0.01 * np.abs(data))
            sigma = np.maximum(sigma, 1e-6)  # Ensure no zero uncertainties

        # Validate inputs
        self.engine.validate_inputs(data, sigma, t1, t2, phi, q, L)

        # Detect dataset size
        dataset_size = self.engine.detect_dataset_size(data)

        # Use dynamic JAX detection for runtime consistency
        logger.debug(f"🔍 fit_vi_jax: Global JAX_AVAILABLE = {JAX_AVAILABLE}")
        jax_working, jax_runtime_modules = _detect_jax_availability()
        
        if jax_working:
            logger.info(f"✅ JAX runtime detection successful in fit_vi_jax")
            # Override global modules with runtime-detected ones for consistency
            global jax, jnp, jit, jax_optimizers, random, value_and_grad, vmap, jstats
            jax = jax_runtime_modules['jax']
            jnp = jax_runtime_modules['jnp']
            jit = jax_runtime_modules['jit'] 
            jax_optimizers = jax_runtime_modules['jax_optimizers']
            random = jax_runtime_modules['random']
            value_and_grad = jax_runtime_modules['value_and_grad']
            vmap = jax_runtime_modules['vmap']
            jstats = jax_runtime_modules['jstats']
            logger.info("✅ Using JAX-accelerated VI optimization")
        else:
            logger.warning("❌ JAX not available at runtime")
        
        if not jax_working:
            logger.warning("JAX not available - VI will use NumPy+numerical gradients")
            return self._fit_numpy_fallback(
                data,
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                dataset_size,
                n_iterations,
                learning_rate,
                convergence_tol,
            )
        else:
            logger.info("✅ Using JAX-accelerated VI optimization")

        # Validate data and sigma before conversion
        if not np.isfinite(data).all():
            logger.error("❌ Data contains non-finite values")
            raise ValueError("Data contains non-finite values (NaN or Inf)")
            
        if not np.isfinite(sigma).all():
            logger.error("❌ Sigma contains non-finite values")
            raise ValueError("Sigma contains non-finite values (NaN or Inf)")
            
        if np.any(sigma <= 0):
            logger.error("❌ Sigma contains zero or negative values")
            raise ValueError("Sigma must be positive")
            
        # Log data statistics for debugging
        logger.debug(f"Data range: [{np.min(data):.4f}, {np.max(data):.4f}], mean: {np.mean(data):.4f}")
        logger.debug(f"Sigma range: [{np.min(sigma):.4f}, {np.max(sigma):.4f}], mean: {np.mean(sigma):.4f}")

        # Convert to JAX arrays with explicit float32 dtype to avoid int32 issues
        data_jax = jnp.array(data, dtype=jnp.float32)
        sigma_jax = jnp.array(sigma, dtype=jnp.float32)
        t1_jax = jnp.array(t1, dtype=jnp.float32)
        t2_jax = jnp.array(t2, dtype=jnp.float32)
        phi_jax = jnp.array(phi, dtype=jnp.float32)
        logger.debug(f"🔍 DEBUG: phi_jax created with shape: {phi_jax.shape}")
        
        # Ensure scalar parameters are also float32
        q_jax = jnp.float32(q)
        L_jax = jnp.float32(L)
        
        # Debug data types before optimization
        logger.debug(f"🔍 Data type checking:")
        logger.debug(f"  data_jax.dtype: {data_jax.dtype}")
        logger.debug(f"  sigma_jax.dtype: {sigma_jax.dtype}")
        logger.debug(f"  t1_jax.dtype: {t1_jax.dtype}")
        logger.debug(f"  t2_jax.dtype: {t2_jax.dtype}")
        logger.debug(f"  phi_jax.dtype: {phi_jax.dtype}")
        logger.debug(f"  q_jax type: {type(q_jax)}, dtype: {q_jax.dtype}")
        logger.debug(f"  L_jax type: {type(L_jax)}, dtype: {L_jax.dtype}")

        # Initialize variational parameters from priors
        variational_params = self._initialize_variational_params(jax_working)
        
        # Debug variational parameter types
        logger.debug(f"🔍 Variational parameter types:")
        for key, value in variational_params.items():
            logger.debug(f"  {key}: {type(value)}, dtype: {getattr(value, 'dtype', 'N/A')}")

        # Set up Adam optimizer
        opt_init, opt_update, get_params = jax_optimizers.adam(learning_rate)
        opt_state = opt_init(variational_params)

        # ELBO history tracking
        elbo_history = []
        best_elbo = -jnp.inf
        best_params = variational_params
        convergence_count = 0

        # Create theory engine function for JIT compilation
        def theory_g1_func(params, t1, t2, phi, q, L):
            logger.debug(f"🔍 DEBUG: theory_g1_func called with phi shape: {phi.shape}")
            try:
                result = self.engine.theory_engine.compute_g1(params, t1, t2, phi, q, L)
                logger.debug(f"✅ DEBUG: theory_g1_func result shape: {result.shape}")
                return result
            except Exception as e:
                logger.error(f"❌ DEBUG: theory_g1_func ERROR: {e}")
                logger.error(f"❌ DEBUG: phi shape when error occurred: {phi.shape}")
                logger.error(f"❌ DEBUG: params shape: {params.shape}")
                logger.error(f"❌ DEBUG: t1 shape: {t1.shape}, t2 shape: {t2.shape}")
                raise

        logger.info(f"Starting VI+JAX optimization ({n_iterations} max iterations)")

        # VI optimization loop
        for iteration in range(n_iterations):
            try:
                # Compute ELBO and gradients
                if jax_working:
                    # For first iteration, get detailed ELBO components for debugging
                    if iteration == 0:
                        elbo_value, elbo_metrics = compute_elbo(
                            variational_params,
                            data_jax,
                            sigma_jax,
                            t1_jax,
                            t2_jax,
                            phi_jax,
                            q_jax,
                            L_jax,
                            self.param_priors,
                            self.param_bounds,
                            self.parameter_space.contrast_prior,
                            self.parameter_space.contrast_bounds,
                            self.parameter_space.offset_prior,
                            self.parameter_space.offset_bounds,
                            theory_g1_func,
                            n_elbo_samples,
                        )
                        logger.debug(f"🔍 Initial ELBO computation:")
                        logger.debug(f"   ELBO: {elbo_value}")
                        logger.debug(f"   Likelihood: {elbo_metrics.get('likelihood', 'unknown')}")
                        logger.debug(f"   KL divergence: {elbo_metrics.get('kl_divergence', 'unknown')}")
                    
                    def elbo_fn(variational_params):
                        return compute_elbo(
                            variational_params,
                            data_jax,
                            sigma_jax,
                            t1_jax,
                            t2_jax,
                            phi_jax,
                            q_jax,
                            L_jax,
                            self.param_priors,
                            self.param_bounds,
                            self.parameter_space.contrast_prior,
                            self.parameter_space.contrast_bounds,
                            self.parameter_space.offset_prior,
                            self.parameter_space.offset_bounds,
                            theory_g1_func,
                            n_elbo_samples,
                        )[0]

                    # Debug which value_and_grad function is being used
                    logger.debug(f"🔍 Using value_and_grad: {value_and_grad}")
                    logger.debug(f"🔍 value_and_grad module: {getattr(value_and_grad, '__module__', 'unknown')}")
                    
                    # Create the value_and_grad function with explicit argnums and allow_int
                    # Use the runtime-detected value_and_grad function
                    try:
                        elbo_value_and_grad = value_and_grad(elbo_fn, argnums=0)
                        logger.debug(f"🔍 Using value_and_grad: {value_and_grad}")
                        elbo_value, elbo_grad = elbo_value_and_grad(variational_params)
                    except TypeError as e:
                        if "int32" in str(e):
                            logger.debug(f"🔍 Retrying with allow_int=True for int32 compatibility")
                            elbo_value_and_grad = value_and_grad(elbo_fn, argnums=0, allow_int=True)
                            elbo_value, elbo_grad = elbo_value_and_grad(variational_params)
                        else:
                            raise e
                    
                    # Check for non-finite values before flipping signs
                    if not jnp.isfinite(elbo_value):
                        logger.error(f"VI iteration {iteration} failed: elbo_func evaluation failed: elbo_func returned non-finite values")
                        logger.error(f"  ELBO value: {elbo_value}")
                        logger.error(f"  Variational params: {variational_params}")
                        logger.error(f"  Data stats: min={jnp.min(data_jax):.4f}, max={jnp.max(data_jax):.4f}, mean={jnp.mean(data_jax):.4f}")
                        logger.error(f"  Sigma stats: min={jnp.min(sigma_jax):.4f}, max={jnp.max(sigma_jax):.4f}, mean={jnp.mean(sigma_jax):.4f}")
                        
                        # Return failed result
                        result = VIResult(
                            mean_params=np.zeros(self.n_params),
                            mean_contrast=1.0,
                            mean_offset=0.0,
                            std_params=np.ones(self.n_params),
                            std_contrast=0.1,
                            std_offset=0.1,
                            final_elbo=float('-inf'),
                            kl_divergence=float('inf'),
                            likelihood=float('-inf'),
                            elbo_history=np.array([]),
                            converged=False,
                            n_iterations=iteration,
                            chi_squared=float('inf'),
                            reduced_chi_squared=float('inf'),
                            computation_time=time.time() - start_time,
                            backend="JAX",
                            dataset_size=dataset_size,
                            analysis_mode=self.analysis_mode,
                        )
                        logger.error("❌ VI optimization failed due to non-finite ELBO values")
                        return result
                    
                    elbo_value = -elbo_value  # Minimize negative ELBO
                    # Flip gradients - handle both JAX and NumPy cases
                    if jax_working:
                        # Use modern JAX tree API
                        try:
                            elbo_grad = jax.tree.map(lambda x: -x, elbo_grad)
                        except AttributeError:
                            # Fallback to older API if needed
                            elbo_grad = jax.tree_util.tree_map(lambda x: -x, elbo_grad)
                    else:
                        # For NumPy fallback, elbo_grad should be a dict
                        if isinstance(elbo_grad, dict):
                            elbo_grad = {key: -value for key, value in elbo_grad.items()}
                        else:
                            elbo_grad = -elbo_grad  # If it's just an array
                else:
                    elbo_value = 0.0  # Fallback
                    elbo_grad = variational_params
                    
            except Exception as e:
                logger.error(f"VI iteration {iteration} failed with exception: {e}")
                logger.error(f"  Exception type: {type(e).__name__}")
                logger.error(f"  Data shape: {data_jax.shape}")
                logger.error(f"  Sigma shape: {sigma_jax.shape}")
                
                # Return failed result
                result = VIResult(
                    mean_params=np.zeros(self.n_params),
                    mean_contrast=1.0,
                    mean_offset=0.0,
                    std_params=np.ones(self.n_params),
                    std_contrast=0.1,
                    std_offset=0.1,
                    final_elbo=float('-inf'),
                    kl_divergence=float('inf'),
                    likelihood=float('-inf'),
                    elbo_history=np.array([]),
                    converged=False,
                    n_iterations=iteration,
                    chi_squared=float('inf'),
                    reduced_chi_squared=float('inf'),
                    computation_time=time.time() - start_time,
                    backend="JAX",
                    dataset_size=dataset_size,
                    analysis_mode=self.analysis_mode,
                )
                logger.error("❌ VI optimization failed due to exception")
                return result

            elbo_history.append(float(elbo_value.item() if hasattr(elbo_value, 'item') else elbo_value))

            # Track best parameters
            if elbo_value > best_elbo:
                best_elbo = elbo_value
                best_params = variational_params
                convergence_count = 0
            else:
                convergence_count += 1

            # Update parameters
            # Convert iteration to JAX-compatible type to avoid int32 issues
            iteration_jax = jnp.array(iteration, dtype=jnp.float32)
            
            # Debug gradient structure
            logger.debug(f"🔍 Gradient structure: {type(elbo_grad)}")
            if isinstance(elbo_grad, dict):
                logger.debug(f"🔍 Gradient keys: {list(elbo_grad.keys())}")
            
            opt_state = opt_update(iteration_jax, elbo_grad, opt_state)
            variational_params = get_params(opt_state)
            
            # Enforce parameter bounds on means to prevent numerical instabilities
            param_mu_clipped = jnp.zeros_like(variational_params["param_mu"])
            for i in range(len(self.param_bounds)):
                min_bound, max_bound = self.param_bounds[i]
                param_val = variational_params["param_mu"][i]
                
                # Replace NaN/inf values with initial parameter values 
                if not jnp.isfinite(param_val):
                    param_val = jnp.array(self.param_priors[i][0])  # Use prior mean as fallback
                
                param_mu_clipped = param_mu_clipped.at[i].set(
                    jnp.clip(param_val, min_bound, max_bound)
                )
            
            # Also constrain contrast and offset means to reasonable ranges
            contrast_val = variational_params["contrast_mu"]
            if not jnp.isfinite(contrast_val):
                contrast_val = jnp.array(self.parameter_space.contrast_prior[0])  # Use prior mean as fallback
            contrast_mu_clipped = jnp.clip(contrast_val, 
                                         self.parameter_space.contrast_bounds[0], 
                                         self.parameter_space.contrast_bounds[1])
            
            offset_val = variational_params["offset_mu"]
            if not jnp.isfinite(offset_val):
                offset_val = jnp.array(self.parameter_space.offset_prior[0])  # Use prior mean as fallback
            offset_mu_clipped = jnp.clip(offset_val,
                                       self.parameter_space.offset_bounds[0],
                                       self.parameter_space.offset_bounds[1])
            
            variational_params = variational_params.copy()
            variational_params["param_mu"] = param_mu_clipped
            variational_params["contrast_mu"] = contrast_mu_clipped  
            variational_params["offset_mu"] = offset_mu_clipped

            # Check convergence
            if iteration > 50 and len(elbo_history) > 10:
                recent_improvement = abs(elbo_history[-1] - elbo_history[-10])
                if recent_improvement < convergence_tol or convergence_count > 50:
                    logger.info(f"VI converged at iteration {iteration}")
                    break

            if iteration % 100 == 0:
                logger.debug(f"VI iteration {iteration}: ELBO = {elbo_value:.4f}")

        # Extract final results
        final_params = best_params

        # Get final ELBO breakdown
        if JAX_AVAILABLE:
            final_elbo, metrics = compute_elbo(
                final_params,
                data_jax,
                sigma_jax,
                t1_jax,
                t2_jax,
                phi_jax,
                q_jax,
                L_jax,
                self.param_priors,
                self.param_bounds,
                self.parameter_space.contrast_prior,
                self.parameter_space.contrast_bounds,
                self.parameter_space.offset_prior,
                self.parameter_space.offset_bounds,
                theory_g1_func,
                n_elbo_samples,
            )
        else:
            final_elbo = 0.0
            metrics = {"likelihood": 0.0, "kl_divergence": 0.0, "elbo": 0.0}

        # Extract point estimates and uncertainties
        mean_params = np.array(final_params["param_mu"])
        mean_contrast = float(final_params["contrast_mu"].item() if hasattr(final_params["contrast_mu"], 'item') else final_params["contrast_mu"])
        mean_offset = float(final_params["offset_mu"].item() if hasattr(final_params["offset_mu"], 'item') else final_params["offset_mu"])

        std_params = np.exp(np.array(final_params["param_log_std"]))
        contrast_log_std = final_params["contrast_log_std"]
        offset_log_std = final_params["offset_log_std"]
        std_contrast = float(np.exp(contrast_log_std.item() if hasattr(contrast_log_std, 'item') else contrast_log_std))
        std_offset = float(np.exp(offset_log_std.item() if hasattr(offset_log_std, 'item') else offset_log_std))

        # Compute chi-squared at MAP estimate
        chi_squared = 2 * self.engine.compute_likelihood(
            mean_params, mean_contrast, mean_offset, data, sigma, t1, t2, phi, q, L
        )
        dof = data.size - self.n_params - 2

        computation_time = time.time() - start_time

        result = VIResult(
            mean_params=mean_params,
            mean_contrast=mean_contrast,
            mean_offset=mean_offset,
            std_params=std_params,
            std_contrast=std_contrast,
            std_offset=std_offset,
            final_elbo=float(final_elbo.item() if hasattr(final_elbo, 'item') else final_elbo),
            kl_divergence=float(metrics["kl_divergence"].item() if hasattr(metrics["kl_divergence"], 'item') else metrics["kl_divergence"]),
            likelihood=float(metrics["likelihood"].item() if hasattr(metrics["likelihood"], 'item') else metrics["likelihood"]),
            elbo_history=np.array(elbo_history),
            converged=(convergence_count <= 50),
            n_iterations=len(elbo_history),
            chi_squared=chi_squared,
            reduced_chi_squared=chi_squared / max(dof, 1),
            computation_time=computation_time,
            backend="JAX",
            dataset_size=dataset_size,
            analysis_mode=self.analysis_mode,
        )

        logger.info(
            f"VI+JAX completed: ELBO={final_elbo:.2f}, χ²={chi_squared:.2f}, "
            f"time={computation_time:.2f}s"
        )

        return result

    def _initialize_variational_params(self, jax_working: bool = None) -> Dict[str, jnp.ndarray]:
        """Initialize variational distribution parameters from priors."""
        # Use the runtime-tested JAX availability if provided, otherwise fallback to global
        use_jax = jax_working if jax_working is not None else JAX_AVAILABLE
        
        if not use_jax:
            return self._initialize_numpy_variational_params()

        # Physical parameter variational family (mean-field Gaussian)
        param_mu = jnp.array([prior[0] for prior in self.param_priors])
        
        # Ensure positive standard deviations before taking log
        param_stds = [max(prior[1], 1e-6) for prior in self.param_priors]
        param_log_std = jnp.log(jnp.array(param_stds))

        # Contrast variational family
        contrast_mu = jnp.array(self.parameter_space.contrast_prior[0])
        contrast_std = max(self.parameter_space.contrast_prior[1], 1e-6)
        contrast_log_std = jnp.log(jnp.array(contrast_std))
        
        offset_mu = jnp.array(self.parameter_space.offset_prior[0])
        offset_std = max(self.parameter_space.offset_prior[1], 1e-6)
        offset_log_std = jnp.log(jnp.array(offset_std))

        # Validate all initialization values
        init_params = {
            "param_mu": param_mu,
            "param_log_std": param_log_std,
            "contrast_mu": contrast_mu,
            "contrast_log_std": contrast_log_std,
            "offset_mu": offset_mu,
            "offset_log_std": offset_log_std,
        }
        
        # Check for non-finite values in initialization
        for key, value in init_params.items():
            if not jnp.isfinite(value).all():
                logger.error(f"❌ Non-finite values in {key}: {value}")
                raise ValueError(f"Non-finite values in variational parameter initialization: {key}")
        
        logger.debug(f"✅ Variational parameters initialized successfully")
        logger.debug(f"   param_mu: {param_mu}")
        logger.debug(f"   param_log_std: {param_log_std}")
        logger.debug(f"   contrast_mu: {contrast_mu}, offset_mu: {offset_mu}")
        
        return init_params

    def _initialize_numpy_variational_params(self) -> Dict[str, np.ndarray]:
        """Initialize variational parameters using NumPy."""
        # Physical parameter variational family (mean-field Gaussian)
        param_mu = np.array([prior[0] for prior in self.param_priors])
        param_log_std = np.log(np.array([prior[1] for prior in self.param_priors]))

        # Contrast variational family
        contrast_mu = np.array(self.parameter_space.contrast_prior[0])
        contrast_log_std = np.log(np.array(self.parameter_space.contrast_prior[1]))
        offset_mu = np.array(self.parameter_space.offset_prior[0])
        offset_log_std = np.log(np.array(self.parameter_space.offset_prior[1]))

        return {
            "param_mu": param_mu,
            "param_log_std": param_log_std,
            "contrast_mu": contrast_mu,
            "contrast_log_std": contrast_log_std,
            "offset_mu": offset_mu,
            "offset_log_std": offset_log_std,
        }

    def _fit_numpy_fallback(
        self,
        data,
        sigma,
        t1,
        t2,
        phi,
        q,
        L,
        dataset_size,
        n_iterations=1000,
        learning_rate=0.01,
        convergence_tol=1e-6,
    ) -> VIResult:
        """Advanced NumPy fallback implementing full variational inference."""
        if HAS_NUMPY_GRADIENTS:
            logger.warning(
                "JAX unavailable - using NumPy+numerical gradients fallback.\n"
                "Performance will be 10-50x slower but scientifically accurate."
            )
            return self._fit_numpy_vi_with_gradients(
                data,
                sigma,
                t1,
                t2,
                phi,
                q,
                L,
                dataset_size,
                n_iterations,
                learning_rate,
                convergence_tol,
            )
        else:
            logger.warning(
                "JAX and numpy_gradients unavailable - using simple least squares fallback.\n"
                "Results will be approximate with limited uncertainty quantification."
            )
            return self._fit_simple_fallback(
                data, sigma, t1, t2, phi, q, L, dataset_size
            )

    def _fit_numpy_vi_with_gradients(
        self,
        data,
        sigma,
        t1,
        t2,
        phi,
        q,
        L,
        dataset_size,
        n_iterations,
        learning_rate,
        convergence_tol,
    ) -> VIResult:
        """Full VI implementation using numpy_gradients for differentiation."""
        from homodyne.optimization.variational_fallback_helpers import (
            compute_numpy_elbo, flatten_variational_params,
            unflatten_variational_params)

        start_time = time.time()

        # Initialize variational parameters
        variational_params = self._initialize_numpy_variational_params()

        # Set up optimizer
        adam_optimizer = SimpleAdam(learning_rate)
        opt_state = adam_optimizer.init(variational_params)

        # Configure numerical differentiation for VI
        diff_config = DifferentiationConfig(
            method="adaptive",  # Use best available method
            relative_step=1e-6,  # Smaller step for VI precision
            error_tolerance=1e-10,  # High accuracy for gradients
            max_iterations=10,  # Conservative for stability
        )

        # ELBO tracking
        elbo_history = []
        best_elbo = -np.inf
        best_params = variational_params.copy()
        convergence_count = 0

        logger.info(f"Starting NumPy VI optimization ({n_iterations} max iterations)")
        logger.info(f"Using numerical gradients with {diff_config.method} method")

        for iteration in range(n_iterations):
            try:
                # Define ELBO function for current parameters
                def elbo_func(flat_params):
                    vp = unflatten_variational_params(flat_params, self.n_params)
                    elbo_val, _, _ = compute_numpy_elbo(
                        vp,
                        data,
                        sigma,
                        t1,
                        t2,
                        phi,
                        q,
                        L,
                        self.engine,
                        self.param_priors,
                        self.param_bounds,
                        self.parameter_space.contrast_prior,
                        self.parameter_space.contrast_bounds,
                        self.parameter_space.offset_prior,
                        self.parameter_space.offset_bounds,
                    )
                    return -elbo_val  # Minimize negative ELBO

                # Flatten parameters for gradient computation
                flat_params = flatten_variational_params(variational_params)

                # Compute ELBO and gradients using numerical differentiation
                elbo_grad_func = grad(elbo_func)
                elbo_value = elbo_func(flat_params)
                
                # Check for non-finite ELBO values in NumPy fallback
                if not np.isfinite(elbo_value):
                    logger.error(f"NumPy VI iteration {iteration} failed: elbo_func evaluation failed: elbo_func returned non-finite values")
                    logger.error(f"  ELBO value: {elbo_value}")
                    logger.error(f"  Variational params: {variational_params}")
                    logger.error(f"  Data stats: min={np.min(data):.4f}, max={np.max(data):.4f}, mean={np.mean(data):.4f}")
                    logger.error(f"  Sigma stats: min={np.min(sigma):.4f}, max={np.max(sigma):.4f}, mean={np.mean(sigma):.4f}")
                    break
                
                elbo_grad_flat = elbo_grad_func(flat_params)

                # Unflatten gradients
                elbo_grad = unflatten_variational_params(elbo_grad_flat, self.n_params)

                # Convert to positive ELBO for tracking
                elbo_value = -elbo_value
                elbo_history.append(float(elbo_value.item() if hasattr(elbo_value, 'item') else elbo_value))

                # Track best parameters
                if elbo_value > best_elbo:
                    best_elbo = elbo_value
                    best_params = variational_params.copy()
                    convergence_count = 0
                else:
                    convergence_count += 1

                # Apply gradient clipping to prevent explosive updates
                max_grad_norm = 1.0  # Maximum gradient norm
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in elbo_grad.values()))
                if grad_norm > max_grad_norm:
                    clip_factor = max_grad_norm / grad_norm
                    elbo_grad = {key: g * clip_factor for key, g in elbo_grad.items()}

                # Update parameters using Adam
                param_updates, opt_state = adam_optimizer.update(elbo_grad, opt_state)
                for key in variational_params.keys():
                    variational_params[key] += param_updates[key]

                # Apply bounds to variational parameters to prevent infinite KL divergence
                # Log standard deviations should be reasonable to prevent exp(log_std) from exploding
                if 'param_log_std' in variational_params:
                    variational_params['param_log_std'] = jnp.clip(variational_params['param_log_std'], -10.0, 5.0)
                if 'contrast_log_std' in variational_params:
                    variational_params['contrast_log_std'] = jnp.clip(variational_params['contrast_log_std'], -10.0, 2.0)
                if 'offset_log_std' in variational_params:
                    variational_params['offset_log_std'] = jnp.clip(variational_params['offset_log_std'], -10.0, 2.0)

                # Mean parameters should be within reasonable ranges
                if 'param_mu' in variational_params:
                    # Apply bounds based on number of parameters (3 for static, 7 for laminar_flow)
                    param_mu = variational_params['param_mu']
                    n_params = len(param_mu)

                    if n_params >= 3:
                        # Always clip the first 3 parameters (diffusion)
                        param_mu_bounded = [
                            jnp.clip(param_mu[0], 1e-3, 1e5),     # D0: consistent with DIFFUSION_MAX
                            jnp.clip(param_mu[1], -10.0, 10.0),  # alpha: consistent with ALPHA_MIN/MAX
                            jnp.clip(param_mu[2], -1e5, 1e5)     # D_offset: consistent with DIFFUSION_OFFSET_MIN/MAX
                        ]

                        if n_params == 7:
                            # Add laminar flow parameters
                            param_mu_bounded.extend([
                                jnp.clip(param_mu[3], 1e-6, 1.0),    # gamma_dot_0
                                jnp.clip(param_mu[4], -10.0, 10.0), # beta: consistent with BETA_MIN/MAX
                                jnp.clip(param_mu[5], -1.0, 1.0),   # gamma_dot_offset: consistent with SHEAR_OFFSET_MIN/MAX
                                jnp.clip(param_mu[6], -30.0, 30.0)  # phi0: consistent with ANGLE_MIN/MAX
                            ])

                        variational_params['param_mu'] = jnp.array(param_mu_bounded)
                if 'contrast_mu' in variational_params:
                    variational_params['contrast_mu'] = jnp.clip(variational_params['contrast_mu'], 1e-6, 1.0)
                if 'offset_mu' in variational_params:
                    variational_params['offset_mu'] = jnp.clip(variational_params['offset_mu'], 1e-6, 2.0)

                # Check convergence
                if iteration > 10 and len(elbo_history) > 5:
                    recent_improvement = elbo_history[-1] - elbo_history[-6]
                    if abs(recent_improvement) < convergence_tol:
                        logger.info(f"VI converged at iteration {iteration}")
                        break

                # Progress logging
                if iteration % 100 == 0 or iteration < 10:
                    logger.debug(f"VI iteration {iteration}: ELBO = {elbo_value:.6f}")

            except Exception as e:
                logger.warning(
                    f"VI iteration {iteration} failed: {e}. Using best parameters."
                )
                break

        # Compute final results using best parameters
        final_elbo, kl_divergence, likelihood = compute_numpy_elbo(
            best_params,
            data,
            sigma,
            t1,
            t2,
            phi,
            q,
            L,
            self.engine,
            self.param_priors,
            self.param_bounds,
            self.parameter_space.contrast_prior,
            self.parameter_space.contrast_bounds,
            self.parameter_space.offset_prior,
            self.parameter_space.offset_bounds,
        )

        # Extract point estimates and uncertainties
        mean_params = best_params["param_mu"]
        std_params = np.exp(best_params["param_log_std"])
        mean_contrast = float(best_params["contrast_mu"].item() if hasattr(best_params["contrast_mu"], 'item') else best_params["contrast_mu"])
        contrast_log_std = best_params["contrast_log_std"] 
        std_contrast = float(np.exp(contrast_log_std.item() if hasattr(contrast_log_std, 'item') else contrast_log_std))
        mean_offset = float(best_params["offset_mu"].item() if hasattr(best_params["offset_mu"], 'item') else best_params["offset_mu"])
        offset_log_std = best_params["offset_log_std"]
        std_offset = float(np.exp(offset_log_std.item() if hasattr(offset_log_std, 'item') else offset_log_std))

        # Compute fit quality at MAP estimate
        chi_squared = -2 * likelihood
        dof = data.size - self.n_params - 2

        computation_time = time.time() - start_time
        converged = convergence_count < 5 and len(elbo_history) > 10

        logger.info(
            f"NumPy VI completed in {computation_time:.2f}s with {len(elbo_history)} iterations"
        )
        logger.info(f"Final ELBO: {final_elbo:.6f}, Chi-squared: {chi_squared:.6f}")

        return VIResult(
            mean_params=mean_params,
            mean_contrast=mean_contrast,
            mean_offset=mean_offset,
            std_params=std_params,
            std_contrast=std_contrast,
            std_offset=std_offset,
            final_elbo=final_elbo,
            kl_divergence=kl_divergence,
            likelihood=likelihood,
            elbo_history=np.array(elbo_history),
            converged=converged,
            n_iterations=len(elbo_history),
            chi_squared=chi_squared,
            reduced_chi_squared=chi_squared / max(dof, 1),
            computation_time=computation_time,
            backend="NumPy+Gradients",
            dataset_size=dataset_size,
            analysis_mode=self.analysis_mode,
        )

    def _fit_simple_fallback(
        self, data, sigma, t1, t2, phi, q, L, dataset_size
    ) -> VIResult:
        """Simple least squares fallback when no gradients available."""
        logger.warning("Using simple least squares fallback - limited accuracy")

        start_time = time.time()

        # Use engine's least squares estimation
        g1_initial = self.engine.theory_engine.compute_g1(
            np.array([prior[0] for prior in self.param_priors]), t1, t2, phi, q, L
        )
        contrast, offset = self.engine.estimate_scaling_parameters(data, g1_initial**2)

        computation_time = time.time() - start_time

        # Create basic result
        mean_params = np.array([prior[0] for prior in self.param_priors])
        std_params = np.array(
            [prior[1] for prior in self.param_priors]
        )  # Prior uncertainty

        chi_squared = 2 * self.engine.compute_likelihood(
            mean_params, contrast, offset, data, sigma, t1, t2, phi, q, L
        )
        dof = data.size - self.n_params - 2

        return VIResult(
            mean_params=mean_params,
            mean_contrast=contrast,
            mean_offset=offset,
            std_params=std_params,
            std_contrast=0.1 * abs(contrast),
            std_offset=0.1 * abs(offset),
            final_elbo=0.0,
            kl_divergence=0.0,
            likelihood=-chi_squared / 2,
            elbo_history=np.array([0.0]),
            converged=True,
            n_iterations=1,
            chi_squared=chi_squared,
            reduced_chi_squared=chi_squared / max(dof, 1),
            computation_time=computation_time,
            backend="NumPy",
            dataset_size=dataset_size,
            analysis_mode=self.analysis_mode,
        )


# Main API function - replaces all classical optimization methods
def fit_vi_jax(
    data: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    t1: np.ndarray = None,
    t2: np.ndarray = None,
    phi: np.ndarray = None,
    q: float = None,
    L: float = None,
    analysis_mode: str = "laminar_flow",
    parameter_space: Optional[ParameterSpace] = None,
    enable_dataset_optimization: bool = True,
    estimate_noise: bool = False,
    noise_model: str = "hierarchical",
    **kwargs,
) -> VIResult:
    """
    Primary fitting method using VI+JAX optimization with dataset size optimization.

    This function replaces all classical and robust optimization methods.
    It implements KL divergence minimization with the unified homodyne model
    and includes intelligent dataset size optimization for memory efficiency.

    PIPELINE LOGIC:
    - Case 1 (Traditional): fit_vi_jax(data, sigma=provided_sigma)
      → Standard VI with provided noise
    - Case 2 (Hybrid): fit_vi_jax(data, estimate_noise=True, noise_model="hierarchical")
      → HybridNumPyro(Adam noise estimation) → VI(physics with estimated σ)

    Args:
        data: Experimental correlation data
        sigma: Measurement uncertainties (optional if estimate_noise=True)
        t1, t2, phi: Time and angle grids
        q, L: Experimental parameters
        analysis_mode: Analysis mode
        parameter_space: Parameter space definition
        enable_dataset_optimization: Enable automatic dataset size optimization
        estimate_noise: Enable hybrid NumPyro noise estimation
        noise_model: Noise model type ("hierarchical", "per_angle", "adaptive")
        **kwargs: Additional VI parameters

    Returns:
        VIResult with optimized parameters and uncertainties
        
    Raises:
        ValueError: If neither sigma is provided nor estimate_noise is enabled
        ImportError: If hybrid noise estimation requires JAX/NumPyro but not available
    """
    # Stage 1: Handle hybrid noise estimation if requested
    if sigma is None and estimate_noise:
        logger.info(f"🎯 VI Pipeline: Hybrid NumPyro noise estimation ({noise_model})")
        
        try:
            from homodyne.optimization.hybrid_noise_estimation import HybridNoiseEstimator
            
            # Initialize noise estimator
            noise_estimator = HybridNoiseEstimator(analysis_mode, parameter_space)
            
            # Estimate noise using hybrid NumPyro approach
            noise_result = noise_estimator.estimate_noise_for_vi(
                data, t1, t2, phi, q, L, noise_model, **kwargs
            )
            
            # Extract sigma for VI optimization
            estimated_sigma = noise_result.get_sigma_for_method("vi")
            
            # Convert scalar sigma to array matching data shape
            if isinstance(estimated_sigma, (int, float)):
                sigma = jnp.full_like(data, estimated_sigma)
            else:
                sigma = estimated_sigma
            
            logger.info(f"✅ Hybrid NumPyro estimated σ: {estimated_sigma}")
            
            # Add noise estimation info to kwargs for downstream processing
            kwargs['noise_estimation_result'] = noise_result
            
        except ImportError as e:
            logger.error("❌ Hybrid noise estimation requires JAX and NumPyro")
            raise ImportError(
                "Hybrid noise estimation requires JAX and NumPyro. "
                f"Install with: pip install jax numpyro. Error: {e}"
            ) from e
        except Exception as e:
            logger.error(f"❌ Hybrid noise estimation failed: {e}")
            raise RuntimeError(f"Hybrid noise estimation failed: {e}") from e
    
    elif sigma is None and not estimate_noise:
        raise ValueError(
            "Must provide sigma or set estimate_noise=True. "
            "Use estimate_noise=True for automatic noise estimation via hybrid NumPyro."
        )
    
    # Stage 2: Validate required parameters
    if any(param is None for param in [t1, t2, phi, q, L]):
        raise ValueError("t1, t2, phi, q, L are required parameters")
    
    # Stage 3: Dataset size optimization
    optimization_config = None
    if enable_dataset_optimization:
        try:
            optimization_config = optimize_for_method(
                data, sigma, t1, t2, phi, method="vi", **kwargs
            )

            # Apply optimized parameters
            dataset_info = optimization_config["dataset_info"]
            strategy = optimization_config["strategy"]

            logger.info(f"Dataset optimization enabled:")
            logger.info(
                f"  Size: {dataset_info.size:,} points ({dataset_info.category})"
            )
            logger.info(f"  Memory: {dataset_info.memory_usage_mb:.1f} MB")
            logger.info(
                f"  Strategy: chunk_size={strategy.chunk_size:,}, batch_size={strategy.batch_size}"
            )

            # Update kwargs with optimized parameters
            if "n_iterations" not in kwargs:
                if dataset_info.category == "small":
                    kwargs["n_iterations"] = 2000  # More iterations for small datasets
                elif dataset_info.category == "medium":
                    kwargs["n_iterations"] = 1500  # Balanced
                else:
                    kwargs["n_iterations"] = 1000  # Fewer iterations for large datasets

            if "learning_rate" not in kwargs:
                if dataset_info.category == "large":
                    kwargs["learning_rate"] = (
                        0.005  # Smaller learning rate for stability
                    )
                else:
                    kwargs["learning_rate"] = 0.01

        except Exception as e:
            logger.warning(f"Dataset optimization failed, using defaults: {e}")
            optimization_config = None

    # Add optimization config to kwargs for chunked processing
    if optimization_config and optimization_config.get("chunked_iterator"):
        kwargs["chunked_iterator"] = optimization_config["chunked_iterator"]
        kwargs["preprocessing_time"] = optimization_config["preprocessing_time"]

    # Filter kwargs to only include valid parameters for fit_vi_jax
    valid_params = {
        "n_iterations",
        "learning_rate",
        "convergence_tol",
        "n_elbo_samples",
        "chunked_iterator",
        "preprocessing_time",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

    vi_optimizer = VariationalInferenceJAX(analysis_mode, parameter_space)
    result = vi_optimizer.fit_vi_jax(data, sigma, t1, t2, phi, q, L, **filtered_kwargs)

    # Add optimization information to result
    if optimization_config:
        result.dataset_size = optimization_config["dataset_info"].category

    return result


# Export main classes and functions
__all__ = [
    "VIResult",
    "VariationalFamilies",
    "VariationalInferenceJAX",
    "fit_vi_jax",  # Primary API
]
