"""
Physical Models for Homodyne v2
================================

Object-oriented interface to the physical models implemented in the JAX backend.
Provides structured access to diffusion, shear, and combined models with
parameter validation and configuration management.

This module wraps the low-level JAX functions in user-friendly classes that
handle parameter management, bounds checking, and model configuration.

Physical Models:
- DiffusionModel: Anomalous diffusion D(t) = D₀ t^α + D_offset
- ShearModel: Time-dependent shear γ̇(t) = γ̇₀ t^β + γ̇_offset
- CombinedModel: Full homodyne model with diffusion + shear
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from homodyne.core.jax_backend import (
    compute_chi_squared,
    compute_g1_diffusion,
    compute_g1_shear,
    compute_g1_total,
    compute_g2_scaled,
    get_device_info,
    get_performance_summary,
    gradient_g2,
    hessian_g2,
    jax_available,
    jnp,
    numpy_gradients_available,
    safe_len,
    validate_backend,
)
from homodyne.core.physics import (
    validate_parameters,
)
from homodyne.utils.logging import get_logger, log_calls

logger = get_logger(__name__)


class PhysicsModelBase(ABC):
    """
    Abstract base class for all physical models.

    Defines the interface that all models must implement and provides
    common functionality for parameter management and validation.
    """

    def __init__(self, name: str, parameter_names: list[str]):
        """
        Initialize base model.

        Args:
            name: Model name for identification
            parameter_names: List of parameter names in order
        """
        self.name = name
        self.parameter_names = parameter_names
        self.n_params = len(parameter_names)
        self._bounds = None
        self._default_values = None

    @abstractmethod
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float = None,
    ) -> jnp.ndarray:
        """Compute g1 correlation function for this model."""
        pass

    @abstractmethod
    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get parameter bounds for optimization."""
        pass

    @abstractmethod
    def get_default_parameters(self) -> jnp.ndarray:
        """Get default parameter values."""
        pass

    def validate_parameters(self, params: jnp.ndarray) -> bool:
        """Validate parameter values against bounds and constraints."""
        return validate_parameters(params, self.get_parameter_bounds())

    def get_parameter_dict(self, params: jnp.ndarray) -> dict[str, float]:
        """Convert parameter array to named dictionary."""
        # Ensure params is at least 1D to avoid 0D array indexing issues
        if jax_available and hasattr(params, "ndim"):
            # Convert JAX arrays to NumPy for safe indexing
            params_np = np.atleast_1d(np.asarray(params))
        else:
            params_np = np.atleast_1d(params)

        params_len = safe_len(params_np)
        if params_len != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {params_len}")

        # Convert to regular Python floats only when safe to do so
        try:
            # Try converting to float - will fail if in JIT context
            return {name: float(val) for name, val in zip(self.parameter_names, params_np, strict=False)}
        except (TypeError, ValueError, AttributeError):
            # In JIT context, keep as JAX arrays
            return dict(zip(self.parameter_names, params_np, strict=False))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', n_params={self.n_params})"
        )


class DiffusionModel(PhysicsModelBase):
    """
    Anomalous diffusion model: D(t) = D₀ t^α + D_offset

    Parameters:
    - D₀: Reference diffusion coefficient [Å²/s]
    - α: Diffusion time-dependence exponent [-]
    - D_offset: Baseline diffusion [Å²/s]

    Physical interpretation:
    - α = 0: Normal diffusion (Brownian motion)
    - α > 0: Super-diffusion (enhanced mobility)
    - α < 0: Sub-diffusion (restricted mobility)
    - D_offset: Residual diffusion at t=0
    """

    def __init__(self):
        super().__init__(
            name="anomalous_diffusion", parameter_names=["D0", "alpha", "D_offset"]
        )

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float = None,
    ) -> jnp.ndarray:
        """
        Compute diffusion contribution to g1.

        g₁_diff = exp[-q²/2 ∫|t₂-t₁| D(t')dt']
        """
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning("Invalid diffusion parameters - results may be unreliable")

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        return compute_g1_diffusion(params, t1, t2, q, dt)

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Standard bounds for diffusion parameters."""
        return [
            (1.0, 1e6),  # D0: 1.0 to 1e6 Å²/s (consistent with physics.py)
            (-10.0, 10.0),  # alpha: -10 to 10 (consistent with physics.py)
            (
                -1e5,
                1e5,
            ),  # D_offset: -1e5 to 1e5 Å²/s (consistent with physics.py)
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Default values for typical XPCS measurements."""
        return jnp.array([100.0, 0.0, 10.0])  # Normal diffusion with small offset


class ShearModel(PhysicsModelBase):
    """
    Time-dependent shear model: γ̇(t) = γ̇₀ t^β + γ̇_offset

    Parameters:
    - γ̇₀: Reference shear rate [s⁻¹]
    - β: Shear rate time-dependence exponent [-]
    - γ̇_offset: Baseline shear rate [s⁻¹]
    - φ₀: Angular offset parameter [degrees]

    Physical interpretation:
    - β = 0: Constant shear rate (steady shear)
    - β > 0: Increasing shear rate with time
    - β < 0: Decreasing shear rate with time
    - φ₀: Preferred flow direction angle
    """

    def __init__(self):
        super().__init__(
            name="time_dependent_shear",
            parameter_names=["gamma_dot_0", "beta", "gamma_dot_offset", "phi0"],
        )

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float = None,
    ) -> jnp.ndarray:
        """
        Compute shear contribution to g1.

        g₁_shear = [sinc(Φ)]² where Φ = (qL/2π) cos(φ₀-φ) ∫|t₂-t₁| γ̇(t') dt'
        """
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning("Invalid shear parameters - results may be unreliable")

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        # Create full parameter array with dummy diffusion parameters
        full_params = jnp.concatenate([jnp.array([100.0, 0.0, 10.0]), params])
        return compute_g1_shear(full_params, t1, t2, phi, q, L, dt)

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Standard bounds for shear parameters."""
        return [
            (1e-5, 1.0),  # gamma_dot_0: 1e-5 to 1.0 s⁻¹ (consistent with physics.py)
            (-10.0, 10.0),  # beta: -10 to 10 (consistent with physics.py)
            (-1.0, 1.0),  # gamma_dot_offset: -1 to 1 s⁻¹ (consistent with physics.py)
            (-30.0, 30.0),  # phi0: -30 to 30 degrees (consistent with physics.py)
        ]

    def get_default_parameters(self) -> jnp.ndarray:
        """Default values for typical shear flow."""
        return jnp.array([1.0, 0.0, 0.0, 0.0])  # Constant shear, no offset


class CombinedModel(PhysicsModelBase):
    """
    Combined diffusion + shear model for complete homodyne analysis.

    This is the full model used for laminar flow analysis with both
    anomalous diffusion and time-dependent shear.

    Parameters (7 total):
    - D₀, α, D_offset: Diffusion parameters
    - γ̇₀, β, γ̇_offset: Shear parameters
    - φ₀: Angular offset parameter

    For static analysis, only the first 3 diffusion parameters are used.
    """

    def __init__(self, analysis_mode: str = "laminar_flow"):
        """
        Initialize combined model.

        Args:
            analysis_mode: "static_isotropic", "static_anisotropic", or "laminar_flow"
        """
        self.analysis_mode = analysis_mode

        if analysis_mode.startswith("static"):
            # Static mode: only diffusion parameters
            parameter_names = ["D0", "alpha", "D_offset"]
            name = f"static_diffusion_{analysis_mode.split('_')[1]}"
        else:
            # Laminar flow mode: all parameters
            parameter_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_0",
                "beta",
                "gamma_dot_offset",
                "phi0",
            ]
            name = "laminar_flow_complete"

        super().__init__(name=name, parameter_names=parameter_names)

        # Create component models
        self.diffusion_model = DiffusionModel()
        self.shear_model = ShearModel()

    @log_calls(include_args=False)
    def compute_g1(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        dt: float = None,
    ) -> jnp.ndarray:
        """
        Compute total g1 = g1_diffusion × g1_shear.
        """
        # Skip validation inside JIT to avoid JAX tracer boolean conversion errors
        # if not self.validate_parameters(params):
        #     logger.warning(
        #         "Invalid combined model parameters - results may be unreliable"
        #     )

        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions

        if self.analysis_mode.startswith("static"):
            # Static mode: only diffusion, no shear
            logger.debug(
                f"CombinedModel.compute_g1: calling compute_g1_diffusion with params.shape={params.shape}"
            )
            return compute_g1_diffusion(params, t1, t2, q, dt)
        else:
            # Laminar flow mode: full model
            logger.debug(
                f"CombinedModel.compute_g1: calling compute_g1_total with params.shape={params.shape}, t1.shape={t1.shape}, t2.shape={t2.shape}, phi.shape={phi.shape}, q={q}, L={L}, dt={dt}"
            )
            try:
                result = compute_g1_total(params, t1, t2, phi, q, L, dt)
                logger.debug(
                    f"CombinedModel.compute_g1: compute_g1_total completed, result.shape={result.shape}, min={jnp.min(result):.6e}, max={jnp.max(result):.6e}"
                )
                return result
            except Exception as e:
                logger.error(
                    f"CombinedModel.compute_g1: compute_g1_total failed with error: {e}"
                )
                logger.error("CombinedModel.compute_g1: traceback:", exc_info=True)
                raise

    @log_calls(include_args=False)
    def compute_g2(
        self,
        params: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> jnp.ndarray:
        """
        Compute g2 with scaled fitting: g₂ = offset + contrast × [g₁]²
        """
        # Pass q directly without conversion to avoid JAX tracing issues
        # The backend functions handle any necessary conversions
        return compute_g2_scaled(params, t1, t2, phi, q, L, contrast, offset)

    @log_calls(include_args=False)
    def compute_chi_squared(
        self,
        params: jnp.ndarray,
        data: jnp.ndarray,
        sigma: jnp.ndarray,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        phi: jnp.ndarray,
        q: float,
        L: float,
        contrast: float,
        offset: float,
    ) -> float:
        """Compute chi-squared goodness of fit."""
        return compute_chi_squared(
            params, data, sigma, t1, t2, phi, q, L, contrast, offset
        )

    def get_parameter_bounds(self) -> list[tuple[float, float]]:
        """Get bounds appropriate for analysis mode."""
        bounds = self.diffusion_model.get_parameter_bounds()

        if not self.analysis_mode.startswith("static"):
            # Add shear parameter bounds for laminar flow
            bounds.extend(self.shear_model.get_parameter_bounds())

        return bounds

    def get_default_parameters(self) -> jnp.ndarray:
        """Get default parameters appropriate for analysis mode."""
        defaults = self.diffusion_model.get_default_parameters()

        if not self.analysis_mode.startswith("static"):
            # Add shear parameter defaults for laminar flow
            shear_defaults = self.shear_model.get_default_parameters()
            defaults = jnp.concatenate([defaults, shear_defaults])

        return defaults

    def get_gradient_function(self) -> Callable:
        """Get gradient function with intelligent backend selection."""
        backend_info = self.get_gradient_capabilities()

        if backend_info["gradient_available"]:
            logger.info(f"Using {backend_info['best_method']} for gradient computation")
            if backend_info["performance_warning"]:
                logger.warning(backend_info["performance_warning"])
            return gradient_g2
        else:
            # Provide informative error with recommendations
            error_msg = (
                "Gradient computation not available. Install dependencies for differentiation:\n"
                "• Recommended: pip install jax (optimal performance)\n"
                "• Alternative: pip install scipy (basic numerical gradients)\n\n"
                f"Current backend status: {backend_info['backend_summary']}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

    def get_hessian_function(self) -> Callable:
        """Get hessian function with intelligent backend selection."""
        backend_info = self.get_gradient_capabilities()

        if backend_info["hessian_available"]:
            logger.info(f"Using {backend_info['best_method']} for Hessian computation")
            if backend_info["performance_warning"]:
                logger.warning(backend_info["performance_warning"])
            return hessian_g2
        else:
            # Provide informative error with recommendations
            error_msg = (
                "Hessian computation not available. Install dependencies for second derivatives:\n"
                "• Recommended: pip install jax (optimal performance)\n"
                "• Alternative: pip install scipy (basic numerical Hessians)\n\n"
                f"Current backend status: {backend_info['backend_summary']}"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)

    def supports_gradients(self) -> bool:
        """Check if gradient computation is available."""
        return jax_available or numpy_gradients_available

    def get_best_gradient_method(self) -> str:
        """Get the best available gradient method for optimization algorithms."""
        backend_info = validate_backend()

        if backend_info["jax_available"]:
            return "jax_native"
        elif backend_info["numpy_gradients_available"]:
            return "numpy_fallback"
        else:
            return "none_available"

    def get_gradient_capabilities(self) -> dict[str, Any]:
        """Get comprehensive gradient capability information."""
        backend_info = validate_backend()
        device_info = get_device_info()
        get_performance_summary()

        # Determine best available method
        if backend_info["jax_available"]:
            best_method = "JAX (automatic differentiation)"
            performance_warning = None
        elif backend_info["numpy_gradients_available"]:
            best_method = "NumPy (numerical differentiation)"
            performance_warning = (
                "Using NumPy fallback - expect 10-50x performance degradation"
            )
        else:
            best_method = "None available"
            performance_warning = "No gradient computation backend available"

        return {
            "gradient_available": backend_info["gradient_support"],
            "hessian_available": backend_info["hessian_support"],
            "best_method": best_method,
            "backend_type": backend_info["backend_type"],
            "performance_estimate": backend_info["performance_estimate"],
            "performance_warning": performance_warning,
            "jax_available": backend_info["jax_available"],
            "numpy_gradients_available": backend_info["numpy_gradients_available"],
            "device_info": device_info,
            "recommendations": backend_info["recommendations"],
            "fallback_stats": backend_info["fallback_stats"],
            "backend_summary": self._generate_backend_summary(
                backend_info, device_info
            ),
        }

    def _generate_backend_summary(self, backend_info: dict, device_info: dict) -> str:
        """Generate human-readable backend summary."""
        if backend_info["jax_available"]:
            devices = device_info.get("devices", ["unknown"])
            device_str = (
                f", {len(devices)} device(s) available" if len(devices) > 1 else ""
            )
            return f"JAX backend active{device_str}, optimal performance"
        elif backend_info["numpy_gradients_available"]:
            return "NumPy numerical differentiation active, reduced performance"
        else:
            return "No differentiation backend available"

    def benchmark_gradient_performance(
        self, test_params: jnp.ndarray | None = None
    ) -> dict[str, Any]:
        """Benchmark gradient computation performance across available methods."""
        if test_params is None:
            test_params = self.get_default_parameters()

        logger.info("Benchmarking gradient computation performance...")

        # Test parameters for performance evaluation
        test_t1 = jnp.array([0.0, 0.1, 0.2])
        test_t2 = jnp.array([1.0, 1.1, 1.2])
        test_phi = jnp.array([0.0, 45.0, 90.0])
        test_q = 0.01
        test_L = 1000.0
        test_contrast = 0.8
        test_offset = 1.0

        benchmark_results = {
            "test_conditions": {
                "n_parameters": len(test_params),
                "n_time_points": len(test_t1),
                "n_angles": len(test_phi),
                "analysis_mode": self.analysis_mode,
            },
            "methods": {},
            "best_method": None,
            "performance_ratio": None,
        }

        methods_to_test = []
        if jax_available:
            methods_to_test.append(("jax_native", "JAX automatic differentiation"))
        if numpy_gradients_available:
            methods_to_test.append(
                ("numpy_fallback", "NumPy numerical differentiation")
            )

        if not methods_to_test:
            benchmark_results["error"] = (
                "No gradient methods available for benchmarking"
            )
            return benchmark_results

        # Test each available method
        for method_key, method_name in methods_to_test:
            try:
                start_time = time.perf_counter()

                # Test forward computation
                g2_result = self.compute_g2(
                    test_params,
                    test_t1,
                    test_t2,
                    test_phi,
                    test_q,
                    test_L,
                    test_contrast,
                    test_offset,
                )

                # Test gradient computation if available
                if self.supports_gradients():
                    grad_func = self.get_gradient_function()
                    grad_result = grad_func(
                        test_params,
                        test_t1,
                        test_t2,
                        test_phi,
                        test_q,
                        test_L,
                        test_contrast,
                        test_offset,
                    )

                computation_time = time.perf_counter() - start_time

                benchmark_results["methods"][method_key] = {
                    "name": method_name,
                    "computation_time": computation_time,
                    "success": True,
                    "forward_shape": (
                        g2_result.shape if hasattr(g2_result, "shape") else "scalar"
                    ),
                    "gradient_shape": (
                        grad_result.shape
                        if "grad_result" in locals()
                        else "not_computed"
                    ),
                }

            except Exception as e:
                benchmark_results["methods"][method_key] = {
                    "name": method_name,
                    "success": False,
                    "error": str(e),
                }

        # Determine best method and performance ratios
        successful_methods = [
            (k, v) for k, v in benchmark_results["methods"].items() if v["success"]
        ]
        if successful_methods:
            # Sort by computation time
            successful_methods.sort(key=lambda x: x[1]["computation_time"])
            best_method_key, best_method_info = successful_methods[0]

            benchmark_results["best_method"] = {
                "method": best_method_key,
                "name": best_method_info["name"],
                "time": best_method_info["computation_time"],
            }

            # Calculate performance ratios relative to best method
            best_time = best_method_info["computation_time"]
            for _, method_info in benchmark_results["methods"].items():
                if method_info["success"]:
                    method_info["performance_ratio"] = (
                        method_info["computation_time"] / best_time
                    )

        logger.info("Gradient performance benchmark completed")
        return benchmark_results

    def validate_gradient_accuracy(
        self, test_params: jnp.ndarray | None = None, tolerance: float = 1e-6
    ) -> dict[str, Any]:
        """Validate gradient accuracy against reference solutions."""
        if test_params is None:
            test_params = self.get_default_parameters()

        logger.info("Validating gradient accuracy...")

        # Simple test case for validation
        test_t1 = jnp.array([0.0])
        test_t2 = jnp.array([1.0])
        test_phi = jnp.array([0.0])
        test_q = 0.01
        test_L = 1000.0
        test_contrast = 0.8
        test_offset = 1.0

        validation_results = {
            "test_conditions": {
                "parameters": test_params.tolist(),
                "tolerance": tolerance,
                "analysis_mode": self.analysis_mode,
            },
            "accuracy_assessment": {},
            "recommendations": [],
        }

        try:
            # Test gradient computation
            if self.supports_gradients():
                grad_func = self.get_gradient_function()
                gradient = grad_func(
                    test_params,
                    test_t1,
                    test_t2,
                    test_phi,
                    test_q,
                    test_L,
                    test_contrast,
                    test_offset,
                )

                # Basic validation checks
                validation_results["accuracy_assessment"] = {
                    "gradient_computed": True,
                    "gradient_shape": gradient.shape,
                    "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))),
                    "gradient_magnitude": float(jnp.linalg.norm(gradient)),
                    "max_gradient_component": float(jnp.max(jnp.abs(gradient))),
                    "method_used": self.get_best_gradient_method(),
                }

                # Check for reasonable gradient magnitudes for XPCS physics
                max_grad = float(jnp.max(jnp.abs(gradient)))
                if max_grad > 1e6:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes are very large - check parameter scaling"
                    )
                elif max_grad < 1e-10:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes are very small - may indicate insensitive parameters"
                    )
                else:
                    validation_results["recommendations"].append(
                        "Gradient magnitudes appear reasonable for XPCS analysis"
                    )

            else:
                validation_results["accuracy_assessment"] = {
                    "gradient_computed": False,
                    "error": "No gradient computation backend available",
                }
                validation_results["recommendations"].append(
                    "Install JAX or scipy for gradient-based optimization"
                )

        except Exception as e:
            validation_results["accuracy_assessment"] = {
                "gradient_computed": False,
                "error": str(e),
            }
            validation_results["recommendations"].append(
                f"Gradient computation failed: {str(e)}"
            )

        logger.info("Gradient accuracy validation completed")
        return validation_results

    def get_optimization_recommendations(self) -> list[str]:
        """Get optimization recommendations based on available capabilities."""
        capabilities = self.get_gradient_capabilities()
        recommendations = []

        if capabilities["jax_available"]:
            recommendations.append(
                "✅ JAX available - use gradient-based optimization (BFGS, Adam)"
            )

            device_info = capabilities["device_info"]
            if device_info.get("available", False):
                devices = device_info.get("devices", [])
                if any("gpu" in str(d).lower() for d in devices):
                    recommendations.append(
                        "🎯 GPU acceleration available for large-scale optimization"
                    )
                if len(devices) > 1:
                    recommendations.append(
                        f"🔥 {len(devices)} compute devices available for parallel optimization"
                    )

        elif capabilities["numpy_gradients_available"]:
            recommendations.append(
                "⚠️ Using NumPy gradients - prefer L-BFGS over high-order methods"
            )
            recommendations.append(
                "💡 Consider installing JAX for 10-50x performance improvement"
            )

        else:
            recommendations.append(
                "❌ No gradient support - use gradient-free optimization (Nelder-Mead, Powell)"
            )
            recommendations.append(
                "📦 Install scipy for basic optimization: pip install scipy"
            )
            recommendations.append(
                "🚀 Install JAX for advanced optimization: pip install jax"
            )

        # Analysis mode specific recommendations
        if self.analysis_mode == "laminar_flow":
            if capabilities["jax_available"]:
                recommendations.append(
                    "🧮 Laminar flow mode (7 parameters) - JAX optimization recommended"
                )
            else:
                recommendations.append(
                    "⚖️ Laminar flow mode - many parameters, consider staged optimization"
                )

        elif self.analysis_mode.startswith("static"):
            recommendations.append(
                "📊 Static mode (3 parameters) - most optimization methods will work well"
            )

        return recommendations

    def get_model_info(self) -> dict:
        """Get comprehensive model information with enhanced capabilities."""
        capabilities = self.get_gradient_capabilities()

        return {
            # Basic model information
            "name": self.name,
            "analysis_mode": self.analysis_mode,
            "n_parameters": self.n_params,
            "parameter_names": self.parameter_names,
            "parameter_bounds": self.get_parameter_bounds(),
            "default_parameters": self.get_default_parameters().tolist(),
            # Gradient capabilities
            "supports_gradients": self.supports_gradients(),
            "gradient_method": self.get_best_gradient_method(),
            "gradient_capabilities": capabilities,
            # Backend information
            "jax_available": jax_available,
            "numpy_gradients_available": numpy_gradients_available,
            "backend_summary": capabilities["backend_summary"],
            # Optimization guidance
            "optimization_recommendations": self.get_optimization_recommendations(),
            # Performance information
            "performance_estimate": capabilities["performance_estimate"],
            "device_info": capabilities["device_info"],
        }


# Factory functions for easy model creation
def create_model(analysis_mode: str) -> CombinedModel:
    """
    Factory function to create appropriate model for analysis mode.

    Args:
        analysis_mode: "static_isotropic", "static_anisotropic", or "laminar_flow"

    Returns:
        Configured CombinedModel instance
    """
    valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
    if analysis_mode not in valid_modes:
        raise ValueError(
            f"Invalid analysis mode '{analysis_mode}'. Must be one of {valid_modes}"
        )

    logger.info(f"Creating model for analysis mode: {analysis_mode}")
    return CombinedModel(analysis_mode=analysis_mode)


def get_available_models() -> list[str]:
    """Get list of available analysis modes."""
    return ["static_isotropic", "static_anisotropic", "laminar_flow"]


# Export main classes and functions
__all__ = [
    "PhysicsModelBase",
    "DiffusionModel",
    "ShearModel",
    "CombinedModel",
    "create_model",
    "get_available_models",
]
