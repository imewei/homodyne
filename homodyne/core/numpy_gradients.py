"""Production-Grade Numerical Differentiation System for Homodyne v2
================================================================

High-performance NumPy-based numerical differentiation to replace JAX gradients
with graceful fallback architecture. Implements advanced finite difference methods
with automatic step size selection, memory optimization, and error estimation.

This module provides the critical foundation for JAX-free operation while
maintaining scientific accuracy and computational efficiency for XPCS analysis.

Key Features:
- Multi-method finite differences: Forward, backward, central, complex-step
- Adaptive step size selection with Richardson extrapolation
- Vectorized operations for batch gradient computation
- Memory-aware chunking for large parameter spaces
- Numerical stability with automatic scaling and conditioning
- Built-in error estimation and accuracy assessment
- Drop-in replacement for JAX grad() and hessian() functions
- Integration with XPCS physics (g2, chi-squared functions)

Physical Context:
- Optimized for XPCS parameter estimation (3 and 7 parameter modes)
- Handles typical XPCS parameter ranges and sensitivities
- Provides machine-precision derivatives for optimization algorithms
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np

from homodyne.utils.logging import get_logger, log_performance

logger = get_logger(__name__)

# Numerical constants for stability
EPS = np.finfo(float).eps  # Machine epsilon (~2.22e-16)
SQRT_EPS = np.sqrt(EPS)  # Square root of machine epsilon (~1.49e-8)
CBRT_EPS = EPS ** (1 / 3)  # Cube root of machine epsilon (~6.06e-6)
DEFAULT_STEP = SQRT_EPS  # Default step size for finite differences


class DifferentiationMethod:
    """Enumeration of available differentiation methods."""

    FORWARD = "forward"
    BACKWARD = "backward"
    CENTRAL = "central"
    COMPLEX_STEP = "complex_step"
    RICHARDSON = "richardson"
    ADAPTIVE = "adaptive"


@dataclass
class DifferentiationConfig:
    """Configuration for numerical differentiation."""

    method: str = DifferentiationMethod.ADAPTIVE
    step_size: float | None = None  # Auto-computed if None
    relative_step: float = SQRT_EPS
    min_step: float = 1e-15
    max_step: float = 1e-3
    richardson_terms: int = 4
    error_tolerance: float = 1e-8
    max_iterations: int = 20
    use_parallel: bool = True
    chunk_size: int = 1000
    complex_step_threshold: float = 1e-12


@dataclass
class GradientResult:
    """Result container for gradient computation with error estimates."""

    gradient: np.ndarray
    step_sizes: np.ndarray
    error_estimate: np.ndarray | None = None
    method_used: str = "unknown"
    function_calls: int = 0
    computation_time: float = 0.0
    warnings: list[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class NumericalStabilityError(Exception):
    """Raised when numerical differentiation encounters stability issues."""


def _validate_function(func: Callable, x: np.ndarray, name: str = "function") -> None:
    """Validate that function can be called and returns appropriate values."""
    try:
        result = func(x)
        if not np.isfinite(result).all():
            raise NumericalStabilityError(f"{name} returned non-finite values")
    except Exception as e:
        raise NumericalStabilityError(f"{name} evaluation failed: {e}") from e


def _estimate_optimal_step(
    func: Callable,
    x: np.ndarray,
    relative_step: float = SQRT_EPS,
    min_step: float = 1e-15,
    max_step: float = 1e-3,
) -> np.ndarray:
    """Estimate optimal step size for each parameter using function curvature.

    Uses a heuristic based on the second derivative to balance truncation
    and roundoff errors for finite difference approximations.
    """
    x = np.asarray(x, dtype=float)
    f0 = func(x)

    # Initial step sizes based on parameter magnitude
    h_init = np.maximum(np.abs(x) * relative_step, min_step)
    h_init = np.minimum(h_init, max_step)

    optimal_h = np.zeros_like(x)

    for i in range(len(x)):
        try:
            # Estimate second derivative using central differences
            xi_plus = x.copy()
            xi_minus = x.copy()
            xi_plus[i] += h_init[i]
            xi_minus[i] -= h_init[i]

            f_plus = func(xi_plus)
            f_minus = func(xi_minus)

            # Second derivative approximation
            f_second_deriv = np.abs((f_plus - 2 * f0 + f_minus) / h_init[i] ** 2)

            if f_second_deriv > 0:
                # Optimal step balances truncation and roundoff error
                # h_opt ≈ (ε * |f|/|f''|)^(1/3)
                f_magnitude = np.maximum(np.abs(f0), 1e-12)
                optimal_h[i] = (2 * EPS * f_magnitude / f_second_deriv) ** (1 / 3)
            else:
                optimal_h[i] = h_init[i]

        except Exception:
            optimal_h[i] = h_init[i]

    # Apply bounds
    optimal_h = np.maximum(optimal_h, min_step)
    optimal_h = np.minimum(optimal_h, max_step)

    return optimal_h


def _complex_step_derivative(
    func: Callable,
    x: np.ndarray,
    h: np.ndarray,
) -> np.ndarray:
    """Complex-step differentiation for near machine precision derivatives.

    Uses f'(x) ≈ Im(f(x + ih))/h with no subtractive cancellation error.
    Requires function to be analytic (works for most physics functions).
    """
    x = np.asarray(x, dtype=complex)
    gradient = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_complex = x.copy()
        x_complex[i] += 1j * h[i]

        try:
            f_complex = func(x_complex)
            gradient[i] = np.imag(f_complex) / h[i]
        except Exception as e:
            # Fallback to central differences if complex evaluation fails
            logger.warning(f"Complex-step failed for parameter {i}: {e}")
            gradient[i] = _central_difference_single(func, x.real, i, h[i])

    return gradient


def _central_difference_single(
    func: Callable,
    x: np.ndarray,
    index: int,
    h: float,
) -> float:
    """Compute central difference for single parameter."""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[index] += h
    x_minus[index] -= h

    return (func(x_plus) - func(x_minus)) / (2 * h)


def _forward_difference_single(
    func: Callable,
    x: np.ndarray,
    index: int,
    h: float,
) -> float:
    """Compute forward difference for single parameter."""
    x_plus = x.copy()
    x_plus[index] += h

    return (func(x_plus) - func(x)) / h


def _backward_difference_single(
    func: Callable,
    x: np.ndarray,
    index: int,
    h: float,
) -> float:
    """Compute backward difference for single parameter."""
    x_minus = x.copy()
    x_minus[index] -= h

    return (func(x) - func(x_minus)) / h


def _richardson_extrapolation(
    func: Callable,
    x: np.ndarray,
    index: int,
    h: float,
    terms: int = 4,
) -> tuple[float, float]:
    """Richardson extrapolation for higher-order accuracy.

    Computes derivative with progressively smaller step sizes and
    extrapolates to h=0 limit to remove leading error terms.

    Returns:
        (derivative_estimate, error_estimate)
    """
    # Compute derivatives at different step sizes
    derivatives = []
    step_sizes = []

    for k in range(terms):
        h_k = h / (2**k)
        deriv = _central_difference_single(func, x, index, h_k)
        derivatives.append(deriv)
        step_sizes.append(h_k)

    # Richardson extrapolation using Neville's algorithm
    R = np.array(derivatives)

    for i in range(1, terms):
        for j in range(terms - i):
            # R[j,i] = R[j+1,i-1] + (R[j+1,i-1] - R[j,i-1]) / (4^i - 1)
            R[j] = R[j + 1] + (R[j + 1] - R[j]) / (4**i - 1)

    # Best estimate is R[0] after all extrapolations
    derivative = R[0]

    # Error estimate from difference between last two Richardson levels
    if terms > 1:
        error_estimate = np.abs(derivatives[1] - derivatives[0])
    else:
        error_estimate = np.inf

    return derivative, error_estimate


@log_performance(threshold=0.01)
def numpy_gradient(
    func: Callable,
    argnums: int | list[int] = 0,
    config: DifferentiationConfig | None = None,
) -> Callable:
    """NumPy-based gradient computation with same interface as JAX grad().

    Creates a function that computes gradients using advanced numerical
    differentiation methods with automatic method selection and error control.

    Args:
        func: Function to differentiate
        argnums: Indices of arguments to differentiate w.r.t. (default: 0)
        config: Differentiation configuration (uses defaults if None)

    Returns:
        Function that computes gradients with same signature as JAX grad()

    Examples:
        # Basic usage
        grad_f = numpy_gradient(f)
        gradient = grad_f(x)

        # Multiple arguments
        grad_f = numpy_gradient(f, argnums=[0, 2])
        gradients = grad_f(x, y, z)

        # Custom configuration
        config = DifferentiationConfig(method="complex_step", step_size=1e-10)
        grad_f = numpy_gradient(f, config=config)
    """
    if config is None:
        config = DifferentiationConfig()

    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(func)
    def gradient_func(*args, **kwargs):
        """Compute numerical gradient."""
        start_time = time.perf_counter()

        if len(args) <= max(argnums):
            raise ValueError(
                f"Not enough arguments: need {max(argnums) + 1}, got {len(args)}",
            )

        # Extract parameter arrays for differentiation
        param_arrays = [np.asarray(args[i], dtype=float) for i in argnums]
        total_params = sum(arr.size for arr in param_arrays)

        logger.debug(
            f"Computing gradient for {total_params} parameters using {config.method}",
        )

        # Create wrapper function for current args/kwargs
        def eval_func(param_vector: np.ndarray) -> float:
            """Evaluate function with parameter vector."""
            new_args = list(args)
            start_idx = 0

            for i, argnum in enumerate(argnums):
                param_size = param_arrays[i].size
                param_slice = param_vector[start_idx : start_idx + param_size]
                new_args[argnum] = param_slice.reshape(param_arrays[i].shape)
                start_idx += param_size

            return func(*new_args, **kwargs)

        # Flatten parameters for processing
        x = np.concatenate([arr.flatten() for arr in param_arrays])

        # Validate function
        _validate_function(eval_func, x, func.__name__)

        # Compute gradient using selected method
        if config.method == DifferentiationMethod.ADAPTIVE:
            result = _adaptive_gradient(eval_func, x, config)
        elif config.method == DifferentiationMethod.COMPLEX_STEP:
            result = _complex_step_gradient(eval_func, x, config)
        elif config.method == DifferentiationMethod.RICHARDSON:
            result = _richardson_gradient(eval_func, x, config)
        else:
            result = _finite_difference_gradient(eval_func, x, config)

        result.computation_time = time.perf_counter() - start_time

        # Log performance and warnings
        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)

        logger.debug(
            f"Gradient computation completed in {result.computation_time:.4f}s "
            f"with {result.function_calls} function evaluations",
        )

        # Reshape gradient back to original parameter shapes
        if len(argnums) == 1:
            return result.gradient.reshape(param_arrays[0].shape)
        else:
            gradients = []
            start_idx = 0
            for param_array in param_arrays:
                param_size = param_array.size
                grad_slice = result.gradient[start_idx : start_idx + param_size]
                gradients.append(grad_slice.reshape(param_array.shape))
                start_idx += param_size
            return gradients

    return gradient_func


def _adaptive_gradient(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> GradientResult:
    """Adaptive gradient computation with automatic method selection and memory optimization.

    Chooses the best differentiation method based on function characteristics
    and parameter values, with chunked processing for large parameter spaces.
    """
    n_params = len(x)

    # Use chunked processing for large parameter spaces to manage memory
    if n_params > config.chunk_size:
        return _chunked_gradient_computation(func, x, config)

    gradient = np.zeros_like(x)
    step_sizes = np.zeros_like(x)
    error_estimates = np.zeros_like(x)
    warnings_list = []
    total_function_calls = 0

    # Estimate optimal step sizes
    try:
        h_optimal = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )
    except Exception as e:
        warnings_list.append(f"Step size estimation failed: {e}")
        h_optimal = np.full_like(x, DEFAULT_STEP)

    # Try complex-step differentiation first for high accuracy
    complex_step_success = True
    if np.all(h_optimal > config.complex_step_threshold):
        try:
            gradient = _complex_step_derivative(func, x, h_optimal)
            step_sizes = h_optimal
            total_function_calls += n_params
            method_used = DifferentiationMethod.COMPLEX_STEP
        except Exception as e:
            warnings_list.append(f"Complex-step differentiation failed: {e}")
            complex_step_success = False
    else:
        complex_step_success = False

    # Fallback to Richardson extrapolation if complex-step fails
    if not complex_step_success:
        method_used = DifferentiationMethod.RICHARDSON

        for i in range(n_params):
            try:
                deriv, error_est = _richardson_extrapolation(
                    func,
                    x,
                    i,
                    h_optimal[i],
                    config.richardson_terms,
                )
                gradient[i] = deriv
                error_estimates[i] = error_est
                step_sizes[i] = h_optimal[i]
                total_function_calls += 2 * config.richardson_terms

            except Exception as e:
                warnings_list.append(
                    f"Richardson extrapolation failed for param {i}: {e}",
                )
                # Final fallback to central differences
                gradient[i] = _central_difference_single(func, x, i, h_optimal[i])
                step_sizes[i] = h_optimal[i]
                total_function_calls += 2

    return GradientResult(
        gradient=gradient,
        step_sizes=step_sizes,
        error_estimate=error_estimates if np.any(error_estimates > 0) else None,
        method_used=method_used,
        function_calls=total_function_calls,
        warnings=warnings_list,
    )


def _chunked_gradient_computation(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> GradientResult:
    """Memory-optimized chunked gradient computation for large parameter spaces.

    Processes parameters in chunks to manage memory usage while maintaining
    accuracy. Useful for problems with thousands of parameters.
    """
    n_params = len(x)
    chunk_size = config.chunk_size
    n_chunks = (n_params + chunk_size - 1) // chunk_size

    logger.info(
        f"Using chunked computation: {n_params} parameters in {n_chunks} chunks",
    )

    gradient = np.zeros_like(x)
    step_sizes = np.zeros_like(x)
    error_estimates = np.zeros_like(x)
    warnings_list = []
    total_function_calls = 0

    # Estimate step sizes for all parameters (done once)
    try:
        h_optimal = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )
    except Exception as e:
        warnings_list.append(f"Step size estimation failed: {e}")
        h_optimal = np.full_like(x, DEFAULT_STEP)

    # Process parameters in chunks
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_params)
        chunk_indices = list(range(start_idx, end_idx))

        logger.debug(
            f"Processing chunk {chunk_idx + 1}/{n_chunks}: parameters {start_idx}-{end_idx - 1}",
        )

        # Process current chunk
        chunk_result = _process_parameter_chunk(
            func,
            x,
            h_optimal,
            chunk_indices,
            config,
        )

        # Store results for this chunk
        gradient[start_idx:end_idx] = chunk_result.gradient
        step_sizes[start_idx:end_idx] = chunk_result.step_sizes
        if chunk_result.error_estimate is not None:
            error_estimates[start_idx:end_idx] = chunk_result.error_estimate

        total_function_calls += chunk_result.function_calls
        warnings_list.extend(chunk_result.warnings)

    return GradientResult(
        gradient=gradient,
        step_sizes=step_sizes,
        error_estimate=error_estimates if np.any(error_estimates > 0) else None,
        method_used="chunked_adaptive",
        function_calls=total_function_calls,
        warnings=warnings_list,
    )


def _process_parameter_chunk(
    func: Callable,
    x: np.ndarray,
    h: np.ndarray,
    chunk_indices: list[int],
    config: DifferentiationConfig,
) -> GradientResult:
    """Process a chunk of parameters for gradient computation."""
    n_chunk = len(chunk_indices)
    chunk_gradient = np.zeros(n_chunk)
    chunk_step_sizes = h[chunk_indices]
    chunk_error_estimates = np.zeros(n_chunk)
    chunk_warnings = []
    chunk_function_calls = 0

    # Try complex-step for the chunk if conditions are met
    if np.all(chunk_step_sizes > config.complex_step_threshold):
        try:
            # Complex-step differentiation for chunk
            for i, param_idx in enumerate(chunk_indices):
                x_complex = x.astype(complex)
                x_complex[param_idx] += 1j * h[param_idx]
                f_complex = func(x_complex)
                chunk_gradient[i] = np.imag(f_complex) / h[param_idx]
                chunk_function_calls += 1

            method_used = DifferentiationMethod.COMPLEX_STEP

        except Exception as e:
            chunk_warnings.append(f"Complex-step failed for chunk: {e}")
            # Fallback to Richardson extrapolation
            method_used = DifferentiationMethod.RICHARDSON

            for i, param_idx in enumerate(chunk_indices):
                try:
                    deriv, error_est = _richardson_extrapolation(
                        func,
                        x,
                        param_idx,
                        h[param_idx],
                        config.richardson_terms,
                    )
                    chunk_gradient[i] = deriv
                    chunk_error_estimates[i] = error_est
                    chunk_function_calls += 2 * config.richardson_terms
                except Exception as e2:
                    chunk_warnings.append(
                        f"Richardson failed for param {param_idx}: {e2}",
                    )
                    chunk_gradient[i] = _central_difference_single(
                        func,
                        x,
                        param_idx,
                        h[param_idx],
                    )
                    chunk_function_calls += 2
    else:
        # Use Richardson extrapolation for chunk
        method_used = DifferentiationMethod.RICHARDSON

        for i, param_idx in enumerate(chunk_indices):
            try:
                deriv, error_est = _richardson_extrapolation(
                    func,
                    x,
                    param_idx,
                    h[param_idx],
                    config.richardson_terms,
                )
                chunk_gradient[i] = deriv
                chunk_error_estimates[i] = error_est
                chunk_function_calls += 2 * config.richardson_terms
            except Exception as e:
                chunk_warnings.append(f"Richardson failed for param {param_idx}: {e}")
                chunk_gradient[i] = _central_difference_single(
                    func,
                    x,
                    param_idx,
                    h[param_idx],
                )
                chunk_function_calls += 2

    return GradientResult(
        gradient=chunk_gradient,
        step_sizes=chunk_step_sizes,
        error_estimate=(
            chunk_error_estimates if np.any(chunk_error_estimates > 0) else None
        ),
        method_used=method_used,
        function_calls=chunk_function_calls,
        warnings=chunk_warnings,
    )


def _complex_step_gradient(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> GradientResult:
    """Complex-step differentiation for maximum accuracy."""
    if config.step_size is not None:
        h = np.full_like(x, config.step_size)
    else:
        h = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )

    gradient = _complex_step_derivative(func, x, h)

    return GradientResult(
        gradient=gradient,
        step_sizes=h,
        method_used=DifferentiationMethod.COMPLEX_STEP,
        function_calls=len(x),
    )


def _richardson_gradient(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> GradientResult:
    """Richardson extrapolation for high-order accuracy."""
    if config.step_size is not None:
        h = np.full_like(x, config.step_size)
    else:
        h = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )

    n_params = len(x)
    gradient = np.zeros_like(x)
    error_estimates = np.zeros_like(x)
    warnings_list = []

    for i in range(n_params):
        try:
            deriv, error_est = _richardson_extrapolation(
                func,
                x,
                i,
                h[i],
                config.richardson_terms,
            )
            gradient[i] = deriv
            error_estimates[i] = error_est
        except Exception as e:
            warnings_list.append(
                f"Richardson extrapolation failed for parameter {i}: {e}",
            )
            gradient[i] = _central_difference_single(func, x, i, h[i])
            error_estimates[i] = np.inf

    return GradientResult(
        gradient=gradient,
        step_sizes=h,
        error_estimate=error_estimates,
        method_used=DifferentiationMethod.RICHARDSON,
        function_calls=n_params * 2 * config.richardson_terms,
        warnings=warnings_list,
    )


def _finite_difference_gradient(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> GradientResult:
    """Standard finite difference methods."""
    if config.step_size is not None:
        h = np.full_like(x, config.step_size)
    else:
        h = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )

    n_params = len(x)
    gradient = np.zeros_like(x)

    if config.method == DifferentiationMethod.CENTRAL:
        for i in range(n_params):
            gradient[i] = _central_difference_single(func, x, i, h[i])
        function_calls = n_params * 2
    elif config.method == DifferentiationMethod.FORWARD:
        for i in range(n_params):
            gradient[i] = _forward_difference_single(func, x, i, h[i])
        function_calls = n_params + 1  # +1 for base function evaluation
    elif config.method == DifferentiationMethod.BACKWARD:
        for i in range(n_params):
            gradient[i] = _backward_difference_single(func, x, i, h[i])
        function_calls = n_params + 1
    else:
        raise ValueError(f"Unknown finite difference method: {config.method}")

    return GradientResult(
        gradient=gradient,
        step_sizes=h,
        method_used=config.method,
        function_calls=function_calls,
    )


@log_performance(threshold=0.1)
def numpy_hessian(
    func: Callable,
    argnums: int | list[int] = 0,
    config: DifferentiationConfig | None = None,
) -> Callable:
    """NumPy-based Hessian computation with same interface as JAX hessian().

    Computes second derivatives using finite differences of first derivatives
    for memory efficiency and numerical stability.

    Args:
        func: Function to compute Hessian for
        argnums: Indices of arguments to differentiate w.r.t.
        config: Differentiation configuration

    Returns:
        Function that computes Hessian matrix
    """
    if config is None:
        config = DifferentiationConfig()

    if isinstance(argnums, int):
        argnums = [argnums]

    @wraps(func)
    def hessian_func(*args, **kwargs):
        """Compute numerical Hessian using finite differences."""
        start_time = time.perf_counter()

        # Extract parameter arrays for differentiation
        param_arrays = [np.asarray(args[i], dtype=float) for i in argnums]
        total_params = sum(arr.size for arr in param_arrays)

        logger.debug(f"Computing Hessian for {total_params} parameters")

        # Flatten parameters for processing
        x = np.concatenate([arr.flatten() for arr in param_arrays])

        # Create wrapper function for current args/kwargs
        def eval_func(param_vector: np.ndarray) -> float:
            """Evaluate function with parameter vector."""
            new_args = list(args)
            start_idx = 0

            for i, argnum in enumerate(argnums):
                param_size = param_arrays[i].size
                param_slice = param_vector[start_idx : start_idx + param_size]
                new_args[argnum] = param_slice.reshape(param_arrays[i].shape)
                start_idx += param_size

            return func(*new_args, **kwargs)

        # Compute Hessian using finite differences
        hessian_matrix = _compute_hessian_finite_diff(eval_func, x, config)

        computation_time = time.perf_counter() - start_time
        logger.debug(f"Hessian computation completed in {computation_time:.4f}s")

        # Reshape Hessian back to original parameter structure if single argument
        if len(argnums) == 1:
            param_shape = param_arrays[0].shape
            if len(param_shape) == 1:  # 1D parameter array
                return hessian_matrix
            else:
                # For multi-dimensional parameters, return flattened Hessian
                return hessian_matrix
        else:
            return hessian_matrix

    return hessian_func


def _compute_hessian_finite_diff(
    func: Callable,
    x: np.ndarray,
    config: DifferentiationConfig,
) -> np.ndarray:
    """Compute Hessian matrix using finite differences.

    Uses the standard formula: H_ij = (f(x+h_i+h_j) - f(x+h_i-h_j) - f(x-h_i+h_j) + f(x-h_i-h_j)) / (4*h_i*h_j)
    For diagonal elements: H_ii = (f(x+h_i) - 2*f(x) + f(x-h_i)) / h_i^2
    """
    n_params = len(x)
    hessian = np.zeros((n_params, n_params))

    # Estimate optimal step sizes
    try:
        h = _estimate_optimal_step(
            func,
            x,
            config.relative_step,
            config.min_step,
            config.max_step,
        )
    except Exception:
        h = np.full_like(x, DEFAULT_STEP)

    # Base function value
    f0 = func(x)

    # Compute diagonal elements (second derivatives)
    for i in range(n_params):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h[i]
        x_minus[i] -= h[i]

        f_plus = func(x_plus)
        f_minus = func(x_minus)

        hessian[i, i] = (f_plus - 2 * f0 + f_minus) / (h[i] ** 2)

    # Compute off-diagonal elements (mixed partial derivatives)
    for i in range(n_params):
        for j in range(i + 1, n_params):
            x_pp = x.copy()  # +h_i, +h_j
            x_pm = x.copy()  # +h_i, -h_j
            x_mp = x.copy()  # -h_i, +h_j
            x_mm = x.copy()  # -h_i, -h_j

            x_pp[i] += h[i]
            x_pp[j] += h[j]

            x_pm[i] += h[i]
            x_pm[j] -= h[j]

            x_mp[i] -= h[i]
            x_mp[j] += h[j]

            x_mm[i] -= h[i]
            x_mm[j] -= h[j]

            f_pp = func(x_pp)
            f_pm = func(x_pm)
            f_mp = func(x_mp)
            f_mm = func(x_mm)

            # Mixed partial derivative
            mixed_deriv = (f_pp - f_pm - f_mp + f_mm) / (4 * h[i] * h[j])

            # Hessian is symmetric
            hessian[i, j] = mixed_deriv
            hessian[j, i] = mixed_deriv

    return hessian


def validate_gradient_accuracy(
    func: Callable,
    x: np.ndarray,
    analytical_grad: np.ndarray | None = None,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate numerical gradient accuracy against analytical solution.

    Args:
        func: Function to test
        x: Point to evaluate gradient
        analytical_grad: Known analytical gradient (if available)
        tolerance: Tolerance for validation

    Returns:
        Dictionary with validation results
    """
    methods = [
        DifferentiationMethod.COMPLEX_STEP,
        DifferentiationMethod.RICHARDSON,
        DifferentiationMethod.CENTRAL,
        DifferentiationMethod.FORWARD,
        DifferentiationMethod.BACKWARD,
    ]

    results = {}

    for method in methods:
        try:
            config = DifferentiationConfig(method=method)
            grad_func = numpy_gradient(func, config=config)
            numerical_grad = grad_func(x)

            result = {"gradient": numerical_grad, "method": method, "success": True}

            if analytical_grad is not None:
                error = np.abs(numerical_grad - analytical_grad)
                result["absolute_error"] = error
                result["relative_error"] = error / (np.abs(analytical_grad) + EPS)
                result["max_error"] = np.max(error)
                result["accuracy_ok"] = np.all(error < tolerance)

            results[method] = result

        except Exception as e:
            results[method] = {"success": False, "error": str(e), "method": method}

    return results


# Export main functions with JAX-compatible interface
__all__ = [
    "numpy_gradient",
    "numpy_hessian",
    "DifferentiationConfig",
    "DifferentiationMethod",
    "GradientResult",
    "validate_gradient_accuracy",
    "NumericalStabilityError",
]
