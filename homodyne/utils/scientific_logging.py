"""
Scientific Computing Logging Contexts for Homodyne v2
=====================================================

Specialized logging utilities for X-ray Photon Correlation Spectroscopy (XPCS)
analysis and scientific computing workflows. Provides domain-specific logging
contexts, validation, and monitoring for:

- XPCS data loading and preprocessing
- Correlation function computation
- Physics parameter validation and bounds checking
- Model fitting and optimization progress
- Numerical stability monitoring
- Analysis mode transitions
- Performance benchmarking for scientific algorithms

This module extends the base logging system with scientific computing
awareness while integrating with JAX logging for GPU-accelerated computations.
"""

import functools
import time
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import threading

# Scientific computing imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

# JAX integration
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

from .logging import get_logger, log_operation
from .jax_logging import jax_operation_context, log_array_stats


@dataclass
class XPCSDataInfo:
    """Information about XPCS data loading and characteristics."""
    filepath: str
    file_format: str
    data_shape: Optional[Tuple[int, ...]] = None
    data_type: Optional[str] = None
    file_size_mb: Optional[float] = None
    q_vectors: Optional[int] = None
    time_points: Optional[int] = None
    phi_angles: Optional[int] = None
    loading_time: Optional[float] = None
    validation_passed: bool = True
    validation_warnings: List[str] = field(default_factory=list)
    preprocessing_applied: List[str] = field(default_factory=list)


@dataclass
class PhysicsValidationResult:
    """Result of physics parameter validation."""
    parameter_name: str
    value: float
    is_valid: bool
    validation_type: str  # bounds, physical_meaning, numerical_stability
    expected_range: Optional[Tuple[float, float]] = None
    warning_message: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class CorrelationComputationMetrics:
    """Metrics for correlation function computation."""
    computation_method: str  # 'vectorized', 'chunked', 'distributed'
    input_data_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    chunk_size: Optional[int] = None
    memory_peak_mb: Optional[float] = None
    computation_time: Optional[float] = None
    numerical_precision: Optional[str] = None  # float32, float64
    cache_hit: bool = False
    cache_size_mb: Optional[float] = None


@dataclass
class FittingProgressSnapshot:
    """Snapshot of model fitting progress."""
    iteration: int
    loss_value: float
    parameter_values: Dict[str, float]
    parameter_gradients: Optional[Dict[str, float]] = None
    convergence_criteria: Optional[Dict[str, float]] = None
    timestamp: float = field(default_factory=time.time)
    optimization_method: Optional[str] = None
    
    @property
    def gradient_norm(self) -> Optional[float]:
        if self.parameter_gradients:
            return sum(g**2 for g in self.parameter_gradients.values())**0.5
        return None


class XPCSDataValidator:
    """Validator for XPCS data quality and physics constraints."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        
        # Physics parameter bounds (typical XPCS ranges)
        self.parameter_bounds = {
            'D0': (1e-15, 1e-8),           # Diffusion coefficient (m²/s)
            'alpha': (0.1, 2.0),           # Anomalous diffusion exponent
            'D_offset': (0.0, 1e-9),       # Diffusion offset (m²/s)
            'gamma_dot_t0': (0.1, 1000.0), # Shear rate at t=0 (s⁻¹)
            'beta': (0.0, 2.0),            # Shear rate exponent
            'gamma_dot_t_offset': (0.0, 100.0), # Shear rate offset (s⁻¹)
            'phi0': (0.0, 2*3.14159),      # Phase offset (radians)
            'contrast': (0.0, 1.0),        # Correlation contrast
            'offset': (0.0, 2.0),          # Baseline offset
        }
        
        # Numerical stability thresholds
        self.stability_thresholds = {
            'min_contrast': 1e-6,
            'max_condition_number': 1e12,
            'min_data_variance': 1e-10,
            'max_parameter_ratio': 1e6
        }
    
    def validate_physics_parameter(self, name: str, value: float) -> PhysicsValidationResult:
        """Validate a single physics parameter."""
        
        if name not in self.parameter_bounds:
            return PhysicsValidationResult(
                parameter_name=name,
                value=value,
                is_valid=True,  # Unknown parameters pass by default
                validation_type="unknown",
                warning_message=f"Unknown parameter {name} - validation skipped"
            )
        
        bounds = self.parameter_bounds[name]
        is_valid = bounds[0] <= value <= bounds[1]
        
        # Check for special cases
        warning = None
        suggestion = None
        
        if not is_valid:
            if value < bounds[0]:
                warning = f"{name}={value:.2e} is below typical range"
                suggestion = f"Consider values >= {bounds[0]:.2e}"
            else:
                warning = f"{name}={value:.2e} is above typical range"
                suggestion = f"Consider values <= {bounds[1]:.2e}"
        
        # Additional physics-specific checks
        if name == 'alpha' and value < 0.5:
            warning = "Sub-diffusive behavior (α < 0.5) detected"
            suggestion = "Verify experimental conditions for sub-diffusion"
        elif name == 'alpha' and value > 1.5:
            warning = "Super-diffusive behavior (α > 1.5) detected"
            suggestion = "Check for directed motion or experimental artifacts"
        
        if name == 'contrast' and value < 0.1:
            warning = "Low contrast value may indicate poor signal quality"
            suggestion = "Verify detector calibration and beam stability"
        
        return PhysicsValidationResult(
            parameter_name=name,
            value=value,
            is_valid=is_valid,
            validation_type="bounds",
            expected_range=bounds,
            warning_message=warning,
            suggestion=suggestion
        )
    
    def validate_correlation_data(self, correlation_data, 
                                q_vectors: Optional[np.ndarray] = None) -> List[str]:
        """Validate correlation function data quality."""
        warnings = []
        
        if not hasattr(correlation_data, 'shape'):
            warnings.append("Correlation data is not array-like")
            return warnings
        
        # Check for NaN or Inf values
        if HAS_NUMPY:
            if not np.isfinite(correlation_data).all():
                nan_count = np.isnan(correlation_data).sum()
                inf_count = np.isinf(correlation_data).sum()
                warnings.append(f"Non-finite values detected: {nan_count} NaN, {inf_count} Inf")
        
        # Check data variance
        if HAS_NUMPY:
            try:
                variance = np.var(correlation_data)
                if variance < self.stability_thresholds['min_data_variance']:
                    warnings.append(f"Very low data variance ({variance:.2e}) may indicate processing issues")
            except:
                warnings.append("Could not compute data variance")
        
        # Check dynamic range
        if HAS_NUMPY:
            try:
                data_min, data_max = np.min(correlation_data), np.max(correlation_data)
                dynamic_range = data_max / max(abs(data_min), 1e-15)
                if dynamic_range > self.stability_thresholds['max_parameter_ratio']:
                    warnings.append(f"Large dynamic range ({dynamic_range:.1e}) may cause numerical issues")
            except:
                warnings.append("Could not assess dynamic range")
        
        # Check q-vector consistency if provided
        if q_vectors is not None and hasattr(q_vectors, 'shape'):
            if len(q_vectors.shape) > 0 and correlation_data.shape[-1] != len(q_vectors):
                warnings.append(f"Q-vector count ({len(q_vectors)}) doesn't match correlation data shape ({correlation_data.shape})")
        
        return warnings


class FittingProgressTracker:
    """Track and log model fitting progress."""
    
    def __init__(self, logger: Optional = None, max_snapshots: int = 1000):
        self.logger = logger or get_logger(__name__)
        self.max_snapshots = max_snapshots
        self._snapshots = []
        self._lock = threading.Lock()
        self._convergence_history = defaultdict(list)
    
    def record_iteration(self, snapshot: FittingProgressSnapshot):
        """Record a fitting iteration."""
        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self.max_snapshots:
                self._snapshots = self._snapshots[-self.max_snapshots//2:]
            
            # Track convergence metrics
            self._convergence_history['loss'].append(snapshot.loss_value)
            if snapshot.gradient_norm:
                self._convergence_history['gradient_norm'].append(snapshot.gradient_norm)
    
    def check_convergence(self, window_size: int = 10, 
                         tolerance: float = 1e-6) -> Tuple[bool, str]:
        """Check if fitting has converged."""
        with self._lock:
            if len(self._snapshots) < window_size:
                return False, f"Need at least {window_size} iterations"
            
            recent_losses = [s.loss_value for s in self._snapshots[-window_size:]]
            
            # Check loss stability
            loss_std = np.std(recent_losses) if HAS_NUMPY else 0
            loss_mean = np.mean(recent_losses) if HAS_NUMPY else recent_losses[-1]
            
            if loss_std / max(abs(loss_mean), 1e-15) < tolerance:
                return True, f"Loss converged (std/mean = {loss_std/abs(loss_mean):.2e})"
            
            # Check gradient norm if available
            if self._convergence_history['gradient_norm']:
                recent_grads = self._convergence_history['gradient_norm'][-window_size:]
                if recent_grads[-1] < tolerance:
                    return True, f"Gradient norm below threshold ({recent_grads[-1]:.2e})"
            
            return False, f"Not converged (loss std/mean = {loss_std/abs(loss_mean):.2e})"
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get fitting progress summary."""
        with self._lock:
            if not self._snapshots:
                return {'status': 'no_data'}
            
            latest = self._snapshots[-1]
            initial = self._snapshots[0]
            
            summary = {
                'total_iterations': len(self._snapshots),
                'current_loss': latest.loss_value,
                'initial_loss': initial.loss_value,
                'improvement': (initial.loss_value - latest.loss_value) / initial.loss_value,
                'latest_parameters': latest.parameter_values,
                'optimization_method': latest.optimization_method
            }
            
            # Convergence check
            is_converged, conv_msg = self.check_convergence()
            summary['converged'] = is_converged
            summary['convergence_message'] = conv_msg
            
            return summary


# Global instances
_xpcs_validator = XPCSDataValidator()
_fitting_tracker = FittingProgressTracker()


@contextmanager
def xpcs_data_loading_context(filepath: Union[str, Path],
                             expected_format: str = "auto",
                             logger: Optional = None):
    """Context manager for XPCS data loading with validation."""
    if logger is None:
        logger = get_logger()
    
    filepath = Path(filepath)
    start_time = time.perf_counter()
    
    # Initialize data info
    data_info = XPCSDataInfo(
        filepath=str(filepath),
        file_format=expected_format
    )
    
    # Log start
    logger.info(f"Loading XPCS data from {filepath.name}")
    
    # Get file size
    if filepath.exists():
        data_info.file_size_mb = filepath.stat().st_size / 1024 / 1024
        logger.debug(f"File size: {data_info.file_size_mb:.1f} MB")
    else:
        logger.error(f"Data file not found: {filepath}")
        raise FileNotFoundError(f"XPCS data file not found: {filepath}")
    
    try:
        yield data_info
        
        # Success logging
        data_info.loading_time = time.perf_counter() - start_time
        
        success_msg = f"XPCS data loaded successfully in {data_info.loading_time:.2f}s"
        if data_info.data_shape:
            success_msg += f" (shape: {data_info.data_shape})"
        if data_info.q_vectors:
            success_msg += f", {data_info.q_vectors} q-vectors"
        
        logger.info(success_msg)
        
        # Log any validation warnings
        if data_info.validation_warnings:
            for warning in data_info.validation_warnings:
                logger.warning(f"Data validation: {warning}")
        
        # Log preprocessing steps
        if data_info.preprocessing_applied:
            logger.info(f"Preprocessing applied: {', '.join(data_info.preprocessing_applied)}")
    
    except Exception as e:
        loading_time = time.perf_counter() - start_time
        logger.error(f"XPCS data loading failed after {loading_time:.2f}s: {e}")
        raise


@contextmanager
def correlation_computation_context(method: str,
                                  input_shape: Tuple[int, ...],
                                  logger: Optional = None,
                                  track_memory: bool = True):
    """Context manager for correlation function computation."""
    if logger is None:
        logger = get_logger()
    
    # Initialize metrics
    metrics = CorrelationComputationMetrics(
        computation_method=method,
        input_data_shape=input_shape
    )
    
    start_time = time.perf_counter()
    logger.info(f"Computing correlation functions using {method} method")
    logger.debug(f"Input data shape: {input_shape}")
    
    # Use JAX context if available for GPU operations
    if HAS_JAX and method in ['jax_vectorized', 'jax_chunked']:
        with jax_operation_context(f"correlation_{method}", logger, track_memory):
            try:
                yield metrics
                
                metrics.computation_time = time.perf_counter() - start_time
                success_msg = f"Correlation computation completed in {metrics.computation_time:.2f}s"
                if metrics.output_shape:
                    success_msg += f" (output: {metrics.output_shape})"
                if metrics.cache_hit:
                    success_msg += " [CACHE HIT]"
                
                logger.info(success_msg)
                
            except Exception as e:
                metrics.computation_time = time.perf_counter() - start_time
                logger.error(f"Correlation computation failed after {metrics.computation_time:.2f}s: {e}")
                raise
    else:
        # Standard CPU computation
        try:
            yield metrics
            
            metrics.computation_time = time.perf_counter() - start_time
            logger.info(f"Correlation computation completed in {metrics.computation_time:.2f}s")
            
        except Exception as e:
            metrics.computation_time = time.perf_counter() - start_time
            logger.error(f"Correlation computation failed after {metrics.computation_time:.2f}s: {e}")
            raise


def log_physics_validation(logger: Optional = None):
    """Decorator to log physics parameter validation."""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            logger.debug(f"Physics validation in {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                # Try to extract and validate parameters if result is a dict
                if isinstance(result, dict):
                    validation_results = []
                    for param_name, value in result.items():
                        if isinstance(value, (int, float)):
                            validation = _xpcs_validator.validate_physics_parameter(param_name, float(value))
                            validation_results.append(validation)
                            
                            if not validation.is_valid:
                                logger.warning(f"Parameter validation failed: {validation.warning_message}")
                                if validation.suggestion:
                                    logger.info(f"Suggestion: {validation.suggestion}")
                    
                    # Log summary
                    valid_count = sum(1 for v in validation_results if v.is_valid)
                    total_count = len(validation_results)
                    if total_count > 0:
                        logger.info(f"Physics validation: {valid_count}/{total_count} parameters valid")
                
                return result
                
            except Exception as e:
                logger.error(f"Physics validation error in {func_name}: {e}")
                raise
        
        return wrapper
    return decorator


@contextmanager  
def model_fitting_context(model_name: str,
                         optimization_method: str,
                         initial_parameters: Dict[str, float],
                         logger: Optional = None):
    """Context manager for model fitting with progress tracking."""
    if logger is None:
        logger = get_logger()
    
    start_time = time.perf_counter()
    logger.info(f"Starting {model_name} fitting with {optimization_method}")
    logger.debug(f"Initial parameters: {initial_parameters}")
    
    # Validate initial parameters
    validation_results = []
    for param_name, value in initial_parameters.items():
        validation = _xpcs_validator.validate_physics_parameter(param_name, value)
        validation_results.append(validation)
        if not validation.is_valid and validation.warning_message:
            logger.warning(f"Initial parameter issue: {validation.warning_message}")
    
    try:
        yield _fitting_tracker
        
        # Success logging
        fitting_time = time.perf_counter() - start_time
        progress = _fitting_tracker.get_progress_summary()
        
        success_msg = f"Model fitting completed in {fitting_time:.2f}s"
        if 'total_iterations' in progress:
            success_msg += f" ({progress['total_iterations']} iterations)"
        if progress.get('converged'):
            success_msg += " [CONVERGED]"
        
        logger.info(success_msg)
        
        if 'improvement' in progress:
            improvement = progress['improvement'] * 100
            logger.info(f"Loss improvement: {improvement:.2f}%")
        
    except Exception as e:
        fitting_time = time.perf_counter() - start_time
        logger.error(f"Model fitting failed after {fitting_time:.2f}s: {e}")
        raise


def log_numerical_stability_check(array_name: str = "data"):
    """Decorator to check and log numerical stability."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            result = func(*args, **kwargs)
            
            # Check result for numerical issues
            if hasattr(result, '__iter__') and not isinstance(result, str):
                for i, item in enumerate(result):
                    if hasattr(item, 'shape'):
                        # Check this array
                        log_array_stats(item, f"{array_name}[{i}]", logger)
                        
                        # Stability checks
                        if HAS_NUMPY:
                            if not np.isfinite(item).all():
                                logger.error(f"Non-finite values in {array_name}[{i}]")
                            
                            condition_number = np.linalg.cond(item) if item.ndim == 2 else None
                            if condition_number and condition_number > _xpcs_validator.stability_thresholds['max_condition_number']:
                                logger.warning(f"High condition number in {array_name}[{i}]: {condition_number:.2e}")
            
            elif hasattr(result, 'shape'):
                # Single array result
                log_array_stats(result, array_name, logger)
            
            return result
        return wrapper
    return decorator


def get_scientific_computing_stats() -> Dict[str, Any]:
    """Get comprehensive scientific computing statistics."""
    return {
        'fitting_progress': _fitting_tracker.get_progress_summary(),
        'validation_thresholds': _xpcs_validator.parameter_bounds,
        'stability_thresholds': _xpcs_validator.stability_thresholds,
        'has_jax': HAS_JAX,
        'has_numpy': HAS_NUMPY,
        'has_hdf5': HAS_HDF5
    }


def clear_scientific_tracking_data():
    """Clear all scientific computing tracking data."""
    global _fitting_tracker
    _fitting_tracker = FittingProgressTracker()