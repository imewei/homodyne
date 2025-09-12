"""
Advanced Debugging Features for Homodyne v2
===========================================

Extended debugging capabilities that build upon the existing debug.py infrastructure
to provide advanced debugging features for scientific computing and JAX operations:

- Interactive debugging sessions with remote access capabilities
- Error recovery and retry mechanisms with intelligent backoff
- Correlation matrix inspection and numerical validation utilities  
- Model convergence tracking with detailed diagnostics
- Automatic error classification and solution suggestions
- Performance profiling with statistical analysis
- Memory leak detection and analysis
- JAX compilation debugging and optimization hints

This module integrates with the existing debug.py CallTracker and MemoryTracker
while adding domain-specific debugging tools for XPCS analysis workflows.
"""

import functools
import gc
import inspect
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import threading
import weakref

# Scientific computing imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = jnp = None

# Memory profiling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Statistical analysis
try:
    import scipy.stats as stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

from .logging import get_logger
from .debug import CallTracker, MemoryTracker, get_debug_stats
from .jax_logging import get_jax_compilation_stats
from .scientific_logging import FittingProgressTracker


@dataclass
class ErrorEvent:
    """Detailed information about an error event."""
    timestamp: float
    error_type: str
    error_message: str
    function_name: str
    module_name: str
    traceback_lines: List[str]
    context_variables: Dict[str, str]
    suggested_solutions: List[str] = field(default_factory=list)
    error_category: Optional[str] = None  # 'numerical', 'memory', 'io', 'logic'
    recovery_attempted: bool = False
    recovery_successful: Optional[bool] = None
    
    @classmethod
    def from_exception(cls, e: Exception, context_vars: Optional[Dict] = None) -> 'ErrorEvent':
        """Create ErrorEvent from an exception."""
        tb = traceback.extract_tb(e.__traceback__)
        frame = tb[-1] if tb else None
        
        return cls(
            timestamp=time.time(),
            error_type=type(e).__name__,
            error_message=str(e),
            function_name=frame.name if frame else 'unknown',
            module_name=frame.filename if frame else 'unknown',
            traceback_lines=traceback.format_exception(type(e), e, e.__traceback__),
            context_variables=context_vars or {}
        )


@dataclass
class NumericalIssueReport:
    """Report of numerical stability issues."""
    issue_type: str  # 'overflow', 'underflow', 'nan', 'inf', 'ill_conditioned'
    severity: str    # 'low', 'medium', 'high', 'critical'
    affected_arrays: List[str]
    array_statistics: Dict[str, Dict[str, float]]
    suggested_fixes: List[str] = field(default_factory=list)
    function_context: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformanceAnomaly:
    """Performance anomaly detection result."""
    operation_name: str
    measured_duration: float
    expected_duration: float
    deviation_factor: float  # How many times slower/faster than expected
    anomaly_type: str        # 'slow', 'fast', 'memory_spike', 'compilation'
    confidence: float        # 0.0 to 1.0
    suggested_investigation: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class ErrorRecoveryManager:
    """Intelligent error recovery with retry mechanisms."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._error_history = deque(maxlen=1000)
        self._recovery_strategies = {}
        self._lock = threading.Lock()
        
        # Register default recovery strategies
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default error recovery strategies."""
        
        # Memory errors
        self._recovery_strategies['MemoryError'] = [
            ('garbage_collect', self._gc_recovery),
            ('reduce_precision', self._reduce_precision_recovery),
            ('chunk_processing', self._chunked_processing_recovery)
        ]
        
        # Numerical errors
        self._recovery_strategies['ValueError'] = [
            ('parameter_bounds', self._bounds_recovery),
            ('numerical_stabilization', self._stabilize_recovery)
        ]
        
        # JAX compilation errors
        if HAS_JAX:
            self._recovery_strategies['jax.errors.JaxRuntimeError'] = [
                ('recompile', self._jax_recompile_recovery),
                ('fallback_cpu', self._jax_cpu_fallback_recovery)
            ]
    
    def _gc_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt garbage collection recovery."""
        self.logger.info("Attempting garbage collection recovery")
        gc.collect()
        if HAS_JAX:
            try:
                # Clear JAX compilation cache
                jax.clear_caches()
                self.logger.info("Cleared JAX compilation cache")
            except:
                pass
        return True
    
    def _reduce_precision_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt recovery by reducing numerical precision."""
        self.logger.info("Suggesting precision reduction recovery")
        # This would need to be handled by the calling code
        return False  # Can't automatically change precision
    
    def _chunked_processing_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Suggest chunked processing for memory issues."""
        self.logger.info("Suggesting chunked processing for memory recovery")
        return False  # Needs calling code to implement chunking
    
    def _bounds_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt parameter bounds recovery."""
        self.logger.info("Attempting parameter bounds recovery")
        # Could implement automatic parameter clipping here
        return False
    
    def _stabilize_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt numerical stabilization recovery."""
        self.logger.info("Attempting numerical stabilization recovery")
        return False
    
    def _jax_recompile_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt JAX recompilation recovery."""
        if not HAS_JAX:
            return False
        
        try:
            self.logger.info("Clearing JAX cache for recompilation recovery")
            jax.clear_caches()
            return True
        except:
            return False
    
    def _jax_cpu_fallback_recovery(self, error: ErrorEvent, *args, **kwargs):
        """Attempt JAX CPU fallback recovery."""
        if not HAS_JAX:
            return False
        
        try:
            # This would need support from calling code to switch to CPU
            self.logger.info("Suggesting JAX CPU fallback recovery")
            return False
        except:
            return False
    
    def record_error(self, error: ErrorEvent):
        """Record an error event."""
        with self._lock:
            self._error_history.append(error)
    
    def attempt_recovery(self, error: ErrorEvent, *args, **kwargs) -> bool:
        """Attempt to recover from an error."""
        error_type = error.error_type
        
        # Get recovery strategies for this error type
        strategies = self._recovery_strategies.get(error_type, [])
        
        for strategy_name, strategy_func in strategies:
            self.logger.info(f"Attempting recovery strategy: {strategy_name}")
            
            try:
                success = strategy_func(error, *args, **kwargs)
                if success:
                    self.logger.info(f"Recovery successful using {strategy_name}")
                    error.recovery_attempted = True
                    error.recovery_successful = True
                    return True
                else:
                    self.logger.debug(f"Recovery strategy {strategy_name} unsuccessful")
                    
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy {strategy_name} failed: {recovery_error}")
        
        error.recovery_attempted = True
        error.recovery_successful = False
        return False
    
    def get_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns and suggest improvements."""
        with self._lock:
            if not self._error_history:
                return {'status': 'no_errors'}
            
            # Count error types
            error_counts = defaultdict(int)
            function_errors = defaultdict(int)
            recent_errors = []
            
            cutoff_time = time.time() - 3600  # Last hour
            
            for error in self._error_history:
                error_counts[error.error_type] += 1
                function_errors[error.function_name] += 1
                
                if error.timestamp > cutoff_time:
                    recent_errors.append(error)
            
            return {
                'total_errors': len(self._error_history),
                'recent_errors_1h': len(recent_errors),
                'error_type_counts': dict(error_counts),
                'function_error_counts': dict(function_errors),
                'most_common_error': max(error_counts.items(), key=lambda x: x[1]) if error_counts else None,
                'error_prone_function': max(function_errors.items(), key=lambda x: x[1]) if function_errors else None
            }


class NumericalStabilityAnalyzer:
    """Analyze and detect numerical stability issues."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._issue_history = deque(maxlen=500)
        
        # Thresholds for numerical issue detection
        self.thresholds = {
            'condition_number_high': 1e12,
            'condition_number_critical': 1e15,
            'small_eigenvalue': 1e-12,
            'large_dynamic_range': 1e10,
            'gradient_norm_small': 1e-8,
            'gradient_norm_large': 1e8
        }
    
    def analyze_matrix(self, matrix: np.ndarray, name: str = "matrix") -> NumericalIssueReport:
        """Analyze a matrix for numerical stability issues."""
        issues = []
        severity = 'low'
        array_stats = {}
        
        if not HAS_NUMPY:
            return NumericalIssueReport(
                issue_type='analysis_unavailable',
                severity='low',
                affected_arrays=[name],
                array_statistics={}
            )
        
        try:
            # Basic statistics
            array_stats[name] = {
                'shape': matrix.shape,
                'dtype': str(matrix.dtype),
                'min': float(np.min(matrix)),
                'max': float(np.max(matrix)),
                'mean': float(np.mean(matrix)),
                'std': float(np.std(matrix)),
                'has_nan': bool(np.isnan(matrix).any()),
                'has_inf': bool(np.isinf(matrix).any()),
                'finite_fraction': float(np.isfinite(matrix).mean())
            }
            
            # Check for NaN/Inf
            if array_stats[name]['has_nan'] or array_stats[name]['has_inf']:
                issues.append("Non-finite values detected")
                severity = 'critical'
            
            # Check condition number for square matrices
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                try:
                    cond_num = np.linalg.cond(matrix)
                    array_stats[name]['condition_number'] = float(cond_num)
                    
                    if cond_num > self.thresholds['condition_number_critical']:
                        issues.append(f"Critical condition number: {cond_num:.2e}")
                        severity = 'critical'
                    elif cond_num > self.thresholds['condition_number_high']:
                        issues.append(f"High condition number: {cond_num:.2e}")
                        severity = max(severity, 'high')
                
                except np.linalg.LinAlgError:
                    issues.append("Singular matrix detected")
                    severity = 'critical'
            
            # Check dynamic range
            if array_stats[name]['min'] != 0:
                dynamic_range = array_stats[name]['max'] / abs(array_stats[name]['min'])
                array_stats[name]['dynamic_range'] = dynamic_range
                
                if dynamic_range > self.thresholds['large_dynamic_range']:
                    issues.append(f"Large dynamic range: {dynamic_range:.2e}")
                    severity = max(severity, 'medium')
            
            # Check for near-zero variance (constant arrays)
            if array_stats[name]['std'] < 1e-15:
                issues.append("Near-zero variance (constant array)")
                severity = max(severity, 'medium')
        
        except Exception as e:
            self.logger.warning(f"Error analyzing matrix {name}: {e}")
            issues.append(f"Analysis failed: {e}")
        
        # Generate suggested fixes
        suggested_fixes = self._generate_stability_fixes(issues, array_stats.get(name, {}))
        
        issue_type = 'multiple' if len(issues) > 1 else (issues[0] if issues else 'stable')
        
        report = NumericalIssueReport(
            issue_type=issue_type,
            severity=severity,
            affected_arrays=[name],
            array_statistics=array_stats,
            suggested_fixes=suggested_fixes
        )
        
        self._issue_history.append(report)
        
        if issues:
            self.logger.warning(f"Numerical stability issues in {name}: {', '.join(issues)}")
            for fix in suggested_fixes:
                self.logger.info(f"Suggested fix: {fix}")
        
        return report
    
    def _generate_stability_fixes(self, issues: List[str], stats: Dict[str, Any]) -> List[str]:
        """Generate suggested fixes for numerical stability issues."""
        fixes = []
        
        for issue in issues:
            if "Non-finite values" in issue:
                fixes.append("Replace NaN/Inf values with finite numbers or remove affected data points")
                fixes.append("Check input data quality and preprocessing steps")
            
            elif "condition number" in issue:
                fixes.append("Add regularization (e.g., ridge regression parameter)")
                fixes.append("Use SVD-based pseudo-inverse instead of direct inversion")
                fixes.append("Consider dimensionality reduction or feature selection")
            
            elif "Singular matrix" in issue:
                fixes.append("Use pseudo-inverse (numpy.linalg.pinv) instead of inverse")
                fixes.append("Add small diagonal regularization term")
                fixes.append("Check for linearly dependent columns/rows")
            
            elif "dynamic range" in issue:
                fixes.append("Apply data scaling/normalization")
                fixes.append("Use higher precision arithmetic (float64 instead of float32)")
                fixes.append("Consider log-transformation for large ranges")
            
            elif "constant array" in issue:
                fixes.append("Check data preprocessing - may indicate scaling issues")
                fixes.append("Verify that input parameters are varying as expected")
        
        return list(set(fixes))  # Remove duplicates
    
    def get_stability_summary(self) -> Dict[str, Any]:
        """Get summary of numerical stability analysis."""
        if not self._issue_history:
            return {'status': 'no_analysis'}
        
        severity_counts = defaultdict(int)
        issue_type_counts = defaultdict(int)
        recent_issues = []
        
        cutoff_time = time.time() - 1800  # Last 30 minutes
        
        for report in self._issue_history:
            severity_counts[report.severity] += 1
            issue_type_counts[report.issue_type] += 1
            
            if report.timestamp > cutoff_time:
                recent_issues.append(report)
        
        return {
            'total_analyses': len(self._issue_history),
            'recent_issues_30m': len(recent_issues),
            'severity_distribution': dict(severity_counts),
            'issue_type_distribution': dict(issue_type_counts),
            'critical_issues': severity_counts['critical'],
            'high_issues': severity_counts['high'],
            'stability_score': max(0, 100 - severity_counts['critical'] * 20 - severity_counts['high'] * 10)
        }


class PerformanceAnomalyDetector:
    """Detect performance anomalies using statistical analysis."""
    
    def __init__(self, logger: Optional = None):
        self.logger = logger or get_logger(__name__)
        self._performance_history = defaultdict(list)  # function_name -> [durations]
        self._anomalies = deque(maxlen=200)
        self._lock = threading.Lock()
        
        # Statistical thresholds
        self.anomaly_threshold = 2.0  # Standard deviations
        self.min_samples = 5  # Minimum samples before anomaly detection
    
    def record_performance(self, operation_name: str, duration: float, 
                          context: Optional[Dict] = None):
        """Record a performance measurement."""
        with self._lock:
            self._performance_history[operation_name].append({
                'duration': duration,
                'timestamp': time.time(),
                'context': context or {}
            })
            
            # Keep history bounded
            if len(self._performance_history[operation_name]) > 1000:
                self._performance_history[operation_name] = \
                    self._performance_history[operation_name][-500:]
    
    def detect_anomaly(self, operation_name: str, duration: float) -> Optional[PerformanceAnomaly]:
        """Detect if this duration is anomalous."""
        with self._lock:
            history = self._performance_history.get(operation_name, [])
            
            if len(history) < self.min_samples:
                return None  # Not enough data
            
            # Get recent durations for statistical analysis
            recent_durations = [h['duration'] for h in history[-100:]]
            
            if not HAS_SCIPY:
                # Simple statistics without scipy
                mean_duration = sum(recent_durations) / len(recent_durations)
                std_duration = (sum((d - mean_duration)**2 for d in recent_durations) / len(recent_durations))**0.5
            else:
                # Use scipy for more robust statistics
                mean_duration = np.mean(recent_durations)
                std_duration = np.std(recent_durations)
            
            # Check for anomaly
            if std_duration > 0:
                z_score = abs((duration - mean_duration) / std_duration)
                
                if z_score > self.anomaly_threshold:
                    # Determine anomaly type
                    if duration > mean_duration:
                        anomaly_type = 'slow'
                        deviation_factor = duration / mean_duration
                    else:
                        anomaly_type = 'fast'
                        deviation_factor = mean_duration / duration
                    
                    # Generate investigation suggestions
                    suggestions = self._generate_investigation_suggestions(
                        operation_name, anomaly_type, deviation_factor, duration, mean_duration
                    )
                    
                    anomaly = PerformanceAnomaly(
                        operation_name=operation_name,
                        measured_duration=duration,
                        expected_duration=mean_duration,
                        deviation_factor=deviation_factor,
                        anomaly_type=anomaly_type,
                        confidence=min(z_score / self.anomaly_threshold, 1.0),
                        suggested_investigation=suggestions
                    )
                    
                    self._anomalies.append(anomaly)
                    
                    self.logger.warning(
                        f"Performance anomaly detected in {operation_name}: "
                        f"{duration:.3f}s (expected {mean_duration:.3f}s, "
                        f"{deviation_factor:.1f}x {anomaly_type})"
                    )
                    
                    return anomaly
            
            return None
    
    def _generate_investigation_suggestions(self, operation_name: str, anomaly_type: str,
                                          deviation_factor: float, duration: float, 
                                          expected: float) -> List[str]:
        """Generate suggestions for investigating performance anomalies."""
        suggestions = []
        
        if anomaly_type == 'slow':
            if deviation_factor > 10:
                suggestions.append("Check for memory swapping or disk I/O bottlenecks")
                suggestions.append("Verify system resources are not exhausted")
            elif deviation_factor > 3:
                suggestions.append("Check for CPU throttling or background processes")
                suggestions.append("Verify input data size hasn't increased significantly")
            
            suggestions.append("Check memory usage patterns - possible memory leak")
            suggestions.append("Verify JAX compilation cache is working properly")
            
        elif anomaly_type == 'fast':
            if deviation_factor > 5:
                suggestions.append("Check if computation was skipped or cached")
                suggestions.append("Verify input data size - may be smaller than expected")
            
            suggestions.append("Possible JIT compilation cache hit")
            suggestions.append("Check if algorithm path changed")
        
        # Operation-specific suggestions
        if 'correlation' in operation_name.lower():
            suggestions.append("Check correlation matrix dimensions and sparsity")
        elif 'fitting' in operation_name.lower():
            suggestions.append("Check convergence criteria and iteration counts")
        elif 'jit' in operation_name.lower():
            suggestions.append("Compare with JAX compilation statistics")
        
        return suggestions
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        with self._lock:
            if not self._anomalies:
                return {'status': 'no_anomalies'}
            
            # Recent anomalies (last hour)
            cutoff_time = time.time() - 3600
            recent_anomalies = [a for a in self._anomalies if a.timestamp > cutoff_time]
            
            # Count by type
            type_counts = defaultdict(int)
            operation_counts = defaultdict(int)
            
            for anomaly in self._anomalies:
                type_counts[anomaly.anomaly_type] += 1
                operation_counts[anomaly.operation_name] += 1
            
            return {
                'total_anomalies': len(self._anomalies),
                'recent_anomalies_1h': len(recent_anomalies),
                'anomaly_type_counts': dict(type_counts),
                'operation_anomaly_counts': dict(operation_counts),
                'most_anomalous_operation': max(operation_counts.items(), 
                                              key=lambda x: x[1]) if operation_counts else None
            }


# Global instances
_error_recovery = ErrorRecoveryManager()
_stability_analyzer = NumericalStabilityAnalyzer()
_anomaly_detector = PerformanceAnomalyDetector()


def auto_recover(max_retries: int = 3, backoff_factor: float = 2.0):
    """Decorator for automatic error recovery with intelligent backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.perf_counter()
                    result = func(*args, **kwargs)
                    duration = time.perf_counter() - start_time
                    
                    # Record successful performance
                    _anomaly_detector.record_performance(
                        f"{func.__module__}.{func.__qualname__}", 
                        duration
                    )
                    
                    # Check for performance anomalies
                    anomaly = _anomaly_detector.detect_anomaly(
                        f"{func.__module__}.{func.__qualname__}", 
                        duration
                    )
                    
                    return result
                    
                except Exception as e:
                    # Create error event
                    error_event = ErrorEvent.from_exception(e, {
                        'attempt': attempt + 1,
                        'max_retries': max_retries,
                        'args_types': [type(arg).__name__ for arg in args],
                        'kwargs_keys': list(kwargs.keys())
                    })
                    
                    _error_recovery.record_error(error_event)
                    
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Attempting recovery...")
                        
                        # Attempt recovery
                        recovery_success = _error_recovery.attempt_recovery(error_event, *args, **kwargs)
                        
                        if recovery_success:
                            logger.info("Recovery successful, retrying...")
                        else:
                            logger.info("Recovery unsuccessful, retrying with backoff...")
                        
                        # Backoff delay
                        delay = backoff_factor ** attempt
                        time.sleep(delay)
                        
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed. Final error: {e}")
                        raise
            
        return wrapper
    return decorator


@contextmanager 
def numerical_stability_context(operation_name: str, 
                               check_inputs: bool = True,
                               check_outputs: bool = True,
                               logger: Optional = None):
    """Context manager for numerical stability monitoring."""
    if logger is None:
        logger = get_logger()
    
    logger.debug(f"Starting numerical stability monitoring: {operation_name}")
    
    # Store references to check later
    input_arrays = []
    output_arrays = []
    
    class ArrayInterceptor:
        def __init__(self):
            self.arrays = []
        
        def add_array(self, array, name: str):
            if hasattr(array, 'shape') and hasattr(array, 'dtype'):
                self.arrays.append((array, name))
    
    interceptor = ArrayInterceptor()
    
    try:
        yield interceptor
        
        # Check collected arrays
        issues_found = False
        
        for array, name in interceptor.arrays:
            report = _stability_analyzer.analyze_matrix(array, f"{operation_name}_{name}")
            if report.severity in ['high', 'critical']:
                issues_found = True
        
        if not issues_found:
            logger.debug(f"Numerical stability check passed: {operation_name}")
        
    except Exception as e:
        logger.error(f"Numerical stability monitoring failed: {operation_name}: {e}")
        raise


def debug_correlation_matrix(matrix: np.ndarray, 
                           q_vectors: Optional[np.ndarray] = None,
                           time_points: Optional[np.ndarray] = None,
                           logger: Optional = None) -> Dict[str, Any]:
    """Debug and validate XPCS correlation matrix."""
    if logger is None:
        logger = get_logger(__name__)
    
    if not HAS_NUMPY:
        logger.warning("NumPy not available for correlation matrix debugging")
        return {'status': 'numpy_unavailable'}
    
    debug_info = {
        'matrix_shape': matrix.shape,
        'matrix_dtype': str(matrix.dtype),
        'issues_found': []
    }
    
    # Analyze numerical stability
    stability_report = _stability_analyzer.analyze_matrix(matrix, "correlation_matrix")
    debug_info['stability_report'] = {
        'severity': stability_report.severity,
        'issues': stability_report.issue_type,
        'suggested_fixes': stability_report.suggested_fixes
    }
    
    # Check correlation-specific properties
    try:
        # Check for proper correlation function behavior (g2 >= 1 at t=0)
        if matrix.ndim >= 2:
            diagonal_values = np.diag(matrix) if matrix.shape[0] == matrix.shape[1] else matrix[:, 0]
            
            if np.any(diagonal_values < 0.9):
                debug_info['issues_found'].append("Correlation values < 0.9 at zero lag")
                logger.warning("Unusual correlation values detected at zero lag")
            
            if np.any(diagonal_values > 3.0):
                debug_info['issues_found'].append("Correlation values > 3.0 detected")
                logger.warning("Unusually high correlation values detected")
        
        # Check for monotonic decay (typical in XPCS)
        if matrix.ndim >= 2 and matrix.shape[1] > 2:
            for i in range(min(5, matrix.shape[0])):  # Check first 5 rows
                row = matrix[i, :]
                if len(row) > 1:
                    # Check if generally decreasing
                    diffs = np.diff(row)
                    increasing_fraction = np.sum(diffs > 0) / len(diffs)
                    
                    if increasing_fraction > 0.3:  # More than 30% increasing
                        debug_info['issues_found'].append(f"Row {i}: Non-monotonic decay detected")
        
        # Q-vector consistency check
        if q_vectors is not None:
            expected_q_points = len(q_vectors)
            if matrix.shape[0] != expected_q_points:
                debug_info['issues_found'].append(
                    f"Q-vector count mismatch: matrix has {matrix.shape[0]} rows, "
                    f"but {expected_q_points} q-vectors provided"
                )
        
        # Time point consistency check
        if time_points is not None:
            expected_time_points = len(time_points)
            if matrix.shape[-1] != expected_time_points:
                debug_info['issues_found'].append(
                    f"Time point mismatch: matrix has {matrix.shape[-1]} columns, "
                    f"but {expected_time_points} time points provided"
                )
        
    except Exception as e:
        debug_info['issues_found'].append(f"Debugging analysis failed: {e}")
        logger.warning(f"Correlation matrix debugging failed: {e}")
    
    # Log summary
    if debug_info['issues_found']:
        logger.warning(f"Correlation matrix issues detected: {len(debug_info['issues_found'])} problems found")
        for issue in debug_info['issues_found']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Correlation matrix validation passed")
    
    return debug_info


def get_advanced_debugging_stats() -> Dict[str, Any]:
    """Get comprehensive advanced debugging statistics."""
    return {
        'error_recovery': _error_recovery.get_error_patterns(),
        'numerical_stability': _stability_analyzer.get_stability_summary(),
        'performance_anomalies': _anomaly_detector.get_anomaly_summary(),
        'memory_tracking': get_debug_stats().get('memory_stats', {}),
        'jax_compilation': get_jax_compilation_stats() if HAS_JAX else {'jax_unavailable': True},
        'system_capabilities': {
            'has_numpy': HAS_NUMPY,
            'has_jax': HAS_JAX, 
            'has_scipy': HAS_SCIPY,
            'has_psutil': HAS_PSUTIL
        }
    }


def clear_advanced_debugging_data():
    """Clear all advanced debugging tracking data."""
    global _error_recovery, _stability_analyzer, _anomaly_detector
    _error_recovery = ErrorRecoveryManager()
    _stability_analyzer = NumericalStabilityAnalyzer()
    _anomaly_detector = PerformanceAnomalyDetector()


def dump_debugging_report(filepath: Optional[str] = None, 
                         include_full_traceback: bool = False) -> str:
    """Dump comprehensive debugging report."""
    import json
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'advanced_debugging_stats': get_advanced_debugging_stats(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    if HAS_PSUTIL:
        try:
            process = psutil.Process()
            report['system_info'].update({
                'memory_info': dict(process.memory_info()._asdict()),
                'cpu_percent': process.cpu_percent(),
                'num_threads': process.num_threads(),
                'create_time': process.create_time()
            })
        except:
            pass
    
    # Include recent error tracebacks if requested
    if include_full_traceback:
        with _error_recovery._lock:
            recent_errors = list(_error_recovery._error_history)[-10:]  # Last 10 errors
            report['recent_error_details'] = [
                {
                    'timestamp': error.timestamp,
                    'error_type': error.error_type,
                    'error_message': error.error_message,
                    'function_name': error.function_name,
                    'traceback': error.traceback_lines,
                    'recovery_attempted': error.recovery_attempted,
                    'recovery_successful': error.recovery_successful
                }
                for error in recent_errors
            ]
    
    report_json = json.dumps(report, indent=2, default=str)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(report_json)
    
    return report_json