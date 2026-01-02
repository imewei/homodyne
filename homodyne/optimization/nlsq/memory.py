"""Memory Management Utilities for NLSQ Optimization.

Provides adaptive memory threshold detection and management for determining
when NLSQ should switch to streaming mode for memory-bounded optimization.

Key Features:
- Cross-platform system memory detection (psutil + os.sysconf fallback)
- Adaptive threshold calculation based on available memory
- Environment variable override support (NLSQ_MEMORY_FRACTION)
- Safe fraction clamping to prevent OOM or underutilization

Usage:
    >>> from homodyne.optimization.nlsq.memory import get_adaptive_memory_threshold
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> print(f"Threshold: {threshold_gb:.1f} GB")
"""

import logging
import os
import warnings
from typing import Any

# Module-level logger
logger = logging.getLogger(__name__)

# Default memory fraction and environment variable name
DEFAULT_MEMORY_FRACTION = 0.75
MEMORY_FRACTION_ENV_VAR = "NLSQ_MEMORY_FRACTION"
FALLBACK_THRESHOLD_GB = 16.0
MIN_MEMORY_FRACTION = 0.1
MAX_MEMORY_FRACTION = 0.9


def detect_total_system_memory() -> float | None:
    """Detect total system memory in bytes using multiple methods.

    Returns
    -------
    float | None
        Total system memory in bytes, or None if detection fails.

    Notes
    -----
    Detection priority:
    1. psutil.virtual_memory().total (preferred, cross-platform)
    2. os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') (Linux fallback)
    """
    # Method 1: psutil (preferred, cross-platform)
    try:
        import psutil

        total_bytes = psutil.virtual_memory().total
        if total_bytes > 0:
            return float(total_bytes)
    except ImportError:
        logger.debug("psutil not available, trying os.sysconf fallback")
    except Exception as e:
        logger.debug(f"psutil memory detection failed: {e}")

    # Method 2: os.sysconf (Linux/Unix fallback)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size > 0 and phys_pages > 0:
            total_bytes = page_size * phys_pages
            return float(total_bytes)
    except (ValueError, OSError, AttributeError) as e:
        logger.debug(f"os.sysconf memory detection failed: {e}")

    return None


def get_adaptive_memory_threshold(
    memory_fraction: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute adaptive memory threshold based on system memory.

    The memory threshold determines when NLSQ switches to streaming mode
    for memory-bounded optimization. Instead of a fixed 16 GB threshold,
    this function computes an adaptive threshold as a fraction of total
    system memory.

    Parameters
    ----------
    memory_fraction : float | None, optional
        Fraction of total system memory to use as threshold (0.1 to 0.9).
        If None, uses:
        1. Environment variable NLSQ_MEMORY_FRACTION (if set)
        2. Default value of 0.75 (75% of total memory)

    Returns
    -------
    threshold_gb : float
        Memory threshold in gigabytes.
    info : dict
        Diagnostic information with keys:
        - 'total_memory_gb': Detected total system memory (GB)
        - 'memory_fraction': Fraction used
        - 'source': How the fraction was determined ('argument', 'env', 'default')
        - 'detection_method': How memory was detected ('psutil', 'sysconf', 'fallback')

    Notes
    -----
    - If total memory cannot be detected, falls back to 16.0 GB with a warning.
    - Memory fraction is clamped to [0.1, 0.9] for safety.
    - Environment variable NLSQ_MEMORY_FRACTION can override the default.

    Examples
    --------
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> pct = info['memory_fraction'] * 100
    >>> tot = info['total_memory_gb']
    >>> print(f"Threshold: {threshold_gb:.1f} GB ({pct:.0f}% of {tot:.1f} GB)")
    Threshold: 24.0 GB (75% of 32.0 GB)

    >>> # Override with specific fraction
    >>> threshold_gb, _ = get_adaptive_memory_threshold(memory_fraction=0.5)

    >>> # Override via environment variable
    >>> import os
    >>> os.environ["NLSQ_MEMORY_FRACTION"] = "0.6"
    >>> threshold_gb, info = get_adaptive_memory_threshold()
    >>> assert info['source'] == 'env'
    """
    info: dict[str, Any] = {}

    # Step 1: Determine memory fraction
    fraction_source = "default"
    effective_fraction = DEFAULT_MEMORY_FRACTION

    if memory_fraction is not None:
        # Use explicit argument
        effective_fraction = memory_fraction
        fraction_source = "argument"
    else:
        # Check environment variable
        env_value = os.environ.get(MEMORY_FRACTION_ENV_VAR)
        if env_value is not None:
            try:
                effective_fraction = float(env_value)
                fraction_source = "env"
            except ValueError:
                warnings.warn(
                    f"Invalid {MEMORY_FRACTION_ENV_VAR}='{env_value}', "
                    f"using default {DEFAULT_MEMORY_FRACTION}",
                    UserWarning,
                    stacklevel=2,
                )

    # Step 2: Clamp fraction to safe range
    original_fraction = effective_fraction
    effective_fraction = max(
        MIN_MEMORY_FRACTION, min(effective_fraction, MAX_MEMORY_FRACTION)
    )

    if effective_fraction != original_fraction:
        warnings.warn(
            f"Memory fraction {original_fraction} clamped to "
            f"[{MIN_MEMORY_FRACTION}, {MAX_MEMORY_FRACTION}]: "
            f"using {effective_fraction}",
            UserWarning,
            stacklevel=2,
        )

    info["memory_fraction"] = effective_fraction
    info["source"] = fraction_source

    # Step 3: Detect total system memory
    total_bytes = detect_total_system_memory()

    if total_bytes is not None:
        total_gb = total_bytes / (1024**3)
        threshold_gb = total_gb * effective_fraction

        # Determine detection method for logging
        try:
            import psutil  # noqa: F401

            info["detection_method"] = "psutil"
        except ImportError:
            info["detection_method"] = "sysconf"

        info["total_memory_gb"] = total_gb

        logger.info(
            f"Adaptive memory threshold: {threshold_gb:.1f} GB "
            f"({effective_fraction * 100:.0f}% of {total_gb:.1f} GB total, "
            f"source={fraction_source}, method={info['detection_method']})"
        )

        return threshold_gb, info

    # Step 4: Fallback if memory detection fails
    warnings.warn(
        f"Could not detect system memory. "
        f"Using fallback threshold of {FALLBACK_THRESHOLD_GB} GB. "
        "Install psutil for accurate memory detection: pip install psutil",
        UserWarning,
        stacklevel=2,
    )

    info["total_memory_gb"] = 0.0
    info["detection_method"] = "fallback"

    logger.warning(
        f"Memory detection failed. "
        f"Using fallback threshold: {FALLBACK_THRESHOLD_GB} GB"
    )

    return FALLBACK_THRESHOLD_GB, info


def estimate_peak_memory_gb(
    n_points: int,
    n_params: int,
    bytes_per_element: int = 8,
    jacobian_overhead: float = 3.0,
) -> float:
    """Estimate peak memory usage for full Jacobian optimization.

    Parameters
    ----------
    n_points : int
        Number of data points
    n_params : int
        Number of parameters
    bytes_per_element : int, optional
        Bytes per float element (default: 8 for float64)
    jacobian_overhead : float, optional
        Multiplicative factor for autodiff intermediates (default: 3.0)

    Returns
    -------
    float
        Estimated peak memory in gigabytes
    """
    # Jacobian matrix: n_points × n_params × bytes
    jacobian_bytes = n_points * n_params * bytes_per_element

    # Total with autodiff overhead
    peak_bytes = jacobian_bytes * jacobian_overhead

    return peak_bytes / (1024**3)


def should_use_streaming(
    n_points: int,
    n_params: int,
    threshold_gb: float | None = None,
    memory_fraction: float | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Determine if streaming mode should be used based on memory requirements.

    Parameters
    ----------
    n_points : int
        Number of data points
    n_params : int
        Number of parameters
    threshold_gb : float | None, optional
        Explicit memory threshold in GB. If None, auto-detect.
    memory_fraction : float | None, optional
        Fraction of system memory for threshold (if threshold_gb not set)

    Returns
    -------
    use_streaming : bool
        True if streaming mode should be used
    info : dict
        Diagnostic information including estimated memory and threshold
    """
    # Get threshold
    if threshold_gb is None:
        threshold_gb, threshold_info = get_adaptive_memory_threshold(memory_fraction)
    else:
        threshold_info = {"source": "explicit", "memory_fraction": None}

    # Estimate memory
    estimated_gb = estimate_peak_memory_gb(n_points, n_params)

    info = {
        "estimated_memory_gb": estimated_gb,
        "threshold_gb": threshold_gb,
        "n_points": n_points,
        "n_params": n_params,
        **threshold_info,
    }

    use_streaming = estimated_gb > threshold_gb

    if use_streaming:
        logger.info(
            f"Streaming mode recommended: estimated {estimated_gb:.1f} GB > "
            f"threshold {threshold_gb:.1f} GB"
        )
    else:
        logger.debug(
            f"Full Jacobian mode OK: estimated {estimated_gb:.1f} GB < "
            f"threshold {threshold_gb:.1f} GB"
        )

    return use_streaming, info


__all__ = [
    "DEFAULT_MEMORY_FRACTION",
    "MEMORY_FRACTION_ENV_VAR",
    "FALLBACK_THRESHOLD_GB",
    "detect_total_system_memory",
    "get_adaptive_memory_threshold",
    "estimate_peak_memory_gb",
    "should_use_streaming",
]
