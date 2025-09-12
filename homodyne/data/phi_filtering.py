"""
Phi Angle Filtering Module

This module provides functionality for filtering phi angles based on target ranges,
implementing the logic from the v1 reference implementation with JAX-first optimizations
and configuration-driven parameters.

Key Features:
- Default target ranges: [-10°, 10°] and [170°, 190°]
- Configuration-driven custom ranges
- Vectorized numpy-based filtering for performance
- Fallback to all angles if no matches found
- Comprehensive logging and error handling

Based on homodyne_v1_reference/homodyne/analysis/core.py lines 3677-3724
"""

import logging
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

# JAX integration with fallback
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False

logger = logging.getLogger(__name__)


class PhiAngleFilter:
    """
    Phi angle filtering utility class for XPCS analysis.
    
    Provides vectorized filtering of phi angles based on configurable target ranges,
    with intelligent fallback behavior when no angles match the specified ranges.
    """
    
    DEFAULT_TARGET_RANGES = [
        (-10.0, 10.0),    # Near 0 degrees
        (170.0, 190.0),   # Near 180 degrees
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the phi angle filter.
        
        Args:
            config: Optional configuration dictionary containing angle filtering settings
        """
        self.config = config or {}
        self._parse_config()
        
    def _parse_config(self) -> None:
        """Parse configuration settings for angle filtering."""
        # Get angle filtering configuration
        angle_config = self.config.get("optimization_config", {}).get("angle_filtering", {})
        
        # Extract target ranges
        config_ranges = angle_config.get("target_ranges", [])
        if config_ranges:
            self.target_ranges = [
                (float(r.get("min_angle", -10.0)), float(r.get("max_angle", 10.0)))
                for r in config_ranges
            ]
        else:
            self.target_ranges = self.DEFAULT_TARGET_RANGES
            
        # Extract fallback behavior
        self.fallback_enabled = angle_config.get("fallback_to_all_angles", True)
        self.filtering_enabled = angle_config.get("enabled", True)
        
        logger.debug(f"Initialized PhiAngleFilter with target_ranges: {self.target_ranges}")
        logger.debug(f"Fallback enabled: {self.fallback_enabled}, Filtering enabled: {self.filtering_enabled}")
    
    def filter_angles_for_optimization(
        self, 
        phi_angles: Union[List[float], np.ndarray], 
        target_ranges: Optional[List[Tuple[float, float]]] = None,
        fallback_enabled: Optional[bool] = None
    ) -> Tuple[List[int], np.ndarray]:
        """
        Filter phi angles based on target ranges for optimization.
        
        This method implements the core logic from the v1 reference implementation,
        using vectorized operations for performance.
        
        Args:
            phi_angles: Array or list of phi angles in degrees
            target_ranges: Optional list of (min_angle, max_angle) tuples.
                          If None, uses configured or default ranges.
            fallback_enabled: Optional override for fallback behavior.
                             If None, uses configured setting.
        
        Returns:
            Tuple of (optimization_indices, filtered_angles) where:
            - optimization_indices: List of indices of angles in target ranges
            - filtered_angles: Array of angles corresponding to optimization_indices
        
        Raises:
            ValueError: If no valid angles are found and fallback is disabled
        """
        if not self.filtering_enabled:
            logger.debug("Angle filtering disabled, returning all angles")
            indices = list(range(len(phi_angles)))
            return indices, np.asarray(phi_angles)
            
        # Use provided parameters or fall back to configured/default values
        ranges = target_ranges if target_ranges is not None else self.target_ranges
        fallback = fallback_enabled if fallback_enabled is not None else self.fallback_enabled
        
        # Convert to numpy array for vectorized operations
        phi_angles_array = np.asarray(phi_angles)
        
        logger.debug(f"Filtering {len(phi_angles_array)} angles with target ranges: {ranges}")
        
        # Vectorized range checking for all ranges at once
        optimization_mask = np.zeros(len(phi_angles_array), dtype=bool)
        
        for min_angle, max_angle in ranges:
            # Ensure valid range
            if min_angle > max_angle:
                logger.warning(f"Invalid range [{min_angle}, {max_angle}], skipping")
                continue
                
            # Apply range filter with vectorized operations
            range_mask = (phi_angles_array >= min_angle) & (phi_angles_array <= max_angle)
            optimization_mask |= range_mask
            
            angles_in_range = np.sum(range_mask)
            if angles_in_range > 0:
                logger.debug(f"Found {angles_in_range} angles in range [{min_angle}, {max_angle}]")
        
        # Get indices of matching angles
        optimization_indices = np.flatnonzero(optimization_mask).tolist()
        
        logger.debug(f"Filtering angles for optimization: using {len(optimization_indices)}/{len(phi_angles)} angles")
        
        if optimization_indices:
            filtered_angles = phi_angles_array[optimization_indices]
            logger.debug(f"Optimization angles: {filtered_angles.tolist()}")
            return optimization_indices, filtered_angles
        else:
            # Handle case where no angles match target ranges
            logger.warning(f"No angles found in target optimization ranges {ranges}")
            
            if fallback:
                logger.warning("Falling back to using all angles for optimization")
                optimization_indices = list(range(len(phi_angles)))
                return optimization_indices, phi_angles_array
            else:
                error_msg = f"No angles found in target ranges {ranges} and fallback is disabled"
                logger.error(error_msg)
                raise ValueError(error_msg)
    
    def validate_target_ranges(self, target_ranges: List[Tuple[float, float]]) -> bool:
        """
        Validate target angle ranges.
        
        Args:
            target_ranges: List of (min_angle, max_angle) tuples
            
        Returns:
            True if all ranges are valid, False otherwise
        """
        valid = True
        for i, (min_angle, max_angle) in enumerate(target_ranges):
            if min_angle > max_angle:
                logger.error(f"Range {i}: min_angle ({min_angle}) > max_angle ({max_angle})")
                valid = False
            if not (-360 <= min_angle <= 360 and -360 <= max_angle <= 360):
                logger.warning(f"Range {i}: angles outside typical range [-360, 360]: [{min_angle}, {max_angle}]")
        return valid
    
    def get_angle_statistics(self, phi_angles: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """
        Get statistics about angle distribution relative to target ranges.
        
        Args:
            phi_angles: Array or list of phi angles in degrees
            
        Returns:
            Dictionary containing statistics
        """
        phi_angles_array = np.asarray(phi_angles)
        stats = {
            "total_angles": len(phi_angles_array),
            "angle_range": {
                "min": float(np.min(phi_angles_array)),
                "max": float(np.max(phi_angles_array)),
                "mean": float(np.mean(phi_angles_array)),
                "std": float(np.std(phi_angles_array))
            },
            "target_ranges": self.target_ranges,
            "angles_per_range": []
        }
        
        for min_angle, max_angle in self.target_ranges:
            mask = (phi_angles_array >= min_angle) & (phi_angles_array <= max_angle)
            count = np.sum(mask)
            angles_in_range = phi_angles_array[mask] if count > 0 else []
            
            range_stats = {
                "range": (min_angle, max_angle),
                "count": int(count),
                "percentage": float(count / len(phi_angles_array) * 100),
                "angles": angles_in_range.tolist() if len(angles_in_range) < 20 else "too_many_to_list"
            }
            stats["angles_per_range"].append(range_stats)
            
        return stats


def filter_phi_angles(
    phi_angles: Union[List[float], np.ndarray],
    config: Optional[Dict[str, Any]] = None,
    target_ranges: Optional[List[Tuple[float, float]]] = None,
    fallback_enabled: Optional[bool] = None
) -> Tuple[List[int], np.ndarray]:
    """
    Convenience function for filtering phi angles.
    
    This is the main entry point for phi angle filtering, providing a simple
    interface while maintaining all the functionality of the PhiAngleFilter class.
    
    Args:
        phi_angles: Array or list of phi angles in degrees
        config: Optional configuration dictionary
        target_ranges: Optional list of (min_angle, max_angle) tuples
        fallback_enabled: Optional override for fallback behavior
        
    Returns:
        Tuple of (optimization_indices, filtered_angles)
        
    Example:
        >>> angles = [0, 45, 90, 135, 180, 225, 270, 315]
        >>> indices, filtered = filter_phi_angles(angles)
        >>> print(f"Selected indices: {indices}")
        >>> print(f"Selected angles: {filtered}")
    """
    filter_obj = PhiAngleFilter(config)
    return filter_obj.filter_angles_for_optimization(
        phi_angles, target_ranges, fallback_enabled
    )


def create_anisotropic_ranges() -> List[Tuple[float, float]]:
    """
    Create default target ranges for anisotropic analysis.
    
    Returns anisotropic-optimized ranges that capture key flow directions:
    - Near 0° (flow direction)
    - Near 90° (perpendicular to flow)  
    - Near 180° (opposite flow direction)
    - Near 270° (perpendicular to flow, opposite side)
    
    Returns:
        List of (min_angle, max_angle) tuples for anisotropic analysis
    """
    return [
        (-10.0, 10.0),      # Flow direction
        (80.0, 100.0),      # Perpendicular to flow
        (170.0, 190.0),     # Opposite flow direction  
        (260.0, 280.0),     # Perpendicular to flow (opposite)
    ]


def create_isotropic_ranges() -> List[Tuple[float, float]]:
    """
    Create default target ranges for isotropic analysis.
    
    For isotropic systems, we typically only need a couple of representative
    angles since the system has rotational symmetry.
    
    Returns:
        List of (min_angle, max_angle) tuples for isotropic analysis
    """
    return [
        (-10.0, 10.0),      # Primary direction
        (170.0, 190.0),     # Opposite direction for symmetry check
    ]


# JAX-accelerated version for high-performance filtering
if HAS_JAX:
    def filter_phi_angles_jax(
        phi_angles: Union[List[float], np.ndarray],
        target_ranges: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        JAX-accelerated version of phi angle filtering.
        
        This function uses JAX for potentially faster computation on GPU/TPU,
        but falls back to numpy if JAX is not available.
        
        Args:
            phi_angles: Array of phi angles in degrees
            target_ranges: List of (min_angle, max_angle) tuples
            
        Returns:
            Tuple of (optimization_indices, filtered_angles) as JAX arrays
        """
        phi_angles_jax = jnp.asarray(phi_angles)
        
        # Create mask for all target ranges
        optimization_mask = jnp.zeros(len(phi_angles_jax), dtype=bool)
        
        for min_angle, max_angle in target_ranges:
            range_mask = (phi_angles_jax >= min_angle) & (phi_angles_jax <= max_angle)
            optimization_mask = optimization_mask | range_mask
        
        # Get indices and filtered angles
        optimization_indices = jnp.where(optimization_mask)[0]
        filtered_angles = phi_angles_jax[optimization_mask]
        
        return optimization_indices, filtered_angles
        
else:
    # Fallback to numpy version if JAX is not available
    filter_phi_angles_jax = None


__all__ = [
    'PhiAngleFilter',
    'filter_phi_angles', 
    'create_anisotropic_ranges',
    'create_isotropic_ranges',
    'filter_phi_angles_jax'
]