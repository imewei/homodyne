"""
Data Filtering Utilities for XPCS Data Loader
=============================================

Comprehensive filtering utilities supporting q-range, quality-based, and frame-based filtering
for XPCS correlation matrices. Integrates with the existing phi_filtering system and provides
unified filtering logic for the data loader.

Key Features:
- Q-range filtering based on wavevector values
- Quality-based filtering using correlation matrix properties
- Frame-based filtering beyond basic start/end limits
- Integration with existing phi angle filtering
- Flexible combination criteria (AND/OR logic)
- Comprehensive error handling and validation

Performance optimizations:
- Vectorized operations using numpy/JAX
- Smart caching of filtering results
- Early termination for empty filter results
- Memory-efficient mask operations
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# JAX integration with fallback
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import jax.numpy as jnp
    HAS_JAX = False

# V2 logging integration
try:
    from homodyne.utils.logging import get_logger, log_performance
    HAS_V2_LOGGING = True
except ImportError:
    import logging
    HAS_V2_LOGGING = False
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Physics validation integration
try:
    from homodyne.core.physics import PhysicsConstants
    HAS_PHYSICS = True
except ImportError:
    HAS_PHYSICS = False
    PhysicsConstants = None

# Data validation integration
try:
    from homodyne.data.validation import validate_xpcs_data
    HAS_VALIDATION = True
except ImportError:
    HAS_VALIDATION = False
    validate_xpcs_data = None

logger = get_logger(__name__)

class FilterCriteria(Enum):
    """Enumeration of available filter criteria combination methods."""
    AND = "AND"
    OR = "OR"

@dataclass
class FilteringResult:
    """Result of data filtering operation."""
    selected_indices: Optional[np.ndarray]
    total_available: int
    total_selected: int
    filters_applied: List[str]
    filter_statistics: Dict[str, Any]
    fallback_used: bool
    warnings: List[str]
    errors: List[str]

class DataFilteringError(Exception):
    """Raised when data filtering encounters an error."""
    pass

class XPCSDataFilter:
    """
    Comprehensive data filter for XPCS correlation matrices.
    
    Provides unified filtering based on configuration parameters including
    q-range, quality thresholds, and frame-based criteria.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data filter.
        
        Args:
            config: Configuration dictionary containing data_filtering parameters
        """
        self.config = config or {}
        self.filtering_config = self.config.get('data_filtering', {})
        self.enabled = self.filtering_config.get('enabled', False)
        self.validation_level = self.filtering_config.get('validation_level', 'basic')
        self.fallback_on_empty = self.filtering_config.get('fallback_on_empty', True)
        self.combine_criteria = FilterCriteria(self.filtering_config.get('combine_criteria', 'AND'))
        
        logger.debug(f"Initialized XPCSDataFilter - enabled: {self.enabled}, "
                    f"validation_level: {self.validation_level}")

    @log_performance(threshold=0.1)
    def apply_filtering(self, 
                       dqlist: np.ndarray, 
                       dphilist: np.ndarray,
                       correlation_matrices: Optional[List[np.ndarray]] = None) -> FilteringResult:
        """
        Apply comprehensive filtering to XPCS data indices.
        
        Args:
            dqlist: Array of q-values (wavevector magnitudes)
            dphilist: Array of phi angles in degrees
            correlation_matrices: Optional list of correlation matrices for quality filtering
            
        Returns:
            FilteringResult containing selected indices and filtering statistics
        """
        total_available = len(dqlist)
        logger.info(f"Starting data filtering on {total_available} data points")
        
        # Initialize result
        result = FilteringResult(
            selected_indices=None,
            total_available=total_available,
            total_selected=0,
            filters_applied=[],
            filter_statistics={},
            fallback_used=False,
            warnings=[],
            errors=[]
        )
        
        if not self.enabled:
            logger.info("Data filtering disabled - returning all indices")
            result.selected_indices = np.arange(total_available)
            result.total_selected = total_available
            return result
        
        try:
            # Collect individual filter masks
            filter_masks = {}
            
            # Apply q-range filtering
            q_mask = self._apply_q_range_filtering(dqlist, result)
            if q_mask is not None:
                filter_masks['q_range'] = q_mask
                result.filters_applied.append('q_range')
            
            # Apply phi-range filtering
            phi_mask = self._apply_phi_range_filtering(dphilist, result)
            if phi_mask is not None:
                filter_masks['phi_range'] = phi_mask
                result.filters_applied.append('phi_range')
            
            # Apply quality-based filtering
            if correlation_matrices is not None:
                quality_mask = self._apply_quality_filtering(correlation_matrices, result)
                if quality_mask is not None:
                    filter_masks['quality'] = quality_mask
                    result.filters_applied.append('quality')
            
            # Apply frame-based filtering
            frame_mask = self._apply_frame_filtering(total_available, result)
            if frame_mask is not None:
                filter_masks['frame'] = frame_mask
                result.filters_applied.append('frame')
            
            # Combine filter masks
            if filter_masks:
                combined_mask = self._combine_filter_masks(filter_masks, result)
                selected_indices = np.where(combined_mask)[0]
                
                if len(selected_indices) > 0:
                    result.selected_indices = selected_indices
                    result.total_selected = len(selected_indices)
                    logger.info(f"Filtering selected {result.total_selected}/{total_available} data points")
                else:
                    # Handle empty result
                    result = self._handle_empty_filter_result(result, total_available)
            else:
                # No filters applied
                logger.info("No filtering criteria specified - returning all indices")
                result.selected_indices = np.arange(total_available)
                result.total_selected = total_available
            
            # Validate result
            self._validate_filtering_result(result)
            
        except Exception as e:
            error_msg = f"Data filtering failed: {str(e)}"
            logger.error(error_msg)
            result.errors.append(error_msg)
            
            if self.fallback_on_empty:
                logger.warning("Falling back to all indices due to filtering error")
                result.selected_indices = np.arange(total_available)
                result.total_selected = total_available
                result.fallback_used = True
            else:
                raise DataFilteringError(error_msg)
        
        return result
    
    def _apply_q_range_filtering(self, dqlist: np.ndarray, result: FilteringResult) -> Optional[np.ndarray]:
        """Apply q-range filtering based on wavevector values."""
        q_range = self.filtering_config.get('q_range', {})
        if not q_range:
            return None
        
        q_min = q_range.get('min')
        q_max = q_range.get('max')
        
        if q_min is None and q_max is None:
            return None
        
        logger.debug(f"Applying q-range filtering: [{q_min}, {q_max}]")
        
        # Create mask
        mask = np.ones(len(dqlist), dtype=bool)
        
        if q_min is not None:
            mask &= (dqlist >= q_min)
        if q_max is not None:
            mask &= (dqlist <= q_max)
        
        # Statistics
        selected_count = np.sum(mask)
        result.filter_statistics['q_range'] = {
            'q_min': q_min,
            'q_max': q_max,
            'selected_count': int(selected_count),
            'data_q_min': float(np.min(dqlist)),
            'data_q_max': float(np.max(dqlist)),
            'selection_fraction': float(selected_count / len(dqlist))
        }
        
        logger.debug(f"Q-range filtering: {selected_count}/{len(dqlist)} points selected")
        
        # Physics validation if available
        if HAS_PHYSICS and self.validation_level == 'strict':
            self._validate_q_range_physics(q_min, q_max, result)
        
        return mask
    
    def _apply_phi_range_filtering(self, dphilist: np.ndarray, result: FilteringResult) -> Optional[np.ndarray]:
        """Apply phi-range filtering based on angle values."""
        phi_range = self.filtering_config.get('phi_range', {})
        if not phi_range:
            return None
        
        phi_min = phi_range.get('min')
        phi_max = phi_range.get('max')
        
        if phi_min is None and phi_max is None:
            return None
        
        logger.debug(f"Applying phi-range filtering: [{phi_min}, {phi_max}]")
        
        # Create mask
        mask = np.ones(len(dphilist), dtype=bool)
        
        if phi_min is not None:
            mask &= (dphilist >= phi_min)
        if phi_max is not None:
            mask &= (dphilist <= phi_max)
        
        # Statistics
        selected_count = np.sum(mask)
        result.filter_statistics['phi_range'] = {
            'phi_min': phi_min,
            'phi_max': phi_max,
            'selected_count': int(selected_count),
            'data_phi_min': float(np.min(dphilist)),
            'data_phi_max': float(np.max(dphilist)),
            'selection_fraction': float(selected_count / len(dphilist))
        }
        
        logger.debug(f"Phi-range filtering: {selected_count}/{len(dphilist)} points selected")
        
        return mask
    
    def _apply_quality_filtering(self, correlation_matrices: List[np.ndarray], 
                                result: FilteringResult) -> Optional[np.ndarray]:
        """Apply quality-based filtering using correlation matrix properties."""
        quality_threshold = self.filtering_config.get('quality_threshold')
        if quality_threshold is None:
            return None
        
        logger.debug(f"Applying quality filtering with threshold: {quality_threshold}")
        
        mask = np.ones(len(correlation_matrices), dtype=bool)
        quality_scores = []
        
        for i, matrix in enumerate(correlation_matrices):
            try:
                # Calculate quality score based on matrix properties
                quality_score = self._calculate_matrix_quality_score(matrix)
                quality_scores.append(quality_score)
                
                # Apply threshold
                if quality_score < quality_threshold:
                    mask[i] = False
                    
            except Exception as e:
                logger.warning(f"Quality calculation failed for matrix {i}: {e}")
                quality_scores.append(0.0)
                mask[i] = False
        
        # Statistics
        selected_count = np.sum(mask)
        result.filter_statistics['quality'] = {
            'threshold': quality_threshold,
            'selected_count': int(selected_count),
            'quality_scores': {
                'mean': float(np.mean(quality_scores)),
                'std': float(np.std(quality_scores)),
                'min': float(np.min(quality_scores)),
                'max': float(np.max(quality_scores))
            },
            'selection_fraction': float(selected_count / len(correlation_matrices))
        }
        
        logger.debug(f"Quality filtering: {selected_count}/{len(correlation_matrices)} matrices selected")
        
        return mask
    
    def _apply_frame_filtering(self, total_count: int, result: FilteringResult) -> Optional[np.ndarray]:
        """Apply frame-based filtering beyond basic start/end."""
        frame_filtering = self.filtering_config.get('frame_filtering', {})
        if not frame_filtering:
            return None
        
        # For now, implement basic stride-based filtering
        # This can be extended with more sophisticated frame selection logic
        stride = frame_filtering.get('stride', 1)
        if stride <= 1:
            return None
        
        logger.debug(f"Applying frame filtering with stride: {stride}")
        
        # Create strided indices
        selected_indices = np.arange(0, total_count, stride)
        mask = np.zeros(total_count, dtype=bool)
        mask[selected_indices] = True
        
        # Statistics
        selected_count = len(selected_indices)
        result.filter_statistics['frame'] = {
            'stride': stride,
            'selected_count': int(selected_count),
            'selection_fraction': float(selected_count / total_count)
        }
        
        logger.debug(f"Frame filtering: {selected_count}/{total_count} frames selected")
        
        return mask
    
    def _calculate_matrix_quality_score(self, matrix: np.ndarray) -> float:
        """
        Calculate quality score for a correlation matrix.
        
        Quality metrics:
        - Finite value fraction
        - Diagonal correlation strength
        - Matrix symmetry
        - Reasonable value ranges
        """
        try:
            # Check for finite values
            finite_fraction = np.sum(np.isfinite(matrix)) / matrix.size
            if finite_fraction < 0.9:
                return 0.0  # Poor quality if too many non-finite values
            
            # Check diagonal values (should be around 1.0 at t=0)
            diagonal = np.diag(matrix)
            t0_correlation = diagonal[0] if len(diagonal) > 0 else 0.0
            
            diagonal_quality = 1.0 if 0.5 <= t0_correlation <= 2.0 else 0.5
            
            # Check matrix symmetry
            if matrix.shape[0] == matrix.shape[1]:
                symmetry_error = np.mean(np.abs(matrix - matrix.T))
                symmetry_quality = max(0.0, 1.0 - symmetry_error * 100)
            else:
                symmetry_quality = 0.5
            
            # Check for reasonable value ranges
            matrix_mean = np.mean(matrix)
            range_quality = 1.0 if 0.1 <= matrix_mean <= 5.0 else 0.5
            
            # Combine quality metrics
            overall_quality = (
                finite_fraction * 0.4 + 
                diagonal_quality * 0.3 + 
                symmetry_quality * 0.2 + 
                range_quality * 0.1
            )
            
            return float(overall_quality)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.0
    
    def _combine_filter_masks(self, filter_masks: Dict[str, np.ndarray], 
                             result: FilteringResult) -> np.ndarray:
        """Combine multiple filter masks using specified criteria."""
        if not filter_masks:
            return np.array([])
        
        masks = list(filter_masks.values())
        
        if self.combine_criteria == FilterCriteria.AND:
            # All filters must pass
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask &= mask
        else:  # OR
            # Any filter can pass
            combined_mask = masks[0].copy()
            for mask in masks[1:]:
                combined_mask |= mask
        
        # Statistics
        result.filter_statistics['combination'] = {
            'criteria': self.combine_criteria.value,
            'individual_selections': {
                name: int(np.sum(mask)) for name, mask in filter_masks.items()
            },
            'combined_selection': int(np.sum(combined_mask))
        }
        
        return combined_mask
    
    def _handle_empty_filter_result(self, result: FilteringResult, total_available: int) -> FilteringResult:
        """Handle case where filtering results in no selected indices."""
        logger.warning("Filtering resulted in no selected data points")
        
        if self.fallback_on_empty:
            logger.warning("Falling back to all indices due to empty filter result")
            result.selected_indices = np.arange(total_available)
            result.total_selected = total_available
            result.fallback_used = True
            result.warnings.append("Empty filter result - fallback to all indices")
        else:
            error_msg = "Filtering resulted in empty dataset and fallback is disabled"
            result.errors.append(error_msg)
            raise DataFilteringError(error_msg)
        
        return result
    
    def _validate_q_range_physics(self, q_min: Optional[float], q_max: Optional[float], 
                                 result: FilteringResult) -> None:
        """Validate q-range against physics constraints."""
        if not HAS_PHYSICS:
            return
        
        warnings = []
        
        if q_min is not None and q_min < PhysicsConstants.Q_MIN_TYPICAL:
            warnings.append(f"q_min ({q_min:.2e}) below typical XPCS range "
                           f"({PhysicsConstants.Q_MIN_TYPICAL:.2e})")
        
        if q_max is not None and q_max > PhysicsConstants.Q_MAX_TYPICAL:
            warnings.append(f"q_max ({q_max:.2e}) above typical XPCS range "
                           f"({PhysicsConstants.Q_MAX_TYPICAL:.2e})")
        
        result.warnings.extend(warnings)
        if warnings:
            logger.warning(f"Physics validation warnings: {warnings}")
    
    def _validate_filtering_result(self, result: FilteringResult) -> None:
        """Validate the filtering result for consistency."""
        if result.selected_indices is not None:
            # Check indices are within bounds
            if np.any(result.selected_indices < 0):
                raise DataFilteringError("Negative indices in filtering result")
            if np.any(result.selected_indices >= result.total_available):
                raise DataFilteringError("Indices exceed available data range")
            
            # Check count consistency
            if len(result.selected_indices) != result.total_selected:
                raise DataFilteringError("Inconsistent selected index count")
            
            # Check for duplicates
            if len(np.unique(result.selected_indices)) != len(result.selected_indices):
                raise DataFilteringError("Duplicate indices in filtering result")

# Convenience functions for integration

def apply_data_filtering(dqlist: np.ndarray,
                        dphilist: np.ndarray, 
                        config: Dict[str, Any],
                        correlation_matrices: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
    """
    Convenience function for applying data filtering.
    
    Args:
        dqlist: Array of q-values
        dphilist: Array of phi angles
        config: Configuration dictionary
        correlation_matrices: Optional correlation matrices for quality filtering
        
    Returns:
        Array of selected indices, or None if no filtering applied
    """
    filter_obj = XPCSDataFilter(config)
    result = filter_obj.apply_filtering(dqlist, dphilist, correlation_matrices)
    
    if result.errors:
        logger.error(f"Filtering errors: {result.errors}")
        if not result.fallback_used:
            raise DataFilteringError(f"Filtering failed: {result.errors}")
    
    if result.warnings:
        logger.warning(f"Filtering warnings: {result.warnings}")
    
    return result.selected_indices

# Export main classes and functions
__all__ = [
    'XPCSDataFilter',
    'FilteringResult',
    'FilterCriteria',
    'DataFilteringError',
    'apply_data_filtering'
]