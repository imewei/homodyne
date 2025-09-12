"""
Analysis Mode Resolution for Homodyne v2
========================================

Intelligent analysis mode detection and resolution combining:
- CLI mode flags (--static-isotropic, --static-anisotropic, --laminar-flow)
- Configuration file mode specifications
- Data characteristics and phi angle analysis
- Fallback to auto-detection algorithms

Key Features:
- Priority-based mode resolution (CLI > config > data analysis)
- Phi angle pattern analysis for mode suggestion
- Validation of mode compatibility with data
- Comprehensive logging of resolution logic
"""

import argparse
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from homodyne.utils.logging import get_logger
from homodyne.config.modes import detect_analysis_mode

@dataclass
class ModeCompatibilityResult:
    """
    Result of mode compatibility analysis.
    
    Attributes:
        is_compatible: Whether mode is compatible with data/config
        confidence: Confidence score (0.0 to 1.0)
        reasons: List of reasons for compatibility/incompatibility
        recommendations: List of mode recommendations
        warnings: List of potential issues
    """
    is_compatible: bool
    confidence: float
    reasons: List[str]
    recommendations: List[str]
    warnings: List[str]

logger = get_logger(__name__)


class ModeResolver:
    """
    Analysis mode resolution coordinator.
    
    Determines the appropriate analysis mode using multiple information sources
    with intelligent fallback and validation logic.
    """
    
    def __init__(self):
        """Initialize mode resolver."""
        self.resolution_priority = [
            'cli_explicit',      # CLI flags have highest priority
            'config_explicit',   # Configuration file specification
            'data_analysis',     # Analysis of data characteristics
            'fallback_default'   # Default fallback mode
        ]
        # Enhanced mode compatibility thresholds
        self.compatibility_thresholds = {
            'high_confidence': 0.8,
            'medium_confidence': 0.6,
            'low_confidence': 0.4
        }
        # Mode-specific data requirements
        self.mode_requirements = {
            'static_isotropic': {
                'min_angles': 1,
                'max_angles': 1,
                'min_data_points': 1000,
                'memory_factor': 1.0,
                'complexity': 'low'
            },
            'static_anisotropic': {
                'min_angles': 2,
                'max_angles': None,
                'min_data_points': 5000,
                'memory_factor': 2.0,
                'complexity': 'medium'
            },
            'laminar_flow': {
                'min_angles': 3,
                'max_angles': None,
                'min_data_points': 10000,
                'memory_factor': 3.5,
                'complexity': 'high'
            }
        }
    
    def resolve_mode(self, 
                    config: Dict[str, Any],
                    cli_args: Optional[argparse.Namespace] = None,
                    data_dict: Optional[Dict[str, Any]] = None) -> str:
        """
        Resolve analysis mode using priority-based logic.
        
        Args:
            config: Configuration dictionary
            cli_args: CLI arguments (optional)
            data_dict: Loaded data dictionary (optional)
            
        Returns:
            Resolved analysis mode string
        """
        logger.info("🔍 Resolving analysis mode")
        
        resolution_results = {}
        
        # Try each resolution method in priority order
        for method in self.resolution_priority:
            try:
                if method == 'cli_explicit':
                    result = self._resolve_from_cli(cli_args)
                elif method == 'config_explicit':
                    result = self._resolve_from_config(config)
                elif method == 'data_analysis':
                    result = self._resolve_from_data(data_dict)
                elif method == 'fallback_default':
                    result = self._resolve_fallback()
                
                resolution_results[method] = result
                
                # If we get a definitive result, use it
                if result and result != 'auto-detect':
                    logger.info(f"✓ Mode resolved via {method}: {result}")
                    self._log_resolution_summary(method, result, resolution_results)
                    return self._validate_and_return_enhanced(result, config, data_dict, resolution_results)
                    
            except Exception as e:
                logger.debug(f"Mode resolution method '{method}' failed: {e}")
                resolution_results[method] = None
        
        # If all methods failed, use fallback
        fallback_mode = 'static_isotropic'
        logger.warning(f"All resolution methods failed, using fallback: {fallback_mode}")
        self._log_resolution_summary('fallback', fallback_mode, resolution_results)
        return fallback_mode
    
    def _resolve_from_cli(self, cli_args: Optional[argparse.Namespace]) -> Optional[str]:
        """
        Resolve mode from CLI arguments.
        
        Args:
            cli_args: Parsed CLI arguments
            
        Returns:
            Mode string or None if not specified
        """
        if not cli_args:
            return None
        
        # Check explicit mode flags
        if hasattr(cli_args, 'static_isotropic') and cli_args.static_isotropic:
            logger.debug("CLI mode: static_isotropic flag detected")
            return 'static_isotropic'
        
        if hasattr(cli_args, 'static_anisotropic') and cli_args.static_anisotropic:
            logger.debug("CLI mode: static_anisotropic flag detected")
            return 'static_anisotropic'
        
        if hasattr(cli_args, 'laminar_flow') and cli_args.laminar_flow:
            logger.debug("CLI mode: laminar_flow flag detected")
            return 'laminar_flow'
        
        logger.debug("No explicit CLI mode flags found")
        return None
    
    def _resolve_from_config(self, config: Dict[str, Any]) -> Optional[str]:
        """
        Resolve mode from configuration file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Mode string or None if auto-detect
        """
        config_mode = config.get('analysis_mode', 'auto-detect')
        
        if config_mode and config_mode != 'auto-detect':
            logger.debug(f"Config mode: {config_mode} specified")
            return config_mode
        
        logger.debug("Config mode: auto-detect specified")
        return None
    
    def _resolve_from_data(self, data_dict: Optional[Dict[str, Any]]) -> Optional[str]:
        """
        Resolve mode from data analysis.
        
        Args:
            data_dict: Loaded experimental data
            
        Returns:
            Suggested mode or None if analysis fails
        """
        if not data_dict:
            logger.debug("No data available for mode analysis")
            return None
        
        try:
            # Analyze phi angles for mode suggestion
            phi_angles = data_dict.get('phi_angles')
            if phi_angles is not None:
                return self._analyze_phi_angles(phi_angles)
            
            # If no phi angle analysis possible, check data structure
            return self._analyze_data_structure(data_dict)
            
        except Exception as e:
            logger.debug(f"Data-based mode resolution failed: {e}")
            return None
    
    def _resolve_fallback(self) -> str:
        """
        Get fallback analysis mode.
        
        Returns:
            Default fallback mode
        """
        logger.debug("Using fallback mode: static_isotropic")
        return 'static_isotropic'
    
    def _analyze_phi_angles(self, phi_angles: np.ndarray) -> Optional[str]:
        """
        Analyze phi angle distribution to suggest analysis mode.
        
        Args:
            phi_angles: Array of phi angles
            
        Returns:
            Suggested mode based on angle analysis
        """
        if not isinstance(phi_angles, np.ndarray):
            phi_angles = np.array(phi_angles)
        
        n_angles = len(phi_angles)
        logger.debug(f"Analyzing {n_angles} phi angles for mode suggestion")
        
        # Single angle suggests isotropic analysis
        if n_angles == 1:
            logger.debug("Single phi angle → static_isotropic suggested")
            return 'static_isotropic'
        
        # Multiple angles - check distribution
        if n_angles >= 2:
            angle_range = np.max(phi_angles) - np.min(phi_angles)
            
            # Wide angle range suggests flow analysis potential
            if angle_range >= np.pi/2:  # 90 degrees or more
                logger.debug(f"Wide angle range ({np.degrees(angle_range):.1f}°) → laminar_flow suggested")
                return 'laminar_flow'
            else:
                logger.debug(f"Moderate angle range ({np.degrees(angle_range):.1f}°) → static_anisotropic suggested")
                return 'static_anisotropic'
        
        return None
    
    def _analyze_data_structure(self, data_dict: Dict[str, Any]) -> Optional[str]:
        """
        Analyze data structure for mode hints.
        
        Args:
            data_dict: Loaded data dictionary
            
        Returns:
            Suggested mode or None
        """
        # Check for time-dependent features in correlation data
        c2_data = data_dict.get('c2_exp')
        if c2_data is not None:
            data_size = self._safe_get_size(c2_data)
            
            # Large datasets might benefit from flow analysis
            if data_size > 10_000_000:  # 10M points
                logger.debug(f"Large dataset ({data_size:,} points) → laminar_flow might be appropriate")
                return 'laminar_flow'
            
            # Medium datasets for anisotropic analysis
            elif data_size > 1_000_000:  # 1M points
                logger.debug(f"Medium dataset ({data_size:,} points) → static_anisotropic suggested")
                return 'static_anisotropic'
        
        # Default fallback
        logger.debug("Data structure analysis inconclusive")
        return None
    
    def _validate_and_return_enhanced(self, 
                                    mode: str,
                                    config: Dict[str, Any],
                                    data_dict: Optional[Dict[str, Any]],
                                    resolution_results: Dict[str, Optional[str]]) -> str:
        """
        Enhanced validation with comprehensive compatibility analysis.
        
        Args:
            mode: Resolved mode
            config: Configuration dictionary
            data_dict: Data dictionary for validation
            resolution_results: Results from all resolution methods
            
        Returns:
            Validated and optimized mode string
        """
        try:
            # Validate mode string
            supported_modes = ['static_isotropic', 'static_anisotropic', 'laminar_flow']
            if mode not in supported_modes:
                logger.warning(f"Invalid mode '{mode}', using fallback")
                return 'static_isotropic'
            
            # Enhanced compatibility analysis
            if data_dict:
                compatibility = self._analyze_comprehensive_compatibility(mode, config, data_dict)
                
                if not compatibility.is_compatible:
                    logger.warning(f"Mode incompatibility detected:")
                    for reason in compatibility.reasons:
                        logger.warning(f"  - {reason}")
                    
                    if compatibility.recommendations:
                        best_recommendation = compatibility.recommendations[0]
                        logger.info(f"Switching to recommended mode: {best_recommendation}")
                        return best_recommendation
                elif compatibility.confidence < self.compatibility_thresholds['medium_confidence']:
                    logger.warning(f"Low confidence ({compatibility.confidence:.2f}) in mode '{mode}'")
                    for warning in compatibility.warnings:
                        logger.warning(f"  - {warning}")
            
            # Performance optimization suggestions
            self._provide_performance_recommendations(mode, config, data_dict)
            
            return mode
            
        except Exception as e:
            logger.error(f"Enhanced mode validation failed: {e}")
            return 'static_isotropic'
    
    def _analyze_comprehensive_compatibility(self, 
                                          mode: str,
                                          config: Dict[str, Any],
                                          data_dict: Dict[str, Any]) -> ModeCompatibilityResult:
        """
        Comprehensive analysis of mode-data-config compatibility.
        
        Args:
            mode: Analysis mode to check
            config: Configuration dictionary
            data_dict: Data dictionary
            
        Returns:
            Detailed compatibility analysis result
        """
        result = ModeCompatibilityResult(
            is_compatible=True,
            confidence=1.0,
            reasons=[],
            recommendations=[],
            warnings=[]
        )
        
        # Get mode requirements
        requirements = self.mode_requirements.get(mode, {})
        
        # Analyze phi angles compatibility
        phi_compatibility = self._analyze_phi_angle_compatibility(mode, data_dict, requirements)
        result.confidence *= phi_compatibility['confidence']
        result.reasons.extend(phi_compatibility['reasons'])
        if not phi_compatibility['compatible']:
            result.is_compatible = False
            result.recommendations.extend(phi_compatibility['suggestions'])
        
        # Analyze data size compatibility
        data_compatibility = self._analyze_data_size_compatibility(mode, data_dict, requirements)
        result.confidence *= data_compatibility['confidence']
        result.reasons.extend(data_compatibility['reasons'])
        if data_compatibility['warnings']:
            result.warnings.extend(data_compatibility['warnings'])
        
        # Analyze computational resource compatibility
        resource_compatibility = self._analyze_resource_compatibility(mode, config, data_dict, requirements)
        result.confidence *= resource_compatibility['confidence']
        result.warnings.extend(resource_compatibility['warnings'])
        result.reasons.extend(resource_compatibility['reasons'])
        
        # Generate alternative recommendations if needed
        if result.confidence < self.compatibility_thresholds['medium_confidence']:
            alternatives = self._generate_mode_alternatives(data_dict, config)
            result.recommendations.extend(alternatives)
        
        return result
    
    def _analyze_phi_angle_compatibility(self, 
                                       mode: str,
                                       data_dict: Dict[str, Any],
                                       requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze phi angle compatibility with mode requirements.
        
        Args:
            mode: Analysis mode
            data_dict: Data dictionary
            requirements: Mode requirements
            
        Returns:
            Compatibility analysis result
        """
        phi_angles = data_dict.get('phi_angles', [])
        n_angles = len(phi_angles) if phi_angles is not None else 0
        
        min_angles = requirements.get('min_angles', 1)
        max_angles = requirements.get('max_angles')
        
        compatible = True
        confidence = 1.0
        reasons = []
        suggestions = []
        
        # Check minimum angle requirement
        if n_angles < min_angles:
            compatible = False
            confidence = 0.0
            reasons.append(f"Insufficient phi angles: {n_angles} < {min_angles} required for {mode}")
            suggestions.extend(self._suggest_modes_for_angle_count(n_angles))
        
        # Check maximum angle constraint
        elif max_angles is not None and n_angles > max_angles:
            compatible = False
            confidence = 0.0
            reasons.append(f"Too many phi angles: {n_angles} > {max_angles} supported for {mode}")
            suggestions.extend(self._suggest_modes_for_angle_count(n_angles))
        
        else:
            # Analyze angle distribution quality
            if n_angles > 1 and phi_angles is not None:
                angle_analysis = self._analyze_angle_distribution_quality(phi_angles)
                confidence *= angle_analysis['quality_score']
                reasons.extend(angle_analysis['observations'])
        
        return {
            'compatible': compatible,
            'confidence': confidence,
            'reasons': reasons,
            'suggestions': suggestions
        }
    
    def _analyze_angle_distribution_quality(self, phi_angles: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the quality of phi angle distribution.
        
        Args:
            phi_angles: Array of phi angles
            
        Returns:
            Quality analysis result
        """
        if not isinstance(phi_angles, np.ndarray):
            phi_angles = np.array(phi_angles)
        
        phi_angles_rad = self._ensure_radians(phi_angles)
        observations = []
        quality_score = 1.0
        
        # Check angle range coverage
        angle_range = np.max(phi_angles_rad) - np.min(phi_angles_rad) if len(phi_angles_rad) > 1 else 0
        
        if angle_range < np.pi/6:  # Less than 30 degrees
            quality_score *= 0.7
            observations.append(f"Limited angular coverage ({np.degrees(angle_range):.1f}°)")
        elif angle_range > 3*np.pi/2:  # More than 270 degrees
            quality_score *= 1.2  # Bonus for good coverage
            observations.append(f"Excellent angular coverage ({np.degrees(angle_range):.1f}°)")
        
        # Check for angle clustering
        if len(phi_angles_rad) >= 3:
            angle_spacings = np.diff(np.sort(phi_angles_rad))
            spacing_std = np.std(angle_spacings)
            spacing_mean = np.mean(angle_spacings)
            
            if spacing_std / spacing_mean > 0.5:  # High variability in spacing
                quality_score *= 0.8
                observations.append("Irregular angle spacing detected")
            else:
                observations.append("Good angle spacing uniformity")
        
        # Check for adequate sampling
        n_angles = len(phi_angles_rad)
        if n_angles >= 8:
            observations.append(f"Good angular sampling: {n_angles} angles")
        elif n_angles >= 4:
            observations.append(f"Adequate angular sampling: {n_angles} angles")
        else:
            quality_score *= 0.9
            observations.append(f"Limited angular sampling: {n_angles} angles")
        
        return {
            'quality_score': min(quality_score, 1.0),  # Cap at 1.0
            'observations': observations
        }
    
    def _analyze_data_size_compatibility(self, 
                                       mode: str,
                                       data_dict: Dict[str, Any],
                                       requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze data size compatibility with mode requirements.
        
        Args:
            mode: Analysis mode
            data_dict: Data dictionary
            requirements: Mode requirements
            
        Returns:
            Data size compatibility analysis
        """
        c2_data = data_dict.get('c2_exp', [])
        data_size = self._safe_get_size(c2_data)
        min_data_points = requirements.get('min_data_points', 1000)
        
        confidence = 1.0
        reasons = []
        warnings = []
        
        # Check minimum data requirement
        if data_size < min_data_points:
            confidence *= max(0.5, data_size / min_data_points)
            reasons.append(
                f"Small dataset ({data_size:,} points) for {mode} mode "
                f"(recommended minimum: {min_data_points:,})"
            )
            warnings.append("Small dataset may lead to poor parameter estimation")
        
        # Assess data size appropriateness
        elif data_size < min_data_points * 2:
            confidence *= 0.8
            reasons.append(f"Adequate dataset size ({data_size:,} points) for {mode}")
        else:
            reasons.append(f"Good dataset size ({data_size:,} points) for {mode}")
        
        # Check for very large datasets
        if data_size > 50_000_000:  # 50M points
            warnings.append(
                f"Very large dataset ({data_size:,} points) may require optimized settings"
            )
        
        return {
            'confidence': confidence,
            'reasons': reasons,
            'warnings': warnings
        }
    
    def _analyze_resource_compatibility(self, 
                                      mode: str,
                                      config: Dict[str, Any],
                                      data_dict: Dict[str, Any],
                                      requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze computational resource compatibility.
        
        Args:
            mode: Analysis mode
            config: Configuration dictionary
            data_dict: Data dictionary
            requirements: Mode requirements
            
        Returns:
            Resource compatibility analysis
        """
        import psutil
        
        confidence = 1.0
        reasons = []
        warnings = []
        
        # Estimate memory requirements
        data_size = self._safe_get_size(data_dict.get('c2_exp', []))
        memory_factor = requirements.get('memory_factor', 1.0)
        estimated_memory_gb = data_size * memory_factor * 8 / (1024**3)  # rough estimate
        
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if estimated_memory_gb > available_memory_gb * 0.8:
            confidence *= 0.6
            warnings.append(
                f"High memory usage expected ({estimated_memory_gb:.1f} GB) "
                f"vs available ({available_memory_gb:.1f} GB)"
            )
        
        # Check optimization method compatibility
        opt_config = config.get('optimization_config', {})
        if mode == 'laminar_flow':
            # Laminar flow benefits from MCMC
            if not opt_config.get('mcmc_sampling', {}).get('enabled', True):
                confidence *= 0.8
                warnings.append("Laminar flow mode benefits from MCMC sampling")
            
            # Check for adequate MCMC settings
            mcmc_config = opt_config.get('mcmc_sampling', {})
            mcmc_draws = mcmc_config.get('draws', 3000)
            if mcmc_draws < 4000:
                warnings.append(
                    f"Consider increasing MCMC draws ({mcmc_draws}) for laminar flow mode"
                )
        
        reasons.append(f"Resource analysis for {mode} mode completed")
        
        return {
            'confidence': confidence,
            'reasons': reasons,
            'warnings': warnings
        }
    
    def _suggest_modes_for_angle_count(self, n_angles: int) -> List[str]:
        """Suggest appropriate modes based on angle count."""
        if n_angles == 1:
            return ['static_isotropic']
        elif 2 <= n_angles <= 4:
            return ['static_anisotropic', 'static_isotropic']
        else:
            return ['laminar_flow', 'static_anisotropic']
    
    def _generate_mode_alternatives(self, 
                                  data_dict: Dict[str, Any],
                                  config: Dict[str, Any]) -> List[str]:
        """
        Generate alternative mode recommendations.
        
        Args:
            data_dict: Data dictionary
            config: Configuration dictionary
            
        Returns:
            List of alternative modes in order of preference
        """
        alternatives = []
        
        phi_angles = data_dict.get('phi_angles', [])
        n_angles = len(phi_angles) if phi_angles is not None else 0
        data_size = self._safe_get_size(data_dict.get('c2_exp', []))
        
        # Score each mode based on data characteristics
        mode_scores = {}
        
        for mode, requirements in self.mode_requirements.items():
            score = 1.0
            
            # Angle compatibility
            if n_angles >= requirements['min_angles']:
                score *= 1.0
            else:
                score *= 0.1  # Heavy penalty for incompatible angle count
            
            # Data size appropriateness
            min_data = requirements['min_data_points']
            if data_size >= min_data:
                score *= 1.0
            else:
                score *= data_size / min_data
            
            # Complexity appropriateness (prefer simpler for smaller datasets)
            complexity_factor = {'low': 1.0, 'medium': 0.9, 'high': 0.8}[requirements['complexity']]
            if data_size < 100_000:  # Small dataset, prefer simpler
                score *= complexity_factor
            
            mode_scores[mode] = score
        
        # Sort by score and return top alternatives
        sorted_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [mode for mode, score in sorted_modes if score > 0.3]
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _provide_performance_recommendations(self, 
                                           mode: str,
                                           config: Dict[str, Any],
                                           data_dict: Optional[Dict[str, Any]]) -> None:
        """
        Provide performance optimization recommendations.
        
        Args:
            mode: Selected analysis mode
            config: Configuration dictionary
            data_dict: Data dictionary
        """
        if data_dict is None:
            return
        
        data_size = self._safe_get_size(data_dict.get('c2_exp', []))
        
        # Large dataset recommendations
        if data_size > 10_000_000:
            logger.info("Performance recommendations for large dataset:")
            logger.info("  - Consider enabling low_memory_mode")
            logger.info("  - Enable disk caching for intermediate results")
            logger.info("  - Use batch processing for multi-configuration analysis")
            
            if mode == 'laminar_flow':
                logger.info("  - For laminar flow: reduce MCMC draws to 3000-4000 for faster analysis")
        
        # Mode-specific recommendations
        complexity = self.mode_requirements.get(mode, {}).get('complexity', 'medium')
        if complexity == 'high':
            logger.info(f"Complex mode '{mode}' selected:")
            logger.info("  - Ensure adequate computational resources")
            logger.info("  - Consider using hybrid optimization for best results")
            logger.info("  - Monitor memory usage during analysis")
    
    def _check_mode_data_compatibility(self, 
                                     mode: str,
                                     data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy compatibility check method (maintained for backward compatibility).
        
        Args:
            mode: Analysis mode to check
            data_dict: Loaded data dictionary
            
        Returns:
            Compatibility check result dictionary
        """
        # Use enhanced compatibility analysis
        compatibility = self._analyze_comprehensive_compatibility(mode, {}, data_dict)
        
        if compatibility.is_compatible and compatibility.confidence >= self.compatibility_thresholds['medium_confidence']:
            return {'compatible': True}
        else:
            reason = compatibility.reasons[0] if compatibility.reasons else "Mode incompatible with data"
            suggested_mode = compatibility.recommendations[0] if compatibility.recommendations else 'static_isotropic'
            
            return {
                'compatible': False,
                'reason': reason,
                'suggested_mode': suggested_mode
            }
    
    def _log_resolution_summary(self, 
                              final_method: str,
                              final_mode: str,
                              all_results: Dict[str, Optional[str]]) -> None:
        """
        Log comprehensive resolution summary.
        
        Args:
            final_method: Method that provided the final result
            final_mode: Final resolved mode
            all_results: Results from all resolution methods
        """
        logger.debug("Mode resolution summary:")
        logger.debug(f"  Final result: {final_mode} (via {final_method})")
        
        for method, result in all_results.items():
            if result:
                status = "✓" if method == final_method else "○"
                logger.debug(f"  {status} {method}: {result}")
            else:
                logger.debug(f"  ✗ {method}: no result")
    
    def suggest_mode_for_data(self, data_dict: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Provide enhanced mode suggestions with comprehensive analysis.
        
        Args:
            data_dict: Loaded experimental data
            config: Optional configuration for enhanced analysis
            
        Returns:
            Dictionary with detailed suggestions and analysis
        """
        if config is None:
            config = {}
        
        suggestions = {
            'primary_suggestion': None,
            'alternatives': [],
            'reasoning': [],
            'confidence': 'low',
            'detailed_analysis': {},
            'performance_notes': [],
            'warnings': []
        }
        
        try:
            # Generate alternative mode rankings
            alternatives = self._generate_mode_alternatives(data_dict, config)
            
            if alternatives:
                suggestions['primary_suggestion'] = alternatives[0]
                suggestions['alternatives'] = alternatives[1:]
                
                # Analyze each mode option
                for mode in alternatives[:3]:  # Analyze top 3
                    compatibility = self._analyze_comprehensive_compatibility(mode, config, data_dict)
                    suggestions['detailed_analysis'][mode] = {
                        'compatibility': compatibility.is_compatible,
                        'confidence': compatibility.confidence,
                        'reasons': compatibility.reasons,
                        'warnings': compatibility.warnings
                    }
                
                # Set overall confidence based on primary suggestion
                primary_analysis = suggestions['detailed_analysis'].get(suggestions['primary_suggestion'], {})
                primary_confidence = primary_analysis.get('confidence', 0.5)
                
                if primary_confidence >= self.compatibility_thresholds['high_confidence']:
                    suggestions['confidence'] = 'high'
                elif primary_confidence >= self.compatibility_thresholds['medium_confidence']:
                    suggestions['confidence'] = 'medium'
                else:
                    suggestions['confidence'] = 'low'
                
                # Compile reasoning from primary analysis
                primary_reasons = primary_analysis.get('reasons', [])
                suggestions['reasoning'].extend(primary_reasons)
                
                # Add performance notes
                self._add_performance_notes(data_dict, suggestions)
                
                # Add warnings from analysis
                for mode_analysis in suggestions['detailed_analysis'].values():
                    suggestions['warnings'].extend(mode_analysis.get('warnings', []))
            
            else:
                # Fallback to simple analysis
                suggestions = self._fallback_mode_suggestion(data_dict)
            
        except Exception as e:
            logger.debug(f"Enhanced mode suggestion analysis failed: {e}")
            suggestions = self._fallback_mode_suggestion(data_dict)
        
        return suggestions
    
    def _add_performance_notes(self, data_dict: Dict[str, Any], suggestions: Dict[str, Any]) -> None:
        """
        Add performance-related notes to suggestions.
        
        Args:
            data_dict: Data dictionary
            suggestions: Suggestions dictionary to update
        """
        data_size = self._safe_get_size(data_dict.get('c2_exp', []))
        phi_angles = data_dict.get('phi_angles', [])
        n_angles = len(phi_angles) if phi_angles is not None else 0
        
        # Performance considerations
        if data_size > 50_000_000:
            suggestions['performance_notes'].append(
                "Very large dataset - consider using optimized settings and adequate computational resources"
            )
        elif data_size > 10_000_000:
            suggestions['performance_notes'].append(
                "Large dataset - may benefit from performance optimizations"
            )
        
        # Angle-specific performance notes
        if n_angles > 20:
            suggestions['performance_notes'].append(
                f"Many phi angles ({n_angles}) - consider angle filtering for improved performance"
            )
        
        # Mode-specific notes
        primary = suggestions.get('primary_suggestion')
        if primary == 'laminar_flow':
            suggestions['performance_notes'].append(
                "Laminar flow mode is computationally intensive - ensure adequate resources"
            )
    
    def _fallback_mode_suggestion(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback mode suggestion using simple heuristics.
        
        Args:
            data_dict: Data dictionary
            
        Returns:
            Simple mode suggestion
        """
        phi_angles = data_dict.get('phi_angles', [])
        n_angles = len(phi_angles) if phi_angles is not None else 0
        data_size = self._safe_get_size(data_dict.get('c2_exp', []))
        
        if n_angles == 1:
            return {
                'primary_suggestion': 'static_isotropic',
                'alternatives': [],
                'reasoning': ['Single phi angle detected'],
                'confidence': 'high',
                'detailed_analysis': {},
                'performance_notes': [],
                'warnings': []
            }
        elif n_angles >= 2:
            return {
                'primary_suggestion': 'static_anisotropic',
                'alternatives': ['laminar_flow'],
                'reasoning': [f'Multiple phi angles ({n_angles}) detected'],
                'confidence': 'medium',
                'detailed_analysis': {},
                'performance_notes': [],
                'warnings': []
            }
        else:
            return {
                'primary_suggestion': 'static_isotropic',
                'alternatives': [],
                'reasoning': ['No angle information available - using safe default'],
                'confidence': 'low',
                'detailed_analysis': {},
                'performance_notes': [],
                'warnings': ['No phi angle data found']
            }
    
    def analyze_mode_transition_feasibility(self, 
                                          current_mode: str,
                                          target_mode: str,
                                          data_dict: Dict[str, Any],
                                          config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze feasibility of transitioning between analysis modes.
        
        Args:
            current_mode: Current analysis mode
            target_mode: Desired target mode
            data_dict: Data dictionary
            config: Configuration dictionary
            
        Returns:
            Transition feasibility analysis
        """
        analysis = {
            'feasible': True,
            'confidence': 1.0,
            'requirements_met': [],
            'requirements_missing': [],
            'recommended_changes': [],
            'performance_impact': 'neutral',
            'warnings': []
        }
        
        current_req = self.mode_requirements.get(current_mode, {})
        target_req = self.mode_requirements.get(target_mode, {})
        
        # Check data compatibility for target mode
        target_compatibility = self._analyze_comprehensive_compatibility(target_mode, config, data_dict)
        
        if not target_compatibility.is_compatible:
            analysis['feasible'] = False
            analysis['requirements_missing'].extend(target_compatibility.reasons)
        else:
            analysis['confidence'] = target_compatibility.confidence
            analysis['requirements_met'].extend(target_compatibility.reasons)
        
        # Analyze performance impact
        current_complexity = current_req.get('complexity', 'medium')
        target_complexity = target_req.get('complexity', 'medium')
        
        complexity_order = {'low': 1, 'medium': 2, 'high': 3}
        if complexity_order[target_complexity] > complexity_order[current_complexity]:
            analysis['performance_impact'] = 'increased_computational_cost'
            analysis['warnings'].append(
                f"Transitioning from {current_mode} to {target_mode} will increase computational requirements"
            )
        elif complexity_order[target_complexity] < complexity_order[current_complexity]:
            analysis['performance_impact'] = 'reduced_computational_cost'
        
        # Generate transition recommendations
        if analysis['feasible']:
            self._generate_transition_recommendations(current_mode, target_mode, config, analysis)
        
        return analysis
    
    def _generate_transition_recommendations(self, 
                                           current_mode: str,
                                           target_mode: str,
                                           config: Dict[str, Any],
                                           analysis: Dict[str, Any]) -> None:
        """
        Generate recommendations for mode transitions.
        
        Args:
            current_mode: Current mode
            target_mode: Target mode  
            config: Configuration
            analysis: Analysis dictionary to update
        """
        recommendations = []
        
        # Mode-specific transition recommendations
        if current_mode == 'static_isotropic' and target_mode in ['static_anisotropic', 'laminar_flow']:
            recommendations.append("Enable angle filtering in optimization configuration")
            recommendations.append("Ensure multiple phi angles are available in your data")
        
        if target_mode == 'laminar_flow':
            recommendations.append("Increase MCMC sampling parameters (draws, tune) for better convergence")
            recommendations.append("Consider using hybrid optimization method")
            recommendations.append("Ensure adequate computational resources (memory, CPU cores)")
        
        if current_mode == 'laminar_flow' and target_mode in ['static_isotropic', 'static_anisotropic']:
            recommendations.append("Reduce optimization complexity - fewer parameters to fit")
            recommendations.append("Consider reducing MCMC parameters for faster analysis")
        
        # Configuration adjustments
        opt_config = config.get('optimization_config', {})
        if target_mode == 'laminar_flow' and not opt_config.get('mcmc_sampling', {}).get('enabled', True):
            recommendations.append("Enable MCMC sampling for laminar flow analysis")
        
        analysis['recommended_changes'] = recommendations
    
    def _ensure_radians(self, angles: np.ndarray) -> np.ndarray:
        """
        Ensure angle array is in radians, converting from degrees if needed.

        Uses heuristic: if max angle > 2π, assume degrees and convert.

        Args:
            angles: Array of angles (potentially in degrees or radians)
            
        Returns:
            Angles in radians
        """
        if not isinstance(angles, np.ndarray):
            angles = np.array(angles)
        
        # Handle empty arrays
        if angles.size == 0:
            return angles
        
        # Heuristic: if max angle > 2π (≈6.28), likely in degrees
        if np.max(np.abs(angles)) > 2 * np.pi:
            logger.debug(f"Converting angles from degrees to radians (max: {np.max(angles):.1f}°)")
            return np.radians(angles)
        else:
            logger.debug(f"Angles appear to be in radians (max: {np.max(angles):.3f} rad)")
            return angles
    
    @staticmethod
    def _ensure_numpy_array(data: Any, name: str = "data") -> np.ndarray:
        """
        Ensure data is a numpy array, with defensive programming.
        
        Args:
            data: Data to convert (list, tuple, array, or scalar)
            name: Name of the data for error messages
            
        Returns:
            Data as numpy array
            
        Raises:
            ValueError: If data cannot be converted to numpy array
        """
        if data is None:
            return np.array([])
        
        try:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            return data
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {name} to numpy array: {e}")
    
    @staticmethod
    def _safe_get_size(data: Any) -> int:
        """
        Safely get the size of data, handling lists and arrays.
        
        Args:
            data: Data to get size of
            
        Returns:
            Size of the data (0 if None or empty)
        """
        if data is None:
            return 0
        
        try:
            if hasattr(data, 'size'):
                return data.size
            elif hasattr(data, '__len__'):
                return len(data)
            else:
                return 1  # scalar
        except (TypeError, AttributeError):
            return 0