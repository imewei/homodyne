"""
Configuration Merger for Homodyne v2
====================================

Advanced configuration merging with priority handling, conflict resolution,
and comprehensive validation integration.

Key Features:
- Hierarchical merging (CLI args + file config + defaults)
- Conflict detection and resolution with user-friendly warnings
- Smart deep merging of nested dictionaries
- Configuration source tracking for debugging
- Validation integration with rollback capabilities
"""

from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass
from copy import deepcopy

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MergeConflict:
    """
    Configuration merge conflict information.
    
    Attributes:
        key_path: Dot-separated path to conflicting key
        cli_value: Value from CLI arguments
        file_value: Value from configuration file
        resolution: How the conflict was resolved
    """
    key_path: str
    cli_value: Any
    file_value: Any
    resolution: str


@dataclass
class MergeResult:
    """
    Configuration merge result with metadata.
    
    Attributes:
        config: Merged configuration dictionary
        conflicts: List of detected conflicts
        sources: Source information for each configuration key
        warnings: List of merge warnings
    """
    config: Dict[str, Any]
    conflicts: List[MergeConflict]
    sources: Dict[str, str]
    warnings: List[str]


class ConfigMerger:
    """
    Configuration merger with intelligent conflict resolution.
    
    Handles complex merging scenarios with priority-based resolution
    and comprehensive tracking of configuration sources.
    """
    
    def __init__(self):
        """Initialize configuration merger."""
        self.merge_priority = ['defaults', 'file', 'cli']  # CLI has highest priority
        self.special_merge_keys = {
            'optimization': self._merge_optimization_config,
            'plotting': self._merge_plotting_config,
            'hardware': self._merge_hardware_config
        }
    
    def merge_configurations(self,
                           cli_config: Dict[str, Any],
                           file_config: Optional[Dict[str, Any]] = None,
                           default_config: Optional[Dict[str, Any]] = None) -> MergeResult:
        """
        Merge configurations with conflict detection and resolution.
        
        Args:
            cli_config: Configuration from CLI arguments
            file_config: Configuration from file (optional)
            default_config: Default configuration (optional)
            
        Returns:
            Merge result with final configuration and metadata
        """
        logger.info("🔀 Merging configurations with conflict detection")
        
        # Initialize configurations with empty dicts if None
        file_config = file_config or {}
        default_config = default_config or self._get_base_defaults()
        
        # Initialize merge result
        result = MergeResult(
            config={},
            conflicts=[],
            sources={},
            warnings=[]
        )
        
        # Start with defaults
        result.config = deepcopy(default_config)
        self._track_sources(result.config, 'defaults', result.sources)
        
        # Merge file configuration
        if file_config:
            self._merge_with_conflict_detection(
                target=result.config,
                source=file_config,
                source_name='file',
                result=result
            )
        
        # Merge CLI configuration (highest priority)
        if cli_config:
            self._merge_with_conflict_detection(
                target=result.config,
                source=cli_config,
                source_name='cli',
                result=result
            )
        
        # Post-process configuration
        self._post_process_config(result)
        
        # Log merge summary
        self._log_merge_summary(result)
        
        return result
    
    def _merge_with_conflict_detection(self,
                                     target: Dict[str, Any],
                                     source: Dict[str, Any],
                                     source_name: str,
                                     result: MergeResult,
                                     key_path: str = "") -> None:
        """
        Merge source into target with conflict detection.
        
        Args:
            target: Target configuration dictionary
            source: Source configuration dictionary
            source_name: Name of configuration source
            result: Merge result to update
            key_path: Current key path for nested tracking
        """
        for key, source_value in source.items():
            current_path = f"{key_path}.{key}" if key_path else key
            
            # Check if key exists in target
            if key not in target:
                # No conflict - add new key
                target[key] = deepcopy(source_value)
                result.sources[current_path] = source_name
                continue
            
            target_value = target[key]
            
            # Handle nested dictionaries
            if isinstance(target_value, dict) and isinstance(source_value, dict):
                # Check for special merge handling
                if key in self.special_merge_keys:
                    merged_value = self.special_merge_keys[key](target_value, source_value)
                    target[key] = merged_value
                    result.sources[current_path] = f"{source_name} (special)"
                else:
                    # Recursive merge
                    self._merge_with_conflict_detection(
                        target[key], source_value, source_name, result, current_path
                    )
                continue
            
            # Detect conflict
            if target_value != source_value:
                conflict = MergeConflict(
                    key_path=current_path,
                    cli_value=source_value if source_name == 'cli' else target_value,
                    file_value=source_value if source_name == 'file' else target_value,
                    resolution=f"{source_name} overrides"
                )
                result.conflicts.append(conflict)
            
            # Apply override
            target[key] = deepcopy(source_value)
            result.sources[current_path] = source_name
    
    def _merge_optimization_config(self,
                                 target: Dict[str, Any],
                                 source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Special merge handling for optimization configuration.
        
        Args:
            target: Target optimization config
            source: Source optimization config
            
        Returns:
            Merged optimization configuration
        """
        merged = deepcopy(target)
        
        # Merge each optimization method separately
        for method in ['vi', 'mcmc', 'hybrid']:
            if method in source:
                if method not in merged:
                    merged[method] = {}
                
                # Deep merge method parameters
                for param_key, param_value in source[method].items():
                    merged[method][param_key] = param_value
        
        return merged
    
    def _merge_plotting_config(self,
                             target: Dict[str, Any],
                             source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Special merge handling for plotting configuration.
        
        Args:
            target: Target plotting config
            source: Source plotting config
            
        Returns:
            Merged plotting configuration
        """
        merged = deepcopy(target)
        
        # Handle boolean flags with OR logic for enabling features
        boolean_flags = ['generate_plots', 'plot_experimental_data', 'plot_simulated_data']
        
        for flag in boolean_flags:
            if flag in source:
                # Use OR logic - if either source enables it, keep it enabled
                merged[flag] = merged.get(flag, False) or source[flag]
        
        # Handle other plotting parameters normally
        for key, value in source.items():
            if key not in boolean_flags:
                merged[key] = value
        
        return merged
    
    def _merge_hardware_config(self,
                             target: Dict[str, Any],
                             source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Special merge handling for hardware configuration.
        
        Args:
            target: Target hardware config
            source: Source hardware config
            
        Returns:
            Merged hardware configuration
        """
        merged = deepcopy(target)
        
        # Handle force_cpu flag - CLI should override file
        if 'force_cpu' in source:
            merged['force_cpu'] = source['force_cpu']
        
        # Handle GPU memory fraction with validation
        if 'gpu_memory_fraction' in source:
            gpu_fraction = source['gpu_memory_fraction']
            if 0.0 < gpu_fraction <= 1.0:
                merged['gpu_memory_fraction'] = gpu_fraction
            else:
                # Invalid value - keep target, add warning
                logger.warning(f"Invalid gpu_memory_fraction {gpu_fraction}, keeping {target.get('gpu_memory_fraction', 0.8)}")
        
        return merged
    
    def _track_sources(self,
                      config: Dict[str, Any],
                      source_name: str,
                      sources: Dict[str, str],
                      key_path: str = "") -> None:
        """
        Track configuration sources recursively.
        
        Args:
            config: Configuration dictionary
            source_name: Name of configuration source
            sources: Sources tracking dictionary
            key_path: Current key path
        """
        for key, value in config.items():
            current_path = f"{key_path}.{key}" if key_path else key
            
            if isinstance(value, dict):
                self._track_sources(value, source_name, sources, current_path)
            else:
                sources[current_path] = source_name
    
    def _post_process_config(self, result: MergeResult) -> None:
        """
        Post-process merged configuration for consistency.
        
        Args:
            result: Merge result to post-process
        """
        config = result.config
        
        # Ensure analysis_mode consistency
        self._ensure_mode_consistency(config, result)
        
        # Validate parameter combinations
        self._validate_parameter_combinations(config, result)
        
        # Apply configuration-specific defaults
        self._apply_conditional_defaults(config, result)
    
    def _ensure_mode_consistency(self, config: Dict[str, Any], result: MergeResult) -> None:
        """
        Ensure analysis mode consistency across configuration.
        
        Args:
            config: Configuration dictionary
            result: Merge result for warnings
        """
        mode = config.get('analysis_mode', 'auto-detect')
        
        # Check for mode-specific inconsistencies
        if mode == 'static_isotropic':
            plotting_config = config.get('plotting', {})
            if plotting_config.get('plot_simulated_data'):
                result.warnings.append(
                    "plot_simulated_data may not be meaningful for static_isotropic mode"
                )
        
        # Ensure optimization parameters match mode complexity
        if mode == 'laminar_flow':
            opt_config = config.get('optimization', {})
            vi_config = opt_config.get('vi', {})
            if vi_config.get('n_iterations', 2000) < 1000:
                result.warnings.append(
                    "laminar_flow mode may need more VI iterations for convergence"
                )
    
    def _validate_parameter_combinations(self, config: Dict[str, Any], result: MergeResult) -> None:
        """
        Validate parameter combinations for consistency.
        
        Args:
            config: Configuration dictionary
            result: Merge result for warnings
        """
        hardware = config.get('hardware', {})
        opt_config = config.get('optimization', {})
        
        # Check hardware-optimization consistency
        if hardware.get('force_cpu') and opt_config:
            mcmc_config = opt_config.get('mcmc', {})
            n_chains = mcmc_config.get('n_chains', 4)
            if n_chains > 4:
                result.warnings.append(
                    f"Using {n_chains} MCMC chains on CPU may be slow - consider reducing or enabling GPU"
                )
        
        # Check data-optimization consistency
        data_config = config.get('data', {})
        if not data_config.get('dataset_optimization', True):
            mode = config.get('analysis_mode')
            if mode == 'laminar_flow':
                result.warnings.append(
                    "Disabled dataset optimization with laminar_flow mode may cause memory issues"
                )
    
    def _apply_conditional_defaults(self, config: Dict[str, Any], result: MergeResult) -> None:
        """
        Apply conditional defaults based on configuration state.
        
        Args:
            config: Configuration dictionary
            result: Merge result for tracking
        """
        # Apply mode-specific optimization defaults
        mode = config.get('analysis_mode', 'auto-detect')
        opt_config = config.setdefault('optimization', {})
        
        if mode == 'laminar_flow':
            # Increase default iterations for complex mode
            vi_config = opt_config.setdefault('vi', {})
            if 'n_iterations' not in vi_config:
                vi_config['n_iterations'] = 3000  # Higher than default 2000
                result.sources['optimization.vi.n_iterations'] = 'conditional_default'
        
        # Apply hardware-dependent defaults
        hardware = config.get('hardware', {})
        if hardware.get('force_cpu'):
            # Reduce default MCMC chains for CPU
            mcmc_config = opt_config.setdefault('mcmc', {})
            if 'n_chains' not in mcmc_config:
                mcmc_config['n_chains'] = 2  # Lower than default 4
                result.sources['optimization.mcmc.n_chains'] = 'conditional_default'
    
    def _get_base_defaults(self) -> Dict[str, Any]:
        """
        Get base default configuration.
        
        Returns:
            Base default configuration dictionary
        """
        return {
            'analysis_mode': 'auto-detect',
            'data': {
                'dataset_optimization': True
            },
            'optimization': {
                'vi': {
                    'n_iterations': 2000,
                    'learning_rate': 0.01,
                    'convergence_tol': 1e-6,
                    'n_elbo_samples': 1
                },
                'mcmc': {
                    'n_samples': 1000,
                    'n_warmup': 1000,
                    'n_chains': 4,
                    'target_accept_prob': 0.8
                },
                'hybrid': {
                    'use_vi_init': True,
                    'convergence_threshold': 0.1
                }
            },
            'hardware': {
                'force_cpu': False,
                'gpu_memory_fraction': 0.8
            },
            'plotting': {
                'generate_plots': True,
                'plot_experimental_data': False,
                'plot_simulated_data': False
            },
            'output': {
                'formats': ['yaml', 'npz'],
                'include_diagnostics': True
            }
        }
    
    def _log_merge_summary(self, result: MergeResult) -> None:
        """
        Log comprehensive merge summary.
        
        Args:
            result: Merge result to summarize
        """
        logger.debug(f"Configuration merge completed:")
        logger.debug(f"  Conflicts detected: {len(result.conflicts)}")
        logger.debug(f"  Warnings generated: {len(result.warnings)}")
        logger.debug(f"  Configuration keys: {len(result.sources)}")
        
        # Log conflicts
        for conflict in result.conflicts:
            logger.debug(f"  Conflict: {conflict.key_path} → {conflict.resolution}")
        
        # Log warnings
        for warning in result.warnings:
            logger.debug(f"  Warning: {warning}")
        
        # Summary of sources
        source_counts = {}
        for source in result.sources.values():
            source_counts[source] = source_counts.get(source, 0) + 1
        
        logger.info("Configuration sources: " + ", ".join(f"{src}({cnt})" for src, cnt in source_counts.items()))
    
    def get_configuration_diff(self,
                             config1: Dict[str, Any],
                             config2: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Get differences between two configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary of differences by category
        """
        differences = {
            'added': {},      # Keys in config2 but not config1
            'removed': {},    # Keys in config1 but not config2
            'changed': {}     # Keys with different values
        }
        
        self._find_differences(config1, config2, differences, "")
        
        return differences
    
    def _find_differences(self,
                         config1: Dict[str, Any],
                         config2: Dict[str, Any],
                         differences: Dict[str, Dict[str, Any]],
                         path: str) -> None:
        """
        Recursively find differences between configurations.
        
        Args:
            config1: First configuration
            config2: Second configuration
            differences: Differences dictionary to update
            path: Current key path
        """
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config1:
                differences['added'][current_path] = config2[key]
            elif key not in config2:
                differences['removed'][current_path] = config1[key]
            elif config1[key] != config2[key]:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    self._find_differences(config1[key], config2[key], differences, current_path)
                else:
                    differences['changed'][current_path] = {
                        'from': config1[key],
                        'to': config2[key]
                    }