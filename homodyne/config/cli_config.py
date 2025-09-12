"""
CLI Configuration Integration for Homodyne v2
============================================

Integrates CLI arguments with the existing configuration system,
providing CLI-specific overrides and effective configuration generation.

Key Features:
- CLI argument to config parameter mapping
- Effective configuration generation (CLI + file)
- Configuration validation and defaults
- Analysis mode resolution integration
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union

from homodyne.utils.logging import get_logger
from homodyne.config.manager import ConfigManager
from homodyne.config.modes import detect_analysis_mode

logger = get_logger(__name__)


class CLIConfigManager:
    """
    CLI-specific configuration manager.
    
    Handles integration between CLI arguments and file-based configuration,
    generating effective configurations for analysis execution.
    """
    
    def __init__(self):
        """Initialize CLI configuration manager."""
        self.base_config_manager: Optional[ConfigManager] = None
        self.effective_config: Optional[Dict[str, Any]] = None
    
    def create_effective_config(self, 
                              config_file: Optional[str],
                              args: argparse.Namespace) -> Dict[str, Any]:
        """
        Create effective configuration from CLI args and config file.
        
        Args:
            config_file: Path to configuration file (optional)
            args: Parsed CLI arguments
            
        Returns:
            Effective configuration dictionary
        """
        logger.info("Building effective configuration from CLI and file")
        
        # Load base configuration from file if provided
        if config_file and Path(config_file).exists():
            logger.info(f"Loading base configuration from: {config_file}")
            self.base_config_manager = ConfigManager(config_file)
            base_config = self.base_config_manager.config
        else:
            logger.info("No config file provided, using CLI arguments only")
            base_config = self._get_default_config()
        
        # Apply CLI overrides
        cli_overrides = self._extract_cli_overrides(args)
        self.effective_config = self._merge_configurations(base_config, cli_overrides)
        
        # Resolve analysis mode
        self._resolve_analysis_mode(args)
        
        # Validate effective configuration
        self._validate_effective_config()
        
        logger.info(f"✓ Effective configuration created: {self._get_config_summary()}")
        return self.effective_config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration structure.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'analysis_mode': 'auto-detect',
            'data': {
                'file_path': None,
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
    
    def _extract_cli_overrides(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Extract configuration overrides from CLI arguments.
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            CLI overrides dictionary
        """
        overrides = {}
        
        # Data file path
        if hasattr(args, 'data_file') and args.data_file:
            overrides['data'] = {'file_path': args.data_file}
        
        # Hardware settings
        hardware_config = {}
        if hasattr(args, 'force_cpu') and args.force_cpu:
            hardware_config['force_cpu'] = True
        if hasattr(args, 'gpu_memory_fraction') and args.gpu_memory_fraction != 0.8:
            hardware_config['gpu_memory_fraction'] = args.gpu_memory_fraction
        if hardware_config:
            overrides['hardware'] = hardware_config
        
        # Dataset optimization
        if hasattr(args, 'disable_dataset_optimization') and args.disable_dataset_optimization:
            if 'data' not in overrides:
                overrides['data'] = {}
            overrides['data']['dataset_optimization'] = False
        
        # Plotting settings
        plotting_config = {}
        if hasattr(args, 'plot_experimental_data') and args.plot_experimental_data:
            plotting_config['plot_experimental_data'] = True
        if hasattr(args, 'plot_simulated_data') and args.plot_simulated_data:
            plotting_config['plot_simulated_data'] = True
        if plotting_config:
            overrides['plotting'] = plotting_config
        
        # Analysis mode flags
        if hasattr(args, 'static_isotropic') and args.static_isotropic:
            overrides['analysis_mode'] = 'static_isotropic'
        elif hasattr(args, 'static_anisotropic') and args.static_anisotropic:
            overrides['analysis_mode'] = 'static_anisotropic'
        elif hasattr(args, 'laminar_flow') and args.laminar_flow:
            overrides['analysis_mode'] = 'laminar_flow'
        
        # Optimization parameters
        optimization_overrides = self._extract_optimization_overrides(args)
        if optimization_overrides:
            overrides['optimization'] = optimization_overrides
        
        # Custom phi angles
        if hasattr(args, 'phi_angles') and args.phi_angles:
            if 'data' not in overrides:
                overrides['data'] = {}
            overrides['data']['custom_phi_angles'] = args.phi_angles
        
        logger.debug(f"CLI overrides extracted: {list(overrides.keys())}")
        return overrides
    
    def _extract_optimization_overrides(self, args: argparse.Namespace) -> Dict[str, Any]:
        """
        Extract optimization-specific overrides from CLI arguments.
        
        Args:
            args: Parsed CLI arguments
            
        Returns:
            Optimization overrides dictionary
        """
        opt_overrides = {}
        
        # VI parameters
        vi_overrides = {}
        if hasattr(args, 'vi_iterations') and args.vi_iterations:
            vi_overrides['n_iterations'] = args.vi_iterations
        if hasattr(args, 'vi_learning_rate') and args.vi_learning_rate:
            vi_overrides['learning_rate'] = args.vi_learning_rate
        if vi_overrides:
            opt_overrides['vi'] = vi_overrides
        
        # MCMC parameters
        mcmc_overrides = {}
        if hasattr(args, 'mcmc_samples') and args.mcmc_samples:
            mcmc_overrides['n_samples'] = args.mcmc_samples
        if hasattr(args, 'mcmc_warmup') and args.mcmc_warmup:
            mcmc_overrides['n_warmup'] = args.mcmc_warmup
        if hasattr(args, 'mcmc_chains') and args.mcmc_chains:
            mcmc_overrides['n_chains'] = args.mcmc_chains
        if mcmc_overrides:
            opt_overrides['mcmc'] = mcmc_overrides
        
        return opt_overrides
    
    def _merge_configurations(self, 
                            base_config: Dict[str, Any],
                            cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with CLI overrides.
        
        Args:
            base_config: Base configuration from file
            cli_overrides: CLI argument overrides
            
        Returns:
            Merged configuration
        """
        merged_config = base_config.copy()
        
        def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively update nested dictionaries."""
            for key, value in overrides.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = _deep_update(base[key], value)
                else:
                    base[key] = value
            return base
        
        return _deep_update(merged_config, cli_overrides)
    
    def _resolve_analysis_mode(self, args: argparse.Namespace) -> None:
        """
        Resolve analysis mode using mode resolver.
        
        Args:
            args: CLI arguments for context
        """
        from homodyne.config.mode_resolver import ModeResolver
        
        resolver = ModeResolver()
        resolved_mode = resolver.resolve_mode(
            config=self.effective_config,
            cli_args=args
        )
        
        if resolved_mode != self.effective_config.get('analysis_mode'):
            logger.info(f"Analysis mode resolved: {self.effective_config.get('analysis_mode')} → {resolved_mode}")
            self.effective_config['analysis_mode'] = resolved_mode
    
    def _validate_effective_config(self) -> None:
        """Validate the effective configuration."""
        from homodyne.config.parameter_validator import ParameterValidator
        
        validator = ParameterValidator()
        validation_result = validator.validate_config(self.effective_config)
        
        if not validation_result.is_valid:
            error_msg = f"Configuration validation failed: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    def _get_config_summary(self) -> str:
        """
        Get configuration summary for logging.
        
        Returns:
            Configuration summary string
        """
        if not self.effective_config:
            return "No configuration"
        
        summary_parts = [
            f"mode={self.effective_config.get('analysis_mode', 'unknown')}"
        ]
        
        # Hardware info
        hardware = self.effective_config.get('hardware', {})
        if hardware.get('force_cpu'):
            summary_parts.append("cpu-only")
        else:
            summary_parts.append(f"gpu({hardware.get('gpu_memory_fraction', 0.8):.1f})")
        
        # Dataset optimization
        data_config = self.effective_config.get('data', {})
        if data_config.get('dataset_optimization', True):
            summary_parts.append("opt-enabled")
        else:
            summary_parts.append("opt-disabled")
        
        return ", ".join(summary_parts)
    
    def save_effective_config(self, output_path: Union[str, Path]) -> None:
        """
        Save effective configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if not self.effective_config:
            raise ValueError("No effective configuration to save")
        
        try:
            import yaml
            from datetime import datetime
            
            output_path = Path(output_path)
            
            # Add metadata
            config_with_metadata = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'source': 'CLI + file configuration',
                    'homodyne_version': getattr(__import__('homodyne'), '__version__', 'unknown')
                },
                'configuration': self.effective_config
            }
            
            # Save as YAML
            with open(output_path, 'w') as f:
                yaml.dump(config_with_metadata, f, default_flow_style=False, indent=2)
            
            logger.debug(f"Effective configuration saved to: {output_path}")
            
        except Exception as e:
            logger.warning(f"Could not save effective configuration: {e}")
    
    def get_analysis_mode(self) -> str:
        """
        Get resolved analysis mode.
        
        Returns:
            Analysis mode string
        """
        if not self.effective_config:
            return 'auto-detect'
        return self.effective_config.get('analysis_mode', 'auto-detect')
    
    def get_parameter_count(self) -> int:
        """
        Get effective parameter count for the analysis mode.
        
        Returns:
            Number of parameters (3 or 7)
        """
        mode = self.get_analysis_mode()
        if mode in ['static_isotropic', 'static_anisotropic']:
            return 3
        elif mode == 'laminar_flow':
            return 7
        else:
            # Auto-detect mode - return default
            return 3