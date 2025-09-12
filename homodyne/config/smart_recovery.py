"""
Smart Error Recovery and Configuration Healing System for Homodyne v2
=====================================================================

Intelligent error recovery system that automatically detects, diagnoses, and
attempts to fix common configuration issues. Provides fallback configurations,
automatic corrections, and guided recovery processes.

Key Features:
- Automatic detection of configuration problems
- Smart fallback and recovery strategies  
- Configuration healing with minimal user intervention
- Progressive error recovery (try simple fixes first)
- Backup and restore functionality
- Interactive recovery guidance
- Performance impact analysis during recovery

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    from .exceptions import (
        HomodyneConfigurationError, 
        ConfigurationFileError,
        ParameterValidationError,
        AnalysisModeError,
        ErrorContext,
        FixSuggestion,
        ConfigurationSuggestionEngine
    )
    from .manager import ConfigManager
    HAS_EXCEPTIONS = True
except ImportError:
    HAS_EXCEPTIONS = False
    HomodyneConfigurationError = Exception
    ConfigManager = None

try:
    from homodyne.utils.logging import get_logger
    HAS_UTILS_LOGGING = True
except ImportError:
    import logging
    HAS_UTILS_LOGGING = False
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class RecoveryResult:
    """Result of an error recovery attempt."""
    success: bool
    method_used: str
    description: str
    backup_created: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    performance_impact: Optional[str] = None
    time_taken: float = 0.0
    
    def __str__(self) -> str:
        status = "✅ Success" if self.success else "❌ Failed"
        result = f"{status}: {self.description}\n"
        result += f"Method: {self.method_used}\n"
        result += f"Time taken: {self.time_taken:.2f}s\n"
        
        if self.backup_created:
            result += f"Backup: {self.backup_created}\n"
        
        if self.performance_impact:
            result += f"Performance impact: {self.performance_impact}\n"
        
        if self.warnings:
            result += "Warnings:\n"
            for warning in self.warnings:
                result += f"  • {warning}\n"
        
        return result.strip()


@dataclass
class RecoveryStrategy:
    """A strategy for recovering from a specific type of error."""
    name: str
    description: str
    applicability_check: Callable[[Exception, Dict[str, Any]], bool]
    recovery_function: Callable[[Exception, Dict[str, Any], Path], RecoveryResult] 
    priority: int = 50  # Lower numbers = higher priority
    requires_backup: bool = True
    estimated_time: str = "1-2 minutes"
    user_interaction: bool = False


class SmartRecoveryEngine:
    """Intelligent configuration recovery system."""
    
    def __init__(self, config_path: Union[str, Path], interactive_mode: bool = True):
        self.config_path = Path(config_path)
        self.interactive_mode = interactive_mode
        self.backup_dir = Path.home() / ".homodyne" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize suggestion engine
        self.suggestion_engine = ConfigurationSuggestionEngine() if HAS_EXCEPTIONS else None
        
        # Register recovery strategies
        self.strategies = []
        self._register_default_strategies()
        
        logger.info(f"Smart recovery engine initialized for {config_path}")
    
    def _register_default_strategies(self):
        """Register built-in recovery strategies."""
        
        # Strategy 1: File not found - create from template
        self.strategies.append(RecoveryStrategy(
            name="create_from_template",
            description="Create missing configuration file from template", 
            applicability_check=lambda e, c: isinstance(e, FileNotFoundError),
            recovery_function=self._create_from_template,
            priority=10,
            requires_backup=False,
            estimated_time="1 minute",
            user_interaction=True
        ))
        
        # Strategy 2: YAML syntax error - attempt auto-fix
        self.strategies.append(RecoveryStrategy(
            name="fix_yaml_syntax", 
            description="Attempt to automatically fix YAML syntax errors",
            applicability_check=lambda e, c: "yaml" in str(type(e).__name__).lower(),
            recovery_function=self._fix_yaml_syntax,
            priority=20,
            estimated_time="30 seconds"
        ))
        
        # Strategy 3: Parameter validation error - auto-correct values
        self.strategies.append(RecoveryStrategy(
            name="correct_parameters",
            description="Automatically correct invalid parameter values",
            applicability_check=lambda e, c: (
                HAS_EXCEPTIONS and isinstance(e, ParameterValidationError)
            ),
            recovery_function=self._correct_parameters,
            priority=30,
            estimated_time="1 minute"
        ))
        
        # Strategy 4: Mode mismatch - adjust mode or parameters
        self.strategies.append(RecoveryStrategy(
            name="fix_mode_mismatch",
            description="Fix analysis mode and parameter count mismatches", 
            applicability_check=lambda e, c: (
                HAS_EXCEPTIONS and isinstance(e, AnalysisModeError)
            ),
            recovery_function=self._fix_mode_mismatch,
            priority=40,
            estimated_time="2 minutes"
        ))
        
        # Strategy 5: Missing sections - add with defaults
        self.strategies.append(RecoveryStrategy(
            name="add_missing_sections",
            description="Add missing configuration sections with default values",
            applicability_check=lambda e, c: "missing" in str(e).lower(),
            recovery_function=self._add_missing_sections,
            priority=50,
            estimated_time="1 minute"
        ))
        
        # Strategy 6: Invalid keys - suggest corrections
        self.strategies.append(RecoveryStrategy(
            name="fix_invalid_keys",
            description="Fix typos and invalid configuration keys",
            applicability_check=lambda e, c: any(key in str(e).lower() for key in ["invalid", "unknown", "key"]),
            recovery_function=self._fix_invalid_keys,
            priority=60,
            estimated_time="2 minutes"
        ))
        
        # Strategy 7: Configuration too complex - simplify
        self.strategies.append(RecoveryStrategy(
            name="simplify_configuration",
            description="Simplify overly complex configuration to basic working state",
            applicability_check=lambda e, c: "complex" in str(e).lower() or len(str(e)) > 500,
            recovery_function=self._simplify_configuration,
            priority=70,
            estimated_time="3 minutes",
            user_interaction=True
        ))
        
        # Strategy 8: Last resort - create minimal config
        self.strategies.append(RecoveryStrategy(
            name="create_minimal",
            description="Create minimal working configuration (last resort)",
            applicability_check=lambda e, c: True,  # Always applicable
            recovery_function=self._create_minimal_config,
            priority=100,
            estimated_time="30 seconds",
            user_interaction=True
        ))
    
    def recover_from_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """Attempt to recover from the given configuration error."""
        start_time = time.time()
        context = context or {}
        
        logger.info(f"Attempting recovery from error: {type(error).__name__}")
        logger.debug(f"Error details: {error}")
        
        # Sort strategies by priority
        applicable_strategies = [
            s for s in sorted(self.strategies, key=lambda x: x.priority)
            if s.applicability_check(error, context)
        ]
        
        if not applicable_strategies:
            return RecoveryResult(
                success=False,
                method_used="none",
                description="No applicable recovery strategies found",
                time_taken=time.time() - start_time
            )
        
        logger.info(f"Found {len(applicable_strategies)} applicable recovery strategies")
        
        # Try strategies in order of priority
        for strategy in applicable_strategies:
            logger.info(f"Trying recovery strategy: {strategy.name}")
            
            # Ask user permission for interactive strategies
            if strategy.user_interaction and self.interactive_mode:
                if not self._ask_user_permission(strategy):
                    logger.info(f"User declined strategy: {strategy.name}")
                    continue
            
            # Create backup if required
            backup_path = None
            if strategy.requires_backup and self.config_path.exists():
                backup_path = self._create_backup()
                if backup_path is None:
                    logger.warning(f"Failed to create backup, skipping {strategy.name}")
                    continue
            
            try:
                result = strategy.recovery_function(error, context, self.config_path)
                result.backup_created = str(backup_path) if backup_path else None
                result.time_taken = time.time() - start_time
                
                if result.success:
                    logger.info(f"Recovery successful using strategy: {strategy.name}")
                    return result
                else:
                    logger.warning(f"Recovery strategy failed: {strategy.name} - {result.description}")
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.name} raised exception: {recovery_error}")
                # Restore backup if recovery failed
                if backup_path and self.config_path.exists():
                    self._restore_backup(backup_path)
        
        # All strategies failed
        return RecoveryResult(
            success=False,
            method_used="all_strategies_failed",
            description=f"All {len(applicable_strategies)} recovery strategies failed",
            time_taken=time.time() - start_time
        )
    
    def _ask_user_permission(self, strategy: RecoveryStrategy) -> bool:
        """Ask user permission for interactive recovery strategies."""
        if not self.interactive_mode:
            return True
        
        print(f"\n🔧 Recovery Strategy Available: {strategy.name}")
        print(f"Description: {strategy.description}")
        print(f"Estimated time: {strategy.estimated_time}")
        
        if strategy.requires_backup:
            print("⚠️  This will modify your configuration file (backup will be created)")
        
        while True:
            response = input("Proceed with this recovery strategy? [y/N/s(skip)]: ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no', 's', 'skip', '']:
                return False
            else:
                print("Please enter 'y' for yes, 'n' for no, or 's' to skip")
    
    def _create_backup(self) -> Optional[Path]:
        """Create backup of current configuration file."""
        if not self.config_path.exists():
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{self.config_path.stem}_backup_{timestamp}{self.config_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        try:
            shutil.copy2(self.config_path, backup_path)
            logger.info(f"Configuration backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def _restore_backup(self, backup_path: Path) -> bool:
        """Restore configuration from backup."""
        try:
            shutil.copy2(backup_path, self.config_path)
            logger.info(f"Configuration restored from backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    def _create_from_template(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Create configuration file from template."""
        logger.info("Creating configuration from template")
        
        # Determine appropriate template based on context
        analysis_mode = context.get("analysis_mode", "static_isotropic")
        
        try:
            # Try to import template generation
            from .exceptions import generate_configuration_example
            
            config_content = generate_configuration_example(analysis_mode)
            
            # Write to file
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(config_content)
            
            return RecoveryResult(
                success=True,
                method_used="create_from_template",
                description=f"Created {analysis_mode} configuration from template",
                performance_impact="No performance impact"
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="create_from_template", 
                description=f"Failed to create template: {e}"
            )
    
    def _fix_yaml_syntax(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Attempt to fix common YAML syntax errors."""
        logger.info("Attempting to fix YAML syntax errors")
        
        if not config_path.exists():
            return RecoveryResult(
                success=False,
                method_used="fix_yaml_syntax",
                description="Configuration file does not exist"
            )
        
        try:
            # Read file content
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            fixes_applied = []
            
            # Fix 1: Remove tabs (replace with spaces)
            if '\t' in content:
                content = content.replace('\t', '    ')
                fixes_applied.append("Replaced tabs with spaces")
            
            # Fix 2: Fix common indentation issues
            lines = content.split('\n')
            fixed_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    # Ensure proper indentation for key-value pairs
                    if ':' in stripped and not line.startswith(' '):
                        # Top-level key
                        fixed_lines.append(stripped)
                    elif ':' in stripped:
                        # Ensure consistent indentation
                        indent = len(line) - len(line.lstrip())
                        if indent % 2 != 0:  # Make indent even
                            fixed_lines.append(' ' * (indent + 1) + stripped)
                            if "indentation" not in str(fixes_applied):
                                fixes_applied.append("Fixed inconsistent indentation")
                        else:
                            fixed_lines.append(line)
                    else:
                        fixed_lines.append(line)
                else:
                    fixed_lines.append(line)
            
            content = '\n'.join(fixed_lines)
            
            # Fix 3: Quote problematic strings
            problematic_patterns = [
                (r':\s*([^"\s][^:\n]*[:\[\]{}])', r': "\1"'),  # Quote strings with special chars
                (r':\s*([^"\s\n].*\s.*[^"\n])', r': "\1"'),   # Quote strings with spaces
            ]
            
            for pattern, replacement in problematic_patterns:
                import re
                new_content = re.sub(pattern, replacement, content)
                if new_content != content:
                    content = new_content
                    fixes_applied.append("Added quotes to problematic strings")
            
            # Test if YAML is valid now
            if HAS_YAML:
                try:
                    yaml.safe_load(content)
                except yaml.YAMLError as yaml_error:
                    return RecoveryResult(
                        success=False,
                        method_used="fix_yaml_syntax",
                        description=f"Could not fix YAML syntax: {yaml_error}",
                        warnings=[f"Applied fixes: {', '.join(fixes_applied)}"]
                    )
            
            # Write fixed content
            if content != original_content:
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return RecoveryResult(
                    success=True,
                    method_used="fix_yaml_syntax",
                    description="Fixed YAML syntax errors",
                    warnings=fixes_applied,
                    performance_impact="No performance impact"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="fix_yaml_syntax", 
                    description="No fixable YAML syntax issues found"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="fix_yaml_syntax",
                description=f"Failed to fix YAML syntax: {e}"
            )
    
    def _correct_parameters(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Automatically correct invalid parameter values."""
        logger.info("Attempting to correct parameter values")
        
        if not HAS_EXCEPTIONS or not isinstance(error, ParameterValidationError):
            return RecoveryResult(
                success=False,
                method_used="correct_parameters",
                description="Not a parameter validation error"
            )
        
        try:
            # Load current configuration
            config_data = self._load_config_data(config_path)
            if config_data is None:
                return RecoveryResult(
                    success=False,
                    method_used="correct_parameters",
                    description="Could not load configuration data"
                )
            
            corrections_made = []
            
            # Get parameter information from error
            param_name = error.parameter_name
            param_value = error.value
            
            # Parameter-specific corrections
            if param_name == "wavevector_q":
                if param_value <= 0:
                    new_value = 0.0054  # Typical XPCS value
                    self._set_nested_value(config_data, ["analyzer_parameters", "scattering", "wavevector_q"], new_value)
                    corrections_made.append(f"Set wavevector_q to {new_value} (typical XPCS value)")
                elif param_value > 1.0:
                    new_value = 0.1  # Upper end of typical range
                    self._set_nested_value(config_data, ["analyzer_parameters", "scattering", "wavevector_q"], new_value)
                    corrections_made.append(f"Reduced wavevector_q to {new_value} (within typical range)")
            
            elif param_name == "dt":
                if param_value <= 0:
                    new_value = 0.1  # Typical time step
                    self._set_nested_value(config_data, ["analyzer_parameters", "temporal", "dt"], new_value)
                    corrections_made.append(f"Set dt to {new_value} seconds")
            
            elif param_name in ["start_frame", "end_frame"]:
                start = self._get_nested_value(config_data, ["analyzer_parameters", "temporal", "start_frame"], 1)
                end = self._get_nested_value(config_data, ["analyzer_parameters", "temporal", "end_frame"], 100)
                
                if start >= end:
                    new_end = start + 1000
                    self._set_nested_value(config_data, ["analyzer_parameters", "temporal", "end_frame"], new_end)
                    corrections_made.append(f"Set end_frame to {new_end} (start_frame + 1000)")
            
            elif param_name == "D0":
                if param_value <= 0:
                    new_value = 100.0
                    self._update_parameter_in_list(config_data, "D0", new_value)
                    corrections_made.append(f"Set D0 to {new_value}")
            
            elif param_name == "alpha":
                if abs(param_value) > 2.0:
                    new_value = 0.0 if param_value > 0 else -1.0
                    self._update_parameter_in_list(config_data, "alpha", new_value)
                    corrections_made.append(f"Set alpha to {new_value} (within [-2, 2])")
            
            # Save corrected configuration
            if corrections_made:
                self._save_config_data(config_path, config_data)
                
                return RecoveryResult(
                    success=True,
                    method_used="correct_parameters",
                    description="Corrected invalid parameter values",
                    warnings=corrections_made,
                    performance_impact="No performance impact"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="correct_parameters",
                    description="No correctable parameter issues found"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="correct_parameters",
                description=f"Failed to correct parameters: {e}"
            )
    
    def _fix_mode_mismatch(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Fix analysis mode and parameter count mismatches."""
        logger.info("Attempting to fix mode mismatch")
        
        if not HAS_EXCEPTIONS or not isinstance(error, AnalysisModeError):
            return RecoveryResult(
                success=False,
                method_used="fix_mode_mismatch",
                description="Not a mode mismatch error"
            )
        
        try:
            config_data = self._load_config_data(config_path)
            if config_data is None:
                return RecoveryResult(
                    success=False,
                    method_used="fix_mode_mismatch",
                    description="Could not load configuration data"
                )
            
            # Get current parameter count
            initial_params = config_data.get("initial_parameters", {})
            values = initial_params.get("values", [])
            param_count = len(values)
            
            corrections_made = []
            
            if param_count == 3:
                # Set to static mode
                self._set_nested_value(config_data, ["analysis_settings", "static_mode"], True)
                self._set_nested_value(config_data, ["analysis_settings", "static_submode"], "anisotropic")
                
                # Disable angle filtering for isotropic if only 3 params and user wants simple
                if self.interactive_mode:
                    print("\n📊 You have 3 parameters. Choose static mode:")
                    print("1. Isotropic (simplest, fastest)")
                    print("2. Anisotropic (more features)")
                    choice = input("Choice [1]: ").strip() or "1"
                    if choice == "1":
                        self._set_nested_value(config_data, ["analysis_settings", "static_submode"], "isotropic")
                        self._set_nested_value(config_data, ["optimization_config", "angle_filtering", "enabled"], False)
                        corrections_made.append("Set to static isotropic mode")
                    else:
                        corrections_made.append("Set to static anisotropic mode")
                else:
                    corrections_made.append("Set to static anisotropic mode")
                
                # Update parameter names
                param_names = ["D0", "alpha", "D_offset"]
                initial_params["parameter_names"] = param_names
                corrections_made.append("Updated parameter names for static mode")
                
            elif param_count == 7:
                # Set to laminar flow mode
                self._set_nested_value(config_data, ["analysis_settings", "static_mode"], False)
                
                # Update parameter names
                param_names = ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
                initial_params["parameter_names"] = param_names
                
                corrections_made.append("Set to laminar flow mode")
                corrections_made.append("Updated parameter names for laminar flow mode")
                
            else:
                # Unknown parameter count, adjust to 3 (static)
                default_values = [100.0, 0.0, 10.0]
                default_names = ["D0", "alpha", "D_offset"]
                
                initial_params["values"] = default_values
                initial_params["parameter_names"] = default_names
                
                self._set_nested_value(config_data, ["analysis_settings", "static_mode"], True)
                self._set_nested_value(config_data, ["analysis_settings", "static_submode"], "anisotropic")
                
                corrections_made.append(f"Adjusted parameter count from {param_count} to 3")
                corrections_made.append("Set to static anisotropic mode")
            
            # Save corrected configuration
            if corrections_made:
                self._save_config_data(config_path, config_data)
                
                return RecoveryResult(
                    success=True,
                    method_used="fix_mode_mismatch",
                    description="Fixed analysis mode and parameter mismatch",
                    warnings=corrections_made,
                    performance_impact="Mode changes may affect computation time"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="fix_mode_mismatch",
                    description="No fixable mode mismatch issues found"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="fix_mode_mismatch",
                description=f"Failed to fix mode mismatch: {e}"
            )
    
    def _add_missing_sections(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Add missing configuration sections with default values."""
        logger.info("Adding missing configuration sections")
        
        try:
            config_data = self._load_config_data(config_path)
            if config_data is None:
                config_data = {}
            
            sections_added = []
            
            # Define required sections with defaults
            required_sections = {
                "metadata": {
                    "config_version": "2.0",
                    "description": "Auto-generated configuration"
                },
                "analyzer_parameters": {
                    "temporal": {"dt": 0.1, "start_frame": 1, "end_frame": 1000},
                    "scattering": {"wavevector_q": 0.0054},
                    "geometry": {"stator_rotor_gap": 2000000}
                },
                "experimental_data": {
                    "data_folder_path": "./data/",
                    "data_file_name": "data.hdf"
                },
                "analysis_settings": {
                    "static_mode": True,
                    "static_submode": "anisotropic"
                },
                "initial_parameters": {
                    "values": [100.0, 0.0, 10.0],
                    "parameter_names": ["D0", "alpha", "D_offset"]
                },
                "optimization_config": {
                    "angle_filtering": {"enabled": True}
                },
                "logging": {
                    "log_to_console": True,
                    "level": "INFO"
                }
            }
            
            # Add missing sections
            for section_name, default_content in required_sections.items():
                if section_name not in config_data:
                    config_data[section_name] = default_content
                    sections_added.append(section_name)
            
            # Save updated configuration
            if sections_added:
                self._save_config_data(config_path, config_data)
                
                return RecoveryResult(
                    success=True,
                    method_used="add_missing_sections",
                    description="Added missing configuration sections",
                    warnings=[f"Added sections: {', '.join(sections_added)}"],
                    performance_impact="No performance impact"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="add_missing_sections",
                    description="No missing sections detected"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="add_missing_sections",
                description=f"Failed to add missing sections: {e}"
            )
    
    def _fix_invalid_keys(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Fix typos and invalid configuration keys."""
        logger.info("Fixing invalid configuration keys")
        
        if not self.suggestion_engine:
            return RecoveryResult(
                success=False,
                method_used="fix_invalid_keys",
                description="Suggestion engine not available"
            )
        
        try:
            config_data = self._load_config_data(config_path)
            if config_data is None:
                return RecoveryResult(
                    success=False,
                    method_used="fix_invalid_keys",
                    description="Could not load configuration data"
                )
            
            # Get auto-correction suggestions
            suggestions = self.suggestion_engine.generate_autocorrection_suggestions(config_data)
            corrections_made = []
            
            # Apply simple key corrections
            for key in list(config_data.keys()):
                if key in self.suggestion_engine.common_typos:
                    correct_key = self.suggestion_engine.common_typos[key]
                    config_data[correct_key] = config_data.pop(key)
                    corrections_made.append(f"Renamed '{key}' to '{correct_key}'")
            
            # Fix mode aliases
            analysis_settings = config_data.get("analysis_settings", {})
            if isinstance(analysis_settings, dict):
                mode = analysis_settings.get("static_submode")
                if mode in self.suggestion_engine.mode_aliases:
                    correct_mode = self.suggestion_engine.mode_aliases[mode]
                    analysis_settings["static_submode"] = correct_mode
                    corrections_made.append(f"Changed mode from '{mode}' to '{correct_mode}'")
            
            # Save corrected configuration
            if corrections_made:
                self._save_config_data(config_path, config_data)
                
                return RecoveryResult(
                    success=True,
                    method_used="fix_invalid_keys",
                    description="Fixed invalid configuration keys",
                    warnings=corrections_made,
                    performance_impact="No performance impact"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="fix_invalid_keys",
                    description="No fixable key issues found"
                )
                
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="fix_invalid_keys",
                description=f"Failed to fix invalid keys: {e}"
            )
    
    def _simplify_configuration(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Simplify overly complex configuration."""
        logger.info("Simplifying complex configuration")
        
        try:
            # Create minimal configuration based on detected mode
            analysis_mode = context.get("analysis_mode", "static_isotropic")
            
            from .exceptions import generate_configuration_example
            simplified_config = generate_configuration_example(analysis_mode)
            
            # If user interaction is enabled, ask which sections to keep
            if self.interactive_mode:
                config_data = self._load_config_data(config_path)
                if config_data:
                    print("\n🔧 Configuration Simplification")
                    print("Current configuration is complex. Which sections would you like to preserve?")
                    print("1. Keep experimental_data paths")
                    print("2. Keep logging settings") 
                    print("3. Keep parameter values")
                    print("4. Start completely fresh")
                    
                    choice = input("Choice [1,2,3]: ").strip() or "1,2,3"
                    preserve_sections = choice.split(",")
                    
                    # Parse simplified config and merge preserved sections
                    if HAS_YAML:
                        new_config = yaml.safe_load(simplified_config)
                    else:
                        new_config = json.loads(simplified_config)
                    
                    if "1" in preserve_sections and "experimental_data" in config_data:
                        new_config["experimental_data"] = config_data["experimental_data"]
                    
                    if "2" in preserve_sections and "logging" in config_data:
                        new_config["logging"] = config_data["logging"]
                    
                    if "3" in preserve_sections and "initial_parameters" in config_data:
                        new_config["initial_parameters"] = config_data["initial_parameters"]
                    
                    # Convert back to string
                    if HAS_YAML:
                        simplified_config = yaml.dump(new_config, default_flow_style=False)
                    else:
                        simplified_config = json.dumps(new_config, indent=2)
            
            # Write simplified configuration
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(simplified_config)
            
            return RecoveryResult(
                success=True,
                method_used="simplify_configuration",
                description=f"Simplified configuration to {analysis_mode} template",
                performance_impact="Simpler config may improve performance",
                warnings=["Complex settings were removed - you may need to re-add specific customizations"]
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="simplify_configuration",
                description=f"Failed to simplify configuration: {e}"
            )
    
    def _create_minimal_config(self, error: Exception, context: Dict[str, Any], config_path: Path) -> RecoveryResult:
        """Create minimal working configuration as last resort."""
        logger.info("Creating minimal configuration (last resort)")
        
        try:
            minimal_config = """
# Minimal Homodyne Configuration (Auto-Generated)
metadata:
  config_version: "2.0"
  description: "Minimal auto-generated configuration"

analyzer_parameters:
  temporal:
    dt: 0.1
    start_frame: 1
    end_frame: 1000
  scattering:
    wavevector_q: 0.0054
  geometry:
    stator_rotor_gap: 2000000

experimental_data:
  data_folder_path: "./data/"
  data_file_name: "your_data.hdf"

analysis_settings:
  static_mode: true
  static_submode: "isotropic"

initial_parameters:
  values: [100.0, 0.0, 10.0]
  parameter_names: ["D0", "alpha", "D_offset"]

optimization_config:
  angle_filtering:
    enabled: false

logging:
  log_to_console: true
  level: "INFO"
            """
            
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(minimal_config.strip())
            
            return RecoveryResult(
                success=True,
                method_used="create_minimal_config",
                description="Created minimal working configuration",
                performance_impact="Minimal config optimized for performance",
                warnings=[
                    "This is a minimal configuration - you will need to customize it for your specific needs",
                    "Update data_file_name and data_folder_path to match your experiment",
                    "Adjust parameters as needed for your analysis"
                ]
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="create_minimal_config",
                description=f"Failed to create minimal config: {e}"
            )
    
    def _load_config_data(self, config_path: Path) -> Optional[Dict[str, Any]]:
        """Load configuration data from file."""
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml'] and HAS_YAML:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config data: {e}")
            return None
    
    def _save_config_data(self, config_path: Path, config_data: Dict[str, Any]) -> bool:
        """Save configuration data to file."""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml'] and HAS_YAML:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config data: {e}")
            return False
    
    def _get_nested_value(self, data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
        """Get nested dictionary value."""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def _set_nested_value(self, data: Dict[str, Any], keys: List[str], value: Any) -> None:
        """Set nested dictionary value."""
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def _update_parameter_in_list(self, config_data: Dict[str, Any], param_name: str, new_value: float) -> None:
        """Update parameter value in initial_parameters list."""
        initial_params = config_data.get("initial_parameters", {})
        param_names = initial_params.get("parameter_names", [])
        values = initial_params.get("values", [])
        
        if param_name in param_names:
            index = param_names.index(param_name)
            if index < len(values):
                values[index] = new_value
    
    def get_backup_history(self) -> List[Dict[str, Any]]:
        """Get list of available configuration backups."""
        backups = []
        if not self.backup_dir.exists():
            return backups
        
        for backup_file in self.backup_dir.glob("*_backup_*.yaml"):
            try:
                stat = backup_file.stat()
                backups.append({
                    "file": str(backup_file),
                    "name": backup_file.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except OSError:
                continue
        
        return sorted(backups, key=lambda x: x["created"], reverse=True)
    
    def restore_from_backup(self, backup_name: str) -> RecoveryResult:
        """Restore configuration from a specific backup."""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            return RecoveryResult(
                success=False,
                method_used="restore_from_backup",
                description=f"Backup file not found: {backup_name}"
            )
        
        try:
            if self._restore_backup(backup_path):
                return RecoveryResult(
                    success=True,
                    method_used="restore_from_backup",
                    description=f"Configuration restored from {backup_name}",
                    performance_impact="No performance impact"
                )
            else:
                return RecoveryResult(
                    success=False,
                    method_used="restore_from_backup",
                    description=f"Failed to restore from {backup_name}"
                )
        except Exception as e:
            return RecoveryResult(
                success=False,
                method_used="restore_from_backup",
                description=f"Error during backup restoration: {e}"
            )


def create_smart_recovery_engine(config_path: Union[str, Path], interactive: bool = True) -> SmartRecoveryEngine:
    """
    Create a smart recovery engine for the given configuration.
    
    Args:
        config_path: Path to configuration file
        interactive: Whether to allow interactive recovery
    
    Returns:
        SmartRecoveryEngine instance
    """
    return SmartRecoveryEngine(config_path, interactive)


def auto_recover_configuration(config_path: Union[str, Path], 
                             error: Optional[Exception] = None,
                             interactive: bool = True) -> RecoveryResult:
    """
    Convenience function to automatically recover from configuration errors.
    
    Args:
        config_path: Path to configuration file
        error: The error to recover from (if None, will attempt general fixes)
        interactive: Whether to allow interactive recovery
    
    Returns:
        RecoveryResult indicating success/failure
    """
    engine = create_smart_recovery_engine(config_path, interactive)
    
    if error is None:
        # Try to load configuration and see what fails
        try:
            if ConfigManager:
                ConfigManager(str(config_path))
            return RecoveryResult(
                success=True,
                method_used="auto_recover",
                description="Configuration loaded successfully, no recovery needed"
            )
        except Exception as auto_detected_error:
            error = auto_detected_error
    
    return engine.recover_from_error(error)