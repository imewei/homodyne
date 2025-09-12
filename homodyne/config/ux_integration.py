"""
User Experience Integration Layer for Homodyne v2 Configuration System
=====================================================================

Integration layer that combines all the user experience enhancements into a
cohesive, easy-to-use interface. Provides unified access to all configuration
tools and utilities with intelligent defaults and progressive disclosure.

Key Features:
- Unified interface for all configuration tools
- Intelligent workflow guidance
- Progressive disclosure of complexity
- Context-aware assistance
- Seamless integration between components
- Smart defaults and recommendations
- Comprehensive error handling and recovery

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Import all the enhanced components
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
    from .smart_recovery import SmartRecoveryEngine, create_smart_recovery_engine, auto_recover_configuration
    from .interactive_helpers import (
        InteractiveConfigurationBuilder,
        ConfigurationValidator,
        ConfigurationComparator,
        create_interactive_builder,
        validate_configuration_file,
        compare_configuration_files
    )
    from .enhanced_console_output import (
        EnhancedConsoleLogger,
        ConfigurationProgressTracker,
        ValidationFeedbackSystem,
        InteractivePromptSystem,
        ConsoleTheme,
        create_console_logger,
        create_progress_tracker,
        create_validation_feedback,
        create_interactive_prompts
    )
    from .template_generator import (
        ConfigurationTemplateGenerator,
        InteractiveTemplateBuilder,
        TemplateMetadata,
        AnalysisMode,
        ExperimentType,
        PerformanceProfile,
        create_template_generator,
        generate_example_configurations,
        create_interactive_builder as create_template_builder
    )
    from .health_monitor import (
        ConfigurationHealthMonitor,
        HealthReport,
        HealthStatus,
        ValidationLevel,
        create_health_monitor,
        quick_health_check,
        comprehensive_health_check
    )
    from .manager import ConfigManager
    HAS_ALL_COMPONENTS = True
except ImportError as e:
    HAS_ALL_COMPONENTS = False
    import logging
    logging.getLogger(__name__).warning(f"Some components not available: {e}")

try:
    from homodyne.utils.logging import get_logger
    HAS_UTILS_LOGGING = True
except ImportError:
    import logging
    HAS_UTILS_LOGGING = False
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


class UserLevel(Enum):
    """User experience level enumeration."""
    NOVICE = "novice"         # New to XPCS, need maximum guidance
    INTERMEDIATE = "intermediate"  # Some experience, balanced interface
    EXPERT = "expert"         # Experienced, minimal guidance needed


class WorkflowMode(Enum):
    """Configuration workflow mode."""
    QUICK_START = "quick_start"       # Get started quickly with defaults
    GUIDED = "guided"                 # Step-by-step guided configuration
    ADVANCED = "advanced"             # Full control for experts
    RECOVERY = "recovery"             # Error recovery and fixing
    VALIDATION = "validation"         # Focus on validation and health checks


@dataclass
class UXConfiguration:
    """User experience configuration settings."""
    user_level: UserLevel = UserLevel.INTERMEDIATE
    use_colors: bool = True
    show_progress: bool = True
    interactive_prompts: bool = True
    auto_recovery: bool = True
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    theme: Optional[ConsoleTheme] = None


class HomodyneConfigurationUX:
    """
    Unified user experience interface for Homodyne v2 configuration system.
    
    This class provides a single, comprehensive interface to all configuration
    tools and utilities, with intelligent workflow guidance and progressive
    complexity disclosure based on user experience level.
    """
    
    def __init__(self, ux_config: Optional[UXConfiguration] = None):
        self.ux_config = ux_config or UXConfiguration()
        
        # Initialize console components
        if HAS_ALL_COMPONENTS:
            self.logger = create_console_logger(theme=self.ux_config.theme)
            self.progress_tracker = create_progress_tracker(self.logger)
            self.validation_feedback = create_validation_feedback()
            self.interactive_prompts = create_interactive_prompts()
        else:
            self.logger = None
            self.progress_tracker = None
            self.validation_feedback = None
            self.interactive_prompts = None
        
        # Initialize core components
        self.template_generator = create_template_generator(self.logger) if HAS_ALL_COMPONENTS else None
        self.health_monitor = create_health_monitor(self.ux_config.validation_level, self.logger) if HAS_ALL_COMPONENTS else None
        self.suggestion_engine = ConfigurationSuggestionEngine() if HAS_ALL_COMPONENTS else None
        
        # Track user session
        self.session_history = []
        
        logger.info("Homodyne Configuration UX initialized")
    
    def welcome(self):
        """Show welcome message and system status."""
        if not self.logger:
            print("Homodyne Configuration System")
            return
        
        self.logger.header("🎯 Homodyne v2 Configuration System")
        
        print("Welcome to the enhanced configuration experience!")
        print("This system provides comprehensive tools for creating, validating,")
        print("and optimizing your XPCS analysis configurations.")
        
        print(f"\nYour experience level: {self.ux_config.user_level.value.title()}")
        
        # Show available features
        features = []
        if HAS_ALL_COMPONENTS:
            features.extend([
                "✅ Interactive configuration builder",
                "✅ Smart error recovery",
                "✅ Template generation", 
                "✅ Health monitoring",
                "✅ Performance optimization",
                "✅ Real-time validation"
            ])
        else:
            features.append("⚠️  Some advanced features not available")
        
        if features:
            print("\nAvailable features:")
            for feature in features:
                print(f"  {feature}")
        
        print()
    
    def quick_start(self, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Quick start workflow for new users.
        
        Creates a working configuration with minimal user input.
        """
        if self.logger:
            self.logger.header("🚀 Quick Start Configuration")
            self.logger.info("Creating a configuration with smart defaults...")
        
        try:
            # Simple questions for quick start
            if self.interactive_prompts:
                experiment_choices = [
                    "Hard sphere colloids (most common)",
                    "Soft colloids/polymers",
                    "Biological systems",
                    "Other/custom system"
                ]
                
                experiment_choice = self.interactive_prompts.ask_choice(
                    "What type of system are you studying?",
                    experiment_choices,
                    default=experiment_choices[0]
                )
                
                # Map choice to experiment type
                if "Hard sphere" in experiment_choice:
                    exp_type = ExperimentType.HARD_SPHERES
                elif "Soft" in experiment_choice or "polymer" in experiment_choice.lower():
                    exp_type = ExperimentType.SOFT_COLLOIDS
                elif "Biological" in experiment_choice:
                    exp_type = ExperimentType.BIOLOGICAL
                else:
                    exp_type = ExperimentType.CUSTOM
                
                # Simple analysis mode choice
                if self.ux_config.user_level == UserLevel.NOVICE:
                    analysis_mode = AnalysisMode.STATIC_ISOTROPIC  # Simplest option
                    self.logger.info("Using static isotropic mode (simplest and fastest)")
                else:
                    simple_mode = self.interactive_prompts.ask_yes_no(
                        "Use simple analysis mode (fastest)?", default=True
                    )
                    analysis_mode = AnalysisMode.STATIC_ISOTROPIC if simple_mode else AnalysisMode.STATIC_ANISOTROPIC
            else:
                # No interactive prompts - use safe defaults
                exp_type = ExperimentType.HARD_SPHERES
                analysis_mode = AnalysisMode.STATIC_ISOTROPIC
            
            # Generate template
            if self.template_generator:
                metadata = TemplateMetadata(
                    name="quick_start",
                    description="Quick start configuration with smart defaults",
                    analysis_mode=analysis_mode,
                    experiment_type=exp_type,
                    performance_profile=PerformanceProfile.FAST,
                    complexity_level=self.ux_config.user_level.value
                )
                
                config_content = self.template_generator.generate_template(metadata)
                
                # Determine output path
                if output_path is None:
                    output_path = "homodyne_quickstart_config.yaml"
                
                output_path = Path(output_path)
                
                # Save configuration
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(config_content)
                
                if self.logger:
                    self.logger.success(f"✅ Quick start configuration created!")
                    self.logger.file_path(output_path)
                    self.logger.info("\nNext steps:")
                    self.logger.info("1. Update 'experimental_data' section with your data paths")
                    self.logger.info("2. Run: homodyne analyze " + str(output_path))
                
                return str(output_path)
            else:
                if self.logger:
                    self.logger.error("Template generator not available")
                return None
                
        except Exception as e:
            logger.error(f"Quick start failed: {e}")
            if self.logger:
                self.logger.error(f"Quick start failed: {e}")
                self.logger.info("Try the guided workflow instead: use_guided_workflow()")
            return None
    
    def guided_workflow(self, output_path: Optional[Union[str, Path]] = None) -> Optional[str]:
        """
        Guided workflow for creating configurations step-by-step.
        
        Provides comprehensive guidance through the configuration process.
        """
        if not HAS_ALL_COMPONENTS:
            if self.logger:
                self.logger.error("Guided workflow requires all components to be available")
            return None
        
        if self.logger:
            self.logger.header("🧭 Guided Configuration Workflow")
        
        try:
            # Use the interactive configuration builder
            builder = create_interactive_builder(output_path)
            if builder:
                config_data = builder.run_interactive_setup()
                
                # The builder handles saving, so get the path from it
                if hasattr(builder, 'output_path') and builder.output_path:
                    output_path = str(builder.output_path)
                    
                    # Log session for learning
                    self.session_history.append({
                        "workflow": "guided",
                        "output_path": output_path,
                        "success": True,
                        "user_level": self.ux_config.user_level.value
                    })
                    
                    return output_path
                else:
                    return None
            else:
                if self.logger:
                    self.logger.error("Could not create interactive builder")
                return None
                
        except Exception as e:
            logger.error(f"Guided workflow failed: {e}")
            if self.logger:
                self.logger.error(f"Guided workflow failed: {e}")
                
                # Offer recovery
                if self.ux_config.auto_recovery:
                    if self.interactive_prompts and self.interactive_prompts.ask_yes_no(
                        "Would you like to try automatic recovery?", default=True
                    ):
                        return self.recover_configuration(str(output_path) if output_path else None)
            
            return None
    
    def validate_configuration(self, config_path: Union[str, Path], 
                             detailed: bool = False) -> Optional[HealthReport]:
        """
        Validate a configuration file with user-friendly feedback.
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            if self.logger:
                self.logger.error(f"Configuration file not found: {config_path}")
                
                # Offer to create one
                if self.interactive_prompts and self.interactive_prompts.ask_yes_no(
                    "Would you like to create a new configuration?", default=True
                ):
                    return self.quick_start(config_path)
            return None
        
        try:
            if self.logger:
                self.logger.section("Configuration Validation")
            
            # Choose validation approach based on user level and detailed flag
            if detailed or self.ux_config.user_level == UserLevel.EXPERT:
                report = comprehensive_health_check(config_path) if HAS_ALL_COMPONENTS else None
            else:
                report = quick_health_check(config_path) if HAS_ALL_COMPONENTS else None
            
            if report:
                # Show user-friendly summary
                if self.logger:
                    self._show_validation_summary(report)
                
                # Offer to fix issues
                if report.overall_status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    if self.ux_config.auto_recovery and self.interactive_prompts:
                        if self.interactive_prompts.ask_yes_no(
                            "Would you like to attempt automatic fixes?", default=True
                        ):
                            self.recover_configuration(config_path)
                
                return report
            else:
                if self.logger:
                    self.logger.warning("Validation not available - basic checks only")
                return None
                
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            if self.logger:
                self.logger.error(f"Validation failed: {e}")
                
                # Offer recovery for validation errors too
                if self.ux_config.auto_recovery and isinstance(e, (ConfigurationFileError, HomodyneConfigurationError)):
                    if self.interactive_prompts and self.interactive_prompts.ask_yes_no(
                        "Configuration has errors. Attempt recovery?", default=True
                    ):
                        return self.recover_configuration(config_path)
            
            return None
    
    def recover_configuration(self, config_path: Union[str, Path]) -> Optional[str]:
        """
        Attempt to recover a problematic configuration.
        """
        if not HAS_ALL_COMPONENTS:
            if self.logger:
                self.logger.error("Recovery requires all components to be available")
            return None
        
        config_path = Path(config_path)
        
        if self.logger:
            self.logger.section("🔧 Configuration Recovery")
            self.logger.info(f"Attempting to recover: {config_path}")
        
        try:
            # Use the smart recovery engine
            recovery_engine = create_smart_recovery_engine(
                config_path, 
                interactive=self.ux_config.interactive_prompts
            )
            
            # Try to detect the error by attempting to load
            try:
                if config_path.exists():
                    ConfigManager(str(config_path))
                    if self.logger:
                        self.logger.success("Configuration is actually valid - no recovery needed")
                    return str(config_path)
            except Exception as config_error:
                # Attempt recovery
                recovery_result = recovery_engine.recover_from_error(config_error)
                
                if self.logger:
                    self.logger.info(str(recovery_result))
                
                if recovery_result.success:
                    # Verify the recovered configuration
                    try:
                        ConfigManager(str(config_path))
                        if self.logger:
                            self.logger.success("✅ Recovery successful - configuration is now valid!")
                        return str(config_path)
                    except Exception as verify_error:
                        if self.logger:
                            self.logger.warning(f"Recovery completed but configuration still has issues: {verify_error}")
                        return str(config_path)  # Return anyway - partial success
                else:
                    if self.logger:
                        self.logger.error("❌ Recovery failed")
                        self.logger.info("Consider using the guided workflow to create a new configuration")
                    return None
                    
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            if self.logger:
                self.logger.error(f"Recovery attempt failed: {e}")
            return None
    
    def compare_configurations(self, config1_path: Union[str, Path], 
                             config2_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """
        Compare two configurations with user-friendly output.
        """
        if not HAS_ALL_COMPONENTS:
            if self.logger:
                self.logger.error("Configuration comparison requires all components")
            return None
        
        try:
            if self.logger:
                self.logger.section("Configuration Comparison")
                self.logger.file_path(config1_path, "Configuration 1")
                self.logger.file_path(config2_path, "Configuration 2")
            
            comparison = compare_configuration_files(config1_path, config2_path)
            
            if self.logger and comparison:
                self._show_comparison_summary(comparison)
            
            return comparison
            
        except Exception as e:
            logger.error(f"Configuration comparison failed: {e}")
            if self.logger:
                self.logger.error(f"Configuration comparison failed: {e}")
            return None
    
    def generate_examples(self, output_dir: Union[str, Path] = "./examples") -> Optional[Dict[str, str]]:
        """
        Generate example configurations.
        """
        if not HAS_ALL_COMPONENTS:
            if self.logger:
                self.logger.error("Example generation requires all components")
            return None
        
        try:
            if self.logger:
                self.logger.section("Generating Example Configurations")
            
            generated = generate_example_configurations(output_dir)
            
            if self.logger and generated:
                self.logger.success(f"Generated {len(generated)} example configurations")
                self.logger.file_path(output_dir)
                
                # Show examples by complexity level
                levels = ["novice", "intermediate", "expert"]
                for level in levels:
                    level_examples = [name for name in generated.keys() if level in name or f"{level}_" in name]
                    if level_examples:
                        self.logger.info(f"{level.title()} level examples:")
                        for example in level_examples[:3]:  # Show top 3
                            self.logger.info(f"  • {example}_config.yaml")
            
            return generated
            
        except Exception as e:
            logger.error(f"Example generation failed: {e}")
            if self.logger:
                self.logger.error(f"Example generation failed: {e}")
            return None
    
    def interactive_template_builder(self) -> Optional[str]:
        """
        Launch interactive template builder.
        """
        if not HAS_ALL_COMPONENTS:
            if self.logger:
                self.logger.error("Interactive template builder requires all components")
            return None
        
        try:
            builder = create_template_builder()
            if builder:
                template_content = builder.build_custom_template()
                return template_content
            else:
                if self.logger:
                    self.logger.error("Could not create template builder")
                return None
                
        except Exception as e:
            logger.error(f"Interactive template builder failed: {e}")
            if self.logger:
                self.logger.error(f"Interactive template builder failed: {e}")
            return None
    
    def get_workflow_recommendation(self, context: Optional[Dict[str, Any]] = None) -> WorkflowMode:
        """
        Get recommended workflow based on user level and context.
        """
        context = context or {}
        
        # Check if user has existing configuration
        if context.get("has_existing_config", False):
            if context.get("config_has_errors", False):
                return WorkflowMode.RECOVERY
            else:
                return WorkflowMode.VALIDATION
        
        # New user workflow recommendation
        if self.ux_config.user_level == UserLevel.NOVICE:
            return WorkflowMode.QUICK_START
        elif self.ux_config.user_level == UserLevel.INTERMEDIATE:
            return WorkflowMode.GUIDED
        else:
            return WorkflowMode.ADVANCED
    
    def run_recommended_workflow(self, config_path: Optional[Union[str, Path]] = None,
                                context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Run the recommended workflow based on user level and context.
        """
        # Determine context if not provided
        if context is None:
            context = {}
            if config_path:
                config_path = Path(config_path)
                context["has_existing_config"] = config_path.exists()
                
                if context["has_existing_config"]:
                    # Quick check for errors
                    try:
                        ConfigManager(str(config_path))
                        context["config_has_errors"] = False
                    except Exception:
                        context["config_has_errors"] = True
        
        # Get recommendation
        recommended_mode = self.get_workflow_recommendation(context)
        
        if self.logger:
            self.logger.info(f"Recommended workflow: {recommended_mode.value}")
        
        # Run recommended workflow
        if recommended_mode == WorkflowMode.QUICK_START:
            return self.quick_start(config_path)
        elif recommended_mode == WorkflowMode.GUIDED:
            return self.guided_workflow(config_path)
        elif recommended_mode == WorkflowMode.RECOVERY:
            return self.recover_configuration(config_path)
        elif recommended_mode == WorkflowMode.VALIDATION:
            report = self.validate_configuration(config_path)
            return str(config_path) if report else None
        else:
            # Advanced mode - let user choose
            if self.logger:
                self.logger.info("Advanced mode - available options:")
                self.logger.info("  • quick_start() - Quick configuration with defaults")
                self.logger.info("  • guided_workflow() - Step-by-step configuration")
                self.logger.info("  • interactive_template_builder() - Custom template creation")
                self.logger.info("  • validate_configuration(path) - Comprehensive validation")
                self.logger.info("  • generate_examples() - Create example configurations")
            return str(config_path) if config_path else None
    
    def _show_validation_summary(self, report: HealthReport):
        """Show user-friendly validation summary."""
        if not self.logger:
            return
        
        # Overall status
        status_colors = {
            HealthStatus.EXCELLENT: "success",
            HealthStatus.GOOD: "success", 
            HealthStatus.WARNING: "warning",
            HealthStatus.CRITICAL: "error"
        }
        
        status_icons = {
            HealthStatus.EXCELLENT: "🌟",
            HealthStatus.GOOD: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "🚨"
        }
        
        icon = status_icons.get(report.overall_status, "")
        self.logger.info(f"{icon} Overall Score: {report.overall_score:.1f}/100 ({report.overall_status.value.upper()})")
        
        # Critical issues
        critical_issues = report.get_critical_issues()
        if critical_issues:
            self.logger.error(f"🚨 {len(critical_issues)} critical issue(s) found:")
            for issue in critical_issues[:3]:  # Show top 3
                self.logger.error(f"  • {issue.message}")
                if issue.suggestions and self.ux_config.user_level != UserLevel.EXPERT:
                    self.logger.info(f"    💡 {issue.suggestions[0]}")
        
        # Warnings
        warnings = report.get_warnings()
        if warnings:
            self.logger.warning(f"⚠️  {len(warnings)} warning(s):")
            for warning in warnings[:2]:  # Show top 2
                self.logger.warning(f"  • {warning.message}")
        
        # Performance highlights
        perf_metrics = report.get_metrics_by_category("performance")
        excellent_perf = [m for m in perf_metrics if m.status == HealthStatus.EXCELLENT]
        if excellent_perf:
            self.logger.success("🚀 Performance highlights:")
            for metric in excellent_perf[:2]:  # Show top 2
                self.logger.success(f"  • {metric.message}")
        
        # Recommendations
        if report.recommendations and self.ux_config.user_level != UserLevel.EXPERT:
            self.logger.info("💡 Top recommendations:")
            for rec in report.recommendations[:3]:  # Show top 3
                self.logger.info(f"  {rec}")
    
    def _show_comparison_summary(self, comparison: Dict[str, Any]):
        """Show user-friendly comparison summary."""
        if not self.logger:
            return
        
        differences = comparison.get("differences", {})
        
        if not differences:
            self.logger.success("✅ Configurations are identical")
            return
        
        self.logger.info(f"Found {len(differences)} difference(s):")
        
        for category, diff in differences.items():
            self.logger.parameter(category.replace("_", " ").title(), "")
            self.logger.info(f"  Config 1: {diff.get('config1', 'N/A')}")
            self.logger.info(f"  Config 2: {diff.get('config2', 'N/A')}")
            
            impact = diff.get('impact', '')
            if impact:
                self.logger.info(f"  Impact: {impact}")
        
        # Performance comparison
        perf_comparison = comparison.get("summary", {}).get("performance_comparison", {})
        if perf_comparison:
            faster_config = perf_comparison.get("faster", "unknown")
            self.logger.info(f"\n🚀 Performance: Config {faster_config} is likely faster")
        
        # Recommendations
        recommendations = comparison.get("recommendations", [])
        if recommendations:
            self.logger.info("\n💡 Recommendations:")
            for rec in recommendations:
                self.logger.info(f"  • {rec}")


# Factory functions and convenience interfaces
def create_ux_interface(user_level: UserLevel = UserLevel.INTERMEDIATE,
                       use_colors: bool = True,
                       interactive: bool = True) -> HomodyneConfigurationUX:
    """
    Create a Homodyne configuration UX interface.
    
    Args:
        user_level: User experience level (novice, intermediate, expert)
        use_colors: Enable colored console output
        interactive: Enable interactive prompts
    
    Returns:
        HomodyneConfigurationUX instance
    """
    ux_config = UXConfiguration(
        user_level=user_level,
        use_colors=use_colors,
        interactive_prompts=interactive,
        validation_level=ValidationLevel.COMPREHENSIVE if user_level == UserLevel.EXPERT else ValidationLevel.STANDARD
    )
    
    return HomodyneConfigurationUX(ux_config)


def quick_setup(output_path: Optional[Union[str, Path]] = None,
               user_level: UserLevel = UserLevel.INTERMEDIATE) -> Optional[str]:
    """
    Quick setup function for creating configurations with minimal fuss.
    
    Args:
        output_path: Where to save the configuration
        user_level: User experience level
    
    Returns:
        Path to created configuration file, or None if failed
    """
    ux = create_ux_interface(user_level=user_level)
    ux.welcome()
    return ux.quick_start(output_path)


def guided_setup(output_path: Optional[Union[str, Path]] = None,
                user_level: UserLevel = UserLevel.INTERMEDIATE) -> Optional[str]:
    """
    Guided setup function for step-by-step configuration creation.
    
    Args:
        output_path: Where to save the configuration  
        user_level: User experience level
    
    Returns:
        Path to created configuration file, or None if failed
    """
    ux = create_ux_interface(user_level=user_level)
    ux.welcome()
    return ux.guided_workflow(output_path)


def smart_validate(config_path: Union[str, Path],
                  user_level: UserLevel = UserLevel.INTERMEDIATE) -> Optional[HealthReport]:
    """
    Smart validation with user-friendly output and recovery suggestions.
    
    Args:
        config_path: Path to configuration file
        user_level: User experience level
    
    Returns:
        Health report, or None if validation failed
    """
    ux = create_ux_interface(user_level=user_level)
    return ux.validate_configuration(config_path)


def auto_fix(config_path: Union[str, Path],
            user_level: UserLevel = UserLevel.INTERMEDIATE) -> Optional[str]:
    """
    Attempt to automatically fix configuration issues.
    
    Args:
        config_path: Path to configuration file
        user_level: User experience level
    
    Returns:
        Path to fixed configuration, or None if fixing failed
    """
    ux = create_ux_interface(user_level=user_level)
    return ux.recover_configuration(config_path)


def smart_workflow(config_path: Optional[Union[str, Path]] = None,
                  user_level: UserLevel = UserLevel.INTERMEDIATE) -> Optional[str]:
    """
    Smart workflow that automatically chooses the best approach based on context.
    
    Args:
        config_path: Path to existing configuration (optional)
        user_level: User experience level
    
    Returns:
        Path to configuration file, or None if failed
    """
    ux = create_ux_interface(user_level=user_level)
    ux.welcome()
    return ux.run_recommended_workflow(config_path)


# CLI integration
def main_ux_cli():
    """Main CLI interface for the enhanced UX system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Homodyne v2 Enhanced Configuration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s quick-start                    # Create config quickly
  %(prog)s guided                         # Step-by-step configuration  
  %(prog)s validate my_config.yaml        # Validate configuration
  %(prog)s fix my_config.yaml             # Attempt to fix issues
  %(prog)s compare config1.yaml config2.yaml  # Compare configurations
  %(prog)s examples                       # Generate example configurations
  %(prog)s smart                          # Smart workflow (recommended)
        """
    )
    
    parser.add_argument("--user-level", choices=["novice", "intermediate", "expert"],
                       default="intermediate", help="User experience level")
    parser.add_argument("--no-colors", action="store_true", help="Disable colored output")
    parser.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts")
    parser.add_argument("--output", "-o", help="Output path for generated configurations")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Quick start command
    quick_parser = subparsers.add_parser("quick-start", help="Create configuration quickly")
    
    # Guided workflow command
    guided_parser = subparsers.add_parser("guided", help="Step-by-step configuration")
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config", help="Configuration file to validate")
    validate_parser.add_argument("--detailed", action="store_true", help="Detailed validation")
    
    # Fix command
    fix_parser = subparsers.add_parser("fix", help="Attempt to fix configuration issues")
    fix_parser.add_argument("config", help="Configuration file to fix")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two configurations")
    compare_parser.add_argument("config1", help="First configuration file")
    compare_parser.add_argument("config2", help="Second configuration file")
    
    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Generate example configurations")
    examples_parser.add_argument("--dir", default="./examples", help="Output directory")
    
    # Template builder command
    template_parser = subparsers.add_parser("template", help="Interactive template builder")
    
    # Smart workflow command (default)
    smart_parser = subparsers.add_parser("smart", help="Smart workflow (recommended)")
    smart_parser.add_argument("config", nargs="?", help="Existing configuration file (optional)")
    
    args = parser.parse_args()
    
    # Create UX interface
    user_level = UserLevel(args.user_level)
    use_colors = not args.no_colors
    interactive = not args.non_interactive
    
    ux = create_ux_interface(user_level=user_level, use_colors=use_colors, interactive=interactive)
    
    # Show welcome message
    ux.welcome()
    
    # Execute command
    try:
        if args.command == "quick-start" or not args.command:
            result = ux.quick_start(args.output)
        elif args.command == "guided":
            result = ux.guided_workflow(args.output)
        elif args.command == "validate":
            report = ux.validate_configuration(args.config, detailed=args.detailed)
            result = "Validation completed" if report else None
        elif args.command == "fix":
            result = ux.recover_configuration(args.config)
        elif args.command == "compare":
            comparison = ux.compare_configurations(args.config1, args.config2)
            result = "Comparison completed" if comparison else None
        elif args.command == "examples":
            generated = ux.generate_examples(args.dir)
            result = f"Generated {len(generated)} examples" if generated else None
        elif args.command == "template":
            result = ux.interactive_template_builder()
        elif args.command == "smart":
            result = ux.run_recommended_workflow(args.config)
        else:
            # Default to smart workflow
            result = ux.run_recommended_workflow()
        
        if result:
            print(f"\n✅ Success: {result}")
        else:
            print("\n❌ Operation failed or was cancelled")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("CLI operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main_ux_cli()