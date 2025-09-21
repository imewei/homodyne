"""
Interactive Configuration Helpers for Homodyne v2
================================================

Interactive tools for building, validating, and optimizing homodyne configurations.
Provides guided configuration setup, real-time validation, and intelligent suggestions.

Key Features:
- Interactive configuration builder with step-by-step guidance
- Real-time validation with immediate feedback
- Configuration wizard for different analysis modes
- Parameter tuning with physics-based suggestions
- Configuration comparison and diff tools
- Performance optimization recommendations
- Migration assistants for v1->v2 configurations

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    from .exceptions import (ErrorContext, FixSuggestion,
                             HomodyneConfigurationError,
                             generate_configuration_example)
    from .manager import ConfigManager
    from .smart_recovery import SmartRecoveryEngine

    HAS_CONFIG_SYSTEM = True
except ImportError:
    HAS_CONFIG_SYSTEM = False
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
class ValidationResult:
    """Result of configuration validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    performance_notes: List[str] = field(default_factory=list)
    score: int = 0  # 0-100 configuration quality score

    def __str__(self) -> str:
        status = "✅ Valid" if self.is_valid else "❌ Invalid"
        result = f"{status} (Score: {self.score}/100)\n"

        if self.errors:
            result += "\n🚨 Errors:\n"
            for error in self.errors:
                result += f"  • {error}\n"

        if self.warnings:
            result += "\n⚠️  Warnings:\n"
            for warning in self.warnings:
                result += f"  • {warning}\n"

        if self.suggestions:
            result += "\n💡 Suggestions:\n"
            for suggestion in self.suggestions:
                result += f"  • {suggestion}\n"

        if self.performance_notes:
            result += "\n🚀 Performance Notes:\n"
            for note in self.performance_notes:
                result += f"  • {note}\n"

        return result.strip()


@dataclass
class ConfigurationChoice:
    """A configuration choice with description and impact."""

    key: str
    value: Any
    description: str
    impact: str = "No impact"
    recommended: bool = False
    advanced: bool = False


class InteractiveConfigurationBuilder:
    """Interactive tool for building homodyne configurations step-by-step."""

    def __init__(self, output_path: Optional[Union[str, Path]] = None):
        self.output_path = Path(output_path) if output_path else None
        self.config_data: Dict[str, Any] = {}
        self.user_level = "intermediate"  # novice, intermediate, expert
        self.analysis_mode = None

        # Color codes for terminal output
        self.colors = {
            "header": "\033[95m",
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "end": "\033[0m",
            "bold": "\033[1m",
            "underline": "\033[4m",
        }

        # Check if terminal supports colors
        self.use_colors = (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and os.getenv("TERM") != "dumb"
        )

        logger.info("Interactive configuration builder initialized")

    def colorize(self, text: str, color: str) -> str:
        """Add color codes to text if terminal supports it."""
        if self.use_colors and color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['end']}"
        return text

    def print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "=" * 60)
        print(self.colorize(text, "header"))
        print("=" * 60)

    def print_section(self, text: str):
        """Print a formatted section header."""
        print(f"\n{self.colorize('📋 ' + text, 'blue')}")
        print("-" * (len(text) + 3))

    def ask_choice(
        self, prompt: str, choices: List[str], default: Optional[str] = None
    ) -> str:
        """Ask user to choose from a list of options."""
        print(f"\n{prompt}")
        for i, choice in enumerate(choices, 1):
            marker = " (default)" if choice == default else ""
            print(f"  {i}. {choice}{marker}")

        while True:
            try:
                response = input(f"\nChoice [1-{len(choices)}]: ").strip()
                if not response and default:
                    return default

                choice_num = int(response)
                if 1 <= choice_num <= len(choices):
                    return choices[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(choices)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nConfiguration cancelled by user")
                sys.exit(1)

    def ask_yes_no(self, prompt: str, default: bool = True) -> bool:
        """Ask user a yes/no question."""
        default_str = "Y/n" if default else "y/N"
        while True:
            try:
                response = input(f"{prompt} [{default_str}]: ").strip().lower()
                if not response:
                    return default
                if response in ["y", "yes", "true", "1"]:
                    return True
                elif response in ["n", "no", "false", "0"]:
                    return False
                else:
                    print("Please enter 'y' for yes or 'n' for no")
            except KeyboardInterrupt:
                print("\nConfiguration cancelled by user")
                sys.exit(1)

    def ask_value(
        self,
        prompt: str,
        value_type: type = str,
        default: Any = None,
        validator: Optional[Callable] = None,
    ) -> Any:
        """Ask user for a value with type checking and validation."""
        default_str = f" (default: {default})" if default is not None else ""

        while True:
            try:
                response = input(f"{prompt}{default_str}: ").strip()
                if not response and default is not None:
                    return default

                if value_type == bool:
                    return response.lower() in ["true", "yes", "y", "1"]
                elif value_type == int:
                    value = int(response)
                elif value_type == float:
                    value = float(response)
                else:
                    value = response

                if validator and not validator(value):
                    print("Invalid value, please try again")
                    continue

                return value

            except ValueError:
                print(f"Please enter a valid {value_type.__name__}")
            except KeyboardInterrupt:
                print("\nConfiguration cancelled by user")
                sys.exit(1)

    def run_interactive_setup(self) -> Dict[str, Any]:
        """Run the complete interactive configuration setup."""
        print(
            self.colorize(
                "\n🎯 Welcome to the Homodyne v2 Interactive Configuration Builder!",
                "bold",
            )
        )
        print(
            "This tool will guide you through creating a complete configuration for your XPCS analysis.\n"
        )

        # Step 1: User experience level
        self._setup_user_level()

        # Step 2: Analysis mode selection
        self._setup_analysis_mode()

        # Step 3: Experimental parameters
        self._setup_experimental_parameters()

        # Step 4: Data configuration
        self._setup_data_configuration()

        # Step 5: Analysis settings
        self._setup_analysis_settings()

        # Step 6: Optimization settings
        self._setup_optimization_settings()

        # Step 7: Logging configuration
        self._setup_logging_configuration()

        # Step 8: Advanced settings (if expert user)
        if self.user_level == "expert":
            self._setup_advanced_settings()

        # Step 9: Review and save
        self._review_and_save()

        return self.config_data

    def _setup_user_level(self):
        """Setup user experience level."""
        self.print_section("User Experience Level")

        levels = ["novice", "intermediate", "expert"]
        level_descriptions = {
            "novice": "New to XPCS analysis - use simple defaults and extensive guidance",
            "intermediate": "Some XPCS experience - balanced options with explanations",
            "expert": "Experienced user - all options available with minimal guidance",
        }

        print("Select your experience level with XPCS analysis:")
        for i, level in enumerate(levels, 1):
            print(f"  {i}. {level.title()}: {level_descriptions[level]}")

        choice_num = int(input("\nChoice [1-3]: ").strip() or "2")
        self.user_level = levels[choice_num - 1]

        print(f"\n✅ Set user level to: {self.user_level}")

    def _setup_analysis_mode(self):
        """Setup analysis mode selection."""
        self.print_section("Analysis Mode Selection")

        print("Choose your analysis mode based on your experimental conditions:\n")

        modes = [
            (
                "static_isotropic",
                "Static system, no directional dependence",
                "Fastest, simplest",
            ),
            (
                "static_anisotropic",
                "Static system with directional effects",
                "Medium complexity",
            ),
            ("laminar_flow", "System under shear flow", "Full capability, slower"),
        ]

        for i, (mode, desc, performance) in enumerate(modes, 1):
            print(f"  {i}. {self.colorize(mode.replace('_', ' ').title(), 'bold')}")
            print(f"     {desc}")
            print(
                f"     Performance: {self.colorize(performance, 'green' if 'Fast' in performance else 'yellow')}"
            )
            print()

        # Provide guidance based on user level
        if self.user_level == "novice":
            print(
                "💡 Recommendation: Start with 'Static Isotropic' for your first analysis"
            )
            default_choice = 1
        else:
            default_choice = 2

        choice_num = int(
            input(f"Choice [1-3, default: {default_choice}]: ").strip()
            or str(default_choice)
        )
        self.analysis_mode = modes[choice_num - 1][0]

        # Set basic mode configuration
        if "static" in self.analysis_mode:
            self.config_data["analysis_settings"] = {
                "static_mode": True,
                "static_submode": self.analysis_mode.split("_")[1],
            }
        else:
            self.config_data["analysis_settings"] = {"static_mode": False}

        print(f"\n✅ Set analysis mode to: {self.analysis_mode}")

    def _setup_experimental_parameters(self):
        """Setup experimental parameters."""
        self.print_section("Experimental Parameters")

        print("Configure the core experimental parameters for your XPCS analysis:\n")

        # Temporal parameters
        print(self.colorize("⏱️  Temporal Parameters:", "bold"))

        if self.user_level == "novice":
            print(
                "These parameters define the time window and resolution of your analysis."
            )

        dt = self.ask_value(
            "Time step (dt) in seconds", float, default=0.1, validator=lambda x: x > 0
        )

        start_frame = self.ask_value(
            "Start frame number", int, default=1, validator=lambda x: x >= 1
        )

        end_frame = self.ask_value(
            "End frame number", int, default=1000, validator=lambda x: x > start_frame
        )

        # Scattering parameters
        print(f"\n{self.colorize('📡 Scattering Parameters:', 'bold')}")

        if self.user_level != "expert":
            print("The q-vector determines the length scale probed by your experiment.")
            print("Typical XPCS values range from 0.001 to 0.1 Å⁻¹")

        q_default = 0.0054  # Typical XPCS value
        wavevector_q = self.ask_value(
            "Wavevector q (Å⁻¹)", float, default=q_default, validator=lambda x: x > 0
        )

        # Geometry parameters
        print(f"\n{self.colorize('📐 Geometry Parameters:', 'bold')}")

        if self.user_level != "expert":
            print("Gap size is typically in micrometers (1e6 nm = 1 mm)")

        gap_size = self.ask_value(
            "Stator-rotor gap size (nm)",
            float,
            default=2000000,  # 2 mm in nm
            validator=lambda x: x > 0,
        )

        # Store parameters
        self.config_data["analyzer_parameters"] = {
            "temporal": {"dt": dt, "start_frame": start_frame, "end_frame": end_frame},
            "scattering": {"wavevector_q": wavevector_q},
            "geometry": {"stator_rotor_gap": gap_size},
        }

        # Analysis summary
        frames = end_frame - start_frame
        total_time = frames * dt
        print(f"\n✅ Configuration summary:")
        print(f"   • Analysis window: {frames} frames ({total_time:.1f} seconds)")
        print(f"   • q-vector: {wavevector_q:.4f} Å⁻¹")
        print(f"   • Gap size: {gap_size / 1e6:.1f} mm")

    def _setup_data_configuration(self):
        """Setup data file configuration."""
        self.print_section("Data Configuration")

        print("Configure paths to your experimental data files:\n")

        # Get data folder path
        if self.user_level == "novice":
            print("💡 Tip: Use relative paths like './data/' for portability")

        data_folder = self.ask_value("Data folder path", str, default="./data/")

        data_file = self.ask_value(
            "Data file name (HDF5 format)", str, default="your_data.hdf"
        )

        # Additional data settings for intermediate/expert users
        additional_settings = {}
        if self.user_level != "novice":
            use_cache = self.ask_yes_no(
                "Enable data caching for faster repeated analysis?", default=True
            )
            if use_cache:
                cache_path = self.ask_value(
                    "Cache folder path", str, default="./cache/"
                )
                additional_settings.update(
                    {
                        "cache_file_path": cache_path,
                        "cache_filename_template": "cached_c2_frames_{start_frame}_{end_frame}.npz",
                        "cache_compression": True,
                    }
                )

        self.config_data["experimental_data"] = {
            "data_folder_path": data_folder,
            "data_file_name": data_file,
            **additional_settings,
        }

        print(f"\n✅ Data configuration set:")
        print(f"   • Data path: {data_folder}{data_file}")
        if additional_settings:
            print(
                f"   • Caching enabled: {additional_settings.get('cache_file_path', 'N/A')}"
            )

    def _setup_analysis_settings(self):
        """Setup analysis-specific settings."""
        self.print_section("Analysis Settings")

        # Parameter configuration
        if "static" in self.analysis_mode:
            param_count = 3
            default_values = [100.0, 0.0, 10.0]
            param_names = ["D0", "alpha", "D_offset"]

            print("Static mode uses 3 parameters:")
            print("  • D0: Diffusion coefficient (μm²/s)")
            print("  • alpha: Anomalous diffusion exponent (-2 to 2)")
            print("  • D_offset: Diffusion offset (μm²/s)")
        else:
            param_count = 7
            default_values = [100.0, 0.0, 10.0, 0.01, 0.0, 0.0, 0.0]
            param_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

            print("Laminar flow mode uses 7 parameters:")
            print("  • D0, alpha, D_offset: Diffusion parameters")
            print("  • gamma_dot_t0, beta, gamma_dot_t_offset: Shear rate parameters")
            print("  • phi0: Phase angle parameter")

        # Allow parameter customization for intermediate/expert users
        if self.user_level != "novice":
            customize = self.ask_yes_no(
                f"Customize initial parameter values? (default values available)",
                default=False,
            )

            if customize:
                print(f"\nEnter initial values for {param_count} parameters:")
                custom_values = []
                for i, (name, default_val) in enumerate(
                    zip(param_names, default_values)
                ):
                    value = self.ask_value(f"  {name}", float, default=default_val)
                    custom_values.append(value)
                default_values = custom_values

        self.config_data["initial_parameters"] = {
            "values": default_values,
            "parameter_names": param_names,
        }

        # Model description
        if self.analysis_mode == "static_isotropic":
            model_desc = "g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt), g₂(t₁,t₂) = [g₁(t₁,t₂)]²"
        elif self.analysis_mode == "static_anisotropic":
            model_desc = "g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt), g₂(t₁,t₂) = [g₁(t₁,t₂)]²"
        else:
            model_desc = (
                "g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂) with shear effects"
            )

        self.config_data["analysis_settings"]["model_description"] = {
            self.analysis_mode: model_desc
        }

        print(f"\n✅ Analysis settings configured for {param_count} parameters")

    def _setup_optimization_settings(self):
        """Setup optimization and angle filtering settings."""
        self.print_section("Optimization Settings")

        # Angle filtering configuration
        if self.analysis_mode == "static_isotropic":
            # Disable angle filtering for isotropic mode
            angle_filtering_enabled = False
            print("🔹 Angle filtering automatically disabled for static isotropic mode")
        else:
            print(
                "Angle filtering improves analysis by focusing on specific angular ranges."
            )
            print("This is recommended for anisotropic and laminar flow analyses.")

            angle_filtering_enabled = self.ask_yes_no(
                "Enable angle filtering?", default=True
            )

        angle_filtering_config = {"enabled": angle_filtering_enabled}

        if angle_filtering_enabled:
            if self.user_level == "expert":
                # Allow custom angle ranges
                print("\nConfigure angle ranges (in degrees):")
                ranges = []

                add_range = True
                range_num = 1
                while add_range:
                    print(f"\nAngle range {range_num}:")
                    min_angle = self.ask_value(
                        "  Minimum angle",
                        float,
                        default=-10.0 if range_num == 1 else 170.0,
                    )
                    max_angle = self.ask_value(
                        "  Maximum angle",
                        float,
                        default=10.0 if range_num == 1 else 190.0,
                    )

                    if min_angle < max_angle:
                        ranges.append({"min_angle": min_angle, "max_angle": max_angle})
                        range_num += 1

                        if range_num <= 2:
                            add_range = self.ask_yes_no(
                                "Add another angle range?", default=range_num == 2
                            )
                        else:
                            add_range = self.ask_yes_no(
                                "Add another angle range?", default=False
                            )
                    else:
                        print("Error: Maximum angle must be greater than minimum angle")

                angle_filtering_config["target_ranges"] = ranges
            else:
                # Use default ranges
                angle_filtering_config["target_ranges"] = [
                    {"min_angle": -10.0, "max_angle": 10.0},
                    {"min_angle": 170.0, "max_angle": 190.0},
                ]

        angle_filtering_config["fallback_to_all_angles"] = True

        self.config_data["optimization_config"] = {
            "angle_filtering": angle_filtering_config
        }

        # Classical optimization settings for expert users
        if self.user_level == "expert":
            print(f"\n{self.colorize('Advanced Optimization Settings:', 'bold')}")

            include_classical = self.ask_yes_no(
                "Include classical optimization methods?", default=True
            )

            if include_classical:
                methods = ["Nelder-Mead", "Powell", "L-BFGS-B"]
                selected_methods = []

                print("Available optimization methods:")
                for method in methods:
                    include = self.ask_yes_no(f"  Include {method}?", default=True)
                    if include:
                        selected_methods.append(method)

                if selected_methods:
                    self.config_data["optimization_config"][
                        "classical_optimization"
                    ] = {
                        "methods": selected_methods,
                        "method_options": {
                            "Nelder-Mead": {
                                "maxiter": 1000,
                                "xatol": 1e-8,
                                "fatol": 1e-8,
                            },
                            "Powell": {"maxiter": 1000, "xtol": 1e-8, "ftol": 1e-8},
                            "L-BFGS-B": {
                                "maxiter": 15000,
                                "ftol": 2.220446049250313e-09,
                            },
                        },
                    }

        print(f"\n✅ Optimization settings configured")
        if angle_filtering_enabled:
            ranges = angle_filtering_config.get("target_ranges", [])
            print(f"   • Angle filtering: {len(ranges)} ranges")

    def _setup_logging_configuration(self):
        """Setup logging configuration."""
        self.print_section("Logging Configuration")

        print("Configure logging for analysis monitoring and debugging:\n")

        # Basic logging settings
        log_to_console = self.ask_yes_no("Enable console logging?", default=True)
        log_level = "INFO"

        if self.user_level != "novice":
            levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
            level_descriptions = {
                "DEBUG": "Detailed diagnostic information",
                "INFO": "General information messages",
                "WARNING": "Warning messages only",
                "ERROR": "Error messages only",
            }

            print("Select logging level:")
            for i, level in enumerate(levels, 1):
                print(f"  {i}. {level}: {level_descriptions[level]}")

            choice = int(input("Choice [1-4, default: 2]: ").strip() or "2")
            log_level = levels[choice - 1]

        # File logging
        log_to_file = False
        if self.user_level != "novice":
            log_to_file = self.ask_yes_no("Enable file logging?", default=False)

        logging_config = {
            "log_to_console": log_to_console,
            "log_to_file": log_to_file,
            "level": log_level,
        }

        if log_to_file:
            log_filename = self.ask_value(
                "Log file name", str, default="homodyne_analysis.log"
            )
            logging_config["log_filename"] = log_filename

            if self.user_level == "expert":
                logging_config["rotation"] = {
                    "max_bytes": 10 * 1024 * 1024,  # 10 MB
                    "backup_count": 3,
                }

        self.config_data["logging"] = logging_config

        print(f"\n✅ Logging configuration set:")
        print(f"   • Console: {log_to_console} ({log_level})")
        print(
            f"   • File: {log_to_file}"
            + (f" ({logging_config.get('log_filename', 'N/A')})" if log_to_file else "")
        )

    def _setup_advanced_settings(self):
        """Setup advanced settings for expert users."""
        self.print_section("Advanced Settings")

        print("Configure advanced features and optimizations:\n")

        # V2 features
        v2_features = {}

        print(self.colorize("🚀 Performance Features:", "bold"))

        v2_features["performance_optimization"] = self.ask_yes_no(
            "Enable performance optimizations?", default=True
        )

        v2_features["parallel_processing"] = self.ask_yes_no(
            "Enable parallel processing?", default=True
        )

        # Try to detect GPU/JAX availability
        gpu_available = False
        try:
            import jax

            devices = jax.devices()
            gpu_available = any("gpu" in str(device).lower() for device in devices)
        except ImportError:
            pass

        if gpu_available:
            print("🔥 GPU acceleration available!")
            v2_features["gpu_acceleration"] = self.ask_yes_no(
                "Enable GPU acceleration?", default=True
            )
        else:
            print("ℹ️  GPU acceleration not available")
            v2_features["gpu_acceleration"] = False

        print(f"\n{self.colorize('🔬 Validation Features:', 'bold')}")

        v2_features["physics_validation"] = self.ask_yes_no(
            "Enable physics validation?", default=True
        )

        validation_levels = ["basic", "comprehensive", "strict"]
        level_desc = {
            "basic": "Essential validation only",
            "comprehensive": "Thorough validation with warnings",
            "strict": "Strict validation, fail on warnings",
        }

        print("Select validation level:")
        for i, level in enumerate(validation_levels, 1):
            print(f"  {i}. {level}: {level_desc[level]}")

        choice = int(input("Choice [1-3, default: 2]: ").strip() or "2")
        v2_features["validation_level"] = validation_levels[choice - 1]

        # Cache strategy
        cache_strategies = ["intelligent", "aggressive", "conservative", "disabled"]
        cache_desc = {
            "intelligent": "Smart caching with automatic cleanup",
            "aggressive": "Cache everything possible",
            "conservative": "Cache only essential data",
            "disabled": "No caching (slower but less disk usage)",
        }

        print("\nSelect cache strategy:")
        for i, strategy in enumerate(cache_strategies, 1):
            print(f"  {i}. {strategy}: {cache_desc[strategy]}")

        choice = int(input("Choice [1-4, default: 1]: ").strip() or "1")
        v2_features["cache_strategy"] = cache_strategies[choice - 1]

        self.config_data["v2_features"] = v2_features

        # Parameter space configuration
        print(f"\n{self.colorize('📊 Parameter Space:', 'bold')}")

        configure_bounds = self.ask_yes_no(
            "Configure custom parameter bounds?", default=False
        )

        if configure_bounds:
            param_names = self.config_data["initial_parameters"]["parameter_names"]
            bounds = []

            default_bounds = {
                "D0": (1.0, 1e6),
                "alpha": (-2.0, 2.0),
                "D_offset": (-100.0, 100.0),
                "gamma_dot_t0": (1e-6, 1.0),
                "beta": (-2.0, 2.0),
                "gamma_dot_t_offset": (-1e-2, 1e-2),
                "phi0": (-10.0, 10.0),
            }

            for param_name in param_names:
                if param_name in default_bounds:
                    default_min, default_max = default_bounds[param_name]
                    print(f"\n{param_name} bounds:")
                    min_val = self.ask_value("  Minimum", float, default=default_min)
                    max_val = self.ask_value("  Maximum", float, default=default_max)

                    bounds.append(
                        {
                            "name": param_name,
                            "min": min_val,
                            "max": max_val,
                            "type": "Normal",
                        }
                    )

            if bounds:
                self.config_data["parameter_space"] = {"bounds": bounds}

        print(f"\n✅ Advanced settings configured")

    def _review_and_save(self):
        """Review configuration and save to file."""
        self.print_section("Configuration Review")

        # Add metadata
        self.config_data["metadata"] = {
            "config_version": "2.0",
            "description": f"Interactive configuration for {self.analysis_mode} analysis",
            "created_by": "Homodyne v2 Interactive Builder",
            "created_date": datetime.now().isoformat(),
            "analysis_mode": self.analysis_mode,
            "user_level": self.user_level,
        }

        # Configuration summary
        print("📋 Configuration Summary:")
        print(f"   • Analysis mode: {self.colorize(self.analysis_mode, 'bold')}")
        print(
            f"   • Parameters: {len(self.config_data['initial_parameters']['parameter_names'])}"
        )

        temporal = self.config_data["analyzer_parameters"]["temporal"]
        frames = temporal["end_frame"] - temporal["start_frame"]
        print(f"   • Analysis frames: {frames}")

        if "optimization_config" in self.config_data:
            angle_enabled = self.config_data["optimization_config"]["angle_filtering"][
                "enabled"
            ]
            print(f"   • Angle filtering: {'enabled' if angle_enabled else 'disabled'}")

        # Show estimated performance
        if self.analysis_mode == "static_isotropic":
            perf_estimate = "Very fast (seconds to minutes)"
        elif self.analysis_mode == "static_anisotropic":
            perf_estimate = "Fast (minutes)"
        else:
            perf_estimate = "Moderate (minutes to hours)"

        print(f"   • Estimated runtime: {self.colorize(perf_estimate, 'green')}")

        # Save options
        print(f"\n{self.colorize('💾 Save Configuration:', 'bold')}")

        # Determine output path
        if self.output_path is None:
            default_name = f"homodyne_{self.analysis_mode}_config.yaml"
            output_path = self.ask_value("Output file name", str, default=default_name)
            self.output_path = Path(output_path)

        # Save format
        if self.output_path.suffix.lower() not in [".yaml", ".yml", ".json"]:
            format_choice = self.ask_choice(
                "Choose output format:",
                ["YAML (recommended)", "JSON"],
                default="YAML (recommended)",
            )

            if "YAML" in format_choice:
                self.output_path = self.output_path.with_suffix(".yaml")
            else:
                self.output_path = self.output_path.with_suffix(".json")

        # Write configuration file
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.output_path, "w", encoding="utf-8") as f:
                if self.output_path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
                    yaml.dump(
                        self.config_data,
                        f,
                        default_flow_style=False,
                        indent=2,
                        sort_keys=False,
                    )
                else:
                    json.dump(self.config_data, f, indent=2)

            print(f"\n{self.colorize('✅ Configuration saved successfully!', 'green')}")
            print(f"File: {self.output_path}")

            # Offer validation
            validate = self.ask_yes_no("Validate configuration now?", default=True)
            if validate:
                validator = ConfigurationValidator()
                result = validator.validate_configuration(self.config_data)
                print(f"\n{result}")

                if not result.is_valid:
                    print(
                        f"\n{self.colorize('⚠️  Configuration has issues but was saved.', 'yellow')}"
                    )
                    print("You can fix these issues and re-run validation later.")

        except Exception as e:
            print(f"\n{self.colorize(f'❌ Failed to save configuration: {e}', 'red')}")
            logger.error(f"Failed to save configuration to {self.output_path}: {e}")

        print(f"\n{self.colorize('🎉 Configuration setup complete!', 'bold')}")
        print("You can now run your analysis with:")
        print(f"  homodyne analyze {self.output_path}")


class ConfigurationValidator:
    """Real-time configuration validator with detailed feedback."""

    def __init__(self):
        self.validation_rules = self._load_validation_rules()

    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules for configuration checking."""
        return {
            "required_sections": [
                "analyzer_parameters",
                "experimental_data",
                "analysis_settings",
                "initial_parameters",
            ],
            "parameter_ranges": {
                "wavevector_q": (1e-6, 10.0),
                "dt": (1e-6, 3600.0),
                "start_frame": (1, 1e6),
                "end_frame": (1, 1e6),
                "stator_rotor_gap": (1.0, 1e9),
                "D0": (1.0, 1e6),
                "alpha": (-10.0, 10.0),
                "D_offset": (-1e5, 1e5),
            },
            "mode_parameter_counts": {
                "static_isotropic": 3,
                "static_anisotropic": 3,
                "laminar_flow": 7,
            },
        }

    def validate_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate a complete configuration."""
        result = ValidationResult(is_valid=True)
        score = 100

        # Check required sections
        missing_sections = []
        for section in self.validation_rules["required_sections"]:
            if section not in config:
                missing_sections.append(section)
                result.errors.append(f"Missing required section: {section}")
                score -= 20

        if missing_sections:
            result.is_valid = False
            result.suggestions.append(
                "Add missing sections using the interactive builder"
            )

        # Validate parameter ranges
        self._validate_parameter_ranges(config, result)

        # Validate mode consistency
        mode_score = self._validate_mode_consistency(config, result)
        score += mode_score

        # Check performance settings
        perf_score = self._validate_performance_settings(config, result)
        score += perf_score

        # Check data file accessibility (if paths exist)
        self._validate_data_accessibility(config, result)

        # Validate logging configuration
        self._validate_logging_config(config, result)

        result.score = max(0, min(100, score))

        if result.score >= 80:
            result.performance_notes.append(
                f"Excellent configuration (score: {result.score}/100)"
            )
        elif result.score >= 60:
            result.performance_notes.append(
                f"Good configuration (score: {result.score}/100)"
            )
            result.suggestions.append("Consider addressing warnings to improve score")
        else:
            result.performance_notes.append(
                f"Configuration needs improvement (score: {result.score}/100)"
            )
            result.suggestions.append("Use interactive builder to fix issues")

        return result

    def _validate_parameter_ranges(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate parameter values are within reasonable ranges."""
        analyzer_params = config.get("analyzer_parameters", {})

        # Check temporal parameters
        temporal = analyzer_params.get("temporal", {})
        for param, value in temporal.items():
            if param in self.validation_rules["parameter_ranges"]:
                min_val, max_val = self.validation_rules["parameter_ranges"][param]
                if not (min_val <= value <= max_val):
                    result.warnings.append(
                        f"{param} = {value} outside typical range [{min_val}, {max_val}]"
                    )
                    result.suggestions.append(
                        f"Consider adjusting {param} to be within typical range"
                    )

        # Check scattering parameters
        scattering = analyzer_params.get("scattering", {})
        q = scattering.get("wavevector_q")
        if q is not None:
            min_q, max_q = self.validation_rules["parameter_ranges"]["wavevector_q"]
            if not (min_q <= q <= max_q):
                result.warnings.append(
                    f"wavevector_q = {q} outside typical XPCS range [{min_q}, {max_q}]"
                )

            if q > 0.1:
                result.performance_notes.append(
                    "Large q-vector may require longer computation time"
                )

        # Check frame range consistency
        start = temporal.get("start_frame", 1)
        end = temporal.get("end_frame", 100)
        if start >= end:
            result.errors.append(
                f"Invalid frame range: start_frame ({start}) >= end_frame ({end})"
            )
            result.is_valid = False

        frame_count = end - start
        if frame_count < 10:
            result.warnings.append(
                f"Very few frames ({frame_count}) may give poor statistics"
            )
        elif frame_count > 10000:
            result.performance_notes.append(
                f"Large frame count ({frame_count}) will increase computation time"
            )

    def _validate_mode_consistency(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> int:
        """Validate analysis mode consistency."""
        score_delta = 0

        analysis_settings = config.get("analysis_settings", {})
        initial_params = config.get("initial_parameters", {})

        # Determine expected mode
        static_mode = analysis_settings.get("static_mode", True)
        static_submode = analysis_settings.get("static_submode", "anisotropic")

        if static_mode:
            if static_submode == "isotropic":
                expected_mode = "static_isotropic"
            else:
                expected_mode = "static_anisotropic"
        else:
            expected_mode = "laminar_flow"

        # Check parameter count consistency
        param_values = initial_params.get("values", [])
        param_count = len(param_values)
        expected_count = self.validation_rules["mode_parameter_counts"].get(
            expected_mode, 3
        )

        if param_count != expected_count:
            result.errors.append(
                f"Parameter count mismatch: {expected_mode} mode expects {expected_count} parameters, got {param_count}"
            )
            result.is_valid = False
            result.suggestions.append(f"Adjust parameter count or change analysis mode")
            score_delta -= 20
        else:
            result.performance_notes.append(
                f"Parameter count correctly matches {expected_mode} mode"
            )
            score_delta += 5

        # Check angle filtering consistency
        opt_config = config.get("optimization_config", {})
        angle_filtering = opt_config.get("angle_filtering", {})
        filtering_enabled = angle_filtering.get("enabled", True)

        if expected_mode == "static_isotropic" and filtering_enabled:
            result.warnings.append(
                "Angle filtering enabled for isotropic mode (will be ignored)"
            )
            result.suggestions.append(
                "Disable angle filtering for isotropic mode to avoid confusion"
            )
            score_delta -= 5
        elif expected_mode != "static_isotropic" and not filtering_enabled:
            result.suggestions.append(
                "Consider enabling angle filtering for better analysis quality"
            )

        return score_delta

    def _validate_performance_settings(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> int:
        """Validate performance-related settings."""
        score_delta = 0

        # Check v2 features
        v2_features = config.get("v2_features", {})
        if v2_features:
            if v2_features.get("performance_optimization", True):
                result.performance_notes.append("Performance optimizations enabled")
                score_delta += 5

            if v2_features.get("parallel_processing", False):
                result.performance_notes.append(
                    "Parallel processing enabled - may speed up analysis"
                )
                score_delta += 5

            if v2_features.get("gpu_acceleration", False):
                result.performance_notes.append(
                    "GPU acceleration enabled - significant speedup expected"
                )
                score_delta += 10

            cache_strategy = v2_features.get("cache_strategy", "intelligent")
            if cache_strategy == "disabled":
                result.performance_notes.append(
                    "Caching disabled - analysis will be slower"
                )
                score_delta -= 5
            elif cache_strategy == "intelligent":
                result.performance_notes.append(
                    "Intelligent caching enabled - good balance"
                )
                score_delta += 3

        return score_delta

    def _validate_data_accessibility(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate data file accessibility."""
        exp_data = config.get("experimental_data", {})
        data_folder = exp_data.get("data_folder_path", "./data/")
        data_file = exp_data.get("data_file_name", "data.hdf")

        if data_folder and data_file:
            data_path = Path(data_folder) / data_file

            if not Path(data_folder).exists():
                result.warnings.append(f"Data folder does not exist: {data_folder}")
                result.suggestions.append(
                    "Create data folder or update path before running analysis"
                )
            elif not data_path.exists():
                result.warnings.append(f"Data file not found: {data_path}")
                result.suggestions.append(
                    "Place your data file in the specified location"
                )
            else:
                result.performance_notes.append("Data file path verified")

    def _validate_logging_config(
        self, config: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate logging configuration."""
        logging_config = config.get("logging", {})

        if not logging_config.get("log_to_console", True) and not logging_config.get(
            "log_to_file", False
        ):
            result.warnings.append(
                "Both console and file logging are disabled - no output will be visible"
            )
            result.suggestions.append(
                "Enable at least console logging to see analysis progress"
            )

        log_level = logging_config.get("level", "INFO")
        if log_level == "DEBUG":
            result.performance_notes.append("DEBUG logging may slow down analysis")
        elif log_level == "ERROR":
            result.warnings.append("ERROR-only logging may hide important information")

    def validate_realtime(self, config_section: str, config_value: Any) -> List[str]:
        """Provide real-time validation feedback for a configuration section."""
        issues = []

        if config_section == "wavevector_q" and isinstance(config_value, (int, float)):
            if config_value <= 0:
                issues.append("Wavevector must be positive")
            elif config_value > 1.0:
                issues.append(
                    "Very large wavevector - typical XPCS range is 0.001-0.1 Å⁻¹"
                )
            elif config_value < 0.0001:
                issues.append("Very small wavevector - may not be physical")

        elif config_section == "frame_range" and isinstance(config_value, dict):
            start = config_value.get("start_frame", 1)
            end = config_value.get("end_frame", 100)
            if start >= end:
                issues.append("End frame must be greater than start frame")
            elif end - start < 10:
                issues.append(
                    "Very few frames - consider using more frames for better statistics"
                )

        return issues


class ConfigurationComparator:
    """Tool for comparing configurations and showing differences."""

    def __init__(self):
        self.comparison_categories = [
            "analysis_mode",
            "parameters",
            "performance",
            "data_paths",
            "optimization",
        ]

    def compare_configurations(
        self, config1: Dict[str, Any], config2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two configurations and return detailed differences."""
        comparison = {"summary": {}, "differences": {}, "recommendations": []}

        # Analysis mode comparison
        mode1 = self._get_analysis_mode(config1)
        mode2 = self._get_analysis_mode(config2)

        if mode1 != mode2:
            comparison["differences"]["analysis_mode"] = {
                "config1": mode1,
                "config2": mode2,
                "impact": "Different analysis approaches - affects parameters and performance",
            }

        # Parameter comparison
        params1 = config1.get("initial_parameters", {}).get("values", [])
        params2 = config2.get("initial_parameters", {}).get("values", [])

        if params1 != params2:
            comparison["differences"]["parameters"] = {
                "config1": f"{len(params1)} parameters: {params1[:3]}...",
                "config2": f"{len(params2)} parameters: {params2[:3]}...",
                "impact": "Different parameter values will affect analysis results",
            }

        # Performance comparison
        perf1 = self._estimate_performance(config1)
        perf2 = self._estimate_performance(config2)

        comparison["summary"]["performance_comparison"] = {
            "config1": perf1,
            "config2": perf2,
            "faster": "config1" if perf1["score"] > perf2["score"] else "config2",
        }

        # Generate recommendations
        if mode1 == "static_isotropic" and mode2 != "static_isotropic":
            comparison["recommendations"].append(
                "Config1 uses static_isotropic mode which is fastest - consider if appropriate for your system"
            )

        if len(params1) != len(params2):
            comparison["recommendations"].append(
                "Parameter count differs - ensure you're using the right mode for your analysis needs"
            )

        return comparison

    def _get_analysis_mode(self, config: Dict[str, Any]) -> str:
        """Extract analysis mode from configuration."""
        analysis = config.get("analysis_settings", {})
        static_mode = analysis.get("static_mode", True)

        if static_mode:
            submode = analysis.get("static_submode", "anisotropic")
            return f"static_{submode}"
        else:
            return "laminar_flow"

    def _estimate_performance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate relative performance of configuration."""
        score = 50  # Base score
        details = []

        # Mode impact
        mode = self._get_analysis_mode(config)
        if mode == "static_isotropic":
            score += 30
            details.append("Static isotropic: fastest mode")
        elif mode == "static_anisotropic":
            score += 15
            details.append("Static anisotropic: medium speed")
        else:
            score -= 10
            details.append("Laminar flow: comprehensive but slower")

        # Frame count impact
        temporal = config.get("analyzer_parameters", {}).get("temporal", {})
        frame_count = temporal.get("end_frame", 100) - temporal.get("start_frame", 1)

        if frame_count < 1000:
            score += 10
            details.append("Small frame count: fast")
        elif frame_count > 5000:
            score -= 15
            details.append("Large frame count: slower")

        # Optimization settings
        angle_filtering = (
            config.get("optimization_config", {})
            .get("angle_filtering", {})
            .get("enabled", True)
        )
        if not angle_filtering:
            score += 5
            details.append("Angle filtering disabled: slight speedup")

        # V2 features
        v2_features = config.get("v2_features", {})
        if v2_features.get("gpu_acceleration", False):
            score += 25
            details.append("GPU acceleration: major speedup")
        elif v2_features.get("parallel_processing", False):
            score += 10
            details.append("Parallel processing: moderate speedup")

        return {
            "score": max(0, min(100, score)),
            "details": details,
            "estimated_time": self._score_to_time_estimate(score),
        }

    def _score_to_time_estimate(self, score: int) -> str:
        """Convert performance score to time estimate."""
        if score >= 80:
            return "seconds to minutes"
        elif score >= 60:
            return "minutes"
        elif score >= 40:
            return "minutes to hours"
        else:
            return "hours or longer"


def create_interactive_builder(
    output_path: Optional[Union[str, Path]] = None,
) -> InteractiveConfigurationBuilder:
    """Create an interactive configuration builder."""
    return InteractiveConfigurationBuilder(output_path)


def validate_configuration_file(config_path: Union[str, Path]) -> ValidationResult:
    """Validate a configuration file and return detailed results."""
    validator = ConfigurationValidator()

    try:
        config_path = Path(config_path)
        if not config_path.exists():
            return ValidationResult(
                is_valid=False,
                errors=[f"Configuration file not found: {config_path}"],
                score=0,
            )

        # Load configuration
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
                config_data = yaml.safe_load(f)
            else:
                config_data = json.load(f)

        return validator.validate_configuration(config_data)

    except Exception as e:
        return ValidationResult(
            is_valid=False, errors=[f"Failed to validate configuration: {e}"], score=0
        )


def compare_configuration_files(
    config1_path: Union[str, Path], config2_path: Union[str, Path]
) -> Dict[str, Any]:
    """Compare two configuration files."""
    comparator = ConfigurationComparator()

    try:
        # Load configurations
        configs = []
        for path in [config1_path, config2_path]:
            path = Path(path)
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in [".yaml", ".yml"] and HAS_YAML:
                    config = yaml.safe_load(f)
                else:
                    config = json.load(f)
                configs.append(config)

        return comparator.compare_configurations(configs[0], configs[1])

    except Exception as e:
        return {
            "error": f"Failed to compare configurations: {e}",
            "summary": {},
            "differences": {},
            "recommendations": [],
        }
