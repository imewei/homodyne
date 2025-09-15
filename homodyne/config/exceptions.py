"""
Enhanced Error Handling and Exception System for Homodyne v2
==========================================================

Comprehensive exception classes with actionable error messages, specific fix suggestions,
code examples, and configuration snippets for the homodyne configuration system.

This module transforms generic error messages into specific, actionable guidance
that helps users quickly resolve configuration issues.

Key Features:
- Actionable error messages with specific fix suggestions
- Inline YAML/JSON configuration examples
- Multiple solution options for complex issues
- Performance impact warnings and recommendations
- Auto-correction suggestions for common mistakes
- Fuzzy matching for configuration key typos
- Context-aware error messages based on current configuration

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class ErrorContext:
    """Context information for enhanced error messages."""

    config_file: Optional[str] = None
    current_config: Optional[Dict[str, Any]] = None
    analysis_mode: Optional[str] = None
    parameter_count: Optional[int] = None
    user_level: str = "intermediate"  # novice, intermediate, expert
    file_location: Optional[Tuple[str, int]] = None  # (filename, line_number)


@dataclass
class FixSuggestion:
    """A specific fix suggestion with code examples."""

    title: str
    description: str
    code_example: Optional[str] = None
    yaml_example: Optional[str] = None
    json_example: Optional[str] = None
    performance_impact: Optional[str] = None
    difficulty: str = "easy"  # easy, medium, hard
    estimated_time: Optional[str] = None


class HomodyneConfigurationError(Exception):
    """Base class for all homodyne configuration errors with enhanced messaging."""

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[FixSuggestion]] = None,
        related_docs: Optional[List[str]] = None,
        error_code: Optional[str] = None,
    ):
        self.original_message = message
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []
        self.related_docs = related_docs or []
        self.error_code = error_code or "CONFIG_ERROR"

        # Generate enhanced error message
        enhanced_message = self._generate_enhanced_message()
        super().__init__(enhanced_message)

    def _generate_enhanced_message(self) -> str:
        """Generate comprehensive, actionable error message."""
        lines = []

        # Header with error code and context
        lines.append(f"🚨 Homodyne Configuration Error [{self.error_code}]")
        lines.append("=" * 60)

        # Original error message
        lines.append(f"Issue: {self.original_message}")

        # Context information
        if self.context.config_file:
            lines.append(f"File: {self.context.config_file}")
        if self.context.file_location:
            filename, line_num = self.context.file_location
            lines.append(f"Location: {filename}:{line_num}")
        if self.context.analysis_mode:
            lines.append(f"Analysis Mode: {self.context.analysis_mode}")

        lines.append("")

        # Fix suggestions
        if self.suggestions:
            lines.append("💡 Suggested Solutions:")
            lines.append("-" * 25)

            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"{i}. {suggestion.title}")
                lines.append(f"   {suggestion.description}")

                if suggestion.difficulty != "easy":
                    lines.append(f"   Difficulty: {suggestion.difficulty}")
                if suggestion.estimated_time:
                    lines.append(f"   Estimated time: {suggestion.estimated_time}")
                if suggestion.performance_impact:
                    lines.append(
                        f"   Performance impact: {suggestion.performance_impact}"
                    )

                # Code examples
                if suggestion.yaml_example:
                    lines.append("   YAML Example:")
                    for yaml_line in suggestion.yaml_example.strip().split("\n"):
                        lines.append(f"   {yaml_line}")

                if suggestion.json_example:
                    lines.append("   JSON Example:")
                    for json_line in suggestion.json_example.strip().split("\n"):
                        lines.append(f"   {json_line}")

                if suggestion.code_example:
                    lines.append("   Code Example:")
                    for code_line in suggestion.code_example.strip().split("\n"):
                        lines.append(f"   {code_line}")

                lines.append("")

        # Related documentation
        if self.related_docs:
            lines.append("📚 Related Documentation:")
            for doc in self.related_docs:
                lines.append(f"   • {doc}")
            lines.append("")

        # Quick help footer
        lines.append("ℹ️  For more help:")
        lines.append(
            "   • Run 'homodyne config --validate' to check your configuration"
        )
        lines.append("   • Use 'homodyne config --interactive' for guided setup")
        lines.append("   • See templates in ~/.homodyne/templates/")

        return "\n".join(lines)


class ConfigurationFileError(HomodyneConfigurationError):
    """Error related to configuration file loading and parsing."""

    def __init__(
        self,
        filename: str,
        original_error: Exception,
        context: Optional[ErrorContext] = None,
    ):
        self.filename = filename
        self.original_error = original_error

        # Determine specific file error type
        if isinstance(original_error, FileNotFoundError):
            error_code = "FILE_NOT_FOUND"
            message = f"Configuration file '{filename}' not found"
            suggestions = self._get_file_not_found_suggestions()
        elif "YAML" in str(type(original_error).__name__):
            error_code = "YAML_SYNTAX_ERROR"
            message = f"YAML syntax error in '{filename}': {original_error}"
            suggestions = self._get_yaml_syntax_suggestions()
        elif "JSON" in str(type(original_error).__name__):
            error_code = "JSON_SYNTAX_ERROR"
            message = f"JSON syntax error in '{filename}': {original_error}"
            suggestions = self._get_json_syntax_suggestions()
        else:
            error_code = "FILE_READ_ERROR"
            message = (
                f"Failed to read configuration file '{filename}': {original_error}"
            )
            suggestions = self._get_file_read_suggestions()

        related_docs = [
            "Configuration File Format: docs/configuration.md",
            "YAML Syntax Guide: docs/yaml_guide.md",
            "Configuration Templates: ~/.homodyne/templates/",
        ]

        super().__init__(message, context, suggestions, related_docs, error_code)

    def _get_file_not_found_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for missing configuration files."""
        return [
            FixSuggestion(
                title="Create configuration from template",
                description="Generate a new configuration file from a built-in template",
                code_example="""
# Create default static isotropic configuration
homodyne config --create-template static_isotropic

# Create advanced configuration with all options
homodyne config --create-template advanced
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Use interactive configuration builder",
                description="Launch the interactive configuration wizard",
                code_example="homodyne config --interactive",
                difficulty="easy",
                estimated_time="5-10 minutes",
            ),
            FixSuggestion(
                title="Copy from another location",
                description="If you have a configuration file elsewhere, copy it to the expected location",
                yaml_example=f"""
# Copy your existing configuration
cp /path/to/your/config.yaml {self.filename}

# Or create a symbolic link
ln -s /path/to/your/config.yaml {self.filename}
                """,
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Check current directory",
                description="Verify you're running homodyne from the correct directory",
                code_example="""
# List files in current directory
ls -la *.yaml *.json

# Find configuration files
find . -name "*.yaml" -o -name "*.json"
                """,
                difficulty="easy",
                estimated_time="1 minute",
            ),
        ]

    def _get_yaml_syntax_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for YAML syntax errors."""
        return [
            FixSuggestion(
                title="Validate YAML syntax",
                description="Use a YAML validator to identify syntax issues",
                code_example="""
# Validate YAML file
python -c "import yaml; yaml.safe_load(open('config.yaml'))"

# Or use online validator: yamllint.readthedocs.io
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Check indentation",
                description="YAML is sensitive to indentation. Use spaces, not tabs",
                yaml_example="""
# CORRECT (using spaces)
analysis_settings:
  static_mode: true
  static_submode: isotropic

# INCORRECT (mixing tabs/spaces or wrong indentation)
analysis_settings:
    static_mode: true
        static_submode: isotropic
                """,
                difficulty="easy",
                estimated_time="3 minutes",
            ),
            FixSuggestion(
                title="Check for special characters",
                description="Quote strings containing special characters",
                yaml_example="""
# CORRECT (quoted strings with special characters)
data_folder_path: "/path/with spaces/data"
comment: "This is a comment with: colons"

# INCORRECT (unquoted special characters)
data_folder_path: /path/with spaces/data
comment: This is a comment with: colons
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Regenerate from template",
                description="Start with a fresh template and re-add your customizations",
                code_example="homodyne config --create-template default --overwrite",
                performance_impact="No performance impact",
                difficulty="medium",
                estimated_time="10 minutes",
            ),
        ]

    def _get_json_syntax_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for JSON syntax errors."""
        return [
            FixSuggestion(
                title="Validate JSON syntax",
                description="Use a JSON validator to identify syntax issues",
                code_example="""
# Validate JSON file
python -c "import json; json.load(open('config.json'))"

# Or use online validator: jsonlint.com
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Check for trailing commas",
                description="JSON doesn't allow trailing commas (unlike JavaScript)",
                json_example="""
{
  "analysis_settings": {
    "static_mode": true,
    "static_submode": "isotropic"
  }
}

// INCORRECT (trailing comma)
{
  "analysis_settings": {
    "static_mode": true,
    "static_submode": "isotropic",
  }
}
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Convert to YAML",
                description="YAML is more forgiving and easier to edit manually",
                code_example="homodyne config --convert json-to-yaml config.json",
                performance_impact="No performance impact, YAML is preferred",
                difficulty="easy",
                estimated_time="2 minutes",
            ),
        ]

    def _get_file_read_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for general file reading errors."""
        return [
            FixSuggestion(
                title="Check file permissions",
                description="Ensure the configuration file is readable",
                code_example=f"""
# Check file permissions
ls -la {self.filename}

# Fix permissions if needed
chmod 644 {self.filename}
                """,
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Check file encoding",
                description="Ensure the file is saved in UTF-8 encoding",
                code_example=f"""
# Check file encoding
file -i {self.filename}

# Convert encoding if needed (from Latin-1 to UTF-8)
iconv -f latin1 -t utf8 {self.filename} > {self.filename}.utf8
mv {self.filename}.utf8 {self.filename}
                """,
                difficulty="medium",
                estimated_time="3 minutes",
            ),
        ]


class ParameterValidationError(HomodyneConfigurationError):
    """Error related to parameter validation with specific guidance."""

    def __init__(
        self,
        parameter_name: str,
        value: Any,
        expected_range: Optional[Tuple[float, float]] = None,
        context: Optional[ErrorContext] = None,
        validation_type: str = "range",
    ):
        self.parameter_name = parameter_name
        self.value = value
        self.expected_range = expected_range
        self.validation_type = validation_type

        message = self._generate_parameter_message()
        suggestions = self._get_parameter_suggestions()
        error_code = f"PARAM_{validation_type.upper()}_ERROR"

        related_docs = [
            "Parameter Reference: docs/parameters.md",
            "Physics Validation Guide: docs/physics_validation.md",
            f"Parameter Bounds for {parameter_name}: docs/parameters.md#{parameter_name.lower()}",
        ]

        super().__init__(message, context, suggestions, related_docs, error_code)

    def _generate_parameter_message(self) -> str:
        """Generate specific parameter validation message."""
        if self.validation_type == "range":
            if self.expected_range:
                min_val, max_val = self.expected_range
                return f"Parameter '{self.parameter_name}' = {self.value} is outside valid range [{min_val}, {max_val}]"
            else:
                return f"Parameter '{self.parameter_name}' = {self.value} is invalid"
        elif self.validation_type == "positive":
            return f"Parameter '{self.parameter_name}' = {self.value} must be positive"
        elif self.validation_type == "type":
            return f"Parameter '{self.parameter_name}' has invalid type: expected number, got {type(self.value).__name__}"
        else:
            return f"Parameter '{self.parameter_name}' = {self.value} failed validation ({self.validation_type})"

    def _get_parameter_suggestions(self) -> List[FixSuggestion]:
        """Get parameter-specific suggestions."""
        suggestions = []

        # Parameter-specific guidance
        if self.parameter_name == "wavevector_q":
            suggestions.extend(self._get_wavevector_suggestions())
        elif self.parameter_name in ["start_frame", "end_frame"]:
            suggestions.extend(self._get_frame_suggestions())
        elif self.parameter_name == "dt":
            suggestions.extend(self._get_time_step_suggestions())
        elif self.parameter_name in ["D0", "alpha", "D_offset"]:
            suggestions.extend(self._get_diffusion_parameter_suggestions())
        elif "gamma_dot" in self.parameter_name:
            suggestions.extend(self._get_shear_parameter_suggestions())
        else:
            suggestions.extend(self._get_generic_parameter_suggestions())

        return suggestions

    def _get_wavevector_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for wavevector parameter issues."""
        return [
            FixSuggestion(
                title="Use typical XPCS wavevector range",
                description="Most XPCS experiments use q-vectors between 0.001-0.1 Å⁻¹",
                yaml_example="""
analyzer_parameters:
  scattering:
    wavevector_q: 0.0054  # Typical value for XPCS
                """,
                performance_impact="No performance impact",
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Calculate from experimental geometry",
                description="Calculate q = (4π/λ)sin(θ/2) from your experimental setup",
                code_example="""
import numpy as np

# Example calculation
wavelength_angstrom = 1.24  # X-ray wavelength in Angstrom
scattering_angle_deg = 0.15  # Scattering angle in degrees
theta_rad = np.radians(scattering_angle_deg / 2)
q = (4 * np.pi / wavelength_angstrom) * np.sin(theta_rad)
print(f"Calculated q-vector: {q:.6f} Å⁻¹")
                """,
                difficulty="medium",
                estimated_time="5 minutes",
            ),
            FixSuggestion(
                title="Check experimental parameters",
                description="Verify your X-ray energy, detector distance, and pixel size",
                difficulty="medium",
                estimated_time="Variable",
            ),
        ]

    def _get_frame_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for frame range issues."""
        return [
            FixSuggestion(
                title="Use reasonable frame range",
                description="Ensure start_frame < end_frame and sufficient frames for analysis",
                yaml_example="""
analyzer_parameters:
  temporal:
    start_frame: 1      # First frame to analyze
    end_frame: 1000     # Last frame (must be > start_frame)
                """,
                performance_impact="More frames = longer computation time",
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Check your data file",
                description="Verify your data file actually contains the specified frame range",
                code_example="""
# Check HDF5 file structure
h5dump -n your_data.hdf

# Or using Python
import h5py
with h5py.File('your_data.hdf', 'r') as f:
    print("Available datasets:", list(f.keys()))
    if 'C2_frames' in f:
        print("Frame shape:", f['C2_frames'].shape)
                """,
                difficulty="medium",
                estimated_time="5 minutes",
            ),
        ]

    def _get_time_step_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for time step parameter issues."""
        return [
            FixSuggestion(
                title="Use experimental time step",
                description="Set dt to match your experimental acquisition time step",
                yaml_example="""
analyzer_parameters:
  temporal:
    dt: 0.1  # Time step in seconds (match your experiment)
                """,
                performance_impact="No performance impact",
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Consider your dynamics timescale",
                description="Choose dt appropriate for the dynamics you want to resolve",
                yaml_example="""
# Fast dynamics (millisecond timescale)
dt: 0.001

# Medium dynamics (0.1 second timescale)  
dt: 0.1

# Slow dynamics (second timescale)
dt: 1.0
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
        ]

    def _get_diffusion_parameter_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for diffusion parameter issues."""
        param_info = {
            "D0": ("Diffusion coefficient", "1e-3 to 1e5", "Å²/s"),
            "alpha": ("Anomalous diffusion exponent", "-10.0 to 10.0", "dimensionless"),
            "D_offset": ("Diffusion offset", "-1e5 to 1e5", "Å²/s"),
        }

        name, desc, unit = param_info.get(
            self.parameter_name, ("Parameter", "See documentation", "")
        )

        return [
            FixSuggestion(
                title=f"Use typical {name.lower()} values",
                description=f"{desc} typically ranges from {param_info[self.parameter_name][1]} {unit}",
                yaml_example=f"""
initial_parameters:
  values: [100.0, 0.0, 10.0]  # [D0, alpha, D_offset]
  parameter_names: ["D0", "alpha", "D_offset"]
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Check physical reasonableness",
                description=f"Ensure {name.lower()} makes physical sense for your system",
                difficulty="medium",
                estimated_time="Variable",
            ),
        ]

    def _get_shear_parameter_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for shear flow parameter issues."""
        return [
            FixSuggestion(
                title="Use laminar flow parameter ranges",
                description="Shear rate parameters should be appropriate for your rheological conditions",
                yaml_example="""
initial_parameters:
  values: [100.0, 0.0, 10.0, 0.01, 0.0, 0.0, 0.0]  # All 7 parameters
  parameter_names: ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Switch to static mode if no shear",
                description="If you don't have shear flow, use static mode instead",
                yaml_example="""
analysis_settings:
  static_mode: true
  static_submode: anisotropic  # or isotropic
                """,
                performance_impact="Static mode is ~3x faster than laminar flow",
                difficulty="easy",
                estimated_time="1 minute",
            ),
        ]

    def _get_generic_parameter_suggestions(self) -> List[FixSuggestion]:
        """Generic suggestions for parameter validation."""
        return [
            FixSuggestion(
                title="Check parameter bounds",
                description="Verify parameter is within its defined bounds",
                yaml_example="""
parameter_space:
  bounds:
    - name: "your_parameter"
      min: 0.0     # Adjust minimum value
      max: 100.0   # Adjust maximum value
      type: "Normal"
                """,
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Use parameter validation tool",
                description="Run the built-in parameter validation",
                code_example="homodyne config --validate-parameters",
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Consult parameter documentation",
                description="Check the parameter reference guide for valid ranges",
                related_docs=["docs/parameters.md"],
                difficulty="easy",
                estimated_time="5 minutes",
            ),
        ]


class AnalysisModeError(HomodyneConfigurationError):
    """Error related to analysis mode configuration and compatibility."""

    def __init__(
        self,
        current_mode: Optional[str] = None,
        parameter_count: Optional[int] = None,
        context: Optional[ErrorContext] = None,
        issue_type: str = "mode_mismatch",
    ):
        self.current_mode = current_mode
        self.parameter_count = parameter_count
        self.issue_type = issue_type

        message = self._generate_mode_message()
        suggestions = self._get_mode_suggestions()
        error_code = f"MODE_{issue_type.upper()}_ERROR"

        related_docs = [
            "Analysis Modes Guide: docs/analysis_modes.md",
            "Parameter Mapping: docs/parameter_modes.md",
            "Performance Comparison: docs/mode_performance.md",
        ]

        super().__init__(message, context, suggestions, related_docs, error_code)

    def _generate_mode_message(self) -> str:
        """Generate mode-specific error message."""
        if self.issue_type == "mode_mismatch":
            return f"Analysis mode '{self.current_mode}' incompatible with {self.parameter_count} parameters"
        elif self.issue_type == "invalid_mode":
            return f"Unknown analysis mode: '{self.current_mode}'"
        elif self.issue_type == "parameter_count":
            return (
                f"Parameter count {self.parameter_count} doesn't match any valid mode"
            )
        else:
            return f"Analysis mode configuration error: {self.issue_type}"

    def _get_mode_suggestions(self) -> List[FixSuggestion]:
        """Get mode-specific suggestions."""
        suggestions = []

        if self.issue_type == "mode_mismatch":
            suggestions.extend(self._get_mismatch_suggestions())
        elif self.issue_type == "invalid_mode":
            suggestions.extend(self._get_invalid_mode_suggestions())
        else:
            suggestions.extend(self._get_generic_mode_suggestions())

        return suggestions

    def _get_mismatch_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for mode/parameter count mismatch."""
        return [
            FixSuggestion(
                title="Use static mode (3 parameters)",
                description="Configure for static analysis with diffusion-only parameters",
                yaml_example="""
analysis_settings:
  static_mode: true
  static_submode: isotropic    # or anisotropic

initial_parameters:
  values: [100.0, 0.0, 10.0]   # [D0, alpha, D_offset]
  parameter_names: ["D0", "alpha", "D_offset"]
                """,
                performance_impact="2-3x faster than laminar flow mode",
                difficulty="easy",
                estimated_time="2 minutes",
            ),
            FixSuggestion(
                title="Use laminar flow mode (7 parameters)",
                description="Configure for full laminar flow analysis with shear parameters",
                yaml_example="""
analysis_settings:
  static_mode: false

initial_parameters:
  values: [100.0, 0.0, 10.0, 0.01, 0.0, 0.0, 0.0]  # All 7 parameters
  parameter_names: ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
                """,
                performance_impact="Full capability but slower computation",
                difficulty="medium",
                estimated_time="5 minutes",
            ),
            FixSuggestion(
                title="Auto-detect mode from parameters",
                description="Let homodyne automatically determine the analysis mode",
                code_example="homodyne config --auto-detect-mode",
                difficulty="easy",
                estimated_time="1 minute",
            ),
        ]

    def _get_invalid_mode_suggestions(self) -> List[FixSuggestion]:
        """Suggestions for invalid mode specification."""
        valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        if self.current_mode:
            closest_match = difflib.get_close_matches(
                self.current_mode, valid_modes, n=1, cutoff=0.5
            )
            if closest_match:
                suggested_mode = closest_match[0]
                title = f"Use '{suggested_mode}' (closest match)"
                description = (
                    f"Did you mean '{suggested_mode}' instead of '{self.current_mode}'?"
                )
            else:
                suggested_mode = "static_isotropic"
                title = "Use valid analysis mode"
                description = f"Replace '{self.current_mode}' with a valid mode"
        else:
            suggested_mode = "static_isotropic"
            title = "Specify analysis mode"
            description = "Add a valid analysis mode to your configuration"

        return [
            FixSuggestion(
                title=title,
                description=description,
                yaml_example=f"""
# Option 1: Static isotropic (simplest, fastest)
analysis_settings:
  static_mode: true
  static_submode: isotropic

# Option 2: Static anisotropic (medium complexity)
analysis_settings:
  static_mode: true
  static_submode: anisotropic

# Option 3: Laminar flow (full capability)
analysis_settings:
  static_mode: false
                """,
                difficulty="easy",
                estimated_time="1 minute",
            ),
            FixSuggestion(
                title="Use interactive mode selector",
                description="Launch interactive wizard to choose appropriate mode",
                code_example="homodyne config --select-mode",
                difficulty="easy",
                estimated_time="3 minutes",
            ),
        ]

    def _get_generic_mode_suggestions(self) -> List[FixSuggestion]:
        """Generic mode configuration suggestions."""
        return [
            FixSuggestion(
                title="Review mode compatibility",
                description="Ensure your configuration matches your analysis needs",
                related_docs=["docs/analysis_modes.md"],
                difficulty="easy",
                estimated_time="5 minutes",
            ),
            FixSuggestion(
                title="Use configuration validator",
                description="Check mode consistency across your configuration",
                code_example="homodyne config --validate-mode",
                difficulty="easy",
                estimated_time="1 minute",
            ),
        ]


def suggest_typo_corrections(
    key: str, available_keys: List[str], max_suggestions: int = 3
) -> List[str]:
    """Suggest corrections for typos in configuration keys using fuzzy matching."""
    if not available_keys:
        return []

    matches = difflib.get_close_matches(
        key, available_keys, n=max_suggestions, cutoff=0.6
    )
    return matches


def generate_configuration_example(analysis_mode: str = "static_isotropic") -> str:
    """Generate a minimal working configuration example for the specified mode."""

    examples = {
        "static_isotropic": """
# Minimal Static Isotropic Configuration
metadata:
  config_version: "2.0"
  analysis_mode: "static_isotropic"

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
    enabled: false  # Disabled for isotropic mode

logging:
  log_to_console: true
  level: "INFO"
        """,
        "static_anisotropic": """
# Minimal Static Anisotropic Configuration  
metadata:
  config_version: "2.0"
  analysis_mode: "static_anisotropic"

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
  static_submode: "anisotropic"

initial_parameters:
  values: [100.0, 0.0, 10.0]
  parameter_names: ["D0", "alpha", "D_offset"]

optimization_config:
  angle_filtering:
    enabled: true
    target_ranges:
      - min_angle: -10.0
        max_angle: 10.0
      - min_angle: 170.0
        max_angle: 190.0

logging:
  log_to_console: true
  level: "INFO"
        """,
        "laminar_flow": """
# Minimal Laminar Flow Configuration
metadata:
  config_version: "2.0"
  analysis_mode: "laminar_flow"

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
  static_mode: false

initial_parameters:
  values: [100.0, 0.0, 10.0, 0.01, 0.0, 0.0, 0.0]
  parameter_names: ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]

optimization_config:
  angle_filtering:
    enabled: true
    target_ranges:
      - min_angle: -10.0
        max_angle: 10.0
      - min_angle: 170.0
        max_angle: 190.0

logging:
  log_to_console: true
  level: "INFO"
        """,
    }

    return examples.get(analysis_mode, examples["static_isotropic"])


class ConfigurationSuggestionEngine:
    """Engine for generating smart configuration suggestions and auto-corrections."""

    def __init__(self):
        self.common_typos = {
            # Common typos in configuration keys
            "analzer_parameters": "analyzer_parameters",
            "analyser_parameters": "analyzer_parameters",
            "analyzer_params": "analyzer_parameters",
            "experimental_data": "experimental_data",
            "experiemental_data": "experimental_data",
            "experiment_data": "experimental_data",
            "optimization_cfg": "optimization_config",
            "optim_config": "optimization_config",
            "logging_config": "logging",
            "log_config": "logging",
            "param_space": "parameter_space",
            "parameter_bounds": "parameter_space",
            "init_params": "initial_parameters",
            "initial_params": "initial_parameters",
        }

        self.mode_aliases = {
            "static": "static_anisotropic",
            "isotropic": "static_isotropic",
            "anisotropic": "static_anisotropic",
            "flow": "laminar_flow",
            "laminar": "laminar_flow",
            "shear": "laminar_flow",
        }

    def suggest_key_correction(
        self, wrong_key: str, available_keys: List[str]
    ) -> Optional[str]:
        """Suggest correction for a mistyped configuration key."""
        # Check direct typo corrections first
        if wrong_key in self.common_typos:
            return self.common_typos[wrong_key]

        # Use fuzzy matching
        matches = suggest_typo_corrections(wrong_key, available_keys, max_suggestions=1)
        return matches[0] if matches else None

    def suggest_mode_correction(self, wrong_mode: str) -> Optional[str]:
        """Suggest correction for analysis mode."""
        if wrong_mode in self.mode_aliases:
            return self.mode_aliases[wrong_mode]

        valid_modes = ["static_isotropic", "static_anisotropic", "laminar_flow"]
        matches = suggest_typo_corrections(wrong_mode, valid_modes, max_suggestions=1)
        return matches[0] if matches else None

    def generate_autocorrection_suggestions(
        self, config_dict: Dict[str, Any]
    ) -> List[FixSuggestion]:
        """Generate auto-correction suggestions for common configuration issues."""
        suggestions = []

        # Check for common key typos
        for key in config_dict.keys():
            if key in self.common_typos:
                correct_key = self.common_typos[key]
                suggestions.append(
                    FixSuggestion(
                        title=f"Rename '{key}' to '{correct_key}'",
                        description=f"Found common typo in configuration key",
                        yaml_example=f"""
# Change this:
{key}: ...

# To this:
{correct_key}: ...
                        """,
                        difficulty="easy",
                        estimated_time="1 minute",
                    )
                )

        # Check analysis mode
        analysis_settings = config_dict.get("analysis_settings", {})
        if isinstance(analysis_settings, dict):
            mode = analysis_settings.get("static_submode")
            if mode and mode in self.mode_aliases:
                correct_mode = self.mode_aliases[mode]
                suggestions.append(
                    FixSuggestion(
                        title=f"Use standard mode name '{correct_mode}'",
                        description=f"'{mode}' is an alias, use the full mode name",
                        yaml_example=f"""
analysis_settings:
  static_submode: "{correct_mode}"
                        """,
                        difficulty="easy",
                        estimated_time="1 minute",
                    )
                )

        return suggestions
