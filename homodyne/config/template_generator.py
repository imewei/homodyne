"""
Configuration Template Generator and Examples System for Homodyne v2
===================================================================

Comprehensive system for generating configuration templates, examples,
and guided configuration building with context-aware suggestions.

Key Features:
- Dynamic template generation for different analysis modes
- Context-aware configuration examples with inline documentation
- Template customization based on experimental conditions
- Migration tools for v1->v2 configuration upgrade
- Best practices integration with performance recommendations
- Validation-aware template generation
- Interactive template builder with guided prompts

Authors: Claude (Anthropic), based on Wei Chen & Hongrui He's design
Institution: Argonne National Laboratory
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

try:
    from .enhanced_console_output import (EnhancedConsoleLogger,
                                          create_console_logger)
    from .interactive_helpers import InteractiveConfigurationBuilder

    HAS_ENHANCED_OUTPUT = True
except ImportError:
    HAS_ENHANCED_OUTPUT = False
    EnhancedConsoleLogger = None
    InteractiveConfigurationBuilder = None

try:
    from homodyne.utils.logging import get_logger

    HAS_UTILS_LOGGING = True
except ImportError:
    import logging

    HAS_UTILS_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class AnalysisMode(Enum):
    """Analysis mode enumeration."""

    STATIC_ISOTROPIC = "static_isotropic"
    STATIC_ANISOTROPIC = "static_anisotropic"
    LAMINAR_FLOW = "laminar_flow"


class ExperimentType(Enum):
    """Experiment type enumeration."""

    HARD_SPHERES = "hard_spheres"
    SOFT_COLLOIDS = "soft_colloids"
    POLYMERS = "polymers"
    LIQUID_CRYSTALS = "liquid_crystals"
    GELS = "gels"
    BIOLOGICAL = "biological"
    CUSTOM = "custom"


class PerformanceProfile(Enum):
    """Performance profile enumeration."""

    FAST = "fast"  # Optimized for speed
    BALANCED = "balanced"  # Balance between speed and accuracy
    ACCURATE = "accurate"  # Optimized for accuracy
    CUSTOM = "custom"


@dataclass
class TemplateMetadata:
    """Metadata for configuration templates."""

    name: str
    description: str
    analysis_mode: AnalysisMode
    experiment_type: ExperimentType
    performance_profile: PerformanceProfile
    complexity_level: str = "intermediate"  # novice, intermediate, expert
    estimated_runtime: str = "minutes"
    parameter_count: int = 3
    requires_gpu: bool = False
    requires_large_memory: bool = False
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "2.0"
    tags: List[str] = field(default_factory=list)


class ConfigurationTemplateGenerator:
    """Generates configuration templates with context-aware documentation."""

    def __init__(self, logger: Optional[EnhancedConsoleLogger] = None):
        self.logger = logger or (
            create_console_logger() if HAS_ENHANCED_OUTPUT else None
        )
        self.templates_cache = {}

        # Pre-defined parameter sets for different experiment types
        self.experiment_parameters = self._load_experiment_parameters()

        # Performance optimization presets
        self.performance_presets = self._load_performance_presets()

        # Documentation templates
        self.doc_templates = self._load_documentation_templates()

    def _load_experiment_parameters(self) -> Dict[ExperimentType, Dict[str, Any]]:
        """Load experiment-specific parameter recommendations."""
        return {
            ExperimentType.HARD_SPHERES: {
                "wavevector_q": 0.0054,
                "dt": 0.1,
                "typical_D0": 100.0,
                "typical_alpha": 1.0,
                "description": "Hard sphere colloids with Brownian motion",
                "typical_size_nm": 100,
                "typical_viscosity_cP": 1.0,
            },
            ExperimentType.SOFT_COLLOIDS: {
                "wavevector_q": 0.008,
                "dt": 0.05,
                "typical_D0": 50.0,
                "typical_alpha": 0.8,
                "description": "Soft colloidal particles with potential interactions",
                "typical_size_nm": 200,
                "typical_viscosity_cP": 2.0,
            },
            ExperimentType.POLYMERS: {
                "wavevector_q": 0.012,
                "dt": 0.2,
                "typical_D0": 10.0,
                "typical_alpha": 0.5,
                "description": "Polymer solutions and melts",
                "typical_size_nm": 50,
                "typical_viscosity_cP": 10.0,
            },
            ExperimentType.LIQUID_CRYSTALS: {
                "wavevector_q": 0.006,
                "dt": 0.1,
                "typical_D0": 200.0,
                "typical_alpha": 1.2,
                "description": "Liquid crystalline systems",
                "typical_size_nm": 150,
                "typical_viscosity_cP": 5.0,
            },
            ExperimentType.GELS: {
                "wavevector_q": 0.004,
                "dt": 0.5,
                "typical_D0": 5.0,
                "typical_alpha": 0.3,
                "description": "Gel networks and viscoelastic systems",
                "typical_size_nm": 500,
                "typical_viscosity_cP": 100.0,
            },
            ExperimentType.BIOLOGICAL: {
                "wavevector_q": 0.010,
                "dt": 0.05,
                "typical_D0": 25.0,
                "typical_alpha": 0.7,
                "description": "Biological systems (cells, proteins, etc.)",
                "typical_size_nm": 1000,
                "typical_viscosity_cP": 1.5,
            },
            ExperimentType.CUSTOM: {
                "wavevector_q": 0.0054,
                "dt": 0.1,
                "typical_D0": 100.0,
                "typical_alpha": 1.0,
                "description": "Custom experimental system",
                "typical_size_nm": 100,
                "typical_viscosity_cP": 1.0,
            },
        }

    def _load_performance_presets(self) -> Dict[PerformanceProfile, Dict[str, Any]]:
        """Load performance optimization presets."""
        return {
            PerformanceProfile.FAST: {
                "description": "Optimized for speed - good for initial analysis",
                "angle_filtering": {"enabled": False},
                "classical_optimization": {"methods": ["Nelder-Mead"]},
                "v2_features": {
                    "performance_optimization": True,
                    "parallel_processing": True,
                    "gpu_acceleration": False,
                    "validation_level": "basic",
                    "cache_strategy": "aggressive",
                },
                "frame_count_max": 1000,
                "estimated_speedup": "3-5x faster",
            },
            PerformanceProfile.BALANCED: {
                "description": "Balance between speed and accuracy - recommended default",
                "angle_filtering": {"enabled": True},
                "classical_optimization": {"methods": ["Nelder-Mead", "Powell"]},
                "v2_features": {
                    "performance_optimization": True,
                    "parallel_processing": True,
                    "gpu_acceleration": True,
                    "validation_level": "comprehensive",
                    "cache_strategy": "intelligent",
                },
                "frame_count_max": 5000,
                "estimated_speedup": "balanced performance",
            },
            PerformanceProfile.ACCURATE: {
                "description": "Optimized for accuracy - best for final analysis",
                "angle_filtering": {"enabled": True},
                "classical_optimization": {
                    "methods": ["Nelder-Mead", "Powell", "L-BFGS-B"]
                },
                "v2_features": {
                    "performance_optimization": True,
                    "parallel_processing": True,
                    "gpu_acceleration": True,
                    "validation_level": "strict",
                    "cache_strategy": "conservative",
                },
                "bayesian_inference": {"mcmc_draws": 2000, "mcmc_tune": 1000},
                "frame_count_max": 20000,
                "estimated_speedup": "most accurate results",
            },
        }

    def _load_documentation_templates(self) -> Dict[str, str]:
        """Load inline documentation templates."""
        return {
            "header": """# Homodyne v2 Configuration
# Generated: {timestamp}
# Analysis Mode: {analysis_mode}
# Experiment Type: {experiment_type}
# Performance Profile: {performance_profile}
# 
# This configuration file defines all parameters for XPCS analysis.
# Sections are organized hierarchically for easy customization.
""",
            "analyzer_parameters": """# Core experimental parameters defining the analysis window and geometry
analyzer_parameters:
  # Temporal parameters - define the time window for analysis
  temporal:
    dt: {dt}                    # Time step in seconds (match your experimental acquisition rate)
    start_frame: {start_frame}  # First frame to analyze (typically 1)
    end_frame: {end_frame}      # Last frame to analyze
  
  # Scattering geometry parameters
  scattering:
    wavevector_q: {wavevector_q}  # Wavevector in Å⁻¹ (typical XPCS: 0.001-0.1)
  
  # Sample geometry parameters
  geometry:
    stator_rotor_gap: {gap_size}  # Gap size in nanometers (rheometer geometry)
""",
            "analysis_mode_static": """# Analysis mode configuration - Static system analysis
analysis_settings:
  static_mode: true           # Enable static mode (no shear flow)
  static_submode: "{submode}" # Submode: "isotropic" (fastest) or "anisotropic" (more features)
  
  # Physical model description
  model_description:
    {analysis_mode}: |
      Static analysis: g₁(t₁,t₂) = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt)
      Correlation: g₂(t₁,t₂) = [g₁(t₁,t₂)]²
      {model_details}
""",
            "analysis_mode_flow": """# Analysis mode configuration - Laminar flow analysis
analysis_settings:
  static_mode: false          # Disable static mode to enable flow analysis
  
  # Physical model description
  model_description:
    laminar_flow: |
      Laminar flow analysis: g₁(t₁,t₂) = g₁_diff(t₁,t₂) × g₁_shear(t₁,t₂)
      Diffusion: g₁_diff = exp(-q² ∫ᵗ²ᵗ¹ D(t)dt) where D(t) = D₀t^α + D_offset
      Shear: g₁_shear = [sinc(Φ)]² where Φ = (1/2π)qL cos(φ₀-φ) ∫|t₂-t₁| γ̇(t')dt'
      Flow rate: γ̇(t) = γ̇₀t^β + γ̇_offset
""",
            "parameters_static": """# Initial parameters for static mode analysis (3 parameters)
initial_parameters:
  values: [{D0}, {alpha}, {D_offset}]  # [D₀, α, D_offset]
  parameter_names: ["D0", "alpha", "D_offset"]
  
  # Parameter descriptions:
  # D0:       Diffusion coefficient (μm²/s) - typical range: 1-10000
  # alpha:    Anomalous diffusion exponent - typical range: -2 to 2
  #           α = 1: normal diffusion, α < 1: subdiffusion, α > 1: superdiffusion  
  # D_offset: Diffusion offset (μm²/s) - typically small compared to D0
""",
            "parameters_flow": """# Initial parameters for laminar flow analysis (7 parameters)
initial_parameters:
  values: [{D0}, {alpha}, {D_offset}, {gamma_dot_t0}, {beta}, {gamma_dot_t_offset}, {phi0}]
  parameter_names: ["D0", "alpha", "D_offset", "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"]
  
  # Diffusion parameters (same as static mode):
  # D0:       Diffusion coefficient (μm²/s)
  # alpha:    Anomalous diffusion exponent
  # D_offset: Diffusion offset (μm²/s)
  
  # Shear flow parameters:
  # gamma_dot_t0:     Shear rate at t=0 (s⁻¹) - typical range: 1e-6 to 1
  # beta:             Shear rate time exponent - typical range: -2 to 2
  # gamma_dot_t_offset: Shear rate offset (s⁻¹) - typically small
  # phi0:             Phase angle (degrees) - typical range: -10 to 10
""",
            "optimization": """# Optimization configuration
optimization_config:
  # Angle filtering - improves analysis by focusing on specific angular ranges
  angle_filtering:
    enabled: {angle_filtering_enabled}     # Enable/disable angle filtering
    {angle_ranges_comment}
    fallback_to_all_angles: true  # Use all angles if targets not found
    
  {classical_optimization}
  
  {bayesian_inference}
""",
            "performance": """# Performance and feature configuration
v2_features:
  output_format: "auto"                    # Output format: "auto", "hdf5", "json"
  validation_level: "{validation_level}"   # Validation: "basic", "comprehensive", "strict"
  physics_validation: true                 # Enable physics parameter validation
  performance_optimization: {performance_optimization}  # Enable performance optimizations
  parallel_processing: {parallel_processing}           # Enable parallel processing
  gpu_acceleration: {gpu_acceleration}                  # Enable GPU acceleration (requires JAX+GPU)
  cache_strategy: "{cache_strategy}"       # Cache strategy: "intelligent", "aggressive", "conservative", "disabled"
  
# Performance settings
performance_settings:
  parallel_execution: {parallel_processing}  # Enable parallel execution
  num_threads: {num_threads}                 # Number of threads (auto-detected: {cpu_count})
""",
            "experimental_data": """# Experimental data configuration
experimental_data:
  data_folder_path: "{data_folder}"      # Path to your data folder
  data_file_name: "{data_file}"          # Name of your HDF5 data file
  phi_angles_path: "{output_folder}"     # Output folder for angle data
  phi_angles_file: "phi_angles.dat"      # Angle data file name
  
  # Data caching for performance (recommended for repeated analysis)
  cache_file_path: "{cache_folder}"      # Cache folder path
  cache_filename_template: "cached_c2_frames_{start_frame}_{end_frame}.npz"
  cache_compression: true                # Enable cache compression
  
  # Advanced data loading options
  exchange_key: "C2_frames"              # HDF5 dataset key for correlation data
  apply_diagonal_correction: true        # Apply diagonal correction to correlation data
""",
            "logging": """# Logging configuration
logging:
  log_to_console: true          # Show log messages in console
  log_to_file: {log_to_file}    # Save log messages to file
  level: "{log_level}"          # Log level: "DEBUG", "INFO", "WARNING", "ERROR"
  {log_file_config}
  
  # Log message formatting
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
""",
        }

    def generate_template(
        self,
        metadata: TemplateMetadata,
        custom_parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a complete configuration template."""

        if self.logger:
            self.logger.info(f"Generating template: {metadata.name}")
            self.logger.parameter("Analysis mode", metadata.analysis_mode.value)
            self.logger.parameter("Experiment type", metadata.experiment_type.value)
            self.logger.parameter(
                "Performance profile", metadata.performance_profile.value
            )

        # Get base parameters for experiment type
        exp_params = self.experiment_parameters[metadata.experiment_type].copy()
        perf_preset = self.performance_presets[metadata.performance_profile].copy()

        # Apply custom parameters if provided
        if custom_parameters:
            exp_params.update(custom_parameters)

        # Build template sections
        sections = []

        # Header comment
        header = self.doc_templates["header"].format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            analysis_mode=metadata.analysis_mode.value,
            experiment_type=metadata.experiment_type.value.replace("_", " ").title(),
            performance_profile=metadata.performance_profile.value.title(),
        )
        sections.append(header)

        # Metadata section
        metadata_section = self._generate_metadata_section(metadata)
        sections.append(metadata_section)

        # Analyzer parameters
        analyzer_section = self._generate_analyzer_parameters_section(
            exp_params, metadata
        )
        sections.append(analyzer_section)

        # Experimental data
        data_section = self._generate_experimental_data_section(metadata)
        sections.append(data_section)

        # Analysis settings
        analysis_section = self._generate_analysis_settings_section(
            metadata, exp_params
        )
        sections.append(analysis_section)

        # Parameters
        parameters_section = self._generate_parameters_section(metadata, exp_params)
        sections.append(parameters_section)

        # Parameter space (bounds)
        bounds_section = self._generate_parameter_bounds_section(metadata)
        sections.append(bounds_section)

        # Optimization
        optimization_section = self._generate_optimization_section(
            metadata, perf_preset
        )
        sections.append(optimization_section)

        # Performance/V2 features
        performance_section = self._generate_performance_section(metadata, perf_preset)
        sections.append(performance_section)

        # Logging
        logging_section = self._generate_logging_section(metadata)
        sections.append(logging_section)

        # Validation rules (for expert users)
        if metadata.complexity_level == "expert":
            validation_section = self._generate_validation_section(metadata)
            sections.append(validation_section)

        # Combine all sections
        template = "\n".join(sections)

        if self.logger:
            self.logger.success(
                f"Template generated successfully ({len(template)} characters)"
            )

        return template

    def _generate_metadata_section(self, metadata: TemplateMetadata) -> str:
        """Generate metadata section."""
        return f"""
# Configuration metadata
metadata:
  config_version: "{metadata.version}"
  description: "{metadata.description}"
  analysis_mode: "{metadata.analysis_mode.value}"
  experiment_type: "{metadata.experiment_type.value}"
  complexity_level: "{metadata.complexity_level}"
  estimated_runtime: "{metadata.estimated_runtime}"
  created_date: "{metadata.created_date}"
  tags: {metadata.tags}
"""

    def _generate_analyzer_parameters_section(
        self, exp_params: Dict[str, Any], metadata: TemplateMetadata
    ) -> str:
        """Generate analyzer parameters section."""

        # Determine frame count based on performance profile
        if metadata.performance_profile == PerformanceProfile.FAST:
            end_frame = 500
        elif metadata.performance_profile == PerformanceProfile.BALANCED:
            end_frame = 2000
        else:
            end_frame = 5000

        return self.doc_templates["analyzer_parameters"].format(
            dt=exp_params["dt"],
            start_frame=1,
            end_frame=end_frame,
            wavevector_q=exp_params["wavevector_q"],
            gap_size=2000000,  # 2 mm in nm
        )

    def _generate_experimental_data_section(self, metadata: TemplateMetadata) -> str:
        """Generate experimental data section."""

        # Determine appropriate file name based on experiment type
        exp_type_name = metadata.experiment_type.value
        default_file = f"{exp_type_name}_data.hdf"

        return self.doc_templates["experimental_data"].format(
            data_folder="./data/",
            data_file=default_file,
            output_folder="./output/",
            cache_folder="./cache/",
        )

    def _generate_analysis_settings_section(
        self, metadata: TemplateMetadata, exp_params: Dict[str, Any]
    ) -> str:
        """Generate analysis settings section."""

        if metadata.analysis_mode == AnalysisMode.LAMINAR_FLOW:
            model_details = "Includes both diffusion and shear flow effects"
            return self.doc_templates["analysis_mode_flow"].format(
                model_details=model_details
            )
        else:
            submode = metadata.analysis_mode.value.split("_")[
                1
            ]  # isotropic or anisotropic
            model_details = exp_params["description"]
            return self.doc_templates["analysis_mode_static"].format(
                submode=submode,
                analysis_mode=metadata.analysis_mode.value,
                model_details=model_details,
            )

    def _generate_parameters_section(
        self, metadata: TemplateMetadata, exp_params: Dict[str, Any]
    ) -> str:
        """Generate parameters section."""

        if metadata.analysis_mode == AnalysisMode.LAMINAR_FLOW:
            return self.doc_templates["parameters_flow"].format(
                D0=exp_params["typical_D0"],
                alpha=exp_params["typical_alpha"],
                D_offset=exp_params["typical_D0"] * 0.1,  # 10% of D0
                gamma_dot_t0=0.01,
                beta=0.0,
                gamma_dot_t_offset=0.0,
                phi0=0.0,
            )
        else:
            return self.doc_templates["parameters_static"].format(
                D0=exp_params["typical_D0"],
                alpha=exp_params["typical_alpha"],
                D_offset=exp_params["typical_D0"] * 0.1,  # 10% of D0
            )

    def _generate_parameter_bounds_section(self, metadata: TemplateMetadata) -> str:
        """Generate parameter bounds section."""

        if metadata.analysis_mode == AnalysisMode.LAMINAR_FLOW:
            bounds = [
                {"name": "D0", "min": 1.0, "max": 100000.0, "type": "log-uniform"},
                {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                {"name": "D_offset", "min": -1000.0, "max": 1000.0, "type": "Normal"},
                {
                    "name": "gamma_dot_t0",
                    "min": 1e-6,
                    "max": 1.0,
                    "type": "log-uniform",
                },
                {"name": "beta", "min": -2.0, "max": 2.0, "type": "Normal"},
                {
                    "name": "gamma_dot_t_offset",
                    "min": -0.01,
                    "max": 0.01,
                    "type": "Normal",
                },
                {"name": "phi0", "min": -10.0, "max": 10.0, "type": "Normal"},
            ]
        else:
            bounds = [
                {"name": "D0", "min": 1.0, "max": 100000.0, "type": "log-uniform"},
                {"name": "alpha", "min": -2.0, "max": 2.0, "type": "Normal"},
                {"name": "D_offset", "min": -1000.0, "max": 1000.0, "type": "Normal"},
            ]

        bounds_yaml = yaml.dump(
            {"parameter_space": {"bounds": bounds}}, default_flow_style=False, indent=2
        )

        return f"""
# Parameter space definition - bounds and priors for optimization
{bounds_yaml}"""

    def _generate_optimization_section(
        self, metadata: TemplateMetadata, perf_preset: Dict[str, Any]
    ) -> str:
        """Generate optimization section."""

        # Angle filtering
        if metadata.analysis_mode == AnalysisMode.STATIC_ISOTROPIC:
            angle_filtering_enabled = "false"
            angle_ranges_comment = "# Angle filtering disabled for isotropic mode"
            angle_ranges = ""
        else:
            angle_filtering_enabled = "true"
            angle_ranges_comment = "# Target angle ranges for optimization (in degrees)"
            angle_ranges = """target_ranges:
      - min_angle: -10.0
        max_angle: 10.0
      - min_angle: 170.0
        max_angle: 190.0"""

        # Classical optimization
        methods = perf_preset["classical_optimization"]["methods"]
        classical_section = f"""# Classical optimization methods
  classical_optimization:
    methods: {methods}
    method_options:
      Nelder-Mead:
        maxiter: 1000
        xatol: 1e-8
        fatol: 1e-8
      Powell:
        maxiter: 1000
        xtol: 1e-8
        ftol: 1e-8
      L-BFGS-B:
        maxiter: 15000
        ftol: 2.220446049250313e-09"""

        # Bayesian inference (only for accurate profile)
        bayesian_section = ""
        if "bayesian_inference" in perf_preset:
            bayesian_config = perf_preset["bayesian_inference"]
            bayesian_section = f"""
  # Bayesian MCMC inference (high accuracy mode)
  bayesian_inference:
    mcmc_draws: {bayesian_config["mcmc_draws"]}    # Number of MCMC samples
    mcmc_tune: {bayesian_config["mcmc_tune"]}      # Number of tuning samples"""

        return self.doc_templates["optimization"].format(
            angle_filtering_enabled=angle_filtering_enabled,
            angle_ranges_comment=angle_ranges_comment,
            angle_ranges=angle_ranges,
            classical_optimization=classical_section,
            bayesian_inference=bayesian_section,
        )

    def _generate_performance_section(
        self, metadata: TemplateMetadata, perf_preset: Dict[str, Any]
    ) -> str:
        """Generate performance section."""

        v2_features = perf_preset["v2_features"]

        import multiprocessing

        cpu_count = multiprocessing.cpu_count()
        num_threads = min(16, cpu_count)  # Cap at 16 threads

        return self.doc_templates["performance"].format(
            validation_level=v2_features["validation_level"],
            performance_optimization=str(
                v2_features["performance_optimization"]
            ).lower(),
            parallel_processing=str(v2_features["parallel_processing"]).lower(),
            gpu_acceleration=str(v2_features["gpu_acceleration"]).lower(),
            cache_strategy=v2_features["cache_strategy"],
            num_threads=num_threads,
            cpu_count=cpu_count,
        )

    def _generate_logging_section(self, metadata: TemplateMetadata) -> str:
        """Generate logging section."""

        # Determine logging level based on complexity
        if metadata.complexity_level == "novice":
            log_level = "INFO"
            log_to_file = False
            log_file_config = "# File logging disabled for simplicity"
        elif metadata.complexity_level == "expert":
            log_level = "DEBUG"
            log_to_file = True
            log_file_config = """log_filename: "homodyne_analysis.log"
  rotation:
    max_bytes: 10485760      # 10 MB
    backup_count: 3          # Keep 3 backup files"""
        else:
            log_level = "INFO"
            log_to_file = True
            log_file_config = 'log_filename: "homodyne_analysis.log"'

        return self.doc_templates["logging"].format(
            log_to_file=str(log_to_file).lower(),
            log_level=log_level,
            log_file_config=log_file_config,
        )

    def _generate_validation_section(self, metadata: TemplateMetadata) -> str:
        """Generate validation rules section (expert mode)."""

        return """
# Validation rules (expert configuration)
validation_rules:
  # Frame range validation
  frame_range:
    minimum_frames: 10          # Minimum number of frames required
    maximum_frames: 50000       # Maximum frames for memory limits
    warn_above_frames: 10000    # Warn if frame count is high
  
  # Parameter validation
  parameter_validation:
    strict_bounds: true         # Enforce strict parameter bounds
    physics_check: true         # Enable physics-based validation
    correlation_check: true     # Validate correlation data quality
  
  # Performance validation
  performance_validation:
    memory_limit_gb: 16         # Maximum memory usage (GB)
    time_limit_hours: 24        # Maximum analysis time (hours)
    warn_long_runtime: true     # Warn about potentially long runtimes
"""

    def generate_example_set(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """Generate a complete set of example configurations."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.logger:
            self.logger.section("Generating Example Configuration Set")

        generated_files = {}

        # Define example templates
        examples = [
            # Beginner examples
            TemplateMetadata(
                name="beginner_hard_spheres",
                description="Beginner-friendly configuration for hard sphere colloids",
                analysis_mode=AnalysisMode.STATIC_ISOTROPIC,
                experiment_type=ExperimentType.HARD_SPHERES,
                performance_profile=PerformanceProfile.FAST,
                complexity_level="novice",
                estimated_runtime="seconds to minutes",
                parameter_count=3,
                tags=["beginner", "fast", "hard_spheres"],
            ),
            TemplateMetadata(
                name="beginner_polymers",
                description="Simple polymer solution analysis",
                analysis_mode=AnalysisMode.STATIC_ISOTROPIC,
                experiment_type=ExperimentType.POLYMERS,
                performance_profile=PerformanceProfile.FAST,
                complexity_level="novice",
                estimated_runtime="seconds to minutes",
                parameter_count=3,
                tags=["beginner", "polymers", "static"],
            ),
            # Intermediate examples
            TemplateMetadata(
                name="intermediate_colloids",
                description="Balanced analysis for colloidal systems",
                analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
                experiment_type=ExperimentType.SOFT_COLLOIDS,
                performance_profile=PerformanceProfile.BALANCED,
                complexity_level="intermediate",
                estimated_runtime="minutes",
                parameter_count=3,
                tags=["intermediate", "colloids", "anisotropic"],
            ),
            TemplateMetadata(
                name="intermediate_biological",
                description="Biological system analysis with angle filtering",
                analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
                experiment_type=ExperimentType.BIOLOGICAL,
                performance_profile=PerformanceProfile.BALANCED,
                complexity_level="intermediate",
                estimated_runtime="minutes to hours",
                parameter_count=3,
                tags=["intermediate", "biological", "filtering"],
            ),
            # Advanced examples
            TemplateMetadata(
                name="advanced_laminar_flow",
                description="Complete laminar flow analysis with all parameters",
                analysis_mode=AnalysisMode.LAMINAR_FLOW,
                experiment_type=ExperimentType.LIQUID_CRYSTALS,
                performance_profile=PerformanceProfile.ACCURATE,
                complexity_level="expert",
                estimated_runtime="hours",
                parameter_count=7,
                requires_gpu=True,
                tags=["expert", "flow", "complete"],
            ),
            TemplateMetadata(
                name="advanced_gel_analysis",
                description="High-accuracy gel network analysis",
                analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
                experiment_type=ExperimentType.GELS,
                performance_profile=PerformanceProfile.ACCURATE,
                complexity_level="expert",
                estimated_runtime="hours",
                parameter_count=3,
                requires_large_memory=True,
                tags=["expert", "gels", "high_accuracy"],
            ),
            # Performance comparison examples
            TemplateMetadata(
                name="performance_fast_demo",
                description="Fast analysis demonstration",
                analysis_mode=AnalysisMode.STATIC_ISOTROPIC,
                experiment_type=ExperimentType.HARD_SPHERES,
                performance_profile=PerformanceProfile.FAST,
                complexity_level="intermediate",
                estimated_runtime="seconds",
                parameter_count=3,
                tags=["demo", "fast", "benchmark"],
            ),
            TemplateMetadata(
                name="performance_accurate_demo",
                description="High-accuracy analysis demonstration",
                analysis_mode=AnalysisMode.STATIC_ANISOTROPIC,
                experiment_type=ExperimentType.HARD_SPHERES,
                performance_profile=PerformanceProfile.ACCURATE,
                complexity_level="intermediate",
                estimated_runtime="minutes",
                parameter_count=3,
                tags=["demo", "accurate", "benchmark"],
            ),
        ]

        # Generate each example
        for example in examples:
            if self.logger:
                self.logger.info(f"Generating: {example.name}")

            template_content = self.generate_template(example)

            # Save to file
            filename = f"{example.name}_config.yaml"
            filepath = output_dir / filename

            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(template_content)

                generated_files[example.name] = str(filepath)

                if self.logger:
                    self.logger.success(f"✓ Generated: {filename}")

            except Exception as e:
                if self.logger:
                    self.logger.error(f"✗ Failed to generate {filename}: {e}")
                logger.error(f"Failed to generate {filename}: {e}")

        # Generate index file
        self._generate_index_file(output_dir, examples, generated_files)

        if self.logger:
            self.logger.success(
                f"Generated {len(generated_files)} example configurations"
            )
            self.logger.file_path(output_dir)

        return generated_files

    def _generate_index_file(
        self,
        output_dir: Path,
        examples: List[TemplateMetadata],
        generated_files: Dict[str, str],
    ):
        """Generate an index file documenting all examples."""

        index_content = f"""# Homodyne v2 Configuration Examples
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This directory contains example configurations for different experimental scenarios
and user experience levels. Choose the example closest to your needs as a starting point.

## Quick Start Guide

1. **Beginners**: Start with `beginner_*.yaml` files
2. **Intermediate**: Use `intermediate_*.yaml` for balanced analysis
3. **Expert**: Try `advanced_*.yaml` for full capabilities
4. **Performance Testing**: Compare `performance_*_demo.yaml` files

## Available Examples

"""

        # Group examples by complexity level
        by_level = {}
        for example in examples:
            level = example.complexity_level
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(example)

        for level in ["novice", "intermediate", "expert"]:
            if level in by_level:
                index_content += f"\n### {level.title()} Level\n\n"

                for example in by_level[level]:
                    filename = f"{example.name}_config.yaml"
                    if example.name in generated_files:
                        index_content += f"- **{filename}**\n"
                        index_content += f"  - {example.description}\n"
                        index_content += f"  - Mode: {example.analysis_mode.value}\n"
                        index_content += f"  - Type: {example.experiment_type.value.replace('_', ' ').title()}\n"
                        index_content += f"  - Performance: {example.performance_profile.value.title()}\n"
                        index_content += f"  - Runtime: {example.estimated_runtime}\n"

                        if example.requires_gpu:
                            index_content += f"  - ⚠️  Requires GPU acceleration\n"
                        if example.requires_large_memory:
                            index_content += f"  - ⚠️  Requires large memory (>8GB)\n"

                        index_content += f"  - Tags: {', '.join(example.tags)}\n\n"

        # Usage instructions
        index_content += """
## Usage Instructions

### Running an Analysis
```bash
# Basic usage
homodyne analyze your_chosen_config.yaml

# With custom data path
homodyne analyze your_chosen_config.yaml --data-path ./your_data/

# Interactive mode
homodyne analyze --interactive
```

### Customizing a Configuration

1. Copy an example file: `cp beginner_hard_spheres_config.yaml my_config.yaml`
2. Edit the configuration:
   - Update `experimental_data` paths to point to your data
   - Adjust `analyzer_parameters` for your experimental conditions
   - Modify `initial_parameters` values if needed
3. Validate: `homodyne config --validate my_config.yaml`
4. Run: `homodyne analyze my_config.yaml`

### Performance Optimization Tips

- **Fast Analysis**: Use `static_isotropic` mode with `performance_profile: fast`
- **Balanced**: Use `static_anisotropic` with `performance_profile: balanced`
- **High Accuracy**: Use appropriate mode with `performance_profile: accurate`
- **GPU Acceleration**: Enable `gpu_acceleration: true` if available
- **Memory Usage**: Adjust `frame_count` and `cache_strategy` for your system

## Configuration Sections Explained

- **metadata**: Information about the configuration
- **analyzer_parameters**: Core experimental parameters (q, dt, frames)
- **experimental_data**: Paths to your data files
- **analysis_settings**: Analysis mode and model selection
- **initial_parameters**: Starting values for optimization
- **parameter_space**: Bounds and priors for parameters
- **optimization_config**: Optimization methods and angle filtering
- **v2_features**: Performance and validation settings
- **logging**: Log output configuration

## Need Help?

- Run `homodyne config --interactive` for guided configuration
- Use `homodyne config --validate your_config.yaml` to check configuration
- Check the documentation at docs/configuration.md
- Run `homodyne --help` for command-line options
"""

        index_file = output_dir / "README.md"
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                f.write(index_content)

            if self.logger:
                self.logger.success("✓ Generated index file: README.md")

        except Exception as e:
            if self.logger:
                self.logger.error(f"✗ Failed to generate index file: {e}")
            logger.error(f"Failed to generate index file: {e}")

    def create_interactive_template_builder(
        self,
    ) -> Optional["InteractiveTemplateBuilder"]:
        """Create an interactive template builder."""
        if HAS_ENHANCED_OUTPUT and InteractiveConfigurationBuilder:
            return InteractiveTemplateBuilder(self)
        else:
            logger.warning(
                "Interactive template builder not available - missing dependencies"
            )
            return None


class InteractiveTemplateBuilder:
    """Interactive builder for custom templates."""

    def __init__(self, generator: ConfigurationTemplateGenerator):
        self.generator = generator
        self.logger = generator.logger or (
            create_console_logger() if HAS_ENHANCED_OUTPUT else None
        )

    def build_custom_template(self) -> Optional[str]:
        """Build a custom template interactively."""

        if self.logger:
            self.logger.header("Interactive Template Builder")
            self.logger.info(
                "This tool will guide you through creating a custom configuration template."
            )

        try:
            # Step 1: Basic information
            if self.logger:
                self.logger.section("Template Information")

            name = input("Template name (no spaces): ").strip().replace(" ", "_")
            description = input("Description: ").strip()

            # Step 2: Analysis mode
            if self.logger:
                self.logger.section("Analysis Mode Selection")
                print("Available analysis modes:")
                print("  1. Static Isotropic - Fastest, simplest")
                print("  2. Static Anisotropic - Medium complexity")
                print("  3. Laminar Flow - Full capability, slower")

            mode_choice = int(input("Choose analysis mode [1-3]: ").strip())
            mode_map = {
                1: AnalysisMode.STATIC_ISOTROPIC,
                2: AnalysisMode.STATIC_ANISOTROPIC,
                3: AnalysisMode.LAMINAR_FLOW,
            }
            analysis_mode = mode_map.get(mode_choice, AnalysisMode.STATIC_ANISOTROPIC)

            # Step 3: Experiment type
            if self.logger:
                self.logger.section("Experiment Type Selection")
                print("Available experiment types:")
                for i, exp_type in enumerate(ExperimentType, 1):
                    print(f"  {i}. {exp_type.value.replace('_', ' ').title()}")

            exp_choice = int(
                input(f"Choose experiment type [1-{len(ExperimentType)}]: ").strip()
            )
            experiment_type = list(ExperimentType)[exp_choice - 1]

            # Step 4: Performance profile
            if self.logger:
                self.logger.section("Performance Profile Selection")
                print("Available performance profiles:")
                print("  1. Fast - Optimized for speed")
                print("  2. Balanced - Balance of speed and accuracy")
                print("  3. Accurate - Optimized for accuracy")

            perf_choice = int(input("Choose performance profile [1-3]: ").strip())
            perf_map = {
                1: PerformanceProfile.FAST,
                2: PerformanceProfile.BALANCED,
                3: PerformanceProfile.ACCURATE,
            }
            performance_profile = perf_map.get(perf_choice, PerformanceProfile.BALANCED)

            # Step 5: Complexity level
            if self.logger:
                self.logger.section("User Experience Level")

            print("Choose complexity level:")
            print("  1. Novice - Simple configuration with defaults")
            print("  2. Intermediate - Balanced options with explanations")
            print("  3. Expert - All options available")

            complex_choice = int(input("Choose complexity level [1-3]: ").strip())
            complexity_map = {1: "novice", 2: "intermediate", 3: "expert"}
            complexity_level = complexity_map.get(complex_choice, "intermediate")

            # Step 6: Custom parameters (optional)
            if self.logger:
                self.logger.section("Custom Parameters (Optional)")

            custom_params = {}
            if input("Customize experimental parameters? [y/N]: ").strip().lower() in [
                "y",
                "yes",
            ]:
                custom_params["wavevector_q"] = float(
                    input("Wavevector q (Å⁻¹) [0.0054]: ").strip() or "0.0054"
                )
                custom_params["dt"] = float(
                    input("Time step dt (s) [0.1]: ").strip() or "0.1"
                )
                custom_params["typical_D0"] = float(
                    input("Typical D0 (μm²/s) [100.0]: ").strip() or "100.0"
                )
                custom_params["typical_alpha"] = float(
                    input("Typical alpha [1.0]: ").strip() or "1.0"
                )

            # Create metadata
            metadata = TemplateMetadata(
                name=name,
                description=description,
                analysis_mode=analysis_mode,
                experiment_type=experiment_type,
                performance_profile=performance_profile,
                complexity_level=complexity_level,
                parameter_count=3 if analysis_mode != AnalysisMode.LAMINAR_FLOW else 7,
                tags=["custom", complexity_level, analysis_mode.value.split("_")[0]],
            )

            # Generate template
            if self.logger:
                self.logger.section("Generating Template")

            template_content = self.generator.generate_template(metadata, custom_params)

            # Save option
            save = input("Save template to file? [Y/n]: ").strip().lower()
            if save != "n" and save != "no":
                filename = f"{name}_config.yaml"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(template_content)

                if self.logger:
                    self.logger.success(f"Template saved to: {filename}")
                    self.logger.file_path(filename)

            return template_content

        except (KeyboardInterrupt, EOFError):
            if self.logger:
                self.logger.warning("Template creation cancelled by user")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to create template: {e}")
            logger.error(f"Failed to create interactive template: {e}")
            return None


# Factory functions
def create_template_generator(
    logger: Optional[EnhancedConsoleLogger] = None,
) -> ConfigurationTemplateGenerator:
    """Create a configuration template generator."""
    return ConfigurationTemplateGenerator(logger=logger)


def generate_example_configurations(
    output_dir: Union[str, Path] = "./examples",
) -> Dict[str, str]:
    """Generate a complete set of example configurations."""
    generator = create_template_generator()
    return generator.generate_example_set(output_dir)


def create_interactive_builder() -> Optional[InteractiveTemplateBuilder]:
    """Create an interactive template builder."""
    generator = create_template_generator()
    return generator.create_interactive_template_builder()


# CLI integration function
def main_generate_examples():
    """Main function for CLI example generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Homodyne v2 configuration examples"
    )
    parser.add_argument(
        "--output", "-o", default="./examples", help="Output directory for examples"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch interactive template builder",
    )

    args = parser.parse_args()

    if args.interactive:
        builder = create_interactive_builder()
        if builder:
            template = builder.build_custom_template()
            if template:
                print("\nTemplate created successfully!")
        else:
            print("Interactive builder not available")
    else:
        print("Generating example configurations...")
        generated = generate_example_configurations(args.output)
        print(f"Generated {len(generated)} example files in {args.output}/")
        for name, path in generated.items():
            print(f"  ✓ {name}: {path}")


if __name__ == "__main__":
    main_generate_examples()
