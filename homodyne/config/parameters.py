"""
Parameter Structure Definitions for Homodyne v2
==============================================

Defines the parameter structures used in different analysis modes.
Physics parameters are loaded from configuration, while scaling parameters
(contrast, offset) are computed during analysis.

Parameter Structure by Mode:
- Static Isotropic: 3 physics params [D₀, α, D_offset]
- Static Anisotropic: 3 physics params [D₀, α, D_offset]
- Laminar Flow: 7 physics params [D₀, α, D_offset, γ̇₀, β, γ̇_offset, φ₀]

Note: Scaling parameters (contrast, offset) are calculated during optimization,
not loaded from configuration files.

Authors: Wei Chen, Hongrui He
Institution: Argonne National Laboratory
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# V2 integration
try:
    from homodyne.utils.logging import get_logger

    HAS_V2_LOGGING = True
except ImportError:
    import logging

    HAS_V2_LOGGING = False

    def get_logger(name):
        return logging.getLogger(name)


logger = get_logger(__name__)


class AnalysisMode(Enum):
    """Analysis mode enumeration."""

    STATIC_ISOTROPIC = "static_isotropic"
    STATIC_ANISOTROPIC = "static_anisotropic"
    LAMINAR_FLOW = "laminar_flow"


class ParameterType(Enum):
    """Parameter type enumeration."""

    PHYSICS = "physics"
    SCALING = "scaling"


@dataclass
class ParameterBounds:
    """Parameter bounds definition."""

    name: str
    min: float
    max: float
    type: str = "Normal"  # "Normal", "TruncatedNormal", "uniform", "log-uniform"
    unit: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        """Validate bounds after initialization."""
        if self.min >= self.max:
            raise ValueError(
                f"Invalid bounds for {self.name}: min={self.min} >= max={self.max}"
            )


@dataclass
class PhysicsParameters:
    """
    Physics parameters loaded from configuration.

    These are the model parameters that define the physical behavior
    and are loaded from configuration files.
    """

    # Core diffusion parameters (all modes)
    D0: float = 100.0  # ∈ [1, 1e6] Å²/s - Diffusion coefficient
    alpha: float = 0.0  # ∈ [-2, 2] - Power-law exponent (can be negative!)
    D_offset: float = 10.0  # ∈ [-100, 100] Å²/s - Baseline diffusion offset

    # Flow parameters (laminar_flow mode only)
    gamma_dot_t0: Optional[float] = None  # ∈ [1e-6, 1.0] s⁻¹ - Reference shear rate
    beta: Optional[float] = None  # ∈ [-2, 2] - Shear rate power-law
    gamma_dot_t_offset: Optional[float] = None  # ∈ [-0.01, 0.01] s⁻¹ - Shear baseline
    phi0: Optional[float] = None  # ∈ [-10, 10] degrees - Angular offset

    # Metadata
    mode: AnalysisMode = AnalysisMode.STATIC_ISOTROPIC
    parameter_names: List[str] = field(
        default_factory=lambda: ["D0", "alpha", "D_offset"]
    )

    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate_parameters()
        self._update_parameter_names()

    def _validate_parameters(self):
        """Validate parameter values against physical constraints."""
        # Validate core diffusion parameters
        if self.D0 <= 0:
            raise ValueError(f"D0 must be positive: {self.D0}")
        if not -2.0 <= self.alpha <= 2.0:
            logger.warning(f"Alpha outside typical range [-2, 2]: {self.alpha}")

        # Validate flow parameters if present
        if self.mode == AnalysisMode.LAMINAR_FLOW:
            if any(
                param is None
                for param in [
                    self.gamma_dot_t0,
                    self.beta,
                    self.gamma_dot_t_offset,
                    self.phi0,
                ]
            ):
                raise ValueError("Laminar flow mode requires all flow parameters")

            if self.gamma_dot_t0 is not None and self.gamma_dot_t0 <= 0:
                raise ValueError(f"gamma_dot_t0 must be positive: {self.gamma_dot_t0}")

    def _update_parameter_names(self):
        """Update parameter names based on mode."""
        if (
            self.mode == AnalysisMode.STATIC_ISOTROPIC
            or self.mode == AnalysisMode.STATIC_ANISOTROPIC
        ):
            self.parameter_names = ["D0", "alpha", "D_offset"]
        elif self.mode == AnalysisMode.LAMINAR_FLOW:
            self.parameter_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]

    def to_array(self) -> List[float]:
        """Convert parameters to array format for optimization."""
        if (
            self.mode == AnalysisMode.STATIC_ISOTROPIC
            or self.mode == AnalysisMode.STATIC_ANISOTROPIC
        ):
            return [self.D0, self.alpha, self.D_offset]
        elif self.mode == AnalysisMode.LAMINAR_FLOW:
            return [
                self.D0,
                self.alpha,
                self.D_offset,
                self.gamma_dot_t0,
                self.beta,
                self.gamma_dot_t_offset,
                self.phi0,
            ]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def from_array(self, values: List[float], mode: AnalysisMode = None):
        """Update parameters from array format."""
        if mode is not None:
            self.mode = mode

        if len(values) < 3:
            raise ValueError(f"Need at least 3 parameters, got {len(values)}")

        # Set core diffusion parameters
        self.D0 = values[0]
        self.alpha = values[1]
        self.D_offset = values[2]

        # Set flow parameters if available
        if len(values) >= 7:
            self.gamma_dot_t0 = values[3]
            self.beta = values[4]
            self.gamma_dot_t_offset = values[5]
            self.phi0 = values[6]
        elif self.mode == AnalysisMode.LAMINAR_FLOW:
            raise ValueError(
                f"Laminar flow mode requires 7 parameters, got {len(values)}"
            )

        # Update metadata
        self._update_parameter_names()
        self._validate_parameters()

    def get_parameter_count(self) -> int:
        """Get expected parameter count for current mode."""
        if (
            self.mode == AnalysisMode.STATIC_ISOTROPIC
            or self.mode == AnalysisMode.STATIC_ANISOTROPIC
        ):
            return 3
        elif self.mode == AnalysisMode.LAMINAR_FLOW:
            return 7
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


@dataclass
class ScalingParameters:
    """
    Scaling parameters computed during analysis.

    These parameters (contrast, offset) are not loaded from configuration
    but are computed during the optimization process. They handle the
    relationship between the theoretical g1 and experimental g2:

    g2 = offset + contrast * g1²
    """

    # Scaling parameters - computed during analysis
    contrasts: Optional[List[float]] = None  # Per angle: each ∈ [0.01, 1.0]
    offsets: Optional[List[float]] = None  # Per angle: each ∈ [0.0, 2.0]

    # Metadata
    n_angles: int = 1
    angle_indices: Optional[List[int]] = None

    def __post_init__(self):
        """Initialize scaling parameters."""
        if self.contrasts is None:
            self.contrasts = [0.1] * self.n_angles
        if self.offsets is None:
            self.offsets = [1.0] * self.n_angles
        if self.angle_indices is None:
            self.angle_indices = list(range(self.n_angles))

    def validate_scaling_parameters(self):
        """Validate scaling parameter values."""
        if len(self.contrasts) != self.n_angles:
            raise ValueError(
                f"Contrasts length {len(self.contrasts)} != n_angles {self.n_angles}"
            )
        if len(self.offsets) != self.n_angles:
            raise ValueError(
                f"Offsets length {len(self.offsets)} != n_angles {self.n_angles}"
            )

        # Check bounds
        for i, contrast in enumerate(self.contrasts):
            if not 0.01 <= contrast <= 1.0:
                logger.warning(
                    f"Contrast[{i}]={contrast} outside typical range [0.01, 1.0]"
                )

        for i, offset in enumerate(self.offsets):
            if not 0.0 <= offset <= 2.0:
                logger.warning(f"Offset[{i}]={offset} outside typical range [0.0, 2.0]")

    def get_total_scaling_parameters(self) -> int:
        """Get total number of scaling parameters."""
        return 2 * self.n_angles  # contrast + offset per angle


@dataclass
class CombinedParameters:
    """
    Combined parameter structure for full analysis.

    Combines physics parameters (from configuration) and scaling parameters
    (computed during analysis) into a unified structure.
    """

    physics: PhysicsParameters
    scaling: ScalingParameters

    def get_total_parameter_count(self) -> int:
        """Get total parameter count (physics + scaling)."""
        return (
            self.physics.get_parameter_count()
            + self.scaling.get_total_scaling_parameters()
        )

    def get_parameter_structure_summary(self) -> Dict[str, Any]:
        """Get summary of parameter structure."""
        return {
            "mode": self.physics.mode.value,
            "physics_params": self.physics.get_parameter_count(),
            "physics_names": self.physics.parameter_names,
            "scaling_params": self.scaling.get_total_scaling_parameters(),
            "n_angles": self.scaling.n_angles,
            "total_params": self.get_total_parameter_count(),
            "structure": f"{self.physics.get_parameter_count()} physics + {self.scaling.get_total_scaling_parameters()} scaling = {self.get_total_parameter_count()} total",
        }


def get_default_parameter_bounds() -> Dict[str, ParameterBounds]:
    """
    Get default parameter bounds for all parameters.

    Returns
    -------
    dict
        Dictionary mapping parameter names to ParameterBounds objects
    """
    bounds = {
        # Core diffusion parameters
        "D0": ParameterBounds(
            name="D0",
            min=1.0,
            max=1e6,
            type="Normal",
            unit="Å²/s",
            description="Diffusion coefficient",
        ),
        "alpha": ParameterBounds(
            name="alpha",
            min=-10.0,
            max=10.0,
            type="Normal",
            unit="dimensionless",
            description="Power-law exponent (can be negative for subdiffusion)",
        ),
        "D_offset": ParameterBounds(
            name="D_offset",
            min=-1e5,
            max=1e5,
            type="Normal",
            unit="Å²/s",
            description="Baseline diffusion offset",
        ),
        # Flow parameters
        "gamma_dot_t0": ParameterBounds(
            name="gamma_dot_t0",
            min=1e-5,
            max=1.0,
            type="Normal",
            unit="s⁻¹",
            description="Reference shear rate",
        ),
        "beta": ParameterBounds(
            name="beta",
            min=-10.0,
            max=10.0,
            type="Normal",
            unit="dimensionless",
            description="Shear rate power-law exponent",
        ),
        "gamma_dot_t_offset": ParameterBounds(
            name="gamma_dot_t_offset",
            min=-1.0,
            max=1.0,
            type="Normal",
            unit="s⁻¹",
            description="Shear baseline offset",
        ),
        "phi0": ParameterBounds(
            name="phi0",
            min=-30.0,
            max=30.0,
            type="Normal",
            unit="degrees",
            description="Angular offset",
        ),
        # Scaling parameters (computed, not configured)
        "contrast": ParameterBounds(
            name="contrast",
            min=0.01,
            max=1.0,
            type="Normal",
            unit="dimensionless",
            description="Correlation contrast (computed during analysis)",
        ),
        "offset": ParameterBounds(
            name="offset",
            min=0.0,
            max=2.0,
            type="Normal",
            unit="dimensionless",
            description="Correlation offset (computed during analysis)",
        ),
    }

    return bounds


def create_parameters_from_config(
    config_dict: Dict[str, Any], mode: AnalysisMode, n_angles: int = 1
) -> CombinedParameters:
    """
    Create parameter structure from configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary with initial_parameters section
    mode : AnalysisMode
        Analysis mode to determine parameter structure
    n_angles : int, optional
        Number of angles for scaling parameters

    Returns
    -------
    CombinedParameters
        Combined parameter structure with physics and scaling parameters
    """
    initial_params = config_dict.get("initial_parameters", {})
    values = initial_params.get("values", [])

    # Create physics parameters
    physics = PhysicsParameters(mode=mode)
    if values:
        physics.from_array(values, mode)

    # Create scaling parameters
    scaling = ScalingParameters(n_angles=n_angles)

    return CombinedParameters(physics=physics, scaling=scaling)


def validate_parameter_configuration(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate parameter configuration.

    Parameters
    ----------
    config_dict : dict
        Configuration dictionary to validate

    Returns
    -------
    dict
        Validation result with 'valid', 'errors', 'warnings', 'parameter_info'
    """
    errors = []
    warnings = []
    parameter_info = {}

    # Check for initial_parameters section
    if "initial_parameters" not in config_dict:
        errors.append("Missing 'initial_parameters' section")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "parameter_info": parameter_info,
        }

    initial_params = config_dict["initial_parameters"]

    # Check values and parameter_names
    values = initial_params.get("values", [])
    parameter_names = initial_params.get("parameter_names", [])

    if not values:
        errors.append("Missing 'values' in initial_parameters")
    if not parameter_names:
        warnings.append("Missing 'parameter_names' in initial_parameters")

    if values and parameter_names:
        if len(values) != len(parameter_names):
            errors.append(
                f"Parameter count mismatch: {len(values)} values vs {len(parameter_names)} names"
            )

    # Validate parameter count against mode
    analysis_settings = config_dict.get("analysis_settings", {})
    static_mode = analysis_settings.get("static_mode", True)

    expected_count = 3 if static_mode else 7
    if values and len(values) != expected_count:
        mode_str = "static" if static_mode else "laminar_flow"
        errors.append(
            f"Parameter count mismatch for {mode_str} mode: expected {expected_count}, got {len(values)}"
        )

    # Parameter bounds validation
    default_bounds = get_default_parameter_bounds()
    for i, (value, name) in enumerate(zip(values, parameter_names)):
        if name in default_bounds:
            bound = default_bounds[name]
            if not bound.min <= value <= bound.max:
                warnings.append(
                    f"{name}={value} outside recommended bounds [{bound.min}, {bound.max}]"
                )

    parameter_info = {
        "parameter_count": len(values),
        "parameter_names": parameter_names,
        "mode": "static" if static_mode else "laminar_flow",
        "expected_count": expected_count,
        "physics_params": expected_count,
        "scaling_params": "computed during analysis",
    }

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "parameter_info": parameter_info,
    }
