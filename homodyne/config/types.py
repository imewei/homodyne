"""Type Definitions for Homodyne Configuration System
==================================================

TypedDict definitions for configuration structures and parameter management.
Provides type safety and IDE autocomplete for configuration dictionaries.
"""

from typing import Any, Literal, TypedDict


class BoundDict(TypedDict, total=False):
    """Parameter bound specification.

    Attributes
    ----------
    name : str
        Parameter name
    min : float
        Minimum value
    max : float
        Maximum value
    type : str
        Bound type (e.g., "Normal", "LogNormal")
    """

    name: str
    min: float
    max: float
    type: str


class InitialParametersConfig(TypedDict, total=False):
    """Initial parameters section of configuration.

    Attributes
    ----------
    parameter_names : list[str]
        List of parameter names to optimize
    values : list[float]
        Initial values for parameters
    active_parameters : list[str], optional
        Subset of parameters to actively optimize
    fixed_parameters : dict[str, float], optional
        Parameters held fixed during optimization
    """

    parameter_names: list[str]
    values: list[float]
    active_parameters: list[str]
    fixed_parameters: dict[str, float]


class ParameterSpaceConfig(TypedDict, total=False):
    """Parameter space section of configuration.

    Attributes
    ----------
    bounds : list[BoundDict]
        Parameter bounds specifications
    priors : list[dict], optional
        Prior distributions for Bayesian methods
    """

    bounds: list[BoundDict]
    priors: list[dict[str, Any]]


class ExperimentalDataConfig(TypedDict, total=False):
    """Experimental data section of configuration.

    Attributes
    ----------
    file_path : str
        Path to HDF5 data file
    data_folder_path : str, optional
        Legacy: folder containing data
    data_file_name : str, optional
        Legacy: data file name
    phi_angles_file : str, optional
        Path to phi angles file
    """

    file_path: str
    data_folder_path: str
    data_file_name: str
    phi_angles_file: str


class StreamingConfig(TypedDict, total=False):
    """Streaming optimization configuration.

    Configuration for NLSQ StreamingOptimizer with checkpoint management
    and fault tolerance for unlimited dataset sizes.

    Attributes
    ----------
    enable_checkpoints : bool
        Enable checkpoint save/resume functionality
    checkpoint_dir : str
        Directory for checkpoint files
    checkpoint_frequency : int
        Save checkpoint every N batches
    resume_from_checkpoint : bool
        Auto-detect and resume from latest checkpoint
    keep_last_checkpoints : int
        Number of recent checkpoints to keep (older ones deleted)
    enable_fault_tolerance : bool
        Enable numerical validation and error recovery
    max_retries_per_batch : int
        Maximum retry attempts per failed batch
    min_success_rate : float
        Minimum batch success rate (0.0-1.0) before failing optimization
    """

    enable_checkpoints: bool
    checkpoint_dir: str
    checkpoint_frequency: int
    resume_from_checkpoint: bool
    keep_last_checkpoints: int
    enable_fault_tolerance: bool
    max_retries_per_batch: int
    min_success_rate: float


class OptimizationConfig(TypedDict, total=False):
    """Optimization section of configuration.

    Attributes
    ----------
    method : str
        Optimization method ("nlsq", "mcmc")
    lsq : dict, optional
        NLSQ-specific settings
    mcmc : dict, optional
        MCMC-specific settings
    angle_filtering : dict, optional
        Angle filtering settings
    streaming : StreamingConfig, optional
        Streaming optimization settings (checkpoint management, fault tolerance)
    """

    method: Literal["nlsq", "mcmc"]
    lsq: dict[str, Any]
    mcmc: dict[str, Any]
    angle_filtering: dict[str, Any]
    streaming: StreamingConfig


class HomodyneConfig(TypedDict, total=False):
    """Complete homodyne configuration structure.

    Attributes
    ----------
    config_version : str
        Configuration file version
    analysis_mode : str
        Analysis mode ("static_isotropic", "static_anisotropic", "laminar_flow")
    experimental_data : ExperimentalDataConfig
        Experimental data specification
    parameter_space : ParameterSpaceConfig
        Parameter bounds and priors
    initial_parameters : InitialParametersConfig
        Initial parameter values
    optimization : OptimizationConfig
        Optimization settings
    output : dict, optional
        Output settings
    """

    config_version: str
    analysis_mode: Literal["static_isotropic", "static_anisotropic", "laminar_flow"]
    experimental_data: ExperimentalDataConfig
    parameter_space: ParameterSpaceConfig
    initial_parameters: InitialParametersConfig
    optimization: OptimizationConfig
    output: dict[str, Any]


# Analysis mode literal type
AnalysisMode = Literal["static_isotropic", "static_anisotropic", "laminar_flow"]

# Parameter names for different modes
STATIC_PARAM_NAMES: list[str] = ["D0", "alpha", "D_offset"]
LAMINAR_FLOW_PARAM_NAMES: list[str] = [
    "D0",
    "alpha",
    "D_offset",
    "gamma_dot_t0",
    "beta",
    "gamma_dot_t_offset",
    "phi0",
]
SCALING_PARAM_NAMES: list[str] = ["contrast", "offset"]


# Parameter name mapping
PARAMETER_NAME_MAPPING: dict[str, str] = {
    "gamma_dot_0": "gamma_dot_t0",
    "gamma_dot_offset": "gamma_dot_t_offset",
    "phi_0": "phi0",
}
