"""
Type Definitions for Homodyne Configuration System
==================================================

TypedDict definitions for configuration structures and parameter management.
Provides type safety and IDE autocomplete for configuration dictionaries.
"""

from typing import Any, Literal, TypedDict


class BoundDict(TypedDict, total=False):
    """
    Parameter bound specification.

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
    """
    Initial parameters section of configuration.

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
    """
    Parameter space section of configuration.

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
    """
    Experimental data section of configuration.

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


class OptimizationConfig(TypedDict, total=False):
    """
    Optimization section of configuration.

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
    """

    method: Literal["nlsq", "mcmc"]
    lsq: dict[str, Any]
    mcmc: dict[str, Any]
    angle_filtering: dict[str, Any]


class HomodyneConfig(TypedDict, total=False):
    """
    Complete homodyne configuration structure.

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
