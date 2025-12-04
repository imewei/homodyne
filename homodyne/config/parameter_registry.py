"""Parameter Registry for Homodyne Analysis

Centralized parameter registry that eliminates 8x duplication of parameter
definitions across the codebase. Provides:
- Parameter metadata (names, types, bounds, defaults, descriptions)
- Per-angle parameter expansion
- Prior information for MCMC
- Validation utilities

This module consolidates parameter information that was previously duplicated in:
- result.py (MCMCResult.get_param_names)
- mcmc_plots.py (8 functions with hardcoded param names)
- coordinator.py (CMC parameter expansion)
- priors.py (MCMC prior bounds)
- core.py (MCMC model sampling)
- backends/multiprocessing.py (worker validation)
- data_prep.py (data preprocessing)
- several test fixtures

Created as part of code quality remediation (Dec 2025).
Addresses code review finding of 8x parameter name duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    pass

AnalysisMode = Literal["static", "static_isotropic", "laminar_flow"]


@dataclass(frozen=True)
class ParameterInfo:
    """Metadata for a single parameter.

    Attributes
    ----------
    name : str
        Canonical parameter name
    description : str
        Human-readable description
    dtype : type
        Python/numpy type (float, int, etc.)
    default : float
        Default value for initialization
    lower_bound : float
        Lower bound for optimization/MCMC
    upper_bound : float
        Upper bound for optimization/MCMC
    prior_mean : float | None
        Prior mean for Bayesian inference (None for uniform prior)
    prior_std : float | None
        Prior standard deviation (None for uniform prior)
    units : str
        Physical units (e.g., 'Å²/s', 'radians')
    is_scaling : bool
        True if this is a per-angle scaling parameter
    is_physical : bool
        True if this is a physical model parameter
    is_flow : bool
        True if this is a flow-specific parameter
    log_space : bool
        True if parameter should be sampled in log space (e.g., D0)
    """

    name: str
    description: str
    dtype: type = float
    default: float = 1.0
    lower_bound: float = 0.0
    upper_bound: float = float("inf")
    prior_mean: float | None = None
    prior_std: float | None = None
    units: str = ""
    is_scaling: bool = False
    is_physical: bool = False
    is_flow: bool = False
    log_space: bool = False


class ParameterRegistry:
    """Centralized registry of all parameter definitions.

    This class provides a single source of truth for parameter metadata,
    eliminating duplication across the codebase.

    Examples
    --------
    >>> registry = ParameterRegistry()
    >>> registry.get_param_names("static")
    ['D0', 'alpha', 'D_offset']

    >>> registry.get_all_param_names("static", n_angles=3, include_scaling=True)
    ['contrast_0', 'contrast_1', 'contrast_2',
     'offset_0', 'offset_1', 'offset_2',
     'D0', 'alpha', 'D_offset']

    >>> registry.get_bounds("D0")
    (100.0, 100000.0)
    """

    # Singleton instance
    _instance: "ParameterRegistry | None" = None

    # Parameter definitions
    _PARAMETERS: dict[str, ParameterInfo] = {
        # Scaling parameters (per-angle)
        "contrast": ParameterInfo(
            name="contrast",
            description="Contrast factor for c2 = contrast × c1² + offset",
            default=0.5,
            lower_bound=0.01,
            upper_bound=1.5,
            prior_mean=0.5,
            prior_std=0.2,
            units="",
            is_scaling=True,
        ),
        "offset": ParameterInfo(
            name="offset",
            description="Baseline offset for c2 = contrast × c1² + offset",
            default=1.0,
            lower_bound=0.5,
            upper_bound=1.5,
            prior_mean=1.0,
            prior_std=0.1,
            units="",
            is_scaling=True,
        ),
        # Physical diffusion parameters
        "D0": ParameterInfo(
            name="D0",
            description="Diffusion coefficient amplitude",
            default=1000.0,
            lower_bound=100.0,
            upper_bound=100000.0,
            prior_mean=None,  # Uniform prior in log space
            prior_std=None,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        "alpha": ParameterInfo(
            name="alpha",
            description="Anomalous diffusion exponent (α < 0: sub, α > 0: super)",
            default=0.5,
            lower_bound=0.1,
            upper_bound=2.0,
            prior_mean=0.5,
            prior_std=0.3,
            units="",
            is_physical=True,
        ),
        "D_offset": ParameterInfo(
            name="D_offset",
            description="Baseline diffusion offset",
            default=10.0,
            lower_bound=0.1,
            upper_bound=1000.0,
            prior_mean=None,  # Uniform prior in log space
            prior_std=None,
            units="Å²/s",
            is_physical=True,
            log_space=True,
        ),
        # Flow parameters (laminar flow mode only)
        "gamma_dot_t0": ParameterInfo(
            name="gamma_dot_t0",
            description="Shear rate amplitude at t=0",
            default=100.0,
            lower_bound=0.1,
            upper_bound=10000.0,
            prior_mean=None,
            prior_std=None,
            units="s⁻¹",
            is_physical=True,
            is_flow=True,
            log_space=True,
        ),
        "beta": ParameterInfo(
            name="beta",
            description="Shear rate time exponent (γ̇(t) = γ̇₀ × t^β)",
            default=0.5,
            lower_bound=0.0,
            upper_bound=2.0,
            prior_mean=0.5,
            prior_std=0.3,
            units="",
            is_physical=True,
            is_flow=True,
        ),
        "gamma_dot_t_offset": ParameterInfo(
            name="gamma_dot_t_offset",
            description="Baseline shear rate offset",
            default=5.0,
            lower_bound=0.01,
            upper_bound=1000.0,
            prior_mean=None,
            prior_std=None,
            units="s⁻¹",
            is_physical=True,
            is_flow=True,
            log_space=True,
        ),
        "phi0": ParameterInfo(
            name="phi0",
            description="Flow direction angle",
            default=0.0,
            lower_bound=-np.pi,
            upper_bound=np.pi,
            prior_mean=0.0,
            prior_std=np.pi / 2,
            units="radians",
            is_physical=True,
            is_flow=True,
        ),
    }

    # Analysis mode definitions
    _MODE_PARAMS: dict[str, list[str]] = {
        "static": ["D0", "alpha", "D_offset"],
        "static_isotropic": ["D0", "alpha", "D_offset"],
        "laminar_flow": [
            "D0", "alpha", "D_offset",
            "gamma_dot_t0", "beta", "gamma_dot_t_offset", "phi0"
        ],
    }

    def __new__(cls) -> "ParameterRegistry":
        """Singleton pattern - return existing instance if available."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_param_info(self, name: str) -> ParameterInfo:
        """Get parameter metadata.

        Parameters
        ----------
        name : str
            Parameter name (e.g., 'D0', 'contrast')

        Returns
        -------
        ParameterInfo
            Parameter metadata

        Raises
        ------
        KeyError
            If parameter name is unknown
        """
        # Handle per-angle parameter names (e.g., contrast_0 -> contrast)
        base_name = name.split("_")[0] if "_" in name else name

        if base_name not in self._PARAMETERS:
            raise KeyError(
                f"Unknown parameter: {name}. "
                f"Valid parameters: {list(self._PARAMETERS.keys())}"
            )
        return self._PARAMETERS[base_name]

    def get_param_names(
        self,
        analysis_mode: AnalysisMode,
    ) -> list[str]:
        """Get physical parameter names for analysis mode.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: 'static', 'static_isotropic', or 'laminar_flow'

        Returns
        -------
        list[str]
            Physical parameter names (without per-angle scaling)
        """
        mode = self._normalize_mode(analysis_mode)
        return self._MODE_PARAMS[mode].copy()

    def get_all_param_names(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> list[str]:
        """Get all parameter names including per-angle scaling.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles for per-angle scaling parameters
        include_scaling : bool
            If True, include contrast_i and offset_i parameters

        Returns
        -------
        list[str]
            Complete parameter names in NumPyro sampling order:
            [contrast_0..n, offset_0..n, D0, alpha, ...]

        Notes
        -----
        NumPyro requires parameters in EXACT order as model.sample().
        This method returns parameters in the correct order for init_to_value().
        """
        names = []

        if include_scaling:
            # Per-angle contrast FIRST
            for i in range(n_angles):
                names.append(f"contrast_{i}")

            # Per-angle offset SECOND
            for i in range(n_angles):
                names.append(f"offset_{i}")

        # Physical parameters LAST
        names.extend(self.get_param_names(analysis_mode))

        return names

    def get_bounds(
        self,
        name: str,
    ) -> tuple[float, float]:
        """Get parameter bounds.

        Parameters
        ----------
        name : str
            Parameter name

        Returns
        -------
        tuple[float, float]
            (lower_bound, upper_bound)
        """
        info = self.get_param_info(name)
        return (info.lower_bound, info.upper_bound)

    def get_all_bounds(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> tuple[list[float], list[float]]:
        """Get bounds for all parameters.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        tuple[list[float], list[float]]
            (lower_bounds, upper_bounds) in parameter order
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)
        lower = []
        upper = []

        for name in names:
            lb, ub = self.get_bounds(name)
            lower.append(lb)
            upper.append(ub)

        return lower, upper

    def get_defaults(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> list[float]:
        """Get default values for all parameters.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        list[float]
            Default values in parameter order
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)
        return [self.get_param_info(name).default for name in names]

    def get_num_params(
        self,
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> int:
        """Get total number of parameters.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Returns
        -------
        int
            Total parameter count
        """
        return len(self.get_all_param_names(analysis_mode, n_angles, include_scaling))

    def validate_param_values(
        self,
        values: dict[str, float] | list[float],
        analysis_mode: AnalysisMode,
        n_angles: int = 1,
        include_scaling: bool = True,
    ) -> None:
        """Validate parameter values against bounds.

        Parameters
        ----------
        values : dict or list
            Parameter values (dict of name->value or list in order)
        analysis_mode : str
            Analysis mode
        n_angles : int
            Number of angles
        include_scaling : bool
            Include per-angle scaling parameters

        Raises
        ------
        ValueError
            If any value is out of bounds
        """
        names = self.get_all_param_names(analysis_mode, n_angles, include_scaling)

        if isinstance(values, dict):
            for name, value in values.items():
                if name not in names:
                    continue  # Skip unknown parameters
                lb, ub = self.get_bounds(name)
                if value < lb or value > ub:
                    raise ValueError(
                        f"Parameter {name}={value} out of bounds [{lb}, {ub}]"
                    )
        else:
            if len(values) != len(names):
                raise ValueError(
                    f"Expected {len(names)} values, got {len(values)}"
                )
            for name, value in zip(names, values, strict=True):
                lb, ub = self.get_bounds(name)
                if value < lb or value > ub:
                    raise ValueError(
                        f"Parameter {name}={value} out of bounds [{lb}, {ub}]"
                    )

    def expand_initial_values(
        self,
        initial_values: dict[str, float],
        n_angles: int,
    ) -> dict[str, float]:
        """Expand scalar scaling values to per-angle parameters.

        Parameters
        ----------
        initial_values : dict
            Initial parameter values (may have 'contrast'/'offset' scalars)
        n_angles : int
            Number of angles

        Returns
        -------
        dict[str, float]
            Expanded values with contrast_i and offset_i

        Examples
        --------
        >>> registry = ParameterRegistry()
        >>> registry.expand_initial_values({'contrast': 0.5, 'offset': 1.0, 'D0': 1000}, n_angles=3)
        {'contrast_0': 0.5, 'contrast_1': 0.5, 'contrast_2': 0.5,
         'offset_0': 1.0, 'offset_1': 1.0, 'offset_2': 1.0,
         'D0': 1000}
        """
        result = {}

        # Expand contrast
        contrast_val = initial_values.get("contrast", self._PARAMETERS["contrast"].default)
        for i in range(n_angles):
            result[f"contrast_{i}"] = contrast_val

        # Expand offset
        offset_val = initial_values.get("offset", self._PARAMETERS["offset"].default)
        for i in range(n_angles):
            result[f"offset_{i}"] = offset_val

        # Copy physical parameters as-is
        for key, value in initial_values.items():
            if key not in ("contrast", "offset"):
                result[key] = value

        return result

    def _normalize_mode(self, mode: str) -> str:
        """Normalize analysis mode string."""
        mode_lower = mode.lower()
        if "static" in mode_lower and "isotropic" in mode_lower:
            return "static_isotropic"
        elif "static" in mode_lower:
            return "static"
        elif "laminar" in mode_lower or "flow" in mode_lower:
            return "laminar_flow"
        else:
            raise ValueError(
                f"Unknown analysis mode: {mode}. "
                f"Expected 'static', 'static_isotropic', or 'laminar_flow'"
            )


# Module-level singleton accessor
_registry: ParameterRegistry | None = None


def get_registry() -> ParameterRegistry:
    """Get the global ParameterRegistry instance.

    Returns
    -------
    ParameterRegistry
        Singleton registry instance
    """
    global _registry
    if _registry is None:
        _registry = ParameterRegistry()
    return _registry


# Convenience functions that delegate to the singleton

def get_param_names(analysis_mode: AnalysisMode) -> list[str]:
    """Get physical parameter names for analysis mode."""
    return get_registry().get_param_names(analysis_mode)


def get_all_param_names(
    analysis_mode: AnalysisMode,
    n_angles: int = 1,
    include_scaling: bool = True,
) -> list[str]:
    """Get all parameter names including per-angle scaling."""
    return get_registry().get_all_param_names(analysis_mode, n_angles, include_scaling)


def get_bounds(name: str) -> tuple[float, float]:
    """Get parameter bounds."""
    return get_registry().get_bounds(name)


def get_defaults(
    analysis_mode: AnalysisMode,
    n_angles: int = 1,
    include_scaling: bool = True,
) -> list[float]:
    """Get default values for all parameters."""
    return get_registry().get_defaults(analysis_mode, n_angles, include_scaling)


__all__ = [
    "ParameterInfo",
    "ParameterRegistry",
    "get_registry",
    "get_param_names",
    "get_all_param_names",
    "get_bounds",
    "get_defaults",
]
