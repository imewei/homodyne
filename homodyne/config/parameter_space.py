"""Parameter Space Configuration for MCMC/CMC
============================================

Defines the ParameterSpace class for loading parameter bounds and prior
distributions from YAML configuration files. This enables config-driven
MCMC initialization without hardcoded priors.

This module is part of the v2.1.0 MCMC simplification implementation.
See: /home/wei/Documents/GitHub/homodyne/agent-os/specs/2025-10-31-mcmc-simplification/spec.md
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homodyne.config.parameter_manager import ParameterManager
from homodyne.config.types import PARAMETER_NAME_MAPPING
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PriorDistribution:
    """Prior distribution specification for a parameter.

    Attributes
    ----------
    dist_type : str
        Distribution type: 'Normal', 'TruncatedNormal', 'Uniform', 'LogNormal'
    mu : float
        Mean (location parameter)
    sigma : float
        Standard deviation (scale parameter)
    min_val : float
        Minimum bound (for truncated distributions)
    max_val : float
        Maximum bound (for truncated distributions)
    """

    dist_type: str  # 'Normal', 'TruncatedNormal', 'Uniform', 'LogNormal'
    mu: float = 0.0
    sigma: float = 1.0
    min_val: float = -np.inf
    max_val: float = np.inf

    def __post_init__(self):
        """Validate distribution parameters."""
        if self.dist_type not in [
            "Normal",
            "TruncatedNormal",
            "Uniform",
            "LogNormal",
        ]:
            logger.warning(
                f"Unknown distribution type '{self.dist_type}', defaulting to TruncatedNormal"
            )
            self.dist_type = "TruncatedNormal"

        # Validate bounds
        if self.min_val >= self.max_val:
            raise ValueError(
                f"Invalid bounds: min_val ({self.min_val}) >= max_val ({self.max_val})"
            )

        # For TruncatedNormal/Uniform, bounds must be finite
        if self.dist_type in ["TruncatedNormal", "Uniform"]:
            if np.isinf(self.min_val) or np.isinf(self.max_val):
                raise ValueError(
                    f"{self.dist_type} requires finite bounds, got [{self.min_val}, {self.max_val}]"
                )

    def to_numpyro_kwargs(self) -> dict[str, Any]:
        """Convert to NumPyro distribution kwargs.

        Returns
        -------
        dict
            Keyword arguments for NumPyro distribution constructors
        """
        if self.dist_type == "Normal":
            return {"loc": self.mu, "scale": self.sigma}
        elif self.dist_type == "TruncatedNormal":
            return {
                "loc": self.mu,
                "scale": self.sigma,
                "low": self.min_val,
                "high": self.max_val,
            }
        elif self.dist_type == "Uniform":
            return {"low": self.min_val, "high": self.max_val}
        elif self.dist_type == "LogNormal":
            return {"loc": self.mu, "scale": self.sigma}
        else:
            # Fallback to TruncatedNormal
            return {
                "loc": self.mu,
                "scale": self.sigma,
                "low": self.min_val,
                "high": self.max_val,
            }


@dataclass
class ParameterSpace:
    """Parameter space definition with bounds and prior distributions.

    This class encapsulates all information needed to define the parameter
    space for MCMC/CMC optimization, including parameter bounds and prior
    distributions loaded from configuration files.

    Attributes
    ----------
    model_type : str
        Model type: 'static' or 'laminar_flow'
    parameter_names : list[str]
        Canonical parameter names (after name mapping)
    bounds : dict[str, tuple[float, float]]
        Parameter bounds: {param_name: (min, max)}
    priors : dict[str, PriorDistribution]
        Prior distributions: {param_name: PriorDistribution}
    units : dict[str, str]
        Parameter units: {param_name: unit_string}

    Examples
    --------
    >>> # From config dict
    >>> config = {
    ...     'parameter_space': {
    ...         'model': 'static',
    ...         'bounds': [
    ...             {'name': 'D0', 'min': 100.0, 'max': 1e5,
    ...              'prior_mu': 1000.0, 'prior_sigma': 1000.0, 'type': 'TruncatedNormal'},
    ...             {'name': 'alpha', 'min': -2.0, 'max': 2.0,
    ...              'prior_mu': -1.2, 'prior_sigma': 0.3, 'type': 'Normal'}
    ...         ]
    ...     }
    ... }
    >>> param_space = ParameterSpace.from_config(config)
    >>> param_space.get_bounds('D0')
    (100.0, 100000.0)
    >>> prior = param_space.get_prior('D0')
    >>> prior.dist_type
    'TruncatedNormal'
    """

    model_type: str
    parameter_names: list[str] = field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    priors: dict[str, PriorDistribution] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config_dict: dict[str, Any],
        analysis_mode: str | None = None,
    ) -> "ParameterSpace":
        """Load ParameterSpace from configuration dictionary.

        This class method constructs a ParameterSpace instance from a YAML
        configuration dict, handling missing values gracefully and integrating
        with the existing ParameterManager for name mapping and defaults.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary (typically loaded from YAML)
        analysis_mode : str, optional
            Analysis mode ('static' or 'laminar_flow'). Auto-detected from
            config if not provided.

        Returns
        -------
        ParameterSpace
            Configured parameter space instance

        Raises
        ------
        ValueError
            If parameter_space section is malformed or missing required fields

        Examples
        --------
        >>> config = {'parameter_space': {'model': 'static', 'bounds': [...]}}
        >>> param_space = ParameterSpace.from_config(config)
        >>> param_space.model_type
        'static'

        Notes
        -----
        - Uses ParameterManager for name mapping (gamma_dot_0 → gamma_dot_t0)
        - Falls back to package defaults if config is incomplete
        - Validates all bounds and prior distribution parameters
        - Logs warnings for missing or invalid config values
        """
        # Extract parameter_space section
        param_space_config = config_dict.get("parameter_space", {})

        # Determine model type
        if analysis_mode is None:
            # Try to get from config
            analysis_mode = (
                param_space_config.get("model")
                or config_dict.get("analysis_mode")
                or "laminar_flow"
            )

        model_type = analysis_mode.lower()

        # Initialize ParameterManager for name mapping and defaults
        param_manager = ParameterManager(config_dict, analysis_mode=model_type)

        # Get parameter names (use ParameterManager to respect active_parameters)
        parameter_names = param_manager.get_active_parameters()

        # Parse bounds from config
        bounds_dict: dict[str, tuple[float, float]] = {}
        priors_dict: dict[str, PriorDistribution] = {}
        units_dict: dict[str, str] = {}

        config_bounds = param_space_config.get("bounds", [])
        if not isinstance(config_bounds, list):
            logger.warning(
                "parameter_space.bounds must be a list, using package defaults"
            )
            config_bounds = []

        # Build lookup dict from config bounds
        config_bounds_lookup: dict[str, dict[str, Any]] = {}
        for bound_entry in config_bounds:
            if not isinstance(bound_entry, dict):
                continue

            param_name = bound_entry.get("name")
            if not param_name:
                continue

            # Apply name mapping
            canonical_name = PARAMETER_NAME_MAPPING.get(param_name, param_name)
            config_bounds_lookup[canonical_name] = bound_entry

        # Load bounds and priors for each parameter
        for param_name in parameter_names:
            # Get config entry (if exists)
            config_entry = config_bounds_lookup.get(param_name, {})

            # Extract bounds (with fallback to ParameterManager defaults)
            if "min" in config_entry and "max" in config_entry:
                min_val = float(config_entry["min"])
                max_val = float(config_entry["max"])
            else:
                # Fallback to ParameterManager defaults
                default_bounds = param_manager.get_parameter_bounds([param_name])
                if default_bounds:
                    min_val = default_bounds[0]["min"]
                    max_val = default_bounds[0]["max"]
                    logger.debug(
                        f"Using default bounds for '{param_name}': [{min_val}, {max_val}]"
                    )
                else:
                    # Ultimate fallback
                    min_val, max_val = 0.0, 1.0
                    logger.warning(
                        f"No bounds found for '{param_name}', using [0.0, 1.0]"
                    )

            bounds_dict[param_name] = (min_val, max_val)

            # Extract prior distribution
            prior_mu = config_entry.get("prior_mu", (min_val + max_val) / 2.0)
            prior_sigma = config_entry.get("prior_sigma", (max_val - min_val) / 4.0)
            dist_type = config_entry.get("type", "TruncatedNormal")

            # Create PriorDistribution object
            try:
                prior = PriorDistribution(
                    dist_type=dist_type,
                    mu=float(prior_mu),
                    sigma=float(prior_sigma),
                    min_val=min_val,
                    max_val=max_val,
                )
                priors_dict[param_name] = prior
            except ValueError as e:
                logger.warning(
                    f"Invalid prior for '{param_name}': {e}. Using default TruncatedNormal."
                )
                # Fallback prior
                priors_dict[param_name] = PriorDistribution(
                    dist_type="TruncatedNormal",
                    mu=(min_val + max_val) / 2.0,
                    sigma=(max_val - min_val) / 4.0,
                    min_val=min_val,
                    max_val=max_val,
                )

            # Extract unit (optional)
            unit = config_entry.get("unit", "")
            if unit:
                units_dict[param_name] = unit

        # Log summary
        logger.info(
            f"Loaded ParameterSpace: model={model_type}, "
            f"n_params={len(parameter_names)}, "
            f"parameters={parameter_names}"
        )

        return cls(
            model_type=model_type,
            parameter_names=parameter_names,
            bounds=bounds_dict,
            priors=priors_dict,
            units=units_dict,
        )

    @classmethod
    def from_defaults(
        cls,
        analysis_mode: str = "laminar_flow",
    ) -> "ParameterSpace":
        """Create ParameterSpace with package defaults (no config file).

        This method creates a ParameterSpace using only the hardcoded
        defaults from ParameterManager, useful when no config file is
        available or for testing.

        Parameters
        ----------
        analysis_mode : str
            Analysis mode: 'static' or 'laminar_flow'

        Returns
        -------
        ParameterSpace
            Parameter space with default bounds and wide priors

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> param_space.parameter_names
        ['D0', 'alpha', 'D_offset']
        """
        logger.info(
            f"Creating ParameterSpace from package defaults (mode={analysis_mode})"
        )

        # Create empty config and let from_config handle defaults
        empty_config: dict[str, Any] = {"analysis_mode": analysis_mode}

        return cls.from_config(empty_config, analysis_mode=analysis_mode)

    def get_bounds(self, param_name: str) -> tuple[float, float]:
        """Get bounds for a specific parameter.

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        tuple[float, float]
            (min_value, max_value)

        Raises
        ------
        KeyError
            If parameter not found in parameter space
        """
        if param_name not in self.bounds:
            raise KeyError(
                f"Parameter '{param_name}' not in parameter space. "
                f"Available: {list(self.bounds.keys())}"
            )
        return self.bounds[param_name]

    def get_prior(self, param_name: str) -> PriorDistribution:
        """Get prior distribution for a specific parameter.

        Parameters
        ----------
        param_name : str
            Parameter name

        Returns
        -------
        PriorDistribution
            Prior distribution specification

        Raises
        ------
        KeyError
            If parameter not found in parameter space
        """
        if param_name not in self.priors:
            raise KeyError(
                f"Parameter '{param_name}' not in parameter space. "
                f"Available: {list(self.priors.keys())}"
            )
        return self.priors[param_name]

    def get_bounds_array(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds as numpy arrays (for optimization).

        Returns
        -------
        lower_bounds : np.ndarray
            Array of lower bounds (in parameter_names order)
        upper_bounds : np.ndarray
            Array of upper bounds (in parameter_names order)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> lower, upper = param_space.get_bounds_array()
        >>> lower.shape
        (3,)
        """
        lower = np.array([self.bounds[name][0] for name in self.parameter_names])
        upper = np.array([self.bounds[name][1] for name in self.parameter_names])
        return lower, upper

    def get_prior_means(self) -> np.ndarray:
        """Get prior means as numpy array (for initialization).

        Returns
        -------
        np.ndarray
            Array of prior means (in parameter_names order)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> means = param_space.get_prior_means()
        >>> means.shape
        (3,)
        """
        return np.array([self.priors[name].mu for name in self.parameter_names])

    def validate_values(
        self, values: dict[str, float], tolerance: float = 1e-10
    ) -> tuple[bool, list[str]]:
        """Validate parameter values against bounds.

        Parameters
        ----------
        values : dict[str, float]
            Parameter values to validate
        tolerance : float
            Tolerance for bounds checking

        Returns
        -------
        is_valid : bool
            True if all values are within bounds
        violations : list[str]
            List of violation messages (empty if valid)

        Examples
        --------
        >>> param_space = ParameterSpace.from_defaults('static')
        >>> values = {'D0': 1000.0, 'alpha': -1.2, 'D_offset': 0.0}
        >>> is_valid, violations = param_space.validate_values(values)
        >>> is_valid
        True
        """
        violations = []

        for param_name, value in values.items():
            if param_name not in self.bounds:
                violations.append(
                    f"Unknown parameter '{param_name}' (not in parameter space)"
                )
                continue

            min_val, max_val = self.bounds[param_name]

            if value < min_val - tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} < min ({min_val:.3e}) "
                    f"by {min_val - value:.3e}"
                )
            elif value > max_val + tolerance:
                violations.append(
                    f"{param_name} = {value:.3e} > max ({max_val:.3e}) "
                    f"by {value - max_val:.3e}"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ParameterSpace(model={self.model_type}, "
            f"n_params={len(self.parameter_names)}, "
            f"params={self.parameter_names})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = [f"ParameterSpace: {self.model_type} model"]
        lines.append(f"  Parameters ({len(self.parameter_names)}):")

        for param_name in self.parameter_names:
            min_val, max_val = self.bounds[param_name]
            prior = self.priors[param_name]
            unit = self.units.get(param_name, "")

            lines.append(
                f"    {param_name:20s}: "
                f"[{min_val:10.3e}, {max_val:10.3e}] "
                f"{prior.dist_type}(μ={prior.mu:.3e}, σ={prior.sigma:.3e}) "
                f"{unit}"
            )

        return "\n".join(lines)
