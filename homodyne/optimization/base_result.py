"""
Base Result Classes for All Optimization Methods
================================================

Unified result class hierarchy to eliminate duplication across
LSQ, VI, and MCMC result wrappers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import numpy as np
from datetime import datetime

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BaseOptimizationResult(ABC):
    """
    Abstract base class for all optimization results.

    Provides common functionality for all optimization methods
    and ensures consistent interface for CLI display.

    Common field naming conventions:
    - mean_params: Physical parameter means
    - mean_contrast, mean_offset: Fitting parameter means
    - std_params, std_contrast, std_offset: Parameter uncertainties
    - converged: Convergence flag
    - n_iterations: Number of iterations
    - computation_time: Total time in seconds
    - backend: "JAX" or "NumPy"
    - analysis_mode: Analysis mode string
    - dataset_size: Size category
    """

    # Common required fields (match actual codebase interfaces)
    mean_params: np.ndarray  # Physical parameter means
    mean_contrast: float     # Contrast mean
    mean_offset: float       # Offset mean
    converged: bool          # Convergence flag
    n_iterations: int        # Number of iterations
    computation_time: float  # Total time (seconds)
    backend: str            # Backend used
    analysis_mode: str      # Analysis mode
    dataset_size: str       # Dataset size category

    # Common optional fields with defaults
    std_params: Optional[np.ndarray] = None      # Parameter uncertainties
    std_contrast: Optional[float] = None         # Contrast uncertainty
    std_offset: Optional[float] = None           # Offset uncertainty
    chi_squared: Optional[float] = None          # Chi-squared value
    reduced_chi_squared: Optional[float] = None  # Reduced chi-squared
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_derived_metrics()

    def _calculate_derived_metrics(self):
        """Calculate derived metrics if not provided."""
        # Calculate reduced chi-squared if chi_squared is available
        if (self.chi_squared is not None and
            self.reduced_chi_squared is None and
            hasattr(self, 'degrees_of_freedom') and self.degrees_of_freedom > 0):
            self.reduced_chi_squared = self.chi_squared / self.degrees_of_freedom

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        pass

    @abstractmethod
    def format_for_display(self) -> str:
        """Format result for CLI display."""
        pass

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the result."""
        summary = {
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "computation_time": self.computation_time,
            "backend": self.backend,
            "analysis_mode": self.analysis_mode,
            "dataset_size": self.dataset_size,
            "mean_params": self.mean_params.tolist() if isinstance(self.mean_params, np.ndarray) else self.mean_params,
            "mean_contrast": self.mean_contrast,
            "mean_offset": self.mean_offset
        }

        if self.std_params is not None:
            summary["std_params"] = self.std_params.tolist() if isinstance(self.std_params, np.ndarray) else self.std_params
        if self.std_contrast is not None:
            summary["std_contrast"] = self.std_contrast
        if self.std_offset is not None:
            summary["std_offset"] = self.std_offset
        if self.chi_squared is not None:
            summary["chi_squared"] = self.chi_squared
        if self.reduced_chi_squared is not None:
            summary["reduced_chi_squared"] = self.reduced_chi_squared

        return summary

    def get_fitted_params_with_errors(self) -> Dict[str, tuple]:
        """Get parameters with their uncertainties as (value, error) tuples."""
        # This method is for compatibility but relies on subclasses to implement parameter mapping
        raise NotImplementedError("Subclasses must implement parameter mapping for this method")


@dataclass
class LSQResult(BaseOptimizationResult):
    """
    Least Squares optimization result.

    Matches the interface in lsq_wrapper.py exactly.
    """

    # LSQ-specific fields (match lsq_wrapper.py interface)
    residual_std: float = 0.0              # Standard deviation of residuals
    max_residual: float = 0.0              # Maximum residual value
    degrees_of_freedom: int = 0            # Degrees of freedom

    # LSQ always sets these to None (no uncertainties available)
    std_params: Optional[np.ndarray] = None
    std_contrast: Optional[float] = None
    std_offset: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.get_summary()
        result.update({
            "residual_std": self.residual_std,
            "max_residual": self.max_residual,
            "degrees_of_freedom": self.degrees_of_freedom
        })
        return result

    def format_for_display(self) -> str:
        """Format for CLI display."""
        lines = [
            f"{'='*60}",
            f"LSQ Optimization Results",
            f"{'='*60}",
            f"Success: {self.success}",
            f"Message: {self.message}",
            f"Iterations: {self.n_iterations}",
            f"Computation time: {self.computation_time:.3f} s",
            f"",
            f"Goodness of Fit:",
            f"  Chi-squared: {self.chi_squared:.6f}",
            f"  Reduced chi-squared: {self.chi_squared_reduced:.6f}" if self.chi_squared_reduced else "",
            f"  R-squared: {self.r_squared:.6f}" if self.r_squared else "",
            f"",
            f"Optimized Parameters:",
        ]

        for param, value in self.optimized_params.items():
            if self.uncertainties and param in self.uncertainties:
                lines.append(f"  {param}: {value:.6e} ± {self.uncertainties[param]:.6e}")
            else:
                lines.append(f"  {param}: {value:.6e}")

        if self.sampling_info:
            lines.extend([
                f"",
                f"Sampling Information:",
                f"  Sampled points: {self.sampling_info.get('n_samples', 'N/A')}",
                f"  Total points: {self.n_data_points}",
                f"  Sampling ratio: {self.sampling_info.get('sampling_ratio', 'N/A'):.2%}"
            ])

        return "\n".join(filter(None, lines))


@dataclass
class VIResult(BaseOptimizationResult):
    """
    Variational Inference optimization result.

    Extends base class with VI-specific fields.
    """

    # VI-specific fields (match actual usage in variational.py)
    final_elbo: Optional[float] = None
    elbo_history: Optional[np.ndarray] = None
    kl_divergence: Optional[float] = None
    likelihood: Optional[float] = None

    # Variational parameters
    variational_params: Optional[Dict[str, Any]] = None
    posterior_samples: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().get_summary()

        if self.final_elbo is not None:
            result["final_elbo"] = self.final_elbo

        if self.likelihood is not None:
            result["likelihood"] = self.likelihood

        if self.kl_divergence is not None:
            result["kl_divergence"] = self.kl_divergence

        if self.variational_params:
            result["variational_params"] = self.variational_params

        return result

    def format_for_display(self) -> str:
        """Format for CLI display."""
        lines = [
            f"{'='*60}",
            f"Variational Inference Results",
            f"{'='*60}",
            f"Success: {self.success}",
            f"Message: {self.message}",
            f"Iterations: {self.n_iterations}",
            f"Computation time: {self.computation_time:.3f} s",
            f"",
            f"Optimization Metrics:",
            f"  Final ELBO: {self.final_elbo:.6f}" if self.final_elbo else "",
            f"  Likelihood: {self.likelihood:.6f}" if self.likelihood else "",
            f"  KL Divergence: {self.kl_divergence:.6f}" if self.kl_divergence else "",
            f"",
            f"Goodness of Fit:",
            f"  Chi-squared: {self.chi_squared:.6f}",
            f"  Reduced chi-squared: {self.chi_squared_reduced:.6f}" if self.chi_squared_reduced else "",
            f"",
            f"Optimized Parameters (posterior means):",
        ]

        for param, value in self.optimized_params.items():
            if self.uncertainties and param in self.uncertainties:
                lines.append(f"  {param}: {value:.6e} ± {self.uncertainties[param]:.6e}")
            else:
                lines.append(f"  {param}: {value:.6e}")

        return "\n".join(filter(None, lines))


@dataclass
class MCMCResult(BaseOptimizationResult):
    """
    MCMC optimization result.

    Extends base class with MCMC-specific fields.
    Maintains compatibility with existing mcmc.py implementation.
    """

    # MCMC samples (match existing mcmc.py structure)
    samples_params: Optional[np.ndarray] = None
    samples_contrast: Optional[np.ndarray] = None
    samples_offset: Optional[np.ndarray] = None

    # Additional sample statistics
    quantiles_params: Optional[np.ndarray] = None
    log_likelihood_trace: Optional[np.ndarray] = None

    # MCMC chain metadata (missing from original consolidated)
    n_chains: Optional[int] = None
    n_warmup: Optional[int] = None
    n_samples: Optional[int] = None
    sampler: Optional[str] = None
    divergences: Optional[int] = None

    # MCMC diagnostics (flexible format for compatibility)
    acceptance_rate: Optional[float] = None
    r_hat: Optional[Union[np.ndarray, Dict[str, float]]] = None
    effective_sample_size: Optional[Union[np.ndarray, Dict[str, float]]] = None

    # Posterior statistics (legacy consolidated fields)
    posterior_mean: Optional[Dict[str, float]] = None
    posterior_median: Optional[Dict[str, float]] = None
    posterior_std: Optional[Dict[str, float]] = None
    credible_intervals: Optional[Dict[str, tuple]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().get_summary()

        # MCMC chain metadata
        if self.n_chains is not None:
            result["n_chains"] = self.n_chains
        if self.n_warmup is not None:
            result["n_warmup"] = self.n_warmup
        if self.n_samples is not None:
            result["n_samples"] = self.n_samples
        if self.sampler is not None:
            result["sampler"] = self.sampler
        if self.divergences is not None:
            result["divergences"] = self.divergences

        # MCMC diagnostics
        if self.acceptance_rate is not None:
            result["acceptance_rate"] = self.acceptance_rate
        if self.r_hat is not None:
            result["r_hat"] = self.r_hat
        if self.effective_sample_size is not None:
            result["effective_sample_size"] = self.effective_sample_size

        # Posterior statistics
        if self.credible_intervals:
            result["credible_intervals"] = self.credible_intervals
        if self.posterior_mean:
            result["posterior_mean"] = self.posterior_mean
        if self.posterior_std:
            result["posterior_std"] = self.posterior_std

        return result

    def format_for_display(self) -> str:
        """Format for CLI display."""
        lines = [
            f"{'='*60}",
            f"MCMC Results",
            f"{'='*60}",
            f"Success: {self.converged}",
            f"Sampler: {self.sampler}" if self.sampler else "",
            f"Backend: {self.backend}",
            f"Total iterations: {self.n_iterations}",
            f"Computation time: {self.computation_time:.3f} s",
            f"",
            f"Chain Configuration:",
            f"  Chains: {self.n_chains}" if self.n_chains else "",
            f"  Warmup samples: {self.n_warmup}" if self.n_warmup else "",
            f"  Posterior samples: {self.n_samples}" if self.n_samples else "",
            f"  Divergences: {self.divergences}" if self.divergences is not None else "",
            f"",
            f"Sampling Metrics:",
            f"  Acceptance rate: {self.acceptance_rate:.2%}" if self.acceptance_rate else "",
            f"",
            f"Goodness of Fit:",
            f"  Chi-squared: {self.chi_squared:.6f}" if self.chi_squared else "",
            f"  Reduced chi-squared: {self.reduced_chi_squared:.6f}" if self.reduced_chi_squared else "",
            f"",
            f"Posterior Statistics:",
        ]

        for param in self.optimized_params:
            line = f"  {param}:"

            if self.posterior_mean and param in self.posterior_mean:
                line += f" mean={self.posterior_mean[param]:.6e}"

            if self.posterior_std and param in self.posterior_std:
                line += f" ± {self.posterior_std[param]:.6e}"

            if self.credible_intervals and param in self.credible_intervals:
                ci = self.credible_intervals[param]
                line += f" CI=[{ci[0]:.6e}, {ci[1]:.6e}]"

            lines.append(line)

        if self.r_hat:
            lines.extend([
                f"",
                f"Convergence Diagnostics (R-hat):"
            ])
            for param, value in self.r_hat.items():
                status = "✓" if value < 1.01 else "✗"
                lines.append(f"  {param}: {value:.4f} {status}")

        return "\n".join(filter(None, lines))


def create_result(method: str, **kwargs) -> BaseOptimizationResult:
    """
    Factory function to create appropriate result object.

    Args:
        method: Optimization method ("LSQ", "VI", "MCMC")
        **kwargs: Method-specific parameters

    Returns:
        Appropriate result object
    """
    method_upper = method.upper()

    if method_upper == "LSQ":
        return LSQResult(**kwargs)
    elif method_upper == "VI":
        return VIResult(**kwargs)
    elif method_upper == "MCMC":
        return MCMCResult(**kwargs)
    else:
        raise ValueError(f"Unknown optimization method: {method}")