"""Extended MCMCResult class with CMC support and ArviZ integration.

This module extends the existing MCMCResult class from homodyne.optimization.mcmc
to support Consensus Monte Carlo (CMC) specific fields while maintaining 100%
backward compatibility with existing code.

Key Features:
- All CMC-specific fields are optional and default to None
- Existing code using MCMCResult continues to work unchanged
- New is_cmc_result() method to detect CMC results
- Serialization support for CMC-specific data
- Per-shard diagnostics and combination method tracking
- ArviZ InferenceData conversion for diagnostics and plotting
- 95% credible intervals (CI) for uncertainty quantification
- Fitted data storage for comparison with experimental data

Backward Compatibility:
- All existing MCMCResult parameters work exactly as before
- CMC fields only used when explicitly provided
- Non-CMC results load and save without CMC data
- No breaking changes to existing API

ArviZ Integration (v2.4.1+):
- to_arviz() method converts MCMCResult to az.InferenceData
- compute_summary() returns pandas DataFrame with diagnostics
- Compatible with az.plot_trace(), az.plot_posterior(), az.plot_pair()
"""

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import arviz as az
    import pandas as pd


class MCMCResult:
    """Extended MCMC result container with CMC support.

    This class extends the standard MCMC result structure to support
    Consensus Monte Carlo (CMC) workflows while maintaining complete
    backward compatibility with existing code.

    Standard MCMC Fields
    --------------------
    mean_params : np.ndarray
        Mean parameter values from posterior
    mean_contrast : float
        Mean contrast parameter
    mean_offset : float
        Mean offset parameter
    std_params : np.ndarray, optional
        Standard deviations of parameters
    std_contrast : float, optional
        Standard deviation of contrast
    std_offset : float, optional
        Standard deviation of offset
    samples_params : np.ndarray, optional
        Full posterior samples for parameters
    samples_contrast : np.ndarray, optional
        Full posterior samples for contrast
    samples_offset : np.ndarray, optional
        Full posterior samples for offset
    converged : bool, default True
        Whether MCMC sampling converged
    n_iterations : int, default 0
        Number of MCMC iterations completed
    computation_time : float, default 0.0
        Total computation time in seconds
    backend : str, default "JAX"
        Backend used for computation
    analysis_mode : str, default "static_isotropic"
        Analysis mode used
    dataset_size : str, default "unknown"
        Size category of dataset
    n_chains : int, default 4
        Number of MCMC chains
    n_warmup : int, default 1000
        Number of warmup iterations
    n_samples : int, default 1000
        Number of sampling iterations per chain
    sampler : str, default "NUTS"
        MCMC sampler algorithm used
    acceptance_rate : float, optional
        Mean acceptance rate across chains
    r_hat : dict[str, float], optional
        Gelman-Rubin convergence diagnostic per parameter
    effective_sample_size : dict[str, float], optional
        Effective sample size per parameter

    CMC-Specific Fields (NEW)
    --------------------------
    per_shard_diagnostics : list[dict], optional
        Diagnostics from each shard's MCMC run. Each dict contains:
        - shard_id : int
        - n_samples : int
        - acceptance_rate : float
        - r_hat : dict[str, float]
        - ess : dict[str, float]
        - converged : bool
        Default: None (for non-CMC results)

    cmc_diagnostics : dict, optional
        Overall CMC diagnostics including:
        - combination_success : bool
        - n_shards_converged : int
        - n_shards_total : int
        - weighted_product_std : float (for weighted combination)
        - combination_time : float
        Default: None (for non-CMC results)

    combination_method : str, optional
        Method used to combine subposteriors:
        - "weighted" : Weighted Gaussian product
        - "average" : Simple averaging
        - "hierarchical" : Hierarchical combination (Phase 2)
        Default: None (for non-CMC results)

    num_shards : int, optional
        Number of data shards used in CMC.
        If None or 1, result is from standard MCMC.
        If > 1, result is from CMC.
        Default: None (for non-CMC results)

    Config-Driven Metadata (v2.1.0+)
    ---------------------------------
    parameter_space_metadata : dict, optional
        Parameter space configuration used for this analysis. Contains:
        - bounds : dict[str, tuple[float, float]]
        - priors : dict[str, dict] (prior distribution specs)
        - model_type : str (static_isotropic or laminar_flow)
        Default: None (for backward compatibility)

    initial_values_metadata : dict[str, float], optional
        Initial parameter values used to initialize MCMC chains. Example:
        {"D0": 1234.5, "alpha": 0.567, "D_offset": 12.34}
        Default: None (for backward compatibility)

    selection_decision_metadata : dict, optional
        Automatic NUTS/CMC selection decision information. Contains:
        - selected_method : str ("NUTS" or "CMC")
        - num_samples : int
        - parallelism_criterion_met : bool
        - memory_criterion_met : bool
        - min_samples_for_cmc : int
        - memory_threshold_pct : float
        - estimated_memory_fraction : float (optional)
        Default: None (for backward compatibility)

    ArviZ Integration Fields (v2.4.1+)
    ----------------------------------
    ci_95_lower : np.ndarray, optional
        Lower bounds of 95% credible intervals for each parameter.
        Computed as 2.5th percentile of posterior samples.
        Default: None (computed lazily from samples_params)

    ci_95_upper : np.ndarray, optional
        Upper bounds of 95% credible intervals for each parameter.
        Computed as 97.5th percentile of posterior samples.
        Default: None (computed lazily from samples_params)

    fitted_data : np.ndarray, optional
        Predicted g2 values computed from posterior mean parameters.
        Shape matches experimental data for comparison plots.
        Default: None (computed by coordinator after sampling)

    param_names : list[str], optional
        Parameter names for ArviZ labeling and diagnostics.
        Example: ['contrast_0', 'offset_0', 'D0', 'alpha', 'D_offset']
        Default: None (auto-generated from analysis_mode)

    Examples
    --------
    Standard MCMC result (backward compatible):

    >>> result = MCMCResult(
    ...     mean_params=np.array([100.0, 1.5, 10.0]),
    ...     mean_contrast=0.5,
    ...     mean_offset=1.0,
    ... )
    >>> result.is_cmc_result()
    False

    CMC result with shard diagnostics:

    >>> result = MCMCResult(
    ...     mean_params=np.array([100.0, 1.5, 10.0]),
    ...     mean_contrast=0.5,
    ...     mean_offset=1.0,
    ...     num_shards=10,
    ...     combination_method="weighted",
    ...     per_shard_diagnostics=[
    ...         {"shard_id": 0, "converged": True, "acceptance_rate": 0.85},
    ...         {"shard_id": 1, "converged": True, "acceptance_rate": 0.82},
    ...     ],
    ... )
    >>> result.is_cmc_result()
    True
    >>> result.num_shards
    10
    """

    def __init__(
        self,
        mean_params: np.ndarray,
        mean_contrast: float,
        mean_offset: float,
        std_params: np.ndarray | None = None,
        std_contrast: float | None = None,
        std_offset: float | None = None,
        samples_params: np.ndarray | None = None,
        samples_contrast: np.ndarray | None = None,
        samples_offset: np.ndarray | None = None,
        converged: bool = True,
        n_iterations: int = 0,
        computation_time: float = 0.0,
        backend: str = "JAX",
        analysis_mode: str = "static_isotropic",
        dataset_size: str = "unknown",
        n_chains: int = 4,
        n_warmup: int = 1000,
        n_samples: int = 1000,
        sampler: str = "NUTS",
        acceptance_rate: float | None = None,
        r_hat: dict[str, float] | None = None,
        effective_sample_size: dict[str, float] | None = None,
        # NEW: CMC-specific fields (optional, backward compatible)
        per_shard_diagnostics: list[dict[str, Any]] | None = None,
        cmc_diagnostics: dict[str, Any] | None = None,
        combination_method: str | None = None,
        num_shards: int | None = None,
        # NEW v2.1.0: Config-driven metadata fields (optional, backward compatible)
        parameter_space_metadata: dict[str, Any] | None = None,
        initial_values_metadata: dict[str, float] | None = None,
        selection_decision_metadata: dict[str, Any] | None = None,
        # NEW v2.4.1: ArviZ integration fields (optional, backward compatible)
        ci_95_lower: np.ndarray | None = None,
        ci_95_upper: np.ndarray | None = None,
        fitted_data: np.ndarray | None = None,
        param_names: list[str] | None = None,
        **kwargs,
    ):
        """Initialize MCMCResult with optional CMC support.

        All CMC-specific parameters default to None for backward compatibility.
        Existing code continues to work without modification.
        """
        # Standard MCMC fields (existing)
        self.mean_params = mean_params
        self.mean_contrast = mean_contrast
        self.mean_offset = mean_offset

        # Uncertainties
        self.std_params = (
            std_params if std_params is not None else np.zeros_like(mean_params)
        )
        self.std_contrast = std_contrast if std_contrast is not None else 0.0
        self.std_offset = std_offset if std_offset is not None else 0.0

        # Samples
        self.samples_params = samples_params
        self.samples_contrast = samples_contrast
        self.samples_offset = samples_offset

        # Metadata
        self.converged = converged
        self.n_iterations = n_iterations
        self.computation_time = computation_time
        self.backend = backend
        self.analysis_mode = analysis_mode
        self.dataset_size = dataset_size

        # MCMC-specific
        self.n_chains = n_chains
        self.n_warmup = n_warmup
        self.n_samples = n_samples
        self.sampler = sampler
        self.acceptance_rate = acceptance_rate
        self.r_hat = r_hat
        self.effective_sample_size = effective_sample_size

        # NEW: CMC-specific attributes (all optional, default None)
        self.per_shard_diagnostics = per_shard_diagnostics
        self.cmc_diagnostics = cmc_diagnostics
        self.combination_method = combination_method
        self.num_shards = num_shards

        # NEW v2.1.0: Config-driven metadata attributes (all optional, default None)
        self.parameter_space_metadata = parameter_space_metadata
        self.initial_values_metadata = initial_values_metadata
        self.selection_decision_metadata = selection_decision_metadata

        # NEW v2.4.1: ArviZ integration attributes (all optional, default None)
        self.ci_95_lower = ci_95_lower
        self.ci_95_upper = ci_95_upper
        self.fitted_data = fitted_data
        self.param_names = param_names

    def is_cmc_result(self) -> bool:
        """Return True when the result came from the CMC pipeline.

        v3.0 made CMC the only MCMC path, and single-shard runs still flow
        through the CMC machinery (per-shard NUTS + combiner bypass). We treat a
        result as CMC if any of the following holds:
        - selection_decision_metadata.method == "CMC"
        - cmc_diagnostics is present
        - num_shards is not None (including the single-shard case)

        Returns
        -------
        bool
            True if produced by CMC; False for legacy/standard MCMC results.

        Examples
        --------
        >>> # Standard MCMC result (legacy)
        >>> result = MCMCResult(mean_params=np.array([1.0]), mean_contrast=0.5, mean_offset=1.0)
        >>> result.is_cmc_result()
        False

        >>> # CMC result with 10 shards
        >>> result = MCMCResult(
        ...     mean_params=np.array([1.0]),
        ...     mean_contrast=0.5,
        ...     mean_offset=1.0,
        ...     num_shards=10
        ... )
        >>> result.is_cmc_result()
        True

        >>> # Single-shard CMC (combiner bypassed but still CMC pipeline)
        >>> result = MCMCResult(
        ...     mean_params=np.array([1.0]),
        ...     mean_contrast=0.5,
        ...     mean_offset=1.0,
        ...     num_shards=1,
        ...     selection_decision_metadata={"method": "CMC"},
        ... )
        >>> result.is_cmc_result()
        True
        """
        if self.selection_decision_metadata:
            method = str(self.selection_decision_metadata.get("method", "")).lower()
            if method == "cmc":
                return True

        if self.cmc_diagnostics is not None:
            return True

        if self.num_shards is not None:
            # In CMC-only architecture, any explicit shard count signals CMC.
            return True

        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert MCMCResult to dictionary for serialization.

        Returns dictionary representation suitable for JSON serialization.
        Numpy arrays are converted to lists. None values are preserved.

        Returns
        -------
        dict
            Dictionary containing all result fields

        Examples
        --------
        >>> result = MCMCResult(
        ...     mean_params=np.array([1.0, 2.0]),
        ...     mean_contrast=0.5,
        ...     mean_offset=1.0,
        ...     num_shards=5
        ... )
        >>> data = result.to_dict()
        >>> data["num_shards"]
        5
        >>> data["mean_params"]
        [1.0, 2.0]
        """
        data = {
            # Standard fields
            "mean_params": self.mean_params.tolist(),
            "mean_contrast": self.mean_contrast,
            "mean_offset": self.mean_offset,
            "std_params": (
                self.std_params.tolist() if self.std_params is not None else None
            ),
            "std_contrast": self.std_contrast,
            "std_offset": self.std_offset,
            "samples_params": (
                self.samples_params.tolist()
                if self.samples_params is not None
                else None
            ),
            "samples_contrast": (
                self.samples_contrast.tolist()
                if self.samples_contrast is not None
                else None
            ),
            "samples_offset": (
                self.samples_offset.tolist()
                if self.samples_offset is not None
                else None
            ),
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "computation_time": self.computation_time,
            "backend": self.backend,
            "analysis_mode": self.analysis_mode,
            "dataset_size": self.dataset_size,
            "n_chains": self.n_chains,
            "n_warmup": self.n_warmup,
            "n_samples": self.n_samples,
            "sampler": self.sampler,
            "acceptance_rate": self.acceptance_rate,
            "r_hat": self.r_hat,
            "effective_sample_size": self.effective_sample_size,
            # CMC-specific fields
            "per_shard_diagnostics": self.per_shard_diagnostics,
            "cmc_diagnostics": self.cmc_diagnostics,
            "combination_method": self.combination_method,
            "num_shards": self.num_shards,
            # v2.1.0 Config-driven metadata fields
            "parameter_space_metadata": self.parameter_space_metadata,
            "initial_values_metadata": self.initial_values_metadata,
            "selection_decision_metadata": self.selection_decision_metadata,
            # v2.4.1 ArviZ integration fields
            "ci_95_lower": (
                self.ci_95_lower.tolist() if self.ci_95_lower is not None else None
            ),
            "ci_95_upper": (
                self.ci_95_upper.tolist() if self.ci_95_upper is not None else None
            ),
            "fitted_data": (
                self.fitted_data.tolist() if self.fitted_data is not None else None
            ),
            "param_names": self.param_names,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCMCResult":
        """Create MCMCResult from dictionary.

        Handles both standard MCMC results and CMC results.
        Missing CMC fields default to None for backward compatibility.

        Parameters
        ----------
        data : dict
            Dictionary containing result fields

        Returns
        -------
        MCMCResult
            Reconstructed result object

        Examples
        --------
        >>> data = {
        ...     "mean_params": [1.0, 2.0],
        ...     "mean_contrast": 0.5,
        ...     "mean_offset": 1.0,
        ...     "num_shards": 5,
        ... }
        >>> result = MCMCResult.from_dict(data)
        >>> result.is_cmc_result()
        True
        >>> result.num_shards
        5
        """
        # Convert lists back to numpy arrays
        mean_params = np.array(data["mean_params"])
        std_params = (
            np.array(data["std_params"]) if data.get("std_params") is not None else None
        )
        samples_params = (
            np.array(data["samples_params"])
            if data.get("samples_params") is not None
            else None
        )
        samples_contrast = (
            np.array(data["samples_contrast"])
            if data.get("samples_contrast") is not None
            else None
        )
        samples_offset = (
            np.array(data["samples_offset"])
            if data.get("samples_offset") is not None
            else None
        )

        return cls(
            mean_params=mean_params,
            mean_contrast=data["mean_contrast"],
            mean_offset=data["mean_offset"],
            std_params=std_params,
            std_contrast=data.get("std_contrast"),
            std_offset=data.get("std_offset"),
            samples_params=samples_params,
            samples_contrast=samples_contrast,
            samples_offset=samples_offset,
            converged=data.get("converged", True),
            n_iterations=data.get("n_iterations", 0),
            computation_time=data.get("computation_time", 0.0),
            backend=data.get("backend", "JAX"),
            analysis_mode=data.get("analysis_mode", "static_isotropic"),
            dataset_size=data.get("dataset_size", "unknown"),
            n_chains=data.get("n_chains", 4),
            n_warmup=data.get("n_warmup", 1000),
            n_samples=data.get("n_samples", 1000),
            sampler=data.get("sampler", "NUTS"),
            acceptance_rate=data.get("acceptance_rate"),
            r_hat=data.get("r_hat"),
            effective_sample_size=data.get("effective_sample_size"),
            # CMC-specific fields (default to None for backward compatibility)
            per_shard_diagnostics=data.get("per_shard_diagnostics"),
            cmc_diagnostics=data.get("cmc_diagnostics"),
            combination_method=data.get("combination_method"),
            num_shards=data.get("num_shards"),
            # v2.1.0 Config-driven metadata fields (default to None for backward compatibility)
            parameter_space_metadata=data.get("parameter_space_metadata"),
            initial_values_metadata=data.get("initial_values_metadata"),
            selection_decision_metadata=data.get("selection_decision_metadata"),
            # v2.4.1 ArviZ integration fields (default to None for backward compatibility)
            ci_95_lower=(
                np.array(data["ci_95_lower"])
                if data.get("ci_95_lower") is not None
                else None
            ),
            ci_95_upper=(
                np.array(data["ci_95_upper"])
                if data.get("ci_95_upper") is not None
                else None
            ),
            fitted_data=(
                np.array(data["fitted_data"])
                if data.get("fitted_data") is not None
                else None
            ),
            param_names=data.get("param_names"),
        )

    def get_param_names(self) -> list[str]:
        """Get parameter names for this result.

        Returns parameter names based on analysis_mode if not explicitly set.
        For per-angle scaling, includes contrast_i and offset_i names.

        Returns
        -------
        list[str]
            Parameter names in correct order for NumPyro/ArviZ.
        """
        if self.param_names is not None:
            return self.param_names

        # Auto-generate based on analysis_mode
        num_params = len(self.mean_params)

        if self.analysis_mode == "static":
            base_names = ["D0", "alpha", "D_offset"]
        elif self.analysis_mode == "laminar_flow":
            base_names = [
                "D0",
                "alpha",
                "D_offset",
                "gamma_dot_t0",
                "beta",
                "gamma_dot_t_offset",
                "phi0",
            ]
        else:
            base_names = [f"param_{i}" for i in range(num_params)]

        # If num_params > len(base_names), assume per-angle scaling
        if num_params > len(base_names):
            n_physical = len(base_names)
            n_scaling = num_params - n_physical
            n_angles = n_scaling // 2  # contrast + offset per angle

            names = []
            names.extend([f"contrast_{i}" for i in range(n_angles)])
            names.extend([f"offset_{i}" for i in range(n_angles)])
            names.extend(base_names)
            return names

        return base_names[:num_params]

    def compute_credible_intervals(
        self, ci_level: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute credible intervals from posterior samples.

        Parameters
        ----------
        ci_level : float, default=0.95
            Credible interval level (e.g., 0.95 for 95% CI).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lower_bounds, upper_bounds) arrays for each parameter.

        Raises
        ------
        ValueError
            If samples_params is None.

        Examples
        --------
        >>> lower, upper = result.compute_credible_intervals(0.95)
        >>> print(f"D0: [{lower[0]:.2f}, {upper[0]:.2f}]")
        """
        if self.samples_params is None:
            raise ValueError("Cannot compute CI: samples_params is None")

        alpha = (1 - ci_level) / 2
        lower_pct = alpha * 100
        upper_pct = (1 - alpha) * 100

        # Handle multi-dimensional samples
        samples = self.samples_params
        if samples.ndim == 3:
            # (n_chains, n_samples, n_params) -> flatten chains
            samples = samples.reshape(-1, samples.shape[-1])

        lower = np.percentile(samples, lower_pct, axis=0)
        upper = np.percentile(samples, upper_pct, axis=0)

        return lower, upper

    def to_arviz(self) -> "az.InferenceData":
        """Convert MCMCResult to ArviZ InferenceData for diagnostics and plotting.

        Creates an InferenceData object compatible with all ArviZ functions:
        - az.plot_trace()
        - az.plot_posterior()
        - az.plot_pair()
        - az.summary()
        - az.rhat()
        - az.ess()

        Returns
        -------
        az.InferenceData
            ArviZ InferenceData object with posterior samples.

        Raises
        ------
        ImportError
            If arviz is not installed.
        ValueError
            If samples_params is None.

        Examples
        --------
        >>> idata = result.to_arviz()
        >>> az.plot_trace(idata)
        >>> az.summary(idata)

        >>> # With custom plots
        >>> az.plot_posterior(idata, var_names=["D0", "alpha"])
        >>> az.plot_pair(idata, var_names=["D0", "alpha", "D_offset"])
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ is required for to_arviz(). Install with: pip install arviz"
            ) from e

        if self.samples_params is None:
            raise ValueError("Cannot convert to ArviZ: samples_params is None")

        # Get parameter names
        param_names = self.get_param_names()

        # Prepare posterior samples
        samples = self.samples_params
        if samples.ndim == 2:
            # (n_samples, n_params) -> (1, n_samples, n_params) for single chain
            samples = samples[np.newaxis, :, :]
        elif samples.ndim == 3:
            # (n_chains, n_samples, n_params) - already correct
            pass
        else:
            raise ValueError(f"Unexpected samples shape: {samples.shape}")

        # Create posterior dict with named parameters
        posterior_dict = {}
        for i, name in enumerate(param_names):
            # Shape: (n_chains, n_samples)
            posterior_dict[name] = samples[:, :, i]

        # Create InferenceData
        idata = az.from_dict(
            posterior=posterior_dict,
            attrs={
                "analysis_mode": self.analysis_mode,
                "n_chains": self.n_chains,
                "n_warmup": self.n_warmup,
                "n_samples": self.n_samples,
                "sampler": self.sampler,
                "converged": self.converged,
                "computation_time": self.computation_time,
                "is_cmc": self.is_cmc_result(),
                "num_shards": self.num_shards,
            },
        )

        return idata

    def compute_summary(self) -> "pd.DataFrame":
        """Compute summary statistics using ArviZ.

        Returns a pandas DataFrame with diagnostics including:
        - mean, std: Posterior mean and standard deviation
        - hdi_2.5%, hdi_97.5%: 95% highest density interval
        - mcse_mean, mcse_sd: Monte Carlo standard error
        - ess_bulk, ess_tail: Effective sample size
        - r_hat: Gelman-Rubin convergence diagnostic

        Returns
        -------
        pd.DataFrame
            Summary statistics for all parameters.

        Raises
        ------
        ImportError
            If arviz is not installed.
        ValueError
            If samples_params is None.

        Examples
        --------
        >>> summary = result.compute_summary()
        >>> print(summary)
        >>> # Filter for physical parameters
        >>> print(summary.loc[["D0", "alpha", "D_offset"]])
        """
        try:
            import arviz as az
        except ImportError as e:
            raise ImportError(
                "ArviZ is required for compute_summary(). Install with: pip install arviz"
            ) from e

        idata = self.to_arviz()
        return az.summary(idata)
