"""Extended MCMCResult class with CMC support.

This module extends the existing MCMCResult class from homodyne.optimization.mcmc
to support Consensus Monte Carlo (CMC) specific fields while maintaining 100%
backward compatibility with existing code.

Key Features:
- All CMC-specific fields are optional and default to None
- Existing code using MCMCResult continues to work unchanged
- New is_cmc_result() method to detect CMC results
- Serialization support for CMC-specific data
- Per-shard diagnostics and combination method tracking

Backward Compatibility:
- All existing MCMCResult parameters work exactly as before
- CMC fields only used when explicitly provided
- Non-CMC results load and save without CMC data
- No breaking changes to existing API
"""

from typing import Any, Dict, List, Optional

import numpy as np


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
        std_params: Optional[np.ndarray] = None,
        std_contrast: Optional[float] = None,
        std_offset: Optional[float] = None,
        samples_params: Optional[np.ndarray] = None,
        samples_contrast: Optional[np.ndarray] = None,
        samples_offset: Optional[np.ndarray] = None,
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
        acceptance_rate: Optional[float] = None,
        r_hat: Optional[Dict[str, float]] = None,
        effective_sample_size: Optional[Dict[str, float]] = None,
        # NEW: CMC-specific fields (optional, backward compatible)
        per_shard_diagnostics: Optional[List[Dict[str, Any]]] = None,
        cmc_diagnostics: Optional[Dict[str, Any]] = None,
        combination_method: Optional[str] = None,
        num_shards: Optional[int] = None,
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

    def is_cmc_result(self) -> bool:
        """Check if this result is from Consensus Monte Carlo.

        Returns
        -------
        bool
            True if result is from CMC (num_shards > 1), False otherwise

        Examples
        --------
        >>> # Standard MCMC result
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

        >>> # Edge case: num_shards=1 is still standard MCMC
        >>> result = MCMCResult(
        ...     mean_params=np.array([1.0]),
        ...     mean_contrast=0.5,
        ...     mean_offset=1.0,
        ...     num_shards=1
        ... )
        >>> result.is_cmc_result()
        False
        """
        return self.num_shards is not None and self.num_shards > 1

    def to_dict(self) -> Dict[str, Any]:
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
            "std_params": self.std_params.tolist() if self.std_params is not None else None,
            "std_contrast": self.std_contrast,
            "std_offset": self.std_offset,
            "samples_params": self.samples_params.tolist() if self.samples_params is not None else None,
            "samples_contrast": self.samples_contrast.tolist() if self.samples_contrast is not None else None,
            "samples_offset": self.samples_offset.tolist() if self.samples_offset is not None else None,
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
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCMCResult":
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
        std_params = np.array(data["std_params"]) if data.get("std_params") is not None else None
        samples_params = np.array(data["samples_params"]) if data.get("samples_params") is not None else None
        samples_contrast = np.array(data["samples_contrast"]) if data.get("samples_contrast") is not None else None
        samples_offset = np.array(data["samples_offset"]) if data.get("samples_offset") is not None else None

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
        )
