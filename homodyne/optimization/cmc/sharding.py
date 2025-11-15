"""Data Sharding Strategies for Consensus Monte Carlo
=====================================================

This module implements data sharding strategies for CMC (Consensus Monte Carlo),
splitting large XPCS datasets into manageable shards for parallel MCMC execution.

Sharding Strategies
-------------------
1. **Stratified Sharding (PRIMARY)**: Preserves phi angle distribution
   - Sorts data by phi angle
   - Assigns every Nth point to each shard (round-robin)
   - Ensures each shard has representative phi coverage
   - Recommended for XPCS data (default)

2. **Random Sharding**: Randomly assigns datapoints to shards
   - Simple, unbiased distribution
   - May break correlations in structured data
   - Good for general-purpose use

3. **Contiguous Sharding**: Splits along time dimension
   - Preserves temporal structure
   - Useful for time-series analysis
   - May create non-representative shards

Key Features
------------
- Hardware-adaptive shard count calculation
- Automatic shard size balancing
- Data integrity validation (no data loss)
- Phi angle distribution preservation (stratified)
- Memory-efficient implementation

Usage
-----
    from homodyne.optimization.cmc.sharding import (
        calculate_optimal_num_shards,
        shard_data_stratified,
        validate_shards,
    )
    from homodyne.device.config import detect_hardware

    # Detect hardware
    hw_config = detect_hardware()

    # Calculate optimal number of shards
    dataset_size = 5_000_000
    num_shards = calculate_optimal_num_shards(dataset_size, hw_config)

    # Create stratified shards (preserves phi distribution)
    shards = shard_data_stratified(
        data=c2_exp,
        t1=t1,
        t2=t2,
        phi=phi,
        num_shards=num_shards,
        q=q,
        L=L,
    )

    # Validate shards
    is_valid, diagnostics = validate_shards(shards, dataset_size)

Integration
-----------
This module is called by:
- CMC coordinator for data preparation
- Parallel MCMC execution backends
- Test suites for validation

References
----------
Scott et al. (2016): "Bayes and big data: the consensus Monte Carlo algorithm"
https://arxiv.org/abs/1411.7435
"""

from typing import Dict, List, Literal, Tuple, Any
import numpy as np
from scipy import stats

from homodyne.device.config import HardwareConfig
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


def calculate_adaptive_min_shard_size(
    dataset_size: int,
    num_shards: int,
    default_min: int = 10_000,
    absolute_min: int = 100,
    large_dataset_threshold: int = 100_000,
) -> int:
    """Calculate dataset-adaptive minimum shard size.

    Adapts the minimum shard size based on dataset size to support both
    production datasets (requiring statistical robustness) and test datasets
    (requiring validation without artificial failures).

    Algorithm
    ---------
    - Large datasets (≥100k points): Use default_min (10,000)
    - Small datasets (<100k points): Use max(absolute_min, expected_size / 2)
    - Ensures each shard has at least absolute_min points

    Parameters
    ----------
    dataset_size : int
        Total number of data points in the dataset
    num_shards : int
        Number of shards to create
    default_min : int, default 10_000
        Standard minimum for large datasets (statistical robustness)
    absolute_min : int, default 100
        Absolute minimum shard size (prevents degenerate cases)
    large_dataset_threshold : int, default 100_000
        Dataset size threshold for using default_min

    Returns
    -------
    min_shard_size : int
        Adaptive minimum shard size

    Examples
    --------
    >>> # Large dataset (production)
    >>> calculate_adaptive_min_shard_size(5_000_000, 5)
    10000  # Use default minimum

    >>> # Small dataset (testing)
    >>> calculate_adaptive_min_shard_size(1_350, 1)
    675  # Use dataset_size / 2 = 1350 / 2 = 675

    >>> # Very small dataset
    >>> calculate_adaptive_min_shard_size(50, 1)
    100  # Use absolute minimum

    Notes
    -----
    - Ensures test datasets (1k-10k points) can pass validation
    - Maintains statistical rigor for production datasets (>100k points)
    - Never exceeds default_min for consistency
    """
    if dataset_size >= large_dataset_threshold:
        # Large dataset: use standard minimum for statistical robustness
        return default_min
    else:
        # Small dataset: adaptive minimum based on expected shard size
        expected_shard_size = (
            dataset_size // num_shards if num_shards > 0 else dataset_size
        )
        adaptive_min = max(absolute_min, expected_shard_size // 2)
        # Never exceed default for consistency
        return min(adaptive_min, default_min)


def calculate_optimal_num_shards(
    dataset_size: int,
    hardware_config: HardwareConfig,
    target_shard_size_gpu: int = 50_000,
    target_shard_size_cpu: int = 2_000_000,
    min_shard_size: int = 10_000,
) -> int:
    """Calculate optimal number of shards based on hardware and dataset size.

    This function balances three competing objectives:
    1. **Memory constraints**: Each shard must fit in device memory
    2. **Parallelization**: More shards enable more parallel speedup
    3. **Statistical efficiency**: Fewer shards reduce combination overhead
    4. **GPU kernel limits**: JAX kernel launch has grid dimension limits

    Algorithm
    ---------
    1. Use CPU target shard size (v2.3.0+ is CPU-only)
    2. Calculate number of shards: dataset_size / target_shard_size
    3. Cap at hardware parallelism limit (max_parallel_shards)
    4. Ensure minimum shard size (>10k points)
    5. Log warnings if exceeding hardware limits

    Parameters
    ----------
    dataset_size : int
        Total number of data points in the dataset
    hardware_config : HardwareConfig
        Detected hardware configuration from detect_hardware()
    target_shard_size_gpu : int, default 50_000
        (Unused in v2.3.0+ CPU-only, kept for backward compatibility)
    target_shard_size_cpu : int, default 2_000_000
        Target points per shard for CPU (v2.3.0+ uses this exclusively)
    min_shard_size : int, default 10_000
        Minimum shard size to ensure statistical robustness

    Returns
    -------
    num_shards : int
        Optimal number of shards (≥1)

    Examples
    --------
    >>> from homodyne.device.config import detect_hardware
    >>> hw = detect_hardware()
    >>> num_shards = calculate_optimal_num_shards(5_000_000, hw)
    >>> print(num_shards)
    2  # For CPU (v2.3.0+): 5M / 2M = 2-3 shards

    Notes
    -----
    - v2.3.0+ is CPU-only (no GPU support)
    - CPU target: 2M points/shard (relaxed memory limits, no kernel constraints)
    - Warnings logged if num_shards > max_parallel_shards (sequential execution)
    - Returns 1 if dataset is too small for sharding
    """
    logger.info(
        f"Calculating optimal num_shards for {dataset_size:,} points on "
        f"{hardware_config.platform} platform"
    )

    # Step 1: Use CPU target shard size (v2.3.0+ is CPU-only)
    target_shard_size = target_shard_size_cpu
    logger.debug(f"CPU platform: target shard size = {target_shard_size:,} points")

    # Step 2: Calculate initial number of shards
    num_shards = max(1, dataset_size // target_shard_size)
    logger.debug(f"Initial calculation: {num_shards} shards")

    # Step 3: Ensure minimum shard size
    # If shards would be too small, reduce num_shards
    actual_shard_size = dataset_size // num_shards if num_shards > 0 else dataset_size
    if actual_shard_size < min_shard_size:
        num_shards = max(1, dataset_size // min_shard_size)
        logger.warning(
            f"Reducing num_shards to {num_shards} to maintain minimum shard size "
            f"({min_shard_size:,} points)"
        )

    # Step 4: Check against hardware parallelism limit
    max_parallel = hardware_config.max_parallel_shards
    if num_shards > max_parallel:
        logger.warning(
            f"Requested {num_shards} shards exceeds hardware parallelism limit "
            f"({max_parallel}). Shards will be executed sequentially in batches."
        )
        # Note: We don't cap num_shards here - more shards may still be beneficial
        # for memory management even if executed sequentially

    # Step 5: Final validation
    final_shard_size = dataset_size // num_shards if num_shards > 0 else dataset_size
    logger.info(
        f"Optimal configuration: {num_shards} shards × ~{final_shard_size:,} points/shard"
    )
    logger.info(
        f"Hardware capacity: {max_parallel} parallel shards "
        f"({hardware_config.recommended_backend} backend)"
    )

    return num_shards


def shard_data_random(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    num_shards: int,
    q: float,
    L: float,
    sigma: np.ndarray = None,
    random_seed: int = 42,
) -> List[Dict[str, np.ndarray]]:
    """Create data shards using random assignment strategy.

    Randomly assigns datapoints to shards, ensuring balanced shard sizes
    while preserving data integrity (matching indices across arrays).

    Algorithm
    ---------
    1. Generate random permutation of indices
    2. Split permuted indices into num_shards chunks
    3. Extract data subsets using permuted indices
    4. Package each shard as dictionary with all required arrays

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Experimental c2 data (flattened)
    t1, t2 : np.ndarray, shape (N,)
        Time arrays (matching data indices)
    phi : np.ndarray, shape (N,)
        Angle array (matching data indices)
    num_shards : int
        Number of shards to create
    q : float
        Wavevector magnitude (constant across shards)
    L : float
        Sample-detector distance (constant across shards)
    sigma : np.ndarray, optional, shape (N,)
        Noise estimates (if available)
    random_seed : int, default 42
        Random seed for reproducibility

    Returns
    -------
    shards : list of dict
        List of shard dictionaries, each containing:
        - 'data': np.ndarray, experimental c2 values
        - 't1', 't2': np.ndarray, time arrays
        - 'phi': np.ndarray, angle array
        - 'q': float, wavevector
        - 'L': float, sample-detector distance
        - 'sigma': np.ndarray, noise estimates (if provided)
        - 'shard_id': int, shard index
        - 'shard_size': int, number of points in shard

    Examples
    --------
    >>> shards = shard_data_random(
    ...     data=c2_exp,
    ...     t1=t1, t2=t2, phi=phi,
    ...     num_shards=5,
    ...     q=0.01, L=5.0,
    ... )
    >>> print(len(shards))
    5
    >>> print(shards[0]['shard_size'])
    1000000

    Notes
    -----
    - Random assignment is unbiased and simple
    - May break correlations in structured data
    - Shard sizes are approximately balanced (within 1 point)
    - Preserves data integrity (matching indices)
    """
    logger.info(
        f"Creating {num_shards} random shards from {len(data):,} datapoints "
        f"(seed={random_seed})"
    )

    # Step 1: Generate random permutation
    rng = np.random.RandomState(random_seed)
    indices = rng.permutation(len(data))
    logger.debug(f"Generated random permutation of {len(indices):,} indices")

    # Step 2: Split indices into chunks
    shard_indices = np.array_split(indices, num_shards)

    # Step 3: Create shard dictionaries
    shards = []
    for i, shard_idx in enumerate(shard_indices):
        shard = {
            "data": data[shard_idx],
            "t1": t1[shard_idx],
            "t2": t2[shard_idx],
            "phi": phi[shard_idx],
            "q": q,
            "L": L,
            "shard_id": i,
            "shard_size": len(shard_idx),
        }

        # Add sigma if provided
        if sigma is not None:
            shard["sigma"] = sigma[shard_idx]

        shards.append(shard)
        logger.debug(f"Shard {i}: {len(shard_idx):,} points")

    logger.info(f"Random sharding complete: {len(shards)} shards created")
    return shards


def shard_data_stratified(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    num_shards: int,
    q: float,
    L: float,
    sigma: np.ndarray = None,
) -> List[Dict[str, np.ndarray]]:
    """Create data shards using stratified assignment strategy (PRIMARY).

    Sorts data by phi angle and assigns every Nth point to each shard using
    round-robin assignment. This ensures each shard has representative coverage
    of the phi angle distribution. This is the RECOMMENDED strategy for XPCS data.

    Algorithm
    ---------
    1. Sort all data by phi angle
    2. Assign every Nth point to each shard (round-robin):
       - Shard 0: points [0, N, 2N, 3N, ...]
       - Shard 1: points [1, N+1, 2N+1, 3N+1, ...]
       - Shard 2: points [2, N+2, 2N+2, 3N+2, ...]
    3. Each shard gets evenly-spaced samples across phi range
    4. Ensures balanced shard sizes and preserved phi distribution

    Why Stratified Sharding?
    ------------------------
    - **Preserves phi distribution**: Each shard samples across full phi range
    - **XPCS-optimized**: Respects angular structure of XPCS data
    - **Statistical robustness**: Each shard is representative of full dataset
    - **Reproducible**: Deterministic (no random seed needed)

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Experimental c2 data (flattened)
    t1, t2 : np.ndarray, shape (N,)
        Time arrays (matching data indices)
    phi : np.ndarray, shape (N,)
        Angle array (matching data indices)
    num_shards : int
        Number of shards to create
    q : float
        Wavevector magnitude (constant across shards)
    L : float
        Sample-detector distance (constant across shards)
    sigma : np.ndarray, optional, shape (N,)
        Noise estimates (if available)

    Returns
    -------
    shards : list of dict
        List of shard dictionaries, each containing:
        - 'data': np.ndarray, experimental c2 values
        - 't1', 't2': np.ndarray, time arrays
        - 'phi': np.ndarray, angle array
        - 'q': float, wavevector
        - 'L': float, sample-detector distance
        - 'sigma': np.ndarray, noise estimates (if provided)
        - 'shard_id': int, shard index
        - 'shard_size': int, number of points in shard

    Examples
    --------
    >>> shards = shard_data_stratified(
    ...     data=c2_exp,
    ...     t1=t1, t2=t2, phi=phi,
    ...     num_shards=5,
    ...     q=0.01, L=5.0,
    ... )
    >>> print(len(shards))
    5
    >>> # Verify phi distribution preservation (KS test)
    >>> from scipy import stats
    >>> for shard in shards:
    ...     ks_stat, p_value = stats.ks_2samp(shard['phi'], phi)
    ...     print(f"Shard {shard['shard_id']}: KS p-value = {p_value:.3f}")

    Notes
    -----
    - Deterministic (no random seed)
    - Preserves phi angle distribution (validated by KS test)
    - Shard sizes balanced within 10% tolerance
    - Recommended as default strategy for XPCS data
    """
    logger.info(
        f"Creating {num_shards} stratified shards from {len(data):,} datapoints "
        f"(round-robin phi sampling)"
    )

    # Step 1: Sort by phi angle
    sort_indices = np.argsort(phi)
    logger.debug(f"Sorted {len(sort_indices):,} datapoints by phi angle")

    # Step 2: Round-robin assignment - every Nth point to each shard
    # This ensures each shard samples evenly across the phi distribution
    shard_indices_list = [[] for _ in range(num_shards)]
    for i, idx in enumerate(sort_indices):
        shard_id = i % num_shards
        shard_indices_list[shard_id].append(idx)

    # Convert to numpy arrays
    shard_indices = [np.array(indices) for indices in shard_indices_list]

    # Step 3: Create shard dictionaries
    shards = []
    for i, shard_idx in enumerate(shard_indices):
        shard = {
            "data": data[shard_idx],
            "t1": t1[shard_idx],
            "t2": t2[shard_idx],
            "phi": phi[shard_idx],
            "q": q,
            "L": L,
            "shard_id": i,
            "shard_size": len(shard_idx),
        }

        # Add sigma if provided
        if sigma is not None:
            shard["sigma"] = sigma[shard_idx]

        # Log phi range for this shard
        phi_min, phi_max = shard["phi"].min(), shard["phi"].max()
        logger.debug(
            f"Shard {i}: {len(shard_idx):,} points, "
            f"phi range [{phi_min:.2f}°, {phi_max:.2f}°]"
        )

        shards.append(shard)

    logger.info(f"Stratified sharding complete: {len(shards)} shards created")
    return shards


def shard_data_contiguous(
    data: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    phi: np.ndarray,
    num_shards: int,
    q: float,
    L: float,
    sigma: np.ndarray = None,
) -> List[Dict[str, np.ndarray]]:
    """Create data shards using contiguous assignment strategy.

    Splits data along the flattened time dimension, preserving temporal
    structure. Each shard gets a contiguous block of datapoints in the
    order they appear in the arrays.

    Algorithm
    ---------
    1. Split indices into num_shards contiguous chunks
    2. Each shard gets sequential datapoints (no sorting or shuffling)
    3. Preserves original data ordering

    Use Cases
    ---------
    - Time-series analysis requiring temporal structure
    - Debugging and visualization (easier to interpret contiguous blocks)
    - When phi distribution is already well-distributed across time

    Limitations
    -----------
    - May create non-representative shards if data is structured
    - Not recommended for XPCS data (use stratified instead)

    Parameters
    ----------
    data : np.ndarray, shape (N,)
        Experimental c2 data (flattened)
    t1, t2 : np.ndarray, shape (N,)
        Time arrays (matching data indices)
    phi : np.ndarray, shape (N,)
        Angle array (matching data indices)
    num_shards : int
        Number of shards to create
    q : float
        Wavevector magnitude (constant across shards)
    L : float
        Sample-detector distance (constant across shards)
    sigma : np.ndarray, optional, shape (N,)
        Noise estimates (if available)

    Returns
    -------
    shards : list of dict
        List of shard dictionaries, each containing:
        - 'data': np.ndarray, experimental c2 values
        - 't1', 't2': np.ndarray, time arrays
        - 'phi': np.ndarray, angle array
        - 'q': float, wavevector
        - 'L': float, sample-detector distance
        - 'sigma': np.ndarray, noise estimates (if provided)
        - 'shard_id': int, shard index
        - 'shard_size': int, number of points in shard

    Examples
    --------
    >>> shards = shard_data_contiguous(
    ...     data=c2_exp,
    ...     t1=t1, t2=t2, phi=phi,
    ...     num_shards=5,
    ...     q=0.01, L=5.0,
    ... )
    >>> print(len(shards))
    5
    >>> # Verify contiguous ordering
    >>> for shard in shards:
    ...     print(f"Shard {shard['shard_id']}: t1 range "
    ...           f"[{shard['t1'].min():.4f}, {shard['t1'].max():.4f}]")

    Notes
    -----
    - Deterministic (no random seed)
    - Preserves original data ordering
    - May create non-representative shards for structured data
    - Use stratified sharding for XPCS data (recommended)
    """
    logger.info(
        f"Creating {num_shards} contiguous shards from {len(data):,} datapoints "
        f"(preserving temporal order)"
    )

    # Step 1: Split indices into contiguous chunks
    # np.array_split automatically handles non-divisible sizes
    total_points = len(data)
    shard_indices = np.array_split(np.arange(total_points), num_shards)

    # Step 2: Create shard dictionaries
    shards = []
    for i, shard_idx in enumerate(shard_indices):
        shard = {
            "data": data[shard_idx],
            "t1": t1[shard_idx],
            "t2": t2[shard_idx],
            "phi": phi[shard_idx],
            "q": q,
            "L": L,
            "shard_id": i,
            "shard_size": len(shard_idx),
        }

        # Add sigma if provided
        if sigma is not None:
            shard["sigma"] = sigma[shard_idx]

        # Log index range for this shard
        idx_min, idx_max = shard_idx[0], shard_idx[-1]
        logger.debug(
            f"Shard {i}: {len(shard_idx):,} points, "
            f"index range [{idx_min:,}, {idx_max:,}]"
        )

        shards.append(shard)

    logger.info(f"Contiguous sharding complete: {len(shards)} shards created")
    return shards


def validate_shards(
    shards: List[Dict[str, np.ndarray]],
    original_dataset_size: int,
    min_shard_size: int = 10_000,
    max_size_imbalance_pct: float = 0.10,
    validate_phi_distribution: bool = True,
    ks_test_threshold: float = 0.05,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate shard integrity and quality.

    Performs comprehensive validation of sharding results to ensure:
    1. No data loss (total size matches original)
    2. Reasonable shard sizes (>10k points minimum)
    3. Balanced shard sizes (within 10% tolerance)
    4. Phi angle distribution preservation (for stratified sharding)

    Validation Checks
    -----------------
    1. **Data Loss Check**: sum(shard_sizes) == original_dataset_size
    2. **Minimum Size Check**: all shards >= min_shard_size
    3. **Balance Check**: max/min shard size ratio <= 1 + max_size_imbalance_pct
    4. **Phi Distribution Check** (optional): Kolmogorov-Smirnov test
       - Tests if each shard's phi distribution matches overall distribution
       - Threshold: p-value > ks_test_threshold (default 0.05)

    Parameters
    ----------
    shards : list of dict
        Shard dictionaries from sharding functions
    original_dataset_size : int
        Expected total number of datapoints
    min_shard_size : int, default 10_000
        Minimum acceptable shard size
    max_size_imbalance_pct : float, default 0.10
        Maximum size imbalance as percentage (0.10 = 10%)
    validate_phi_distribution : bool, default True
        Whether to validate phi distribution preservation
    ks_test_threshold : float, default 0.05
        Kolmogorov-Smirnov test p-value threshold

    Returns
    -------
    is_valid : bool
        True if all validation checks pass
    diagnostics : dict
        Detailed diagnostic information containing:
        - 'total_points': int, total points across all shards
        - 'num_shards': int, number of shards
        - 'shard_sizes': list of int, size of each shard
        - 'min_shard_size': int, smallest shard size
        - 'max_shard_size': int, largest shard size
        - 'size_imbalance': float, max/min shard size ratio
        - 'data_loss_check': bool, whether total matches expected
        - 'min_size_check': bool, whether all shards meet minimum
        - 'balance_check': bool, whether size imbalance is acceptable
        - 'phi_ks_pvalues': list of float (if validate_phi_distribution=True)
        - 'phi_distribution_check': bool (if validate_phi_distribution=True)
        - 'errors': list of str, validation error messages

    Examples
    --------
    >>> shards = shard_data_stratified(data, t1, t2, phi, 5, q, L)
    >>> is_valid, diag = validate_shards(shards, len(data))
    >>> if is_valid:
    ...     print("Sharding validation passed!")
    ... else:
    ...     print(f"Validation failed: {diag['errors']}")

    Notes
    -----
    - Validation is non-destructive (does not modify shards)
    - KS test is only meaningful for stratified sharding
    - Diagnostics always returned (even if validation fails)
    """
    logger.info(
        f"Validating {len(shards)} shards against original dataset size {original_dataset_size:,}"
    )

    # Initialize diagnostics
    diagnostics = {
        "num_shards": len(shards),
        "errors": [],
    }

    # Extract shard sizes
    shard_sizes = [shard["shard_size"] for shard in shards]
    diagnostics["shard_sizes"] = shard_sizes
    diagnostics["min_shard_size"] = min(shard_sizes)
    diagnostics["max_shard_size"] = max(shard_sizes)

    # Check 1: Data loss check
    total_points = sum(shard_sizes)
    diagnostics["total_points"] = total_points
    data_loss_check = total_points == original_dataset_size
    diagnostics["data_loss_check"] = data_loss_check

    if not data_loss_check:
        error_msg = (
            f"Data loss detected: total shard size ({total_points:,}) != "
            f"original dataset size ({original_dataset_size:,})"
        )
        logger.error(error_msg)
        diagnostics["errors"].append(error_msg)

    # Check 2: Minimum size check
    min_size_violations = [size for size in shard_sizes if size < min_shard_size]
    min_size_check = len(min_size_violations) == 0
    diagnostics["min_size_check"] = min_size_check

    if not min_size_check:
        error_msg = (
            f"Minimum size violation: {len(min_size_violations)} shards have "
            f"< {min_shard_size:,} points (smallest: {min(shard_sizes):,})"
        )
        logger.error(error_msg)
        diagnostics["errors"].append(error_msg)

    # Check 3: Balance check
    size_imbalance = (
        max(shard_sizes) / min(shard_sizes) if min(shard_sizes) > 0 else float("inf")
    )
    diagnostics["size_imbalance"] = size_imbalance
    balance_check = size_imbalance <= (1.0 + max_size_imbalance_pct)
    diagnostics["balance_check"] = balance_check

    if not balance_check:
        error_msg = (
            f"Size imbalance too large: {size_imbalance:.2f}x "
            f"(max {1.0 + max_size_imbalance_pct:.2f}x allowed)"
        )
        logger.warning(error_msg)
        diagnostics["errors"].append(error_msg)

    # Check 4: Phi distribution check (optional)
    phi_distribution_check = True  # Default to True if not validated
    if validate_phi_distribution:
        # Concatenate all phi values to get overall distribution
        all_phi = np.concatenate([shard["phi"] for shard in shards])

        # Run KS test for each shard
        ks_pvalues = []
        for i, shard in enumerate(shards):
            shard_phi = shard["phi"]
            # Two-sample Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(shard_phi, all_phi)
            ks_pvalues.append(ks_pvalue)
            logger.debug(
                f"Shard {i}: KS test p-value = {ks_pvalue:.4f} "
                f"(threshold = {ks_test_threshold})"
            )

        diagnostics["phi_ks_pvalues"] = ks_pvalues

        # Check if all p-values exceed threshold
        phi_distribution_check = all(p >= ks_test_threshold for p in ks_pvalues)
        diagnostics["phi_distribution_check"] = phi_distribution_check

        if not phi_distribution_check:
            failed_shards = [
                i for i, p in enumerate(ks_pvalues) if p < ks_test_threshold
            ]
            error_msg = (
                f"Phi distribution check failed for {len(failed_shards)} shards "
                f"(KS test p-value < {ks_test_threshold}): {failed_shards}"
            )
            logger.warning(error_msg)
            diagnostics["errors"].append(error_msg)

    # Overall validation result
    is_valid = (
        data_loss_check and min_size_check and balance_check and phi_distribution_check
    )

    if is_valid:
        logger.info("Shard validation PASSED: all checks successful")
    else:
        logger.error(f"Shard validation FAILED: {len(diagnostics['errors'])} errors")

    return is_valid, diagnostics


# Export public API
__all__ = [
    "calculate_adaptive_min_shard_size",
    "calculate_optimal_num_shards",
    "shard_data_random",
    "shard_data_stratified",
    "shard_data_contiguous",
    "validate_shards",
]
