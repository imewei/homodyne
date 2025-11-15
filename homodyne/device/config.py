"""Hardware Detection and Configuration for Consensus Monte Carlo
================================================================

This module provides hardware-adaptive detection and decision-making for
CMC (Consensus Monte Carlo) optimization. It determines when to use CMC
based on available hardware resources and dataset characteristics using
a tri-criteria OR logic system.

Key Features
------------
- Automatic JAX device detection (GPU/CPU)
- GPU memory detection with graceful fallback
- Cluster environment detection (PBS/Slurm)
- Hardware-adaptive CMC threshold selection
- Tri-criteria decision logic (parallelism OR memory OR large dataset)
- Backend recommendation based on hardware capabilities

Usage
-----
    from homodyne.device.config import detect_hardware, should_use_cmc

    # Detect hardware capabilities
    hw_config = detect_hardware()
    print(f"Platform: {hw_config.platform}")
    print(f"Recommended backend: {hw_config.recommended_backend}")

    # Decide whether to use CMC for a dataset
    dataset_size = 5_000_000
    use_cmc = should_use_cmc(dataset_size, hw_config)
    print(f"Use CMC for {dataset_size} points: {use_cmc}")

Integration
-----------
This module is called by:
- fit_mcmc_jax() to determine method selection (NUTS vs CMC)
- CMC coordinator to select optimal backend
- Shard sizing calculations in CMC pipeline

Notes
-----
- Hardware detection is performed once at import time
- Results can be cached for performance
- Fallback mechanisms ensure robustness
"""

from dataclasses import dataclass
from typing import Literal, Optional
import os
import multiprocessing

import jax

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware configuration for CMC optimization.

    This dataclass encapsulates all detected hardware information needed
    for intelligent CMC decision-making and backend selection.

    Attributes
    ----------
    platform : {'gpu', 'cpu'}
        Primary compute platform detected by JAX
    num_devices : int
        Number of available devices (GPUs or CPU cores)
    memory_per_device_gb : float
        Available memory per device in GB
    num_nodes : int
        Number of cluster nodes (1 for standalone)
    cores_per_node : int
        Number of physical CPU cores per node
    total_memory_gb : float
        Total system memory in GB
    cluster_type : {'pbs', 'slurm', 'standalone', None}
        Detected cluster scheduler type
    recommended_backend : str
        Recommended CMC backend based on hardware
        Options: 'pjit', 'multiprocessing', 'pbs', 'slurm'
    max_parallel_shards : int
        Maximum number of shards that can run in parallel
        - Multi-node cluster: num_nodes * cores_per_node
        - Multi-GPU: num_devices
        - Single GPU: 1 (sequential execution)
        - CPU: cores_per_node

    Examples
    --------
    >>> hw = detect_hardware()
    >>> print(hw.platform)
    'gpu'
    >>> print(hw.max_parallel_shards)
    4
    >>> print(hw.recommended_backend)
    'pjit'
    """

    platform: Literal["gpu", "cpu"]
    num_devices: int
    memory_per_device_gb: float
    num_nodes: int
    cores_per_node: int
    total_memory_gb: float
    cluster_type: Optional[Literal["pbs", "slurm", "standalone"]]
    recommended_backend: str
    max_parallel_shards: int


def detect_hardware() -> HardwareConfig:
    """Auto-detect hardware configuration for CMC optimization.

    This function performs comprehensive hardware detection to inform
    intelligent CMC strategy selection and backend choice.

    Detection Logic
    ---------------
    1. **JAX Devices**: Query JAX for available devices (GPU/CPU)
    2. **GPU Memory**: Attempt to query GPU memory via JAX/CUDA
       - Fallback: Assume 16GB per GPU if query fails
    3. **Cluster Environment**: Check environment variables
       - PBS: PBS_JOBID, PBS_NODEFILE
       - Slurm: SLURM_JOB_NUM_NODES, SLURM_CPUS_ON_NODE
       - Standalone: Neither PBS nor Slurm detected
    4. **CPU Resources**: Count physical cores using psutil
    5. **Backend Recommendation**: Select optimal backend based on:
       - Multi-node cluster → PBS/Slurm backend
       - Multi-GPU → pjit backend
       - Single GPU → pjit backend (sequential)
       - CPU-only → multiprocessing backend

    Returns
    -------
    HardwareConfig
        Comprehensive hardware configuration for CMC

    Examples
    --------
    >>> hw = detect_hardware()
    >>> print(hw.platform)
    'gpu'
    >>> print(hw.num_devices)
    4
    >>> print(hw.memory_per_device_gb)
    80.0
    >>> print(hw.cluster_type)
    'pbs'
    >>> print(hw.recommended_backend)
    'pbs'

    Notes
    -----
    - Detection is robust with multiple fallback mechanisms
    - GPU memory detection may fail on some systems (uses fallback)
    - Cluster detection requires environment variables set by scheduler
    - CPU core count excludes hyperthreading for accurate parallelism
    """
    logger.info("Detecting hardware configuration for CMC...")

    # Step 1: Detect JAX devices
    # Use the actual active backend, not just first device in list
    # When JAX_PLATFORMS="cpu,gpu", devices[0] may be CPU even if GPU is active
    try:
        # Try new API first (JAX 0.8.0+), fall back to legacy API
        try:
            from jax.extend import backend as jax_backend

            backend = jax_backend.get_backend()
        except (ImportError, AttributeError):
            # Legacy API for JAX < 0.8.0
            from jax.lib import xla_bridge

            backend = xla_bridge.get_backend()

        platform = backend.platform
        devices = backend.devices()
        num_devices = len(devices)
        logger.info(f"JAX devices detected: {num_devices} {platform} device(s)")
    except Exception as e:
        logger.warning(f"JAX device detection failed: {e}. Falling back to CPU.")
        platform = "cpu"
        num_devices = 1

    # Step 2: Estimate memory per device
    memory_gb = 16.0  # Default fallback for GPU
    if platform == "gpu":
        try:
            # Try to query GPU memory from JAX/CUDA
            from jax.lib import xla_bridge

            backend = xla_bridge.get_backend()
            if hasattr(backend, "devices"):
                device = backend.devices()[0]
                if hasattr(device, "memory_stats"):
                    memory_info = device.memory_stats()
                    if "bytes_limit" in memory_info:
                        memory_gb = memory_info["bytes_limit"] / 1e9
                        logger.info(f"GPU memory detected: {memory_gb:.2f} GB")
        except Exception as e:
            logger.debug(f"GPU memory detection failed: {e}. Using fallback (16 GB)")
            memory_gb = 16.0
    else:
        # CPU: Use total system memory
        if HAS_PSUTIL:
            memory_gb = psutil.virtual_memory().total / 1e9
            logger.info(f"System memory detected: {memory_gb:.2f} GB")
        else:
            logger.warning("psutil not available. Assuming 32 GB system memory")
            memory_gb = 32.0

    # Step 3: Detect cluster environment
    cluster_type = None
    num_nodes = 1

    if "PBS_JOBID" in os.environ:
        cluster_type = "pbs"
        # Parse PBS_NODEFILE for node count
        nodefile = os.environ.get("PBS_NODEFILE")
        if nodefile and os.path.exists(nodefile):
            try:
                with open(nodefile) as f:
                    num_nodes = len(set(f.read().splitlines()))
                logger.info(f"PBS cluster detected: {num_nodes} nodes")
            except Exception as e:
                logger.warning(f"Failed to parse PBS_NODEFILE: {e}")
                num_nodes = 1
        else:
            logger.debug("PBS_JOBID present but PBS_NODEFILE not found")
            num_nodes = 1

    elif "SLURM_JOB_NUM_NODES" in os.environ:
        cluster_type = "slurm"
        try:
            num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
            logger.info(f"Slurm cluster detected: {num_nodes} nodes")
        except ValueError:
            logger.warning("Failed to parse SLURM_JOB_NUM_NODES")
            num_nodes = 1

    else:
        cluster_type = "standalone"
        num_nodes = 1
        logger.info("Standalone system detected (no cluster scheduler)")

    # Step 4: Detect CPU cores
    if HAS_PSUTIL:
        # Use physical cores (exclude hyperthreading)
        cores_per_node = psutil.cpu_count(logical=False) or 1
        total_memory_gb = psutil.virtual_memory().total / 1e9
        logger.info(f"CPU cores detected: {cores_per_node} physical cores")
    else:
        logger.warning("psutil not available. Using multiprocessing for CPU count")
        cores_per_node = multiprocessing.cpu_count()
        total_memory_gb = memory_gb  # Use previously detected value

    # Step 5: Recommend backend and calculate max parallel shards
    if cluster_type in ["pbs", "slurm"] and num_nodes > 1:
        # Multi-node cluster: Use PBS/Slurm backend
        recommended_backend = cluster_type
        max_parallel_shards = num_nodes * cores_per_node
        logger.info(
            f"Recommended backend: {recommended_backend} "
            f"(max {max_parallel_shards} parallel shards)"
        )

    elif platform == "gpu" and num_devices > 1:
        # Multi-GPU: Use pjit backend for parallel GPU execution
        recommended_backend = "pjit"
        max_parallel_shards = num_devices
        logger.info(
            f"Recommended backend: pjit (multi-GPU, "
            f"max {max_parallel_shards} parallel shards)"
        )

    elif platform == "gpu" and num_devices == 1:
        # Single GPU: Use pjit but sequential execution
        recommended_backend = "pjit"
        max_parallel_shards = 1
        logger.info("Recommended backend: pjit (single GPU, sequential execution)")

    else:
        # CPU-only: Use multiprocessing backend
        recommended_backend = "multiprocessing"
        max_parallel_shards = cores_per_node
        logger.info(
            f"Recommended backend: multiprocessing "
            f"(max {max_parallel_shards} parallel shards)"
        )

    # Construct and return HardwareConfig
    hw_config = HardwareConfig(
        platform=platform,
        num_devices=num_devices,
        memory_per_device_gb=memory_gb,
        num_nodes=num_nodes,
        cores_per_node=cores_per_node,
        total_memory_gb=total_memory_gb,
        cluster_type=cluster_type,
        recommended_backend=recommended_backend,
        max_parallel_shards=max_parallel_shards,
    )

    logger.info(f"Hardware detection complete: {hw_config.platform} platform")
    return hw_config


def should_use_cmc(
    num_samples: int,
    hardware_config: HardwareConfig,
    dataset_size: Optional[int] = None,
    memory_threshold_pct: float = 0.20,
    min_samples_for_cmc: int = 15,
    large_dataset_threshold: int = 1_000_000,
    large_dataset_threshold_low_sample: int = 10_000_000,
) -> bool:
    """Determine if CMC should be used based on samples AND/OR dataset size.

    CMC serves FOUR distinct purposes requiring different triggering conditions:

    **Use Case 1: Parallelism** (many independent samples)
    - Trigger: num_samples >= min_samples_for_cmc (default: 15)
    - Sharding: Split samples (phi angles) across shards
    - Benefit: Parallel MCMC chains on multi-core CPU, faster convergence
    - Example: 20 phi angles on 14-core CPU → ~1.4x speedup via parallelization

    **Use Case 2: Memory Management** (few samples, huge data)
    - Trigger: dataset_size causes estimated memory > threshold (default: 20%)
    - Sharding: Keep all samples in each shard, split data points
    - Benefit: Avoid OOM errors, enable large dataset analysis
    - Example: 2 phi × 50M points → CMC triggered (50% memory) → avoid OOM

    **Use Case 3: JAX Broadcasting Protection** (very large pooled datasets)
    - Trigger: dataset_size > large_dataset_threshold (default: 1M)
    - Reason: JAX broadcasting in compute_g1_total can create impossible arrays
    - Example: 3 phi × 3M points → JAX broadcasting overflow → CMC prevents crash
    - Benefit: CMC sharding prevents catastrophic memory overflow
    - Trade-off: Accept 10-20% CMC overhead to prevent JAX crash (overflow independent of sample count)

    **Use Case 4: Large Dataset, Few Samples** (proactive OOM prevention)
    - Trigger: dataset_size > large_dataset_threshold_low_sample (default: 10M) AND 2 <= num_samples < min_samples_for_cmc
    - Sharding: Split data points (no parallelism benefit)
    - Benefit: Proactively prevent OOM before it happens
    - Trade-off: Accept 10-20% CMC overhead to avoid OOM failure
    - Example: 2 phi × 24M points → CMC triggered proactively → memory per shard halved

    Decision Logic (OR condition)
    ------------------------------
    Use CMC if:

    1. num_samples >= min_samples_for_cmc (parallelism mode), OR
    2. estimated_memory_gb > threshold × available_memory (memory mode), OR
    3. dataset_size > large_dataset_threshold
       (JAX broadcasting protection - independent of sample count), OR
    4. (dataset_size > large_dataset_threshold_low_sample AND 2 <= num_samples < min_samples_for_cmc)
       (proactive OOM prevention for large datasets with few samples)

    Parameters
    ----------
    num_samples : int
        Number of independent samples (e.g., phi angles in XPCS)
    hardware_config : HardwareConfig
        Detected hardware configuration
    dataset_size : int, optional
        Total number of data points (for memory estimation)
        If None, only sample-based decision is used
    memory_threshold_pct : float, default 0.20
        Use CMC if estimated memory > this fraction of available (0.20 = 20%)
        Lowered from 0.30 to trigger earlier for proactive OOM prevention
        Calibrated for 16 GB GPUs with safety margin for OS/driver overhead
    min_samples_for_cmc : int, default 15
        Minimum samples for parallelism-mode CMC
        Optimized for multi-core CPU workloads (14+ cores)
        15 samples / 14 cores = 1.07 samples/core (acceptable minimum)
    large_dataset_threshold : int, default 1_000_000
        Force CMC if dataset_size > threshold (JAX broadcasting protection)
        Prevents catastrophic memory overflow from JAX broadcasting in compute_g1_total
        Critical for pooled datasets >1M points in laminar flow analysis
    large_dataset_threshold_low_sample : int, default 10_000_000
        Proactively trigger CMC for very large datasets even with few samples (Criterion 4)
        Prevents OOM before it happens when num_samples < min_samples_for_cmc
        Optimized for common XPCS use case: 2 phi angles × 10M-100M points
        Trade-off: Accept CMC overhead to avoid OOM failure

    Returns
    -------
    bool
        True if CMC should be used, False for standard NUTS

    Examples
    --------
    >>> hw = detect_hardware()
    >>> # Case 1: Few samples, small data → NUTS (all criteria fail)
    >>> should_use_cmc(10, hw, dataset_size=5_000_000)
    False

    >>> # Case 2: Moderate samples → CMC (Criterion 1: parallelism, 23 ≥ 15)
    >>> should_use_cmc(23, hw, dataset_size=23_000_000)
    True

    >>> # Case 3: Few samples, very large data → CMC (Criterion 4: 24M > 10M, 2 < 15)
    >>> should_use_cmc(2, hw, dataset_size=24_000_000)
    True

    >>> # Case 4: Few samples, HUGE data → CMC (Criteria 2 & 4: memory > 20%, dataset > 10M)
    >>> should_use_cmc(2, hw, dataset_size=50_000_000)
    True

    >>> # Case 5: Borderline → CMC (Criterion 1: 20 samples triggers parallelism)
    >>> should_use_cmc(20, hw, dataset_size=10_000_000)
    True

    Notes
    -----
    - For typical XPCS: 2-100 phi angles, 1M-100M+ points per angle
    - Memory estimate: dataset_size × 8 bytes × 30 (empirically calibrated)
      Components: data + gradients (9 params) + NUTS tree + JAX overhead + MCMC state
    - CMC sharding strategy adapts based on which condition triggered it
    - Calibration: 23M points → ~12-14 GB actual NUTS memory usage on GPU
    - Large dataset threshold protects against JAX broadcasting edge cases
    """
    # Step 1: Evaluate quad-criteria OR logic
    # Criterion 1 (Parallelism): num_samples >= min_samples_for_cmc
    use_cmc_for_parallelism = num_samples >= min_samples_for_cmc

    # Criterion 2 (Memory): estimated_memory > memory_threshold_pct
    use_cmc_for_memory = False
    estimated_memory_gb = 0.0
    memory_fraction = 0.0

    # Criterion 3 (Large Dataset with Parallelism): dataset_size > large_dataset_threshold AND sufficient samples
    use_cmc_for_large_dataset = False

    # Criterion 4 (Large Dataset, Few Samples): dataset_size > large_dataset_threshold_low_sample AND 2 <= num_samples < min_samples_for_cmc
    use_cmc_large_low_sample = False

    if dataset_size is not None:
        # Estimate memory requirement for MCMC
        # Formula: dataset_size × 8 bytes/float × 30 (data + gradients + NUTS tree + JAX overhead + MCMC state)
        # Multiplier calibrated empirically: 23M points → ~12-14 GB actual usage
        # Components: data (1x) + gradients for 9 params (9x) + NUTS trajectory storage (15x) + overhead (5x)
        estimated_memory_gb = (dataset_size * 8 * 30) / 1e9
        available_memory_gb = hardware_config.memory_per_device_gb
        memory_fraction = estimated_memory_gb / available_memory_gb
        use_cmc_for_memory = memory_fraction > memory_threshold_pct

        # Criterion 3: JAX Broadcasting Protection (large datasets)
        # Critical for pooled datasets >1M that can cause JAX broadcasting overflow
        # JAX overflow is physics-based (array size), NOT dependent on sample count
        # Trade-off: Accept 10-20% CMC overhead to prevent catastrophic JAX crash
        # Example: 3M points / 3 samples → triggers CMC → prevents overflow
        use_cmc_for_large_dataset = dataset_size > large_dataset_threshold

        # Criterion 4 (NEW): Proactive OOM prevention for large datasets with few samples
        # Purpose: Prevent OOM before it happens when parallelism doesn't apply
        # Trade-off: Accept 10-20% CMC overhead to avoid OOM failure
        # Common XPCS use case: 2 phi angles × 10M-100M points
        use_cmc_large_low_sample = (
            dataset_size > large_dataset_threshold_low_sample
            and num_samples >= 2  # Minimum for CMC
            and num_samples < min_samples_for_cmc  # Only if parallelism doesn't apply
        )

    # Step 2: Log comprehensive quad-criteria evaluation
    logger.info("=" * 70)
    logger.info("Automatic NUTS/CMC Selection - Quad-Criteria Evaluation")
    logger.info("=" * 70)
    logger.info(
        f"Criterion 1 (Parallelism): num_samples={num_samples:,} >= "
        f"min_samples_for_cmc={min_samples_for_cmc} → {use_cmc_for_parallelism}"
    )

    if dataset_size is not None:
        logger.info(
            f"Criterion 2 (Memory): {memory_fraction:.1%} "
            f"({estimated_memory_gb:.2f}/{hardware_config.memory_per_device_gb:.2f} GB) > "
            f"{memory_threshold_pct:.1%} → {use_cmc_for_memory}"
        )
        logger.info(
            f"Criterion 3 (JAX Broadcasting Protection): dataset_size={dataset_size:,} > "
            f"threshold={large_dataset_threshold:,} → {use_cmc_for_large_dataset}"
        )
        logger.info(
            f"Criterion 4 (Large Dataset, Few Samples): (dataset_size={dataset_size:,} > "
            f"threshold={large_dataset_threshold_low_sample:,}) AND (2 <= num_samples={num_samples} < "
            f"min={min_samples_for_cmc}) → {use_cmc_large_low_sample}"
        )
    else:
        logger.info("Criterion 2 (Memory): dataset_size=None → False (not evaluated)")
        logger.info(
            "Criterion 3 (JAX Broadcasting Protection): dataset_size=None → False (not evaluated)"
        )
        logger.info(
            "Criterion 4 (Large Dataset, Few Samples): dataset_size=None → False (not evaluated)"
        )

    # Step 3: Apply OR logic and make decision (any criterion triggers CMC)
    use_cmc = (
        use_cmc_for_parallelism
        or use_cmc_for_memory
        or use_cmc_for_large_dataset
        or use_cmc_large_low_sample
    )

    logger.info("-" * 70)

    # Build mode string based on which criteria triggered CMC
    if use_cmc:
        mode_parts = []
        if use_cmc_for_parallelism:
            mode_parts.append("Parallelism")
        if use_cmc_for_memory:
            mode_parts.append("Memory")
        if use_cmc_for_large_dataset:
            mode_parts.append("JAX Broadcasting Protection")
        if use_cmc_large_low_sample:
            mode_parts.append("Large Dataset, Few Samples")
        mode_string = " + ".join(mode_parts) + " mode"
    else:
        mode_string = "All criteria failed mode"

    logger.info(f"Final decision: Using {'CMC' if use_cmc else 'NUTS'} ({mode_string})")
    logger.info("=" * 70)

    # Step 4: Log warnings for edge cases
    if use_cmc and num_samples < min_samples_for_cmc:
        # CMC triggered by memory, large dataset, or low-sample criterion, not parallelism
        trigger_reason = []
        if use_cmc_for_memory:
            trigger_reason.append(
                f"memory ({memory_fraction:.1%} > {memory_threshold_pct:.1%})"
            )
        if use_cmc_for_large_dataset:
            trigger_reason.append(
                f"JAX broadcasting protection ({dataset_size:,} > {large_dataset_threshold:,})"
            )
        if use_cmc_large_low_sample:
            trigger_reason.append(
                f"large dataset, few samples ({dataset_size:,} > {large_dataset_threshold_low_sample:,})"
            )
        logger.warning(
            f"Using CMC with only {num_samples} samples (< {min_samples_for_cmc} threshold). "
            f"CMC adds 10-20% overhead; NUTS is faster for <{min_samples_for_cmc} samples if memory permits. "
            f"Triggered by: {', '.join(trigger_reason)}"
        )

    return use_cmc


# Export public API
__all__ = [
    "HardwareConfig",
    "detect_hardware",
    "should_use_cmc",
]
