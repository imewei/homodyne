"""Hardware Detection and Configuration for Consensus Monte Carlo
================================================================

This module provides hardware-adaptive detection and decision-making for
CMC (Consensus Monte Carlo) optimization. It determines when to use CMC
based on available hardware resources and dataset characteristics.

Key Features
------------
- Automatic JAX device detection (GPU/CPU)
- GPU memory detection with graceful fallback
- Cluster environment detection (PBS/Slurm)
- Hardware-adaptive CMC threshold selection
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
    try:
        devices = jax.devices()
        platform = devices[0].platform if devices else "cpu"
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
    dataset_size: int,
    hardware_config: HardwareConfig,
    memory_threshold_pct: float = 0.8,
    min_points_for_cmc: int = 500_000,
) -> bool:
    """Determine if CMC should be used for a given dataset.

    This function implements hardware-adaptive decision logic to determine
    when Consensus Monte Carlo is beneficial compared to standard NUTS MCMC.

    Decision Logic
    --------------
    1. **Minimum Threshold**: Never use CMC below min_points_for_cmc
       - Default: 500k points (CMC overhead not worth it for smaller datasets)

    2. **Memory-Based Threshold**: Use CMC if estimated memory exceeds
       threshold percentage of available memory
       - Estimate: dataset_size * 8 bytes/float64 * 3 arrays * 2x MCMC overhead
       - Default threshold: 80% of available memory

    3. **Hardware-Specific Fallback Thresholds**:
       - GPU 16GB: 1M points
       - GPU 80GB: 10M points
       - CPU: 20M points

    Parameters
    ----------
    dataset_size : int
        Number of data points in the dataset
    hardware_config : HardwareConfig
        Detected hardware configuration from detect_hardware()
    memory_threshold_pct : float, default 0.8
        Memory usage threshold as percentage (0.8 = 80%)
    min_points_for_cmc : int, default 500_000
        Minimum dataset size to consider CMC

    Returns
    -------
    bool
        True if CMC should be used, False for standard NUTS

    Examples
    --------
    >>> hw = detect_hardware()
    >>> should_use_cmc(100_000, hw)
    False  # Below minimum threshold
    >>> should_use_cmc(5_000_000, hw)
    True   # Exceeds memory or hardware threshold

    Notes
    -----
    - Conservative defaults favor robustness over performance
    - Users can override thresholds via configuration
    - Memory estimation is approximate (actual usage varies)
    - Fallback thresholds ensure consistent behavior across hardware
    """
    # Step 1: Check minimum threshold
    if dataset_size < min_points_for_cmc:
        logger.debug(
            f"Dataset size {dataset_size:,} < min_points_for_cmc "
            f"({min_points_for_cmc:,}). Using standard NUTS."
        )
        return False

    # Step 2: Estimate memory usage
    # Assumptions:
    # - 8 bytes per float64
    # - 3 arrays: data, t1, t2
    # - 2x overhead for MCMC (samples, gradients, etc.)
    estimated_memory_gb = (dataset_size * 8 * 3 * 2) / 1e9

    # Check if estimated memory exceeds threshold
    available_memory = hardware_config.memory_per_device_gb
    memory_fraction = estimated_memory_gb / available_memory

    if memory_fraction > memory_threshold_pct:
        logger.info(
            f"Dataset requires ~{estimated_memory_gb:.2f} GB "
            f"({memory_fraction:.1%} of {available_memory:.2f} GB available). "
            f"Using CMC."
        )
        return True

    # Step 3: Hardware-specific fallback thresholds
    if hardware_config.platform == "gpu":
        if hardware_config.memory_per_device_gb <= 20:
            # 16GB GPU: Use CMC above 1M points
            threshold = 1_000_000
            if dataset_size > threshold:
                logger.info(
                    f"Dataset size {dataset_size:,} > {threshold:,} "
                    f"(16GB GPU threshold). Using CMC."
                )
                return True
        else:
            # 80GB GPU: Use CMC above 10M points
            threshold = 10_000_000
            if dataset_size > threshold:
                logger.info(
                    f"Dataset size {dataset_size:,} > {threshold:,} "
                    f"(80GB GPU threshold). Using CMC."
                )
                return True
    else:
        # CPU: Use CMC above 20M points
        threshold = 20_000_000
        if dataset_size > threshold:
            logger.info(
                f"Dataset size {dataset_size:,} > {threshold:,} "
                f"(CPU threshold). Using CMC."
            )
            return True

    # Default: Use standard NUTS
    logger.debug(
        f"Dataset size {dataset_size:,} within standard NUTS capacity. "
        f"Memory fraction: {memory_fraction:.1%}"
    )
    return False


# Export public API
__all__ = [
    "HardwareConfig",
    "detect_hardware",
    "should_use_cmc",
]
