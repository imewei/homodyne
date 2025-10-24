"""Backend Selection Logic for Consensus Monte Carlo
====================================================

This module implements intelligent backend selection based on detected hardware
configuration and user preferences. It provides automatic backend selection with
sensible defaults and supports manual override for advanced users.

Selection Strategy
------------------
Auto-selection follows this priority order:

1. **Multi-node HPC cluster** (PBS/Slurm detected, num_nodes > 1):
   → PBS/Slurm backend (virtually unlimited parallelism)

2. **Multi-GPU system** (num_gpus > 1):
   → pjit backend (parallel GPU execution)

3. **Single GPU** (num_gpus == 1):
   → pjit backend (sequential execution on single GPU)

4. **CPU-only** (no GPUs detected):
   → multiprocessing backend (parallel CPU execution)

User Override
-------------
Users can override auto-selection via configuration:

    cmc_config = {
        'backend': {
            'type': 'pjit',  # Force specific backend
        }
    }

This is useful for:
- Testing backend performance
- Debugging backend-specific issues
- Forcing specific execution strategy

Backend Registry
----------------
Available backends are registered in _BACKEND_REGISTRY. Backends are lazy-loaded
to avoid import errors if dependencies are missing (e.g., PBS not available on
standalone systems).

Usage Examples
--------------
    from homodyne.device.config import detect_hardware
    from homodyne.optimization.cmc.backends import select_backend

    # Auto-select based on hardware
    hw = detect_hardware()
    backend = select_backend(hw)
    print(f"Selected: {backend.get_backend_name()}")

    # Manual override
    backend = select_backend(hw, user_override='multiprocessing')

    # Get backend by name (factory function)
    backend = get_backend_by_name('pjit')

Integration Points
------------------
- Called by CMC coordinator during initialization
- Uses HardwareConfig from homodyne.device.config
- Instantiates backend classes from backends/ submodules

Error Handling
--------------
- Invalid backend names raise clear ValueError
- Missing backend implementations raise ImportError with helpful message
- Backend compatibility is validated before instantiation
"""

from typing import Optional

from homodyne.device.config import HardwareConfig
from homodyne.optimization.cmc.backends.base import CMCBackend
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


# Backend registry - lazy loaded to avoid import errors
_BACKEND_REGISTRY = {
    "pjit": "homodyne.optimization.cmc.backends.pjit.PjitBackend",
    "multiprocessing": "homodyne.optimization.cmc.backends.multiprocessing.MultiprocessingBackend",
    "pbs": "homodyne.optimization.cmc.backends.pbs.PBSBackend",
}


def select_backend(
    hardware_config: HardwareConfig,
    user_override: Optional[str] = None,
) -> CMCBackend:
    """Select optimal CMC backend based on hardware configuration.

    This function implements the auto-selection logic for choosing the best
    execution backend based on detected hardware capabilities. Users can
    override the automatic selection if needed.

    Auto-Selection Logic
    --------------------
    1. Multi-node cluster (PBS/Slurm, num_nodes > 1) → PBS backend
    2. Multi-GPU system (num_gpus > 1) → pjit backend
    3. Single GPU (num_gpus == 1) → pjit backend (sequential)
    4. CPU-only → multiprocessing backend

    Parameters
    ----------
    hardware_config : HardwareConfig
        Detected hardware configuration from detect_hardware()
    user_override : str, optional
        Force specific backend by name (e.g., 'pjit', 'multiprocessing', 'pbs')
        Overrides automatic selection

    Returns
    -------
    CMCBackend
        Instantiated backend ready for parallel MCMC execution

    Raises
    ------
    ValueError
        If user_override specifies an invalid backend name
    ImportError
        If the selected backend is not implemented yet

    Examples
    --------
    >>> from homodyne.device.config import detect_hardware
    >>> hw = detect_hardware()
    >>> backend = select_backend(hw)
    >>> print(backend.get_backend_name())
    'pjit'

    >>> # Force multiprocessing backend
    >>> backend = select_backend(hw, user_override='multiprocessing')
    >>> print(backend.get_backend_name())
    'multiprocessing'

    Notes
    -----
    - Logs selected backend and reasoning for transparency
    - Validates backend compatibility with hardware
    - Backends are lazy-loaded to avoid unnecessary imports
    - Selection can be overridden for testing and debugging
    """
    # Handle user override
    if user_override:
        logger.info(f"User override: forcing '{user_override}' backend")
        backend = get_backend_by_name(user_override)
        _validate_backend_compatibility(backend, hardware_config)
        return backend

    # Auto-selection based on hardware
    logger.info("Auto-selecting CMC backend based on hardware configuration...")

    # Priority 1: Multi-node HPC cluster
    if hardware_config.cluster_type in ["pbs", "slurm"] and hardware_config.num_nodes > 1:
        backend_name = hardware_config.cluster_type
        logger.info(
            f"Multi-node {hardware_config.cluster_type.upper()} cluster detected "
            f"({hardware_config.num_nodes} nodes). Selecting '{backend_name}' backend."
        )
        backend = get_backend_by_name(backend_name)
        return backend

    # Priority 2: Multi-GPU system
    if hardware_config.platform == "gpu" and hardware_config.num_devices > 1:
        backend_name = "pjit"
        logger.info(
            f"Multi-GPU system detected ({hardware_config.num_devices} GPUs). "
            f"Selecting '{backend_name}' backend for parallel GPU execution."
        )
        backend = get_backend_by_name(backend_name)
        return backend

    # Priority 3: Single GPU
    if hardware_config.platform == "gpu" and hardware_config.num_devices == 1:
        backend_name = "pjit"
        logger.info(
            f"Single GPU detected. Selecting '{backend_name}' backend "
            f"(sequential execution)."
        )
        backend = get_backend_by_name(backend_name)
        return backend

    # Priority 4: CPU-only
    backend_name = "multiprocessing"
    logger.info(
        f"CPU-only system detected ({hardware_config.cores_per_node} cores). "
        f"Selecting '{backend_name}' backend."
    )
    backend = get_backend_by_name(backend_name)
    return backend


def get_backend_by_name(backend_name: str) -> CMCBackend:
    """Factory function to instantiate a backend by name.

    This function performs lazy loading of backend classes to avoid import
    errors if backend dependencies are not available (e.g., PBS not installed
    on standalone systems).

    Parameters
    ----------
    backend_name : str
        Name of the backend to instantiate
        Valid options: 'pjit', 'multiprocessing', 'pbs'

    Returns
    -------
    CMCBackend
        Instantiated backend instance

    Raises
    ------
    ValueError
        If backend_name is not recognized
    ImportError
        If the backend module is not implemented yet or dependencies are missing

    Examples
    --------
    >>> backend = get_backend_by_name('pjit')
    >>> print(backend.get_backend_name())
    'pjit'

    >>> backend = get_backend_by_name('invalid')
    ValueError: Unknown backend: 'invalid'. Available: ['pjit', 'multiprocessing', 'pbs']

    Notes
    -----
    - Backends are lazy-loaded on first access
    - Import errors provide helpful messages about missing implementations
    - Backend instances are stateless (can create multiple instances)
    """
    # Validate backend name
    if backend_name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown backend: '{backend_name}'. "
            f"Available backends: {available}"
        )

    # Lazy load backend class
    backend_path = _BACKEND_REGISTRY[backend_name]
    module_path, class_name = backend_path.rsplit(".", 1)

    try:
        # Import module
        import importlib
        module = importlib.import_module(module_path)

        # Get class
        backend_class = getattr(module, class_name)

        # Instantiate and return
        backend = backend_class()
        logger.debug(f"Instantiated {backend_name} backend: {backend_class.__name__}")
        return backend

    except ImportError as e:
        raise ImportError(
            f"Backend '{backend_name}' is not implemented yet. "
            f"This backend will be added in Task Group 5. "
            f"Original error: {str(e)}"
        )
    except AttributeError as e:
        raise ImportError(
            f"Backend class '{class_name}' not found in module '{module_path}'. "
            f"Original error: {str(e)}"
        )


def _validate_backend_compatibility(
    backend: CMCBackend,
    hardware_config: HardwareConfig,
) -> None:
    """Validate that a backend is compatible with detected hardware.

    This function performs sanity checks to warn users if they're using a
    backend that may not be optimal for their hardware.

    Parameters
    ----------
    backend : CMCBackend
        Backend instance to validate
    hardware_config : HardwareConfig
        Detected hardware configuration

    Notes
    -----
    - Warnings are non-fatal (execution continues)
    - Helps users avoid common misconfigurations
    - Examples:
      - Using pjit backend on CPU-only system
      - Using PBS backend on standalone system
    """
    backend_name = backend.get_backend_name()

    # Check GPU backend on CPU system
    if backend_name == "pjit" and hardware_config.platform == "cpu":
        logger.warning(
            f"Using '{backend_name}' backend on CPU-only system. "
            f"Consider using 'multiprocessing' backend for better performance."
        )

    # Check cluster backend on standalone system
    if backend_name in ["pbs", "slurm"] and hardware_config.cluster_type == "standalone":
        logger.warning(
            f"Using '{backend_name}' backend on standalone system. "
            f"PBS/Slurm scheduler may not be available. "
            f"Consider using 'pjit' (GPU) or 'multiprocessing' (CPU) backend."
        )

    # Check multiprocessing on GPU system
    if (
        backend_name == "multiprocessing"
        and hardware_config.platform == "gpu"
        and hardware_config.num_devices > 1
    ):
        logger.warning(
            f"Using '{backend_name}' backend on multi-GPU system. "
            f"Consider using 'pjit' backend to leverage GPU acceleration."
        )


# Export public API
__all__ = [
    "select_backend",
    "get_backend_by_name",
]
