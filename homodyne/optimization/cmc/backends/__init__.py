"""Backend Infrastructure for Consensus Monte Carlo
====================================================

This module provides the abstract backend interface and selection logic for
executing parallel MCMC sampling across data shards in Consensus Monte Carlo.

Backend Types
-------------
The following backends are implemented in Phase 1 (Task Group 5):

1. **pjit Backend** (`pjit.py`):
   - JAX-based parallel execution for single or multi-GPU systems
   - Best for: GPU workstations, single-node GPU clusters
   - Parallelism: Up to num_gpus (sequential on single GPU)
   - Status: ✅ IMPLEMENTED

2. **Multiprocessing Backend** (`multiprocessing.py`):
   - Python multiprocessing.Pool for CPU-based parallelism
   - Best for: CPU workstations, single-node CPU clusters
   - Parallelism: Up to num_cpu_cores
   - Status: ✅ IMPLEMENTED

3. **PBS Backend** (`pbs.py`):
   - PBS job array submission for HPC clusters
   - Best for: Multi-node PBS clusters
   - Parallelism: Virtually unlimited (100+ nodes)
   - Status: ✅ IMPLEMENTED

Backend Selection
-----------------
The select_backend() function automatically chooses the optimal backend based
on detected hardware configuration:

Auto-selection priority:
1. Multi-node HPC cluster (PBS/Slurm) → PBS backend
2. Multi-GPU system → pjit backend
3. Single GPU → pjit backend (sequential execution)
4. CPU-only → multiprocessing backend

Users can override auto-selection via configuration:
    cmc_config = {'backend': {'type': 'multiprocessing'}}

Usage Examples
--------------
    from homodyne.optimization.cmc.backends import select_backend
    from homodyne.device.config import detect_hardware

    # Auto-select backend
    hardware = detect_hardware()
    backend = select_backend(hardware)
    print(f"Selected backend: {backend.get_backend_name()}")

    # Manual override
    backend = select_backend(hardware, user_override='pjit')

    # Run parallel MCMC
    results = backend.run_parallel_mcmc(
        shards=data_shards,
        mcmc_config={'num_warmup': 500, 'num_samples': 2000},
        init_params={'D0': 1234.5, 'alpha': 0.567},  # From config
        inv_mass_matrix=identity_matrix,  # Identity, adapted during warmup
    )

    # Direct instantiation
    from homodyne.optimization.cmc.backends import get_backend_by_name

    backend = get_backend_by_name('multiprocessing')
    results = backend.run_parallel_mcmc(...)

Integration Points
------------------
- Called by CMC coordinator to instantiate execution backend
- Backend instances are stateless and can be reused
- All backends follow the CMCBackend abstract interface
- Results are passed to combination.py for subposterior merging

Notes
-----
- All three backends fully implemented in Task Group 5 ✅
- Backends are lazy-loaded to avoid import errors if dependencies missing
- Each backend has comprehensive error handling and logging
- PBS backend requires cluster access (dry-run mode for testing)

Implementation Status
---------------------
✅ Task Group 4: Backend infrastructure (base class, selection logic)
✅ Task Group 5: Backend implementations (pjit, multiprocessing, PBS)
"""

from homodyne.optimization.cmc.backends.base import CMCBackend
from homodyne.optimization.cmc.backends.selection import (
    select_backend,
    get_backend_by_name,
)

# Import backend implementations for direct access
# These are lazy-loaded by get_backend_by_name() but can be imported directly
try:
    from homodyne.optimization.cmc.backends.pjit import PjitBackend
    PJIT_AVAILABLE = True
except ImportError:
    PJIT_AVAILABLE = False

try:
    from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False

try:
    from homodyne.optimization.cmc.backends.pbs import PBSBackend
    PBS_AVAILABLE = True
except ImportError:
    PBS_AVAILABLE = False

__all__ = [
    # Base class
    "CMCBackend",
    # Selection functions
    "select_backend",
    "get_backend_by_name",
    # Backend implementations (if available)
    "PjitBackend",
    "MultiprocessingBackend",
    "PBSBackend",
    # Availability flags
    "PJIT_AVAILABLE",
    "MULTIPROCESSING_AVAILABLE",
    "PBS_AVAILABLE",
]
