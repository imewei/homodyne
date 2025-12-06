"""CMC execution backends.

This module provides different execution backends for running
CMC shards in parallel:

- MultiprocessingBackend: CPU parallelism via Python multiprocessing
- PjitBackend: JAX distributed execution via pjit
- PBSBackend: HPC cluster execution via PBS job scheduler
"""

from homodyne.optimization.cmc.backends.base import CMCBackend, select_backend
from homodyne.optimization.cmc.backends.multiprocessing import MultiprocessingBackend
from homodyne.optimization.cmc.backends.pbs import PBSBackend
from homodyne.optimization.cmc.backends.pjit import PjitBackend

__all__ = [
    "CMCBackend",
    "MultiprocessingBackend",
    "PBSBackend",
    "PjitBackend",
    "select_backend",
]
