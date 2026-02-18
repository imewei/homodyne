.. _adr_jax_cpu_only:

ADR-001: JAX CPU-Only Backend
==============================

:Status: Accepted
:Date: 2024
:Deciders: Core team

Context
-------

Homodyne performs large-scale numerical optimization over two-time correlation matrices from
XPCS experiments. The correlation matrices can reach :math:`N_t^2 \times N_\phi` elements
(e.g., :math:`1000^2 \times 23 \approx 23 \times 10^6` values), and the CMC backend spawns
multiple worker processes, each running NUTS chains via JAX-compiled NumPyro models.

The key question: which computational backend should homodyne target — CPU, GPU, or both?

The target deployment environment is **synchrotron beamlines**, which typically provide:

- Multi-core CPU workstations (32–128 cores), often with NUMA topology.
- No GPUs at most beamlines, or GPUs shared across multiple users.
- Strict software environment controls (no arbitrary CUDA versions).
- Requirements for **reproducibility** and **portability** across institutions.


Decision
--------

Homodyne targets **CPU-only JAX** (no GPU/TPU support). The JAX backend is used for:

1. JIT compilation of the physics kernel (``jax_backend.py``, ``physics_cmc.py``).
2. Automatic differentiation for Jacobians (NLSQ) and NUTS gradients (CMC).
3. Vectorized operations via ``jax.vmap`` over angles and time points.

The CMC backend explicitly sets:

.. code-block:: python

   # In multiprocessing.py worker initialization
   os.environ["JAX_ENABLE_X64"] = "1"
   os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

This creates 4 virtual JAX devices per worker process, enabling NumPyro's ``parallel``
chain execution mode (``pmap`` over 4 devices) without requiring a real GPU.


Rationale
---------

**1. Deployment environment**

Synchrotron beamline workstations universally have multi-core CPUs. GPU availability is
inconsistent across facilities and cannot be assumed. A CPU-only implementation guarantees
portability.

**2. Performance is adequate on CPU**

The JIT compilation boundary means that after the first call, numerical kernels run as
compiled XLA code at near-native speed. For the problem sizes typical in XPCS (correlation
matrices up to :math:`\sim 10^8` elements per shard), CPU throughput is sufficient when
parallelism is structured correctly (worker pool + virtual devices).

Empirical benchmark (laminar_flow, 5K points/shard, 23 angles, 4 virtual devices):

- ``parallel`` chain execution: 4.9 s wall time per shard.
- ``vectorized`` chain execution: 101 s wall time per shard.

The ``parallel`` mode with virtual CPU devices achieves 20x speedup over the naive
``vectorized`` mode, making CPU-only performance viable.

**3. Reproducibility**

CPU execution is deterministic (given the same PRNG seed) on a given machine.
GPU execution is non-deterministic by default due to non-deterministic floating-point
reductions, which would compromise result reproducibility across runs.

**4. 64-bit precision**

The physics model requires 64-bit floating-point arithmetic for accurate computation of
transport integrals (differences of large numbers). On CPU, JAX 64-bit is enabled by
setting ``JAX_ENABLE_X64=1``. On GPU, 64-bit operations are significantly slower and
often unavailable on consumer-grade hardware.

**5. Simplicity**

Maintaining a single backend (CPU) avoids the complexity of:

- Conditional code paths for GPU/CPU.
- GPU memory management (sharding strategies differ on GPU).
- GPU-specific debugging tools.

The codebase remains simpler and easier to audit.


Consequences
------------

**Positive**:

- Portability: works on any Python 3.12+ system with JAX installed.
- Reproducibility: deterministic computation with explicit PRNG keys.
- Full 64-bit precision throughout.
- Simple deployment (``uv sync`` is sufficient; no CUDA setup).

**Negative / Accepted trade-offs**:

- Cannot use GPU-accelerated matrix operations for very large correlation matrices.
- The multiprocessing backend must spawn N separate processes rather than using GPU
  thread parallelism, which has higher per-process startup overhead.
- For extremely large datasets (:math:`> 10^9` data points), GPU acceleration would
  provide significant speedup that the current CPU implementation cannot match.


Alternatives Considered
-----------------------

**A. GPU-first JAX**

Would enable faster matrix operations for the NLSQ Jacobian computation and NUTS leapfrog
steps. Rejected because: GPU not universally available at beamlines; 32-bit precision
limitation; non-deterministic results.

**B. NumPy / SciPy only**

Would eliminate the JAX dependency, simplifying installation. Rejected because: no
automatic differentiation (would require finite-difference Jacobians, ~100x slower);
no JIT compilation; cannot leverage the NumPyro ecosystem for NUTS.

**C. PyTorch backend**

PyTorch has mature CPU and GPU support with autograd. Rejected because: the NumPyro
probabilistic programming library is JAX-native, and rewriting NUTS from scratch would
be a significant engineering effort; JAX's ``jit``/``vmap``/``pmap`` composability is
a better fit for the parallelism patterns in CMC.

**D. Mixed CPU/GPU**

Support both, dispatch at runtime. Rejected for the current version because: doubles the
testing matrix; GPU code path is untested and risky; adds complexity without clear benefit
for the target deployment environment. May be reconsidered in a future version.


.. seealso::

   - :ref:`developer_architecture` — overall system design
   - :ref:`adr_nlsq_primary` — NLSQ optimizer choice
   - :ref:`adr_cmc_consensus` — CMC multiprocessing architecture
