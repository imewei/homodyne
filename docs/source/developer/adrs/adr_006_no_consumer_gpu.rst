.. _adr_no_consumer_gpu:

ADR-006: No GPU Acceleration on Consumer Hardware
===================================================

:Status: Accepted
:Date: 2026-03
:Deciders: Core team

Context
-------

:ref:`ADR-001 <adr_jax_cpu_only>` established a CPU-only backend to match the
synchrotron beamline deployment environment.  A separate question arose: would
**consumer GPU acceleration** (specifically an NVIDIA RTX 4090 Laptop, 16 GB
VRAM, CUDA 13.1) improve performance for local development and pre-beamline
analysis?

This ADR documents the quantitative assessment.


Decision
--------

**Do not implement GPU acceleration on consumer-class GPUs.**  The CPU-only
configuration (``JAX_PLATFORMS=cpu``) is enforced in ``device/cpu.py`` and
remains the only supported backend.  No changes to ``pyproject.toml``, the
Makefile, or device configuration are required.


Rationale
---------

**1. Float64 throughput penalty on consumer GPUs**

Homodyne parameters span 7 orders of magnitude
(:math:`D_0 \sim 10^4`,  :math:`\dot\gamma_0 \sim 10^{-3}`).  The positivity
floor ``epsilon_abs = 1e-12`` is below float32 machine epsilon
(:math:`\sim 1.2 \times 10^{-7}`).  Float32 would cause NUTS leapfrog
divergence and NLSQ Jacobian collapse.  **Float64 is non-negotiable.**

Consumer GPUs penalize float64 severely:

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Hardware
     - float64 (TFLOPS)
     - vs 20-core CPU
     - float64 : float32
   * - RTX 4090 Laptop
     - ~1.3
     - 1.3--2.6x
     - 1 : 64
   * - A100 SXM4
     - ~19.5
     - 20--40x
     - 1 : 2
   * - H100 SXM5
     - ~67
     - 67--130x
     - 1 : 2

The generic "20--100x" speedup claim in GPU acceleration guides assumes float32
workloads.  For float64 physics on consumer hardware the net advantage is
**1.3--2.6x before transfer overhead**.

**2. NLSQ path: PCIe overhead exceeds compute savings**

The external ``nlsq`` C extension forces a CPU round-trip every Levenberg--Marquardt
iteration:

.. code-block:: text

   GPU kernel -> block_until_ready -> np.asarray (D2H)
     -> NLSQ optimizer step (CPU)
     -> jnp.asarray (H2D) -> GPU kernel

For a typical dataset (``n_time=1000``, ``n_phi=23``):

- Jacobian size: :math:`(23, 1000, 1000, 7)` float64 = **1.23 GB**.
- PCIe transfer per iteration: ~70 ms at ~20 GB/s.
- Optimistic kernel speedup: 2x (967 ms to ~484 ms).
- Net with transfer: ~559 ms GPU vs 967 ms CPU = **1.7x** -- reduced to
  **~1.3--1.5x** after synchronization barriers.

For 10M+ point datasets the expanded Jacobian (~22 GB) exceeds 16 GB VRAM,
forcing a fallback to the CPU out-of-core solver anyway.

**3. CMC path: architectural incompatibility**

The CMC backend (``backends/multiprocessing.py``) is structurally incompatible
with GPU execution:

- **Virtual CPU devices**: ``--xla_force_host_platform_device_count=4`` is
  undefined on the CUDA backend.
- **CUDA context overhead**: Each spawned worker creates an independent CUDA
  context (300--800 MB).  With 9 workers: 2.7--7.2 GB consumed before any
  computation.
- **Shared memory**: ``SharedDataManager`` uses POSIX shared memory
  (CPU RAM only); each worker would need to re-transfer shard data to VRAM.
- **Single-process alternative**: Eliminating spawn-based parallelism to run
  sequential shards on GPU would sacrifice 9-way concurrency, resulting in a
  net **3--6x slowdown**.


Consequences
------------

**Positive**:

- No additional complexity in the device configuration layer.
- NLSQ and CMC paths remain unchanged and fully tested.
- No CUDA version management or GPU driver dependencies.
- The existing 2718-test suite runs unmodified.

**Negative / Accepted trade-offs**:

- Users with consumer GPUs cannot offload computation to the GPU.
- Local development cannot exploit GPU parallelism for faster iteration.


When to Revisit
---------------

GPU acceleration becomes viable when **all three** conditions are met:

1. **Datacenter GPU** with :math:`\geq` 1 : 2 float64 ratio (A100 / H100).
2. **NLSQ library boundary eliminated** -- migrate from ``nlsq.curve_fit`` to
   a pure-JAX optimizer (e.g., ``jaxopt.LevenbergMarquardt``), removing the
   per-iteration CPU round-trip.
3. **CMC refactored** to single-process ``jax.vmap``-over-chains, replacing
   spawn-based multiprocessing.

.. list-table::
   :header-rows: 1
   :widths: 40 25 25

   * - Upgrade path
     - Estimated speedup
     - Engineering effort
   * - A100 + current code
     - 2--4x (NLSQ boundary limited)
     - Low
   * - A100/H100 + jaxopt LM rewrite
     - 10--30x (NLSQ)
     - High
   * - A100/H100 + CMC pmap refactor
     - 5--15x (CMC)
     - Medium
   * - A100/H100 + both rewrites
     - 10--30x (end-to-end)
     - High


.. seealso::

   - :ref:`adr_jax_cpu_only` -- foundational CPU-only decision
   - :ref:`adr_cmc_consensus` -- CMC multiprocessing architecture
   - :ref:`adr_nlsq_primary` -- NLSQ / CMC architectural split
