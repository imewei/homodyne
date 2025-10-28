Architecture Documentation
=========================

This section contains detailed architectural documentation for advanced features of the Homodyne XPCS analysis package.

Overview
--------

The architecture documentation covers:

* **CMC (Consensus Monte Carlo)**: Dual-criteria decision logic and distributed MCMC
* **NUTS Chain Parallelization**: Platform-specific execution modes and convergence diagnostics
* **Backend Selection**: Automatic hardware-based backend selection

These documents provide deep technical details beyond the user-facing guides in :doc:`advanced-topics/index`.

Quick Navigation
----------------

CMC (Consensus Monte Carlo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Logic:**
   :doc:`architecture/cmc-dual-mode-strategy`
      Comprehensive design document (3,500+ words) covering:

      - Dual-criteria decision logic (parallelism OR memory)
      - Current implementation (sample-level sharding)
      - Future enhancement (data-level sharding, planned v2.1.0)
      - Technical design with code examples

   :doc:`architecture/cmc-decision-quick-reference`
      Quick lookup table for developers:

      - Memory calculation formula
      - Configuration examples
      - Decision tree

NUTS Chain Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Execution Strategy:**
   :doc:`architecture/nuts-chain-parallelization`
      Complete guide to NUTS chain execution (4,000+ words):

      - Platform-specific behavior (CPU/GPU/multi-GPU)
      - Why 4 chains on single GPU (even though sequential)
      - Convergence diagnostics (R-hat, ESS, divergences)
      - Performance characteristics

   :doc:`architecture/nuts-chain-parallelization-quick-reference`
      Quick reference guide:

      - Configuration presets (fast/balanced/high-quality)
      - Platform comparison table
      - Common issues and solutions

Architecture Hub
~~~~~~~~~~~~~~~~

:doc:`architecture/README`
   Central navigation hub with:

   - Decision trees for method selection
   - Performance summary tables
   - Troubleshooting guide
   - Links to all architecture docs

Related Documentation
---------------------

**User-Facing Guides:**

* :doc:`advanced-topics/cmc-large-datasets` - CMC user guide
* :doc:`advanced-topics/mcmc-uncertainty` - MCMC user guide
* :doc:`user-guide/configuration` - Configuration system

**Developer Guides:**

* :doc:`developer-guide/architecture` - Overall system architecture
* :doc:`developer-guide/performance` - Performance optimization

**Theoretical Background:**

* :doc:`theoretical-framework/index` - Physics equations and models

Document Structure
------------------

.. toctree::
   :maxdepth: 2
   :caption: Architecture Documentation

   architecture/README
   architecture/cmc-dual-mode-strategy
   architecture/cmc-decision-quick-reference
   architecture/nuts-chain-parallelization
   architecture/nuts-chain-parallelization-quick-reference

Key Concepts
------------

CMC (Consensus Monte Carlo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Distributed MCMC for large datasets or many samples

**Triggering Conditions** (OR logic):

1. **Parallelism**: ``num_samples >= 100`` (many phi angles)
2. **Memory**: ``dataset_size`` exceeds 50% of available memory

**Example**::

   # 2 phi × 100M each = 200M total
   # Memory: 9.6 GB = 60% of 16 GB → CMC triggered
   use_cmc = should_use_cmc(num_samples=2, hw, dataset_size=200_000_000)
   # Result: True (memory threshold exceeded)

**Implementation**:

- ✅ Sample-level sharding (current)
- 🚧 Data-level sharding (planned v2.1.0)

NUTS Chains
~~~~~~~~~~~

**Purpose**: Multiple independent MCMC chains for convergence diagnostics

**Default Configuration**:

- 4 chains (enables R-hat, ESS diagnostics)
- Sequential on single GPU (4× time)
- Parallel on CPU (1.1× time)

**Platform Behavior**:

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Platform
     - Chains
     - Mode
     - Performance
   * - CPU
     - 4
     - Parallel
     - ~1.1× single-chain time
   * - Single GPU
     - 4
     - Sequential
     - 4× single-chain time
   * - Multi-GPU
     - 4
     - Parallel
     - ~1.1× single-chain time

**Why Multiple Chains**:

- ✅ R-hat statistic (convergence detection)
- ✅ Effective Sample Size (true independent samples)
- ✅ Divergence detection (problematic regions)
- ✅ Better uncertainty quantification

Decision Trees
--------------

Method Selection (Automatic)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   Dataset Analysis
   ├─ num_samples >= 100?
   │  ├─ YES → CMC (parallelism mode)
   │  └─ NO → Check memory
   │     ├─ dataset_size > 50% memory?
   │     │  ├─ YES → CMC (memory mode)
   │     │  └─ NO → NUTS
   │     └─ dataset_size unknown?
   │        └─ NUTS

Chain Configuration
~~~~~~~~~~~~~~~~~~~

::

   Hardware Platform
   ├─ CPU (multi-core)?
   │  └─ Use 4-8 chains (parallel)
   ├─ Single GPU?
   │  └─ Use 4 chains (sequential, but with diagnostics)
   └─ Multi-GPU (N)?
      └─ Use N chains (1 per GPU, parallel)

Implementation Status
---------------------

✅ Implemented
~~~~~~~~~~~~~~

1. **Dual-Criteria CMC Decision**

   - Parallelism-based triggering (num_samples)
   - Memory-based triggering (dataset_size)
   - OR logic (either condition triggers CMC)

2. **NUTS Chain Parallelization**

   - CPU parallel chains (via ``set_host_device_count``)
   - GPU sequential chains (automatic)
   - Multi-GPU distribution

3. **Backend Selection**

   - Auto-detection (hardware-based)
   - Manual override support
   - Platform compatibility validation

🚧 Planned
~~~~~~~~~~

1. **Data-Level Sharding** (v2.1.0)

   - Split time points within samples
   - Enable memory reduction for few-sample scenarios
   - Adaptive sharding mode selection

2. **Hybrid Sharding** (v3.0.0)

   - Both sample AND data sharding
   - Automatic shard size optimization
   - Custom sharding strategies

Changelog
---------

**October 28, 2025**:

- ✅ Implemented dual-criteria CMC decision logic
- ✅ Fixed backend='auto' handling
- ✅ Added memory-based CMC triggering
- ✅ Documented NUTS chain parallelization
- ✅ Created comprehensive architecture docs

**Next Steps**:

- Implement data-level sharding for CMC
- Add performance benchmarks
- Create visual decision trees
- Add troubleshooting examples

See Also
--------

* :doc:`advanced-topics/index` - User-facing advanced topics
* :doc:`developer-guide/index` - Developer documentation
* :doc:`api-reference/index` - API reference
