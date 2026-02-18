Configuration
==============

This section documents Homodyne's comprehensive YAML configuration system,
which controls all aspects of XPCS analysis including data input, analysis mode,
optimization methods, and output generation.

Overview
--------

Homodyne uses **YAML configuration files** to specify:

- **Analysis mode**: Static diffusion or laminar flow
- **Data input**: HDF5 file paths and phi angle filtering
- **Initial parameters**: Starting values and bounds for optimization
- **Optimization method**: NLSQ (fast) or CMC (uncertainty quantification)
- **Output options**: Formats, visualization, and logging

Configuration Sections
^^^^^^^^^^^^^^^^^^^^^^

A complete Homodyne configuration includes:

1. **Metadata**: Version and description
2. **Analysis Mode**: Static or laminar_flow
3. **Analyzer Parameters**: Experimental settings (dt, frame range, q-vector)
4. **Experimental Data**: HDF5 file paths and caching
5. **Phi Angle Filtering**: Angle selection and quality control
6. **Initial Parameters**: Starting values and bounds
7. **Optimization**: Method selection and settings (NLSQ/CMC)
8. **Performance**: JAX compilation and memory management
9. **Output**: File formats and directory structure
10. **Logging**: Console and file logging levels

.. toctree::
   :maxdepth: 2
   :caption: Configuration Resources

   templates
   options

Quick Links
-----------

- **Templates**: :doc:`templates` - Complete configuration examples
- **Options Reference**: :doc:`options` - All configuration parameters
- **Developer Guide**: :doc:`/developer/index` - Development setup

Working with Configurations
----------------------------

Interactive Configuration Builder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use the interactive configuration builder for guided setup:

.. code-block:: bash

    homodyne-config --interactive

This prompts you through required fields and generates a configuration file.

Validating Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^

Check if a configuration file is valid before running:

.. code-block:: bash

    homodyne-config --validate my_config.yaml

This validates:
- YAML syntax
- Required fields present
- Parameter bounds physically reasonable
- Data files exist
- Compatibility with selected analysis mode

Running Analysis with Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run NLSQ optimization:

.. code-block:: bash

    homodyne --config my_config.yaml --method nlsq

Run CMC inference:

.. code-block:: bash

    homodyne --config my_config.yaml --method cmc

Both methods use the same configuration file.

Analysis Modes
--------------

Two analysis modes are supported:

Static Mode (3 physical parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For equilibrium systems with pure diffusion:

.. code-block:: yaml

    analysis_mode: "static"

    initial_parameters:
      parameter_names:
        - D0           # Diffusion coefficient prefactor
        - alpha        # Anomalous diffusion exponent
        - D_offset     # Baseline diffusion

Laminar Flow Mode (7 physical parameters)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For non-equilibrium systems under shear:

.. code-block:: yaml

    analysis_mode: "laminar_flow"

    initial_parameters:
      parameter_names:
        - D0               # Diffusion coefficient prefactor
        - alpha            # Anomalous diffusion exponent
        - D_offset         # Baseline diffusion
        - gamma_dot_t0     # Shear rate prefactor
        - beta             # Shear rate exponent
        - gamma_dot_t_offset  # Baseline shear rate
        - phi0             # Flow angle

Per-Angle Scaling
^^^^^^^^^^^^^^^^^

**Mandatory in v2.4.0+**: Each scattering angle has unique optical and detector
properties, requiring per-angle scaling parameters.

Total parameters = physical parameters + 2 × number of scattering angles

Example with 3 filtered angles:
- Physical: 3 parameters (static) or 7 parameters (laminar_flow)
- Per-angle: 2 × 3 = 6 parameters (3 contrast + 3 offset)
- Total: 9 parameters (static) or 13 parameters (laminar_flow)

See :doc:`templates` for complete examples.

Configuration Features
----------------------

Data Filtering and Caching
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Homodyne supports:
- **Phi angle filtering**: Select specific scattering angles (e.g., 0°, 90°, 180°)
- **Data caching**: Save processed C2 data for reuse
- **Stratification**: Automatic angle-aware chunking for large datasets
- **Sequential optimization**: Per-angle NLSQ with result combination

See :doc:`options` for detailed parameter descriptions.

Optimization Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

NLSQ Configuration:
- Trust region scaling
- Convergence tolerance
- Maximum iterations
- Automatic strategy selection (STANDARD/LARGE/CHUNKED/STREAMING)

CMC Configuration:
- Consensus Monte Carlo only in v2.4.1+
- Per-shard NUTS sampling
- Subposterior combination method
- Convergence diagnostics
- Checkpoint and resume capability

Performance Tuning
^^^^^^^^^^^^^^^^^^

Memory management:
- Chunk size for processing
- Maximum memory limits
- Caching strategies

JAX optimization:
- JIT compilation control
- CPU thread count
- Vectorization level

See :doc:`options` for all performance parameters.

Output Configuration
^^^^^^^^^^^^^^^^^^^^

Control output generation:
- Output directory and subdirectory structure
- File formats (HDF5, JSON, CSV)
- Compression settings
- Visualization settings (plots, format, DPI)

Validation and Quality Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Built-in validation checks:
- Parameter bounds enforcement
- Mode compatibility verification
- File existence checks
- Angle coverage validation

Configuration Examples
----------------------

Complete configuration examples are provided in :doc:`templates`:

- **Static mode**: Minimal and comprehensive examples
- **Laminar flow**: Minimal and comprehensive examples
- **Feature demonstrations**: Per-angle scaling, filtering, stratification

Each template includes:
- Detailed parameter descriptions and units
- Physical meaning and typical ranges
- Usage notes and workflow guidance
- Production-ready defaults

Version Information
-------------------

- **Required Python**: 3.12+
- **JAX**: ≥0.8.2 (CPU-only)
- **Analysis Modes**: static, laminar_flow

Environment Validation
----------------------

Before running analysis, validate your system:

.. code-block:: bash

    python -m homodyne.runtime.utils.system_validator --quick

This checks:
- Python version (3.12+)
- JAX installation and version (≥0.8.2)
- CPU device availability
- Required dependencies

For Questions
--------------

- See :doc:`/configuration/options` for all configuration parameters
- See :doc:`/developer/index` for development setup
