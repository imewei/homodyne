I/O Module
==========

The :mod:`homodyne.io` module provides functions for saving and loading optimization results, experimental data, and analysis outputs.


Overview
--------

**Key Features**:

- JSON serialization of optimization results
- NPZ file output for numerical data
- JAX/NumPy array handling
- MCMC diagnostic output formatting

Module Contents
---------------

.. automodule:: homodyne.io
   :members:
   :undoc-members:
   :show-inheritance:

Primary Functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :nosignatures:

   homodyne.io.save_nlsq_json_files
   homodyne.io.save_nlsq_npz_file
   homodyne.io.create_mcmc_parameters_dict
   homodyne.io.create_mcmc_analysis_dict
   homodyne.io.create_mcmc_diagnostics_dict
   homodyne.io.json_safe
   homodyne.io.json_serializer

NLSQ Writers
------------

Functions for saving NLSQ optimization results.

.. automodule:: homodyne.io.nlsq_writers
   :members:
   :undoc-members:
   :show-inheritance:

Output Files
~~~~~~~~~~~~

NLSQ optimization produces two output files:

**JSON Files** (human-readable):

- ``fit_results.json``: Best-fit parameters, uncertainties, and metadata
- ``optimization_diagnostics.json``: Convergence information and iteration history

**NPZ Files** (numerical data):

- ``fitted_data.npz``: Arrays including fitted C2 surface, residuals, and per-angle scaling

MCMC Writers
------------

Functions for saving MCMC/CMC optimization results.

.. automodule:: homodyne.io.mcmc_writers
   :members:
   :undoc-members:
   :show-inheritance:

Output Structure
~~~~~~~~~~~~~~~~

MCMC results are formatted for:

- ArviZ-compatible posterior analysis
- Publication-quality uncertainty reporting
- Convergence diagnostic summaries

JSON Utilities
--------------

Helper functions for JSON serialization of scientific data.

.. automodule:: homodyne.io.json_utils
   :members:
   :undoc-members:
   :show-inheritance:

Supported Types
~~~~~~~~~~~~~~~

The JSON utilities handle:

- NumPy arrays and scalars
- JAX arrays (converted to NumPy)
- Complex numbers
- NaN and Inf values
- Nested dictionaries and lists

Usage Examples
--------------

**Saving NLSQ Results**::

    from pathlib import Path
    from homodyne.io import save_nlsq_json_files, save_nlsq_npz_file

    # Save JSON files (3 separate dicts: parameters, analysis, convergence)
    save_nlsq_json_files(
        param_dict=param_dict,
        analysis_dict=analysis_dict,
        convergence_dict=convergence_dict,
        output_dir=Path("./results"),
    )

    # Save NPZ file (arrays for fitted data, residuals, scaling)
    save_nlsq_npz_file(
        phi_angles=phi_angles,
        c2_exp=c2_exp,
        c2_raw=c2_raw,
        c2_scaled=c2_scaled,
        c2_solver=c2_solver,
        per_angle_scaling=per_angle_scaling,
        per_angle_scaling_solver=per_angle_scaling_solver,
        residuals=residuals,
        output_dir=Path("./results"),
    )

**Creating MCMC Output**::

    from homodyne.io import create_mcmc_parameters_dict, create_mcmc_diagnostics_dict

    # Format parameters for output (takes the full CMCResult)
    params = create_mcmc_parameters_dict(result=cmc_result)

    # Format diagnostics (takes the full CMCResult)
    diagnostics = create_mcmc_diagnostics_dict(result=cmc_result)

**JSON-Safe Conversion**::

    from homodyne.io import json_safe
    import numpy as np

    # Convert arrays to JSON-serializable format
    data = {'array': np.array([1.0, 2.0, 3.0])}
    safe_data = json_safe(data)

See Also
--------

- :mod:`homodyne.optimization` - Optimization result classes
- :mod:`homodyne.viz` - Visualization of results
- :mod:`homodyne.config` - Configuration management
