Angle Filtering
===============

Overview
--------

Angle filtering selects specific phi angle ranges before optimization, reducing parameter count and improving convergence for targeted analysis of directional dynamics.

**Key Features:**

- **Angle Normalization**: All angles in [-180°, 180°] range
- **Wrap-Aware Filtering**: Handles boundary-crossing ranges (e.g., [170°, -170°])
- **Multi-Range Support**: Define multiple angular regimes
- **Pre-Optimization Integration**: Applied before NLSQ and MCMC
- **Configuration Validation**: Warnings for non-overlapping ranges

When to Use Angle Filtering
----------------------------

Use angle filtering to:

- **Reduce parameter count**: From 7+2n to 7+2m (where m < n)
- **Focus on specific angles**: Study particular flow/shear directions
- **Improve convergence**: Fewer parameters = faster, more stable fitting
- **Targeted analysis**: Extract direction-specific properties

Example: 10 experimental angles → Filter to 2-3 angles of interest → Faster, simpler analysis

Configuration
--------------

Basic Setup
~~~~~~~~~~~

Enable angle filtering with target ranges:

.. code-block:: yaml

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -10.0
          max_angle: 10.0
          description: "Parallel to flow (0°)"
        - min_angle: 85.0
          max_angle: 95.0
          description: "Perpendicular to flow (90°)"

**Effect**: Only angles in [-10°, 10°] or [85°, 95°] are retained for analysis.

Angle Normalization
~~~~~~~~~~~~~~~~~~~~

All angles are automatically normalized to [-180°, 180°]:

- Input: Any angle value
- Internal: Converted to [-180°, 180°]
- Output: Same normalized range

Examples:

.. code-block:: text

    Input       Normalized
    ----        ----------
    45°      →  45°
    270°     →  -90°
    -180°    →  -180° (boundary)
    180°     →  180° (same as -180°)
    -200°    →  160°
    400°     →  40°

Wrap-Aware Range Checking
~~~~~~~~~~~~~~~~~~~~~~~~~~

Handles ranges spanning the ±180° boundary:

**Example 1**: Near 0°
.. code-block:: yaml

    target_ranges:
      - min_angle: -10.0
        max_angle: 10.0

Works correctly for angles near 0°.

**Example 2**: Near ±180° (boundary crossing)
.. code-block:: yaml

    target_ranges:
      - min_angle: 170.0
        max_angle: -170.0  # or (180 - 10 = 170)

This range includes:
- 170° to 180°
- -180° to -170°

The algorithm correctly handles wrapping across the ±180° boundary.

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

Multiple range definitions:

.. code-block:: yaml

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -10.0
          max_angle: 10.0
          description: "Parallel (0°)"
        - min_angle: 85.0
          max_angle: 95.0
          description: "Perpendicular (90°)"
        - min_angle: 170.0
          max_angle: -170.0
          description: "Opposite (180°)"

**Result**: Keep angles in all 3 ranges, discard others.

Impact on Parameter Counting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Parameter count changes with filtering:

**Static Isotropic:**
- No filtering: 3 + 2n parameters
- With filtering: 3 + 2m parameters (m = filtered angles)

**Laminar Flow:**
- No filtering: 7 + 2n parameters
- With filtering: 7 + 2m parameters

Example: 10 angles → filter to 3 → reduce from 27 to 13 parameters (laminar)

Integration Points
------------------

Angle filtering is applied at three points in analysis:

1. **Before NLSQ Optimization**
   - Reduces parameter space for fitting
   - Faster convergence
   - Fewer parameters to estimate

2. **Before MCMC Sampling**
   - Simpler posterior to sample
   - Better mixing
   - Faster convergence diagnostics

3. **In Plotting Functions**
   - Plots show only filtered angles
   - Cleaner visualizations
   - Matches optimization data

Configuration Validation
~~~~~~~~~~~~~~~~~~~~~~~~

Homodyne validates angle configurations:

.. code-block:: yaml

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -45.0
          max_angle: -35.0
          description: "Custom range"

**Warnings Produced For:**

- Empty angle overlap (no data in specified ranges)
- Unusual ranges (e.g., > 90° wide)
- Boundary crossing without wrap-aware syntax
- Non-overlapping multiple ranges

Workflow Examples
-----------------

Static Isotropic with Angle Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplified analysis for static system:

.. code-block:: yaml

    # config_static_filtered.yaml
    experimental_data:
      file_path: "./data/experiment.hdf"

    parameter_space:
      model: "static_isotropic"
      bounds:
        - name: D0
          min: 100.0
          max: 100000.0
        - name: alpha
          min: 0.0
          max: 2.0
        - name: D_offset
          min: -100.0
          max: 100.0

    initial_parameters:
      parameter_names: ["D0", "alpha", "D_offset"]

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -15.0
          max_angle: 15.0

    optimization:
      method: "nlsq"

Run:

.. code-block:: bash

    homodyne --config config_static_filtered.yaml --output-dir results_filtered

Effect: Only analyze angles near 0° (isotropic direction).

Laminar Flow with Directional Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Separate analysis for different flow directions:

.. code-block:: yaml

    # config_laminar_filtered.yaml
    parameter_space:
      model: "laminar_flow"
      bounds:
        # ... all 7 physical parameters ...

    phi_filtering:
      enabled: true
      target_ranges:
        - min_angle: -10.0
          max_angle: 10.0
          description: "Parallel to flow"
        - min_angle: 80.0
          max_angle: 100.0
          description: "Perpendicular to flow"

    optimization:
      method: "nlsq"
      nlsq:
        max_iterations: 100

Run multiple times or use a wrapper to analyze each directional regime.

Comparing Filtered vs. Unfiltered
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze impact of filtering:

.. code-block:: python

    import json

    # Load unfiltered results (all angles)
    with open('results_unfiltered/parameters.json') as f:
        unfiltered = json.load(f)

    # Load filtered results (selected angles)
    with open('results_filtered/parameters.json') as f:
        filtered = json.load(f)

    # Compare D0
    print("D0 Comparison:")
    print(f"  Unfiltered: {unfiltered['D0']['value']:.1f} ± {unfiltered['D0']['uncertainty']:.1f}")
    print(f"  Filtered: {filtered['D0']['value']:.1f} ± {filtered['D0']['uncertainty']:.1f}")

    # Filtered typically has:
    # - Smaller uncertainty (fewer competing parameters)
    # - Possibly different value (if angles had directional bias)

Validation and Diagnostics
---------------------------

Check Filtered Data
~~~~~~~~~~~~~~~~~~~

Verify which angles were kept:

.. code-block:: python

    import json

    with open('analysis_results_nlsq.json') as f:
        results = json.load(f)

    print(f"Original angles: {results['dataset_info'].get('original_num_angles', 'N/A')}")
    print(f"Filtered angles: {results['dataset_info']['num_angles']}")
    print(f"Points per angle: {results['dataset_info']['points_per_angle']}")

Validation Warnings
~~~~~~~~~~~~~~~~~~~

Watch logs for filtering warnings:

.. code-block:: text

    WARNING: Angle range [-10, 10] contains 0 datapoints
    WARNING: No overlap between specified ranges and experimental angles

These indicate:

1. Configured ranges don't match data
2. Filtering removed all data
3. Configuration error in angle ranges

Solution: Adjust target_ranges to match experimental angles.

Parameter Impact
~~~~~~~~~~~~~~~~

Check parameter count reduction:

.. code-block:: python

    import json

    with open('analysis_results_nlsq.json') as f:
        results = json.load(f)

    param_space = results['parameter_space']
    print(f"Total parameters: {param_space['num_parameters']}")
    print(f"Active parameters: {param_space['active_parameters']}")
    print(f"Fixed parameters: {len(param_space.get('fixed_parameters', {}))}")

Troubleshooting
---------------

No Data After Filtering
~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: "All angles filtered out" or empty results

**Solution**:

1. Check experimental angles in data
2. Verify range syntax (wrap-aware if crossing ±180°)
3. Expand ranges temporarily to debug
4. Examine data: `homodyne --config config.yaml --plot-experimental-data`

Unexpected Parameter Values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Filtered results differ significantly from unfiltered

**Causes**:

1. **Angle selection bias**: Filtered angles may have different signal
2. **Reduced constraints**: Fewer angles = fewer constraints
3. **Degeneracies**: Some parameters only constrained by certain angles

**Solution**: Include more angles or use unfiltered results for global estimates.

See Also
--------

- :doc:`../theoretical-framework/parameter-models` - Parameter counting
- :doc:`nlsq-optimization` - NLSQ optimization
- :doc:`mcmc-uncertainty` - MCMC with filtered angles
- :doc:`../user-guide/configuration` - Configuration details
