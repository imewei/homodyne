.. _batch_processing:

Batch Processing Multiple Datasets
====================================

.. rubric:: Learning Objectives

By the end of this section you will understand:

- How to process multiple HDF5 files in sequence
- Script patterns for batch NLSQ and CMC analysis
- How to aggregate results across datasets
- Strategies for parallel processing on HPC clusters

---

Overview
---------

XPCS experiments often produce many datasets: multiple samples, multiple q-values,
or multiple time points. Homodyne provides no built-in batch runner, but the
Python API makes it straightforward to build one.

---

Sequential Batch Script
------------------------

A minimal script to process a list of HDF5 files:

.. code-block:: python

   """
   batch_nlsq.py — Process multiple XPCS files with NLSQ.
   Usage: uv run python batch_nlsq.py --config base_config.yaml --files data/*.h5
   """

   import argparse
   import json
   from pathlib import Path
   import numpy as np

   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data, XPCSDataLoader, XPCSDataFormatError
   from homodyne.optimization.nlsq import fit_nlsq_jax
   from homodyne.utils.logging import get_logger

   logger = get_logger(__name__)


   def process_file(h5_path: Path, base_config_path: Path, output_dir: Path):
       """Process one HDF5 file and save results."""
       stem = h5_path.stem

       # Override file_path in config for this specific file
       config = ConfigManager.from_yaml(str(base_config_path))
       config.data.file_path = str(h5_path)

       try:
           data = load_xpcs_data(config=config)
       except XPCSDataFormatError as e:
           logger.error(f"Data loading failed for {h5_path.name}: {e}")
           return None

       result = fit_nlsq_jax(data, config)

       if result.success:
           # Build result dictionary
           result_dict = {
               "file": str(h5_path),
               "convergence_status": result.convergence_status,
               "reduced_chi_squared": result.reduced_chi_squared,
               "parameters": result.parameters.tolist(),
               "uncertainties": result.uncertainties.tolist(),
               "execution_time": result.execution_time,
           }
           # Save to JSON
           out_json = output_dir / f"{stem}_nlsq.json"
           with open(out_json, "w") as f:
               json.dump(result_dict, f, indent=2)
           logger.info(f"Saved {out_json}")
           return result_dict
       else:
           logger.warning(f"Fit failed for {h5_path.name}: {result.message}")
           return None


   def main():
       parser = argparse.ArgumentParser(description="Batch NLSQ processing")
       parser.add_argument("--config", required=True, help="Base YAML config")
       parser.add_argument("--files", nargs="+", required=True, help="HDF5 files")
       parser.add_argument("--output", default="./results", help="Output directory")
       args = parser.parse_args()

       output_dir = Path(args.output)
       output_dir.mkdir(parents=True, exist_ok=True)

       results = []
       for h5_file in args.files:
           h5_path = Path(h5_file)
           logger.info(f"Processing {h5_path.name}")
           result = process_file(h5_path, Path(args.config), output_dir)
           if result:
               results.append(result)

       # Save aggregate summary
       summary_path = output_dir / "batch_summary.json"
       with open(summary_path, "w") as f:
           json.dump(results, f, indent=2)
       logger.info(f"Batch complete: {len(results)}/{len(args.files)} succeeded")
       logger.info(f"Summary saved to {summary_path}")


   if __name__ == "__main__":
       main()

**Run the batch script:**

.. code-block:: bash

   uv run python batch_nlsq.py \
     --config base_config.yaml \
     --files data/sample_*.h5 \
     --output results/

---

Multiple q-Values
------------------

If each HDF5 file contains data at multiple q-values, iterate over q:

.. code-block:: python

   import numpy as np
   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax

   q_values = [0.020, 0.030, 0.054, 0.080, 0.110]  # Å⁻¹

   results_by_q = {}
   for q in q_values:
       config = ConfigManager.from_yaml("config_template.yaml")
       config.data.q_value = q

       data = load_xpcs_data(config=config)
       result = fit_nlsq_jax(data, config)

       results_by_q[q] = {
           "D0": result.parameters[0],
           "D0_err": result.uncertainties[0],
           "chi2_nu": result.reduced_chi_squared,
       }

   # Check q-dependence of D0 (should be constant if Stokes-Einstein applies)
   for q, res in sorted(results_by_q.items()):
       print(f"q={q:.3f}: D0 = {res['D0']:.2f} ± {res['D0_err']:.2f}")

---

Result Aggregation
-------------------

After batch processing, aggregate results for analysis:

.. code-block:: python

   import json
   import numpy as np
   from pathlib import Path

   results_dir = Path("results/")
   all_results = []

   for json_file in sorted(results_dir.glob("*_nlsq.json")):
       with open(json_file) as f:
           all_results.append(json.load(f))

   # Extract D0 values across samples
   D0_values = np.array([r['parameters'][0] for r in all_results
                          if r['convergence_status'] == 'converged'])
   D0_errors = np.array([r['uncertainties'][0] for r in all_results
                          if r['convergence_status'] == 'converged'])

   print(f"D0 mean:   {D0_values.mean():.2f} Å²/s")
   print(f"D0 std:    {D0_values.std():.2f} Å²/s")
   print(f"D0 range:  [{D0_values.min():.2f}, {D0_values.max():.2f}] Å²/s")

---

HPC Parallel Processing
------------------------

On HPC clusters, use job arrays to process files in parallel:

**SLURM job array example (batch_slurm.sh):**

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=homodyne_batch
   #SBATCH --array=0-99           # 100 files, one per job
   #SBATCH --cpus-per-task=8
   #SBATCH --mem=32G
   #SBATCH --time=02:00:00
   #SBATCH --output=logs/job_%A_%a.out

   # Get file for this array index
   FILE_LIST=files.txt              # One file path per line
   FILE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$FILE_LIST")

   echo "Processing: $FILE"
   uv run python process_one.py \
     --config config.yaml \
     --input "$FILE" \
     --output "results/job_${SLURM_ARRAY_TASK_ID}/"

**process_one.py:**

.. code-block:: python

   """process_one.py — Single file processor for SLURM array jobs."""

   import argparse
   import json
   from pathlib import Path

   from homodyne.config import ConfigManager
   from homodyne.data import load_xpcs_data
   from homodyne.optimization.nlsq import fit_nlsq_jax

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument("--config")
       parser.add_argument("--input")
       parser.add_argument("--output")
       args = parser.parse_args()

       output_dir = Path(args.output)
       output_dir.mkdir(parents=True, exist_ok=True)

       config = ConfigManager.from_yaml(args.config)
       config.data.file_path = args.input

       data = load_xpcs_data(config=config)
       result = fit_nlsq_jax(data, config)

       result_dict = {
           "file": args.input,
           "parameters": result.parameters.tolist(),
           "uncertainties": result.uncertainties.tolist(),
           "chi2_nu": result.reduced_chi_squared,
           "converged": result.success,
       }

       out_path = output_dir / "result.json"
       with open(out_path, "w") as f:
           json.dump(result_dict, f, indent=2)

   if __name__ == "__main__":
       main()

---

Collecting SLURM Results
--------------------------

After all array jobs complete, collect results:

.. code-block:: python

   import json
   from pathlib import Path
   import numpy as np

   results = []
   for job_dir in sorted(Path("results").glob("job_*")):
       result_file = job_dir / "result.json"
       if result_file.exists():
           with open(result_file) as f:
               results.append(json.load(f))

   converged = [r for r in results if r['converged']]
   print(f"Converged: {len(converged)}/{len(results)}")

   D0_array = np.array([r['parameters'][0] for r in converged])
   print(f"D0: {D0_array.mean():.2f} ± {D0_array.std():.2f} Å²/s")

---

See Also
---------

- :doc:`configuration` — YAML configuration reference
- :doc:`performance_tuning` — Optimizing for HPC
- :doc:`../05_appendices/troubleshooting` — Batch job failure troubleshooting
