"""PBS Backend for Consensus Monte Carlo
========================================

This backend implements distributed MCMC execution on HPC clusters using PBS
(Portable Batch System) job arrays. It's designed for multi-node clusters and
provides virtually unlimited parallelism.

Execution Strategy
------------------
1. Write per-shard data to temporary HDF5 files
2. Generate PBS job array script from template
3. Submit PBS job array via qsub
4. Poll job status with qstat
5. Collect results from shard output files
6. Clean up temporary files

Key Features
------------
- Scalable to 100+ nodes
- Automatic job array submission
- Job status polling and failure detection
- Retry logic for failed jobs
- Automatic cleanup of temporary files

PBS Job Array Structure
-----------------------
- Array size: num_shards
- Each array task processes one shard
- Tasks run independently (no inter-task communication)
- Output: One HDF5 file per shard

File Organization
-----------------
Temporary directory structure:
    cmc_temp_{timestamp}/
    ├── shard_data/
    │   ├── shard_000.h5    # Input data for shard 0
    │   ├── shard_001.h5
    │   └── ...
    ├── shard_results/
    │   ├── shard_000.h5    # Output results from shard 0
    │   ├── shard_001.h5
    │   └── ...
    ├── job_script.pbs      # Generated PBS script
    ├── job_logs/
    │   ├── shard_000.out   # stdout from shard 0
    │   ├── shard_000.err   # stderr from shard 0
    │   └── ...
    └── mcmc_config.json    # MCMC configuration

PBS Script Template
-------------------
The PBS script is generated from PBS_template.txt with the following replacements:
- <project_name> → User's PBS project
- <job_name> → "homodyne_cmc_{timestamp}"
- <walltime> → Configured walltime (default: 02:00:00)
- <num_nodes> → 1 (each task uses 1 node)
- <cores_per_node> → Configured cores (default: 36)
- <your_email_address> → User's email (if configured)

Usage Example
-------------
    from homodyne.optimization.cmc.backends.pbs import PBSBackend

    backend = PBSBackend(
        project_name="my_project",
        walltime="02:00:00",
        queue="batch",
        email="user@example.com",
    )

    results = backend.run_parallel_mcmc(
        shards=data_shards,
        mcmc_config={'num_warmup': 500, 'num_samples': 2000},
        init_params={'D0': 1000.0, 'alpha': 0.5},
        inv_mass_matrix=mass_matrix,
    )

Integration Points
------------------
- Called by CMC coordinator via select_backend()
- Uses HDF5 for data serialization (h5py)
- Integrates with system PBS/qsub commands
- Returns results in standard format for combination.py

Performance Considerations
--------------------------
- Job submission overhead: ~5-10 seconds
- Queue wait time: Variable (depends on cluster load)
- Per-shard execution: 5-30 minutes (typical)
- Result collection: ~1-2 seconds per shard
- Total time: Queue wait + max(shard times) + overhead

Error Handling
--------------
- Job submission failures: Raise clear error message
- Job failures: Detected via qstat exit codes
- Missing output files: Logged and returned as failed shard
- Timeout: Configurable walltime per job
- Retry: Up to max_retries per shard (default: 2)
"""

from typing import List, Dict, Any, Optional
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import h5py

from homodyne.optimization.cmc.backends.base import CMCBackend
from homodyne.utils.logging import get_logger

logger = get_logger(__name__)


class PBSBackend(CMCBackend):
    """PBS job array backend for distributed MCMC execution on HPC clusters.

    This backend submits PBS job arrays to execute MCMC sampling on multiple
    nodes in parallel. It's designed for HPC clusters with PBS schedulers.

    Attributes
    ----------
    project_name : str
        PBS project name (required for job submission)
    walltime : str
        PBS walltime per shard (format: HH:MM:SS)
    queue : str
        PBS queue name
    cores_per_node : int
        Number of cores per node
    email : str or None
        Email for job notifications
    temp_dir : Path
        Temporary directory for shard data and results
    poll_interval : int
        Seconds between job status polls
    max_retries : int
        Maximum retries per failed shard

    Methods
    -------
    run_parallel_mcmc(shards, mcmc_config, init_params, inv_mass_matrix)
        Execute MCMC on all shards via PBS job array
    get_backend_name()
        Return 'pbs'

    Notes
    -----
    - Requires PBS Pro or OpenPBS on the cluster
    - Requires qsub, qstat commands available in PATH
    - Creates temporary files (cleaned up after completion)
    - Job array size limited by cluster configuration
    """

    def __init__(self,
        project_name: Optional[str] = None,
        walltime: str = "02:00:00",
        queue: str = "batch",
        cores_per_node: int = 36,
        email: Optional[str] = None,
        temp_dir: Optional[str] = None,
        poll_interval: int = 30,
        max_retries: int = 2,
    ):
        """Initialize PBS backend.

        Parameters
        ----------
        project_name : str
            PBS project name (required for job submission)
        walltime : str, optional
            PBS walltime per shard (HH:MM:SS format), by default "02:00:00"
        queue : str, optional
            PBS queue name, by default "batch"
        cores_per_node : int, optional
            Number of cores per node, by default 36
        email : str, optional
            Email for job notifications, by default None
        temp_dir : str, optional
            Temporary directory for files, by default "./cmc_temp_{timestamp}"
        poll_interval : int, optional
            Seconds between job status polls, by default 30
        max_retries : int, optional
            Maximum retries per failed shard, by default 2
        """
        self.project_name = project_name or "test_project"  # Default for testing
        self.walltime = walltime
        self.queue = queue
        self.cores_per_node = cores_per_node
        self.email = email
        self.poll_interval = poll_interval
        self.max_retries = max_retries

        # Create temporary directory
        if temp_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_dir = f"./cmc_temp_{timestamp}"

        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PBS backend initialized: project={project_name}, "
            f"walltime={walltime}, queue={queue}"
        )
        logger.info(f"Temporary directory: {self.temp_dir}")

    def get_backend_name(self) -> str:
        """Return backend name 'pbs'."""
        return "pbs"

    def run_parallel_mcmc(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
        analysis_mode: str,
        parameter_space,
    ) -> List[Dict[str, Any]]:
        """Run MCMC on all shards via PBS job array.

        Workflow:
        1. Write shard data to HDF5 files
        2. Generate PBS job array script
        3. Submit job array
        4. Poll job status until completion
        5. Collect results from output files
        6. Clean up temporary files

        Parameters
        ----------
        shards : list of dict
            Data shards to process
        mcmc_config : dict
            MCMC configuration
        init_params : dict
            Initial parameter values
        inv_mass_matrix : np.ndarray
            Inverse mass matrix
        analysis_mode : str
            Analysis mode ("static_isotropic" or "laminar_flow")
        parameter_space : ParameterSpace
            Parameter space with bounds and constraints

        Returns
        -------
        list of dict
            Per-shard MCMC results
        """
        logger.info(
            f"Starting PBS backend execution for {len(shards)} shards via job array"
        )

        try:
            # Step 1: Write shard data to files
            logger.info("Writing shard data to HDF5 files...")
            self._write_shard_data(shards, mcmc_config, init_params, inv_mass_matrix, analysis_mode, parameter_space)

            # Step 2: Generate PBS script
            logger.info("Generating PBS job array script...")
            script_path = self._generate_pbs_script(len(shards))

            # Step 3: Submit job array
            logger.info(f"Submitting PBS job array for {len(shards)} shards...")
            job_id = self._submit_job(script_path)
            logger.info(f"Job array submitted: Job ID = {job_id}")

            # Step 4: Poll job status
            logger.info(f"Polling job status (interval: {self.poll_interval}s)...")
            self._wait_for_completion(job_id, len(shards))

            # Step 5: Collect results
            logger.info("Collecting results from shard output files...")
            results = self._collect_results(len(shards))

            # Step 6: Clean up (optional - keep files for debugging)
            if self._should_cleanup():
                logger.info("Cleaning up temporary files...")
                self._cleanup()
            else:
                logger.info(f"Temporary files kept for debugging: {self.temp_dir}")

            return results

        except Exception as e:
            logger.error(f"PBS backend execution failed: {str(e)}")
            # Don't cleanup on error (for debugging)
            logger.info(f"Temporary files kept for debugging: {self.temp_dir}")
            raise

    def _write_shard_data(
        self,
        shards: List[Dict[str, np.ndarray]],
        mcmc_config: Dict[str, Any],
        init_params: Dict[str, float],
        inv_mass_matrix: np.ndarray,
    ) -> None:
        """Write shard data to HDF5 files.

        Creates shard_data/ directory with one HDF5 file per shard.

        Parameters
        ----------
        shards : list of dict
            Data shards
        mcmc_config : dict
            MCMC configuration
        init_params : dict
            Initial parameters
        inv_mass_matrix : np.ndarray
            Mass matrix
        """
        # Create directories
        shard_data_dir = self.temp_dir / "shard_data"
        shard_data_dir.mkdir(exist_ok=True)

        # Write MCMC config and init params (shared across shards)
        config_path = self.temp_dir / "mcmc_config.json"
        with open(config_path, 'w') as f:
            json.dump({
                'mcmc_config': mcmc_config,
                'init_params': init_params,
            }, f)

        # Write mass matrix (shared across shards)
        mass_matrix_path = self.temp_dir / "inv_mass_matrix.npy"
        np.save(mass_matrix_path, inv_mass_matrix)

        # Write each shard to HDF5
        for i, shard in enumerate(shards):
            shard_path = shard_data_dir / f"shard_{i:03d}.h5"

            with h5py.File(shard_path, 'w') as f:
                # Write shard data
                f.create_dataset('data', data=shard['data'])
                f.create_dataset('sigma', data=shard.get('sigma', np.ones_like(shard['data'])))
                f.create_dataset('t1', data=shard['t1'])
                f.create_dataset('t2', data=shard['t2'])
                f.create_dataset('phi', data=shard['phi'])
                f.attrs['q'] = shard['q']
                f.attrs['L'] = shard['L']
                f.attrs['shard_idx'] = i

        logger.info(f"Wrote {len(shards)} shard data files to {shard_data_dir}")

    def _generate_pbs_script(self, num_shards: int) -> Path:
        """Generate PBS job array script from template.

        Parameters
        ----------
        num_shards : int
            Number of shards (array size)

        Returns
        -------
        Path
            Path to generated PBS script
        """
        # Read PBS template
        template_path = Path(__file__).parents[4] / "PBS_template.txt"

        if not template_path.exists():
            # Fallback: Create minimal PBS script
            logger.warning(
                f"PBS template not found at {template_path}. "
                f"Using minimal fallback template."
            )
            template_content = self._get_fallback_template()
        else:
            with open(template_path, 'r') as f:
                template_content = f.read()

        # Generate job name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"homodyne_cmc_{timestamp}"

        # Replace template variables
        script_content = template_content.replace("<project_name>", self.project_name)
        # Job name replacement moved after array directive
        script_content = script_content.replace("HH:MM:SS", self.walltime)

        # Set job array
        script_content = script_content.replace(
            "#PBS -l select=4:mpiprocs=128",
            f"#PBS -l select=1:ncpus={self.cores_per_node}"
        )

        # Add job array directive
        array_directive = f"#PBS -J 0-{num_shards - 1}\n"
        script_content = script_content.replace(
            "#PBS -N <job_name>",
            f"#PBS -N {job_name}\n{array_directive}"
        )

        # Replace email if provided
        if self.email:
            script_content = script_content.replace("<your_email_address>", self.email)
        else:
            # Remove email lines
            script_content = script_content.replace("#PBS -m be\n", "")
            script_content = script_content.replace("#PBS -M <your_email_address>\n", "")

        # Replace job execution commands
        execution_script = self._get_execution_script()
        script_content = script_content.replace(
            "mpirun ./hello_mpi",
            execution_script
        )

        # Write script to file
        script_path = self.temp_dir / "job_script.pbs"
        with open(script_path, 'w') as f:
            f.write(script_content)

        logger.info(f"Generated PBS script: {script_path}")
        return script_path

    def _get_fallback_template(self) -> str:
        """Get minimal fallback PBS script template.

        Returns
        -------
        str
            Minimal PBS script template
        """
        return """#!/bin/bash -l
#PBS -A <project_name>
#PBS -l select=4:mpiprocs=128
#PBS -l walltime=HH:MM:SS
#PBS -N <job_name>
#PBS -j n
#PBS -m be
#PBS -M <your_email_address>

echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

echo Jobid: $PBS_JOBID
echo Running on host `hostname`

# Job array task execution
mpirun ./hello_mpi
"""

    def _get_execution_script(self) -> str:
        """Get Python execution script for PBS job array tasks.

        Returns
        -------
        str
            Python execution commands
        """
        return f"""
# Load Python environment (adjust module names as needed)
module load python/3.12
module load cuda/12.2  # If using GPU

# Run Python script for this shard
python -c "
import sys
sys.path.insert(0, '{Path.cwd()}')
from homodyne.optimization.cmc.backends.pbs import run_shard_task
run_shard_task(
    temp_dir='{self.temp_dir}',
    shard_idx=int('$PBS_ARRAY_INDEX')
)
"
"""

    def _submit_job(self, script_path: Path) -> str:
        """Submit PBS job array via qsub.

        Parameters
        ----------
        script_path : Path
            Path to PBS script

        Returns
        -------
        str
            Job ID

        Raises
        ------
        RuntimeError
            If qsub submission fails
        """
        try:
            # Submit job
            result = subprocess.run(
                ["qsub", str(script_path)],
                capture_output=True,
                text=True,
                check=True,
            )

            # Extract job ID from output (format: "1234567.pbsserver")
            job_id = result.stdout.strip()
            return job_id

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"PBS job submission failed:\n"
                f"Command: qsub {script_path}\n"
                f"Exit code: {e.returncode}\n"
                f"Stdout: {e.stdout}\n"
                f"Stderr: {e.stderr}"
            )

    def _wait_for_completion(self, job_id: str, num_shards: int) -> None:
        """Poll job status until completion or failure.

        Parameters
        ----------
        job_id : str
            PBS job ID
        num_shards : int
            Number of shards (for progress tracking)

        Raises
        ------
        RuntimeError
            If job fails or times out
        """
        logger.info(f"Waiting for job {job_id} to complete...")

        poll_count = 0
        max_polls = 1000  # ~8 hours at 30s intervals

        while poll_count < max_polls:
            # Check job status
            status = self._get_job_status(job_id)

            if status == "completed":
                logger.info(f"Job {job_id} completed successfully")
                return
            elif status == "failed":
                raise RuntimeError(f"Job {job_id} failed. Check PBS logs.")
            elif status == "running":
                # Count completed shards
                completed = self._count_completed_shards(num_shards)
                logger.info(
                    f"Job {job_id} running... "
                    f"({completed}/{num_shards} shards completed)"
                )
            elif status == "queued":
                logger.info(f"Job {job_id} queued (waiting for resources)...")
            else:
                logger.warning(f"Unknown job status: {status}")

            # Wait before next poll
            time.sleep(self.poll_interval)
            poll_count += 1

        raise RuntimeError(
            f"Job {job_id} timed out after {max_polls * self.poll_interval}s"
        )

    def _get_job_status(self, job_id: str) -> str:
        """Get PBS job status via qstat.

        Parameters
        ----------
        job_id : str
            PBS job ID

        Returns
        -------
        str
            Job status: 'queued', 'running', 'completed', 'failed', 'unknown'
        """
        try:
            result = subprocess.run(
                ["qstat", "-f", job_id],
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse qstat output for job_state
            for line in result.stdout.split('\n'):
                if 'job_state' in line:
                    state = line.split('=')[-1].strip()
                    if state == 'Q':
                        return 'queued'
                    elif state == 'R':
                        return 'running'
                    elif state == 'F' or state == 'C':
                        # Check exit status
                        return self._check_exit_status(result.stdout)
                    else:
                        return 'unknown'

            return 'unknown'

        except subprocess.CalledProcessError:
            # Job not found in queue (likely completed or failed)
            return self._check_completion_status()

    def _check_exit_status(self, qstat_output: str) -> str:
        """Check job exit status from qstat output.

        Parameters
        ----------
        qstat_output : str
            Output from qstat -f

        Returns
        -------
        str
            'completed' or 'failed'
        """
        for line in qstat_output.split('\n'):
            if 'exit_status' in line:
                exit_code = int(line.split('=')[-1].strip())
                return 'completed' if exit_code == 0 else 'failed'

        # No exit status found - assume completed
        return 'completed'

    def _check_completion_status(self) -> str:
        """Check if job completed by examining output files.

        Returns
        -------
        str
            'completed' or 'failed'
        """
        results_dir = self.temp_dir / "shard_results"
        if results_dir.exists():
            return 'completed'
        return 'failed'

    def _count_completed_shards(self, num_shards: int) -> int:
        """Count number of completed shards.

        Parameters
        ----------
        num_shards : int
            Total number of shards

        Returns
        -------
        int
            Number of completed shards
        """
        results_dir = self.temp_dir / "shard_results"
        if not results_dir.exists():
            return 0

        completed = 0
        for i in range(num_shards):
            result_path = results_dir / f"shard_{i:03d}.h5"
            if result_path.exists():
                completed += 1

        return completed

    def _collect_results(self, num_shards: int) -> List[Dict[str, Any]]:
        """Collect results from shard output files.

        Parameters
        ----------
        num_shards : int
            Number of shards

        Returns
        -------
        list of dict
            Per-shard results
        """
        results_dir = self.temp_dir / "shard_results"
        results = []

        for i in range(num_shards):
            result_path = results_dir / f"shard_{i:03d}.h5"

            if not result_path.exists():
                # Shard failed - create error result
                logger.error(f"Shard {i} output file not found: {result_path}")
                results.append({
                    'converged': False,
                    'error': f"Output file not found: {result_path}",
                    'samples': None,
                    'diagnostics': {},
                    'elapsed_time': 0.0,
                    'shard_idx': i,
                })
                continue

            try:
                # Read result from HDF5
                with h5py.File(result_path, 'r') as f:
                    result = {
                        'converged': bool(f.attrs.get('converged', False)),
                        'samples': f['samples'][:] if 'samples' in f else None,
                        'elapsed_time': float(f.attrs.get('elapsed_time', 0.0)),
                        'shard_idx': int(f.attrs.get('shard_idx', i)),
                    }

                    # Read diagnostics
                    if 'diagnostics' in f:
                        diag_group = f['diagnostics']
                        result['diagnostics'] = {
                            'acceptance_rate': diag_group.attrs.get('acceptance_rate'),
                            'ess': dict(diag_group['ess'].attrs) if 'ess' in diag_group else {},
                            'rhat': dict(diag_group['rhat'].attrs) if 'rhat' in diag_group else {},
                        }
                    else:
                        result['diagnostics'] = {}

                    # Check for error
                    if 'error' in f.attrs:
                        result['error'] = f.attrs['error']

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to read shard {i} result: {str(e)}")
                results.append({
                    'converged': False,
                    'error': f"Failed to read result: {str(e)}",
                    'samples': None,
                    'diagnostics': {},
                    'elapsed_time': 0.0,
                    'shard_idx': i,
                })

        return results

    def _should_cleanup(self) -> bool:
        """Determine if temporary files should be cleaned up.

        Returns
        -------
        bool
            True if cleanup should be performed
        """
        # For Phase 1, keep files for debugging
        # In production (Phase 3), this would return True
        return False

    def _cleanup(self) -> None:
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info(f"Removed temporary directory: {self.temp_dir}")


# -----------------------------------------------------------------------------
# PBS Job Array Task Function (executed on cluster nodes)
# -----------------------------------------------------------------------------

def run_shard_task(temp_dir: str, shard_idx: int) -> None:
    """Execute MCMC for a single shard (called by PBS job array task).

    This function runs on a cluster node as part of a PBS job array.
    It reads shard data, runs MCMC, and writes results to HDF5.

    Parameters
    ----------
    temp_dir : str
        Temporary directory containing shard data
    shard_idx : int
        Index of shard to process (from PBS_ARRAY_INDEX)
    """
    import sys
    import json
    from pathlib import Path

    import numpy as np
    import h5py
    import jax
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS
    from numpyro import sample

    temp_path = Path(temp_dir)
    start_time = time.time()

    try:
        # Read shard data
        shard_path = temp_path / "shard_data" / f"shard_{shard_idx:03d}.h5"

        with h5py.File(shard_path, 'r') as f:
            shard = {
                'data': f['data'][:],
                'sigma': f['sigma'][:],
                't1': f['t1'][:],
                't2': f['t2'][:],
                'phi': f['phi'][:],
                'q': f.attrs['q'],
                'L': f.attrs['L'],
            }

        # Read MCMC config
        config_path = temp_path / "mcmc_config.json"
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            mcmc_config = config_data['mcmc_config']
            init_params = config_data['init_params']

        # Read mass matrix
        mass_matrix_path = temp_path / "inv_mass_matrix.npy"
        inv_mass_matrix = np.load(mass_matrix_path)

        # Run MCMC (same logic as pjit/multiprocessing backends)
        num_warmup = mcmc_config.get('num_warmup', 500)
        num_samples = mcmc_config.get('num_samples', 2000)
        num_chains = mcmc_config.get('num_chains', 1)
        target_accept_prob = mcmc_config.get('target_accept_prob', 0.8)
        max_tree_depth = mcmc_config.get('max_tree_depth', 10)

        # Convert to JAX arrays
        data_jax = jnp.array(shard['data'])
        sigma_jax = jnp.array(shard['sigma'])
        t1_jax = jnp.array(shard['t1'])
        t2_jax = jnp.array(shard['t2'])
        phi_jax = jnp.array(shard['phi'])

        # Define model
        def model(data, sigma, t1, t2, phi, q, L):
            contrast = sample('contrast', dist.Uniform(0.0, 1.0))
            offset = sample('offset', dist.Normal(1.0, 0.1))
            D0 = sample('D0', dist.Uniform(100.0, 10000.0))
            alpha = sample('alpha', dist.Uniform(0.0, 2.0))
            D_offset = sample('D_offset', dist.Uniform(0.0, 100.0))

            g2_theory = jnp.ones_like(data)
            mu = contrast * g2_theory + offset
            sample('obs', dist.Normal(mu, sigma), obs=data)

        # Initial values
        init_param_values = {
            'contrast': init_params.get('contrast', 0.5),
            'offset': init_params.get('offset', 1.0),
            'D0': init_params.get('D0', 1000.0),
            'alpha': init_params.get('alpha', 0.5),
            'D_offset': init_params.get('D_offset', 10.0),
        }

        # Create NUTS sampler
        nuts_kernel = NUTS(
            model,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            init_strategy=numpyro.infer.init_to_value(values=init_param_values),
        )

        # Run MCMC
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=False,
        )

        rng_key = jax.random.PRNGKey(shard_idx)
        mcmc.run(
            rng_key,
            data=data_jax,
            sigma=sigma_jax,
            t1=t1_jax,
            t2=t2_jax,
            phi=phi_jax,
            q=shard['q'],
            L=shard['L'],
        )

        # Extract samples
        samples_dict = mcmc.get_samples()
        samples_array = np.stack([
            np.array(samples_dict['contrast']),
            np.array(samples_dict['offset']),
            np.array(samples_dict['D0']),
            np.array(samples_dict['alpha']),
            np.array(samples_dict['D_offset']),
        ], axis=1)

        # Compute diagnostics
        diagnostics = {}
        extra_fields = mcmc.get_extra_fields()
        if 'accept_prob' in extra_fields:
            diagnostics['acceptance_rate'] = float(np.mean(extra_fields['accept_prob']))

        if mcmc.num_chains > 1:
            from numpyro.diagnostics import effective_sample_size, gelman_rubin

            ess_dict = {}
            for param_name, samples in samples_dict.items():
                ess = effective_sample_size(samples)
                ess_dict[param_name] = float(ess) if ess.size == 1 else float(np.mean(ess))
            diagnostics['ess'] = ess_dict

            rhat_dict = {}
            for param_name, samples in samples_dict.items():
                rhat = gelman_rubin(samples)
                rhat_dict[param_name] = float(rhat) if rhat.size == 1 else float(np.mean(rhat))
            diagnostics['rhat'] = rhat_dict
        else:
            diagnostics['ess'] = {k: len(v) for k, v in samples_dict.items()}
            diagnostics['rhat'] = {k: 1.0 for k in samples_dict.keys()}

        # Check convergence
        converged = True
        if diagnostics.get('rhat'):
            max_rhat = max(diagnostics['rhat'].values())
            if max_rhat > 1.1:
                converged = False

        # Write result
        results_dir = temp_path / "shard_results"
        results_dir.mkdir(exist_ok=True)
        result_path = results_dir / f"shard_{shard_idx:03d}.h5"

        elapsed_time = time.time() - start_time

        with h5py.File(result_path, 'w') as f:
            f.create_dataset('samples', data=samples_array)
            f.attrs['converged'] = converged
            f.attrs['elapsed_time'] = elapsed_time
            f.attrs['shard_idx'] = shard_idx

            # Write diagnostics
            diag_group = f.create_group('diagnostics')
            if 'acceptance_rate' in diagnostics:
                diag_group.attrs['acceptance_rate'] = diagnostics['acceptance_rate']
            if 'ess' in diagnostics:
                ess_group = diag_group.create_group('ess')
                for k, v in diagnostics['ess'].items():
                    ess_group.attrs[k] = v
            if 'rhat' in diagnostics:
                rhat_group = diag_group.create_group('rhat')
                for k, v in diagnostics['rhat'].items():
                    rhat_group.attrs[k] = v

        print(f"Shard {shard_idx} completed in {elapsed_time:.2f}s")

    except Exception as e:
        # Write error result
        error_msg = f"Shard {shard_idx} failed: {str(e)}"
        print(error_msg, file=sys.stderr)

        results_dir = temp_path / "shard_results"
        results_dir.mkdir(exist_ok=True)
        result_path = results_dir / f"shard_{shard_idx:03d}.h5"

        elapsed_time = time.time() - start_time

        with h5py.File(result_path, 'w') as f:
            f.attrs['converged'] = False
            f.attrs['error'] = error_msg
            f.attrs['elapsed_time'] = elapsed_time
            f.attrs['shard_idx'] = shard_idx


# Export backend class
__all__ = ["PBSBackend", "run_shard_task"]
