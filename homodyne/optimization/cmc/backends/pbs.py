"""PBS (Portable Batch System) backend for CMC HPC cluster execution.

This module provides distributed MCMC execution on HPC clusters
using PBS job scheduling.

Note: This backend requires:
- PBS/Torque job scheduler (qsub, qstat commands)
- Shared filesystem accessible from all nodes
- homodyne installed on compute nodes
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess  # nosec B404
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from homodyne.optimization.cmc.backends.base import CMCBackend, combine_shard_samples
from homodyne.utils.logging import get_logger

if TYPE_CHECKING:
    from homodyne.optimization.cmc.config import CMCConfig
    from homodyne.optimization.cmc.data_prep import PreparedData
    from homodyne.optimization.cmc.sampler import MCMCSamples

logger = get_logger(__name__)

_SAFE_PATH_RE = re.compile(r"^[/a-zA-Z0-9._-]+$")
_PBS_JOB_ID_RE = re.compile(r"^\d+(\.\w+)*$")

# Default PBS job template
PBS_JOB_TEMPLATE = """#!/bin/bash
#PBS -N cmc_shard_{shard_id}
#PBS -l nodes=1:ppn={ppn}
#PBS -l walltime={walltime}
#PBS -l mem={memory}
#PBS -o {output_dir}/shard_{shard_id}.out
#PBS -e {output_dir}/shard_{shard_id}.err
#PBS -q {queue}

cd $PBS_O_WORKDIR

# Activate environment if specified
{activate_env}

# Run shard worker
python -m homodyne.optimization.cmc.backends.pbs_worker \\
    --shard-file {shard_file} \\
    --config-file {config_file} \\
    --output-file {result_file}
"""


class PBSBackend(CMCBackend):
    """PBS backend for HPC cluster MCMC execution.

    Submits each data shard as a separate PBS job and combines
    results after all jobs complete.

    Parameters
    ----------
    queue : str
        PBS queue name (default: "batch").
    ppn : int
        Processors per node (default: 4).
    walltime : str
        Job walltime (default: "04:00:00").
    memory : str
        Memory per job (default: "8gb").
    poll_interval : int
        Seconds between job status checks (default: 30).
    max_wait_time : int
        Maximum wait time in seconds (default: 14400 = 4 hours).
    """

    def __init__(
        self,
        queue: str = "batch",
        ppn: int = 4,
        walltime: str = "04:00:00",
        memory: str = "8gb",
        poll_interval: int = 30,
        max_wait_time: int = 14400,
    ) -> None:
        """Initialize PBS backend."""
        self.queue = queue
        self.ppn = ppn
        self.walltime = walltime
        self.memory = memory
        self.poll_interval = poll_interval
        self.max_wait_time = max_wait_time

        self._validate_pbs_available()

    def _validate_pbs_available(self) -> None:
        """Check if PBS commands are available."""
        try:
            qstat_path = shutil.which("qstat")
            if not qstat_path:
                raise FileNotFoundError("qstat not found")
            subprocess.run(  # noqa: S603 - qstat_path from shutil.which is trusted
                [qstat_path, "--version"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            logger.info("PBSBackend: PBS scheduler detected")
        except FileNotFoundError:
            logger.warning(
                "PBSBackend: qstat not found. PBS commands may not be available."
            )
        except subprocess.TimeoutExpired:
            logger.warning("PBSBackend: qstat timed out")

    def get_name(self) -> str:
        """Get backend name.

        Returns
        -------
        str
            Backend identifier.
        """
        return "pbs"

    def is_available(self) -> bool:
        """Check if PBS backend is available.

        Returns
        -------
        bool
            True if PBS commands are accessible.
        """
        try:
            qsub_path = shutil.which("qsub")
            if not qsub_path:
                return False
            result = subprocess.run(  # noqa: S603 - qsub_path from shutil.which is trusted
                [qsub_path],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def run(
        self,
        model: Callable,
        model_kwargs: dict[str, Any],
        config: CMCConfig,
        shards: list[PreparedData] | None = None,
    ) -> MCMCSamples:
        """Run MCMC sampling via PBS job submission.

        Parameters
        ----------
        model : Callable
            NumPyro model function (not directly used - workers import it).
        model_kwargs : dict[str, Any]
            Common model arguments.
        config : CMCConfig
            CMC configuration.
        shards : list[PreparedData] | None
            Data shards for parallel execution.

        Returns
        -------
        MCMCSamples
            Combined samples from all PBS jobs.

        Raises
        ------
        RuntimeError
            If jobs fail or timeout.
        """
        if shards is None or len(shards) == 0:
            raise ValueError("PBSBackend requires sharded data")

        logger.info(f"PBSBackend: Submitting {len(shards)} PBS jobs")

        # Create temporary directory for job files
        with tempfile.TemporaryDirectory(prefix="cmc_pbs_") as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Save config
            config_file = tmpdir_path / "config.json"
            self._save_config(config, model_kwargs, config_file)

            # Submit jobs for each shard
            job_ids: list[str] = []
            result_files: list[Path] = []

            for i, shard in enumerate(shards):
                shard_file = tmpdir_path / f"shard_{i}.npz"
                result_file = tmpdir_path / f"result_{i}.npz"

                # Save shard data
                self._save_shard(shard, shard_file)
                result_files.append(result_file)

                # Submit PBS job
                job_id = self._submit_job(
                    shard_id=i,
                    shard_file=shard_file,
                    config_file=config_file,
                    result_file=result_file,
                    output_dir=tmpdir_path,
                )
                job_ids.append(job_id)
                logger.info(f"Submitted shard {i} as job {job_id}")

            # Wait for all jobs to complete
            self._wait_for_jobs(job_ids)

            # Load and combine results
            shard_results = self._load_results(result_files)

        # Combine samples
        combined = combine_shard_samples(
            shard_results,
            method=(
                config.combination_method
                if hasattr(config, "combination_method")
                else "weighted_gaussian"
            ),
        )

        logger.info("PBSBackend: All jobs completed successfully")
        return combined

    def _save_config(
        self,
        config: CMCConfig,
        model_kwargs: dict[str, Any],
        path: Path,
    ) -> None:
        """Save configuration for PBS workers."""
        # Convert config to serializable dict
        config_dict = {
            "num_warmup": config.num_warmup,
            "num_samples": config.num_samples,
            "num_chains": config.num_chains,
            "target_accept_prob": config.target_accept_prob,
            "max_tree_depth": getattr(config, "max_tree_depth", 10),
        }

        # Add model kwargs (excluding non-serializable)
        serializable_kwargs = {}
        for key, value in model_kwargs.items():
            if isinstance(value, (int, float, str, bool, list, dict)):
                serializable_kwargs[key] = value
            elif isinstance(value, np.ndarray):
                # Skip arrays - they're in shard files
                pass

        combined = {
            "config": config_dict,
            "model_kwargs": serializable_kwargs,
        }

        with open(path, "w") as f:
            json.dump(combined, f, indent=2)

    def _save_shard(self, shard: PreparedData, path: Path) -> None:
        """Save shard data for PBS worker."""
        np.savez_compressed(
            path,
            data=shard.data,
            t1=shard.t1,
            t2=shard.t2,
            phi=shard.phi,
            phi_indices=shard.phi_indices,
            phi_unique=shard.phi_unique,
            n_phi=shard.n_phi,
            noise_scale=shard.noise_scale,
        )

    def _submit_job(
        self,
        shard_id: int,
        shard_file: Path,
        config_file: Path,
        result_file: Path,
        output_dir: Path,
    ) -> str:
        """Submit a PBS job for one shard.

        Returns
        -------
        str
            PBS job ID.
        """
        # Check for conda/venv activation (validate paths for shell safety)
        activate_env = ""
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        virtual_env = os.environ.get("VIRTUAL_ENV", "")
        if conda_prefix:
            if not _SAFE_PATH_RE.match(conda_prefix):
                raise ValueError(f"Unsafe characters in CONDA_PREFIX: {conda_prefix!r}")
            activate_env = f"source activate {conda_prefix}"
        elif virtual_env:
            if not _SAFE_PATH_RE.match(virtual_env):
                raise ValueError(f"Unsafe characters in VIRTUAL_ENV: {virtual_env!r}")
            activate_env = f"source {virtual_env}/bin/activate"

        # Generate job script
        job_script = PBS_JOB_TEMPLATE.format(
            shard_id=shard_id,
            ppn=self.ppn,
            walltime=self.walltime,
            memory=self.memory,
            queue=self.queue,
            output_dir=output_dir,
            activate_env=activate_env,
            shard_file=shard_file,
            config_file=config_file,
            result_file=result_file,
        )

        # Write job script
        script_file = output_dir / f"job_{shard_id}.pbs"
        with open(script_file, "w") as f:
            f.write(job_script)

        # Submit job
        qsub_path = shutil.which("qsub") or "qsub"
        result = subprocess.run(  # noqa: S603 - qsub_path from shutil.which is trusted
            [qsub_path, str(script_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip()
        if not _PBS_JOB_ID_RE.match(job_id):
            raise ValueError(f"Unexpected PBS job ID format: {job_id!r}")
        return job_id

    def _wait_for_jobs(self, job_ids: list[str]) -> None:
        """Wait for all PBS jobs to complete.

        Raises
        ------
        RuntimeError
            If jobs fail or timeout.
        """
        start_time = time.time()
        pending_jobs = set(job_ids)

        while pending_jobs:
            elapsed = time.time() - start_time
            if elapsed > self.max_wait_time:
                raise RuntimeError(
                    f"PBS jobs timed out after {self.max_wait_time}s. "
                    f"Remaining jobs: {pending_jobs}"
                )

            # Check job status
            completed = set()
            for job_id in pending_jobs:
                status = self._get_job_status(job_id)
                if status == "C":  # Completed
                    completed.add(job_id)
                    logger.debug(f"Job {job_id} completed")
                elif status == "E":  # Exiting: PBS/Torque epilogue/cleanup (normal)
                    # P1-R5-02: PBS "E" means "Exiting" (job completing normally),
                    # NOT "Error". Every successful PBS job passes through "E" state
                    # during its teardown/epilogue phase. Treating it as failure
                    # aborts every successful CMC job. Wait for the job to leave
                    # qstat entirely (returncode != 0 in _get_job_status -> "C").
                    pass

            pending_jobs -= completed

            if pending_jobs:
                logger.info(
                    f"Waiting for {len(pending_jobs)} jobs... ({int(elapsed)}s elapsed)"
                )
                time.sleep(self.poll_interval)

    def _get_job_status(self, job_id: str) -> str:
        """Get PBS job status.

        Returns
        -------
        str
            Job status per PBS/Torque conventions:
            Q = queued, R = running, E = exiting (epilogue/cleanup, normal
            completion phase), C = completed (returned when job no longer
            appears in qstat output). Note: there is no "error" job_state in
            standard PBS/Torque; failures are detected via exit_status in
            qstat -f output, not via job_state.
        """
        try:
            qstat_path = shutil.which("qstat") or "qstat"
            result = subprocess.run(  # noqa: S603 - qsub_path from shutil.which is trusted
                [qstat_path, "-f", job_id],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

            if result.returncode != 0:
                # Job no longer in queue - assume completed
                return "C"

            # Parse status and exit_status from output
            state = None
            exit_status = None
            for line in result.stdout.split("\n"):
                stripped = line.strip()
                if "job_state" in stripped:
                    state = stripped.split("=")[-1].strip()
                if "exit_status" in stripped:
                    try:
                        exit_status = int(stripped.split("=")[-1].strip())
                    except ValueError:
                        pass

            if state is None:
                return "C"  # Assume completed if can't parse

            if state == "C" and exit_status is not None and exit_status != 0:
                logger.warning(f"PBS job {job_id} completed with non-zero exit_status={exit_status}")

            return state

        except subprocess.TimeoutExpired:
            logger.warning(f"qstat timeout for job {job_id}")
            return "R"  # Assume still running

    def _load_results(self, result_files: list[Path]) -> list[MCMCSamples]:
        """Load results from completed PBS jobs."""
        from homodyne.optimization.cmc.sampler import MCMCSamples

        results = []
        for path in result_files:
            if not path.exists():
                raise RuntimeError(f"Result file not found: {path}")

            data = np.load(path, allow_pickle=False)

            # Reconstruct samples dict from prefixed arrays
            param_names = list(data["param_names"])
            samples_dict = {
                name: data[f"sample_{name}"]
                for name in param_names
                if f"sample_{name}" in data
            }
            # Fallback for old format with single "samples" key
            if not samples_dict and "samples" in data:
                samples_arr = data["samples"]
                samples_dict = {
                    name: samples_arr[..., i] for i, name in enumerate(param_names)
                }

            # Reconstruct extra_fields from prefixed arrays
            extra_fields: dict[str, Any] = {}
            for key in data.files:
                if key.startswith("extra_"):
                    field_name = key[6:]  # Remove "extra_" prefix
                    extra_fields[field_name] = data[key]

            # Derive actual chain/sample counts from array shape (not serialized metadata)
            first_param = next(iter(samples_dict.values()), None)
            actual_n_chains = first_param.shape[0] if first_param is not None else int(data["n_chains"])
            actual_n_samples = first_param.shape[1] if first_param is not None and first_param.ndim >= 2 else int(data["n_samples"])

            samples = MCMCSamples(
                samples=samples_dict,
                param_names=param_names,
                n_chains=actual_n_chains,
                n_samples=actual_n_samples,
                extra_fields=extra_fields,
            )
            results.append(samples)

        return results
