"""
Distributed Computing Logging Support for Homodyne v2
=====================================================

Specialized logging utilities for distributed and parallel computing environments
in scientific workflows. Supports:

- Multi-process logging coordination with process-safe file handling
- HPC cluster integration with node-aware logging
- Resource usage monitoring across compute nodes
- MPI-aware logging for distributed scientific computing
- SLURM/PBS job integration with job-aware log organization
- Cross-node performance monitoring and aggregation
- Distributed error tracking and recovery logging

This module extends the base logging system with distributed computing
capabilities while maintaining compatibility with single-node operations.
"""

import functools
import json
import logging
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

# System monitoring
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# MPI support
try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

# Distributed computing libraries
try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False
    ray = None

try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    dask = None

from .logging import LoggerManager, get_logger


@dataclass
class NodeInfo:
    """Information about a compute node."""

    hostname: str
    ip_address: str
    process_id: int
    cpu_count: int
    memory_total_gb: float
    node_rank: Optional[int] = None  # For MPI
    job_id: Optional[str] = None  # For SLURM/PBS
    partition: Optional[str] = None  # For SLURM

    @classmethod
    def get_current_node(cls) -> "NodeInfo":
        """Get information about the current compute node."""
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "unknown"

        process_id = os.getpid()
        cpu_count = mp.cpu_count()

        # Memory info
        memory_total_gb = 0
        if HAS_PSUTIL:
            try:
                memory_total_gb = psutil.virtual_memory().total / 1024**3
            except:
                pass

        # MPI rank if available
        node_rank = None
        if HAS_MPI:
            try:
                comm = MPI.COMM_WORLD
                node_rank = comm.Get_rank()
            except:
                pass

        # Job scheduler info
        job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("PBS_JOBID")
        partition = os.environ.get("SLURM_JOB_PARTITION")

        return cls(
            hostname=hostname,
            ip_address=ip_address,
            process_id=process_id,
            cpu_count=cpu_count,
            memory_total_gb=memory_total_gb,
            node_rank=node_rank,
            job_id=job_id,
            partition=partition,
        )


@dataclass
class ResourceUsageSnapshot:
    """Snapshot of resource usage across nodes."""

    timestamp: float
    node_hostname: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: Optional[float] = None
    network_io_mbps: Optional[Tuple[float, float]] = None  # (sent, received)
    gpu_usage: Optional[Dict[str, float]] = None  # GPU utilization if available
    process_count: Optional[int] = None
    load_average: Optional[Tuple[float, float, float]] = None  # 1m, 5m, 15m

    @classmethod
    def take_snapshot(
        cls, node_hostname: Optional[str] = None
    ) -> "ResourceUsageSnapshot":
        """Take a resource usage snapshot."""
        if node_hostname is None:
            node_hostname = socket.gethostname()

        snapshot = cls(
            timestamp=time.time(),
            node_hostname=node_hostname,
            cpu_percent=0.0,
            memory_percent=0.0,
            memory_used_gb=0.0,
        )

        if HAS_PSUTIL:
            try:
                # CPU usage
                snapshot.cpu_percent = psutil.cpu_percent(interval=0.1)

                # Memory usage
                memory = psutil.virtual_memory()
                snapshot.memory_percent = memory.percent
                snapshot.memory_used_gb = memory.used / 1024**3

                # Disk usage
                try:
                    disk = psutil.disk_usage("/")
                    snapshot.disk_usage_percent = (disk.used / disk.total) * 100
                except:
                    pass

                # Network I/O
                try:
                    net_io = psutil.net_io_counters()
                    # Simple rate calculation (not accurate for single snapshot)
                    snapshot.network_io_mbps = (
                        net_io.bytes_sent / 1024**2,
                        net_io.bytes_recv / 1024**2,
                    )
                except:
                    pass

                # Process count
                try:
                    snapshot.process_count = len(psutil.pids())
                except:
                    pass

                # Load average (Unix only)
                try:
                    snapshot.load_average = os.getloadavg()
                except:
                    pass

            except Exception as e:
                logger = get_logger(__name__)
                logger.debug(f"Error taking resource snapshot: {e}")

        return snapshot


class DistributedLoggerManager:
    """Manager for distributed logging across multiple processes/nodes."""

    def __init__(self, base_log_dir: Optional[Path] = None):
        self.base_log_dir = (
            base_log_dir or Path.home() / ".homodyne" / "distributed_logs"
        )
        self.node_info = NodeInfo.get_current_node()
        self._resource_snapshots = []
        self._lock = threading.Lock()

        # Create node-specific log directory
        self.node_log_dir = self._setup_node_log_directory()

        # Initialize base logger manager with distributed settings
        self._logger_manager = LoggerManager()
        self._configure_distributed_logging()

    def _setup_node_log_directory(self) -> Path:
        """Set up node-specific logging directory."""
        # Create hierarchical log structure
        if self.node_info.job_id:
            job_dir = self.base_log_dir / f"job_{self.node_info.job_id}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_dir = self.base_log_dir / f"session_{timestamp}"

        node_dir = job_dir / f"node_{self.node_info.hostname}"
        if self.node_info.node_rank is not None:
            node_dir = node_dir / f"rank_{self.node_info.node_rank}"

        node_dir.mkdir(parents=True, exist_ok=True)

        # Create symlink for easy access to current session
        current_link = self.base_log_dir / "current"
        if current_link.is_symlink():
            current_link.unlink()
        try:
            current_link.symlink_to(job_dir)
        except:
            pass  # Symlink creation might fail on some systems

        return node_dir

    def _configure_distributed_logging(self):
        """Configure logging for distributed environment."""

        # Node-specific log file
        node_log_file = self.node_log_dir / f"homodyne_node.log"

        # Process-specific log file
        process_log_file = (
            self.node_log_dir / f"homodyne_pid_{self.node_info.process_id}.log"
        )

        # Configure with process-safe file handling
        config = {
            "level": "INFO",
            "console_output": True,
            "file_output": True,
            "log_dir": str(self.node_log_dir),
            "max_file_size": 50 * 1024 * 1024,  # 50MB per process
            "backup_count": 10,
        }

        self._logger_manager.configure(**config)

        # Add distributed context to all loggers
        self._add_distributed_context()

    def _add_distributed_context(self):
        """Add distributed computing context to log records."""

        class DistributedContextFilter(logging.Filter):
            def __init__(self, node_info: NodeInfo):
                super().__init__()
                self.node_info = node_info

            def filter(self, record):
                # Add node context
                record.hostname = self.node_info.hostname
                record.process_id = self.node_info.process_id
                if self.node_info.node_rank is not None:
                    record.mpi_rank = self.node_info.node_rank
                if self.node_info.job_id:
                    record.job_id = self.node_info.job_id

                return True

        # Add filter to root logger
        root_logger = logging.getLogger("homodyne")
        distributed_filter = DistributedContextFilter(self.node_info)
        root_logger.addFilter(distributed_filter)

    def get_distributed_logger(self, name: str) -> logging.Logger:
        """Get a logger configured for distributed computing."""
        return self._logger_manager.get_logger(name)

    def take_resource_snapshot(self, operation_context: Optional[str] = None):
        """Take a resource usage snapshot."""
        snapshot = ResourceUsageSnapshot.take_snapshot(self.node_info.hostname)
        if operation_context:
            # Store context in a custom attribute
            setattr(snapshot, "operation_context", operation_context)

        with self._lock:
            self._resource_snapshots.append(snapshot)
            # Keep only recent snapshots
            if len(self._resource_snapshots) > 1000:
                self._resource_snapshots = self._resource_snapshots[-500:]

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary for this node."""
        with self._lock:
            if not self._resource_snapshots:
                return {"status": "no_data"}

            recent_snapshots = self._resource_snapshots[-10:]  # Last 10 snapshots

            return {
                "node_hostname": self.node_info.hostname,
                "node_rank": self.node_info.node_rank,
                "job_id": self.node_info.job_id,
                "current_cpu_percent": recent_snapshots[-1].cpu_percent,
                "current_memory_percent": recent_snapshots[-1].memory_percent,
                "current_memory_used_gb": recent_snapshots[-1].memory_used_gb,
                "avg_cpu_percent": sum(s.cpu_percent for s in recent_snapshots)
                / len(recent_snapshots),
                "avg_memory_percent": sum(s.memory_percent for s in recent_snapshots)
                / len(recent_snapshots),
                "peak_cpu_percent": max(s.cpu_percent for s in recent_snapshots),
                "peak_memory_percent": max(s.memory_percent for s in recent_snapshots),
                "snapshot_count": len(self._resource_snapshots),
            }

    def export_resource_data(self, filepath: Optional[Path] = None) -> str:
        """Export resource usage data to JSON."""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.node_log_dir / f"resource_usage_{timestamp}.json"

        data = {
            "node_info": {
                "hostname": self.node_info.hostname,
                "ip_address": self.node_info.ip_address,
                "process_id": self.node_info.process_id,
                "cpu_count": self.node_info.cpu_count,
                "memory_total_gb": self.node_info.memory_total_gb,
                "node_rank": self.node_info.node_rank,
                "job_id": self.node_info.job_id,
                "partition": self.node_info.partition,
            },
            "resource_snapshots": [],
        }

        with self._lock:
            for snapshot in self._resource_snapshots:
                snapshot_data = {
                    "timestamp": snapshot.timestamp,
                    "cpu_percent": snapshot.cpu_percent,
                    "memory_percent": snapshot.memory_percent,
                    "memory_used_gb": snapshot.memory_used_gb,
                    "disk_usage_percent": snapshot.disk_usage_percent,
                    "network_io_mbps": snapshot.network_io_mbps,
                    "process_count": snapshot.process_count,
                    "load_average": snapshot.load_average,
                }

                # Add operation context if available
                if hasattr(snapshot, "operation_context"):
                    snapshot_data["operation_context"] = snapshot.operation_context

                data["resource_snapshots"].append(snapshot_data)

        # Export to JSON
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        return str(filepath)


# Global distributed logger manager
_distributed_manager = None


def get_distributed_logger_manager() -> DistributedLoggerManager:
    """Get or create the global distributed logger manager."""
    global _distributed_manager
    if _distributed_manager is None:
        _distributed_manager = DistributedLoggerManager()
    return _distributed_manager


def get_distributed_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a distributed-aware logger."""
    manager = get_distributed_logger_manager()
    if name is None:
        # Auto-discover caller's module
        import inspect

        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back
            if caller_frame:
                module_name = caller_frame.f_globals.get("__name__", "unknown")
                name = module_name
        finally:
            del frame

    return manager.get_distributed_logger(name or "unknown")


@contextmanager
def distributed_operation_context(
    operation_name: str,
    logger: Optional = None,
    monitor_resources: bool = True,
    resource_snapshot_interval: float = 10.0,
):
    """Context manager for distributed operations with resource monitoring."""
    if logger is None:
        logger = get_distributed_logger()

    manager = get_distributed_logger_manager()
    start_time = time.perf_counter()

    # Log operation start with node context
    node_info = manager.node_info
    start_msg = f"Distributed operation started: {operation_name}"
    if node_info.node_rank is not None:
        start_msg += f" [Rank {node_info.node_rank}]"
    start_msg += f" [Node: {node_info.hostname}]"

    logger.info(start_msg)

    # Take initial resource snapshot
    if monitor_resources:
        manager.take_resource_snapshot(f"start_{operation_name}")

    # Resource monitoring thread
    monitoring_thread = None
    stop_monitoring = threading.Event()

    if monitor_resources and resource_snapshot_interval > 0:

        def resource_monitor():
            while not stop_monitoring.wait(resource_snapshot_interval):
                manager.take_resource_snapshot(f"during_{operation_name}")

        monitoring_thread = threading.Thread(target=resource_monitor, daemon=True)
        monitoring_thread.start()

    try:
        yield logger

        # Success logging
        end_time = time.perf_counter()
        duration = end_time - start_time

        success_msg = (
            f"Distributed operation completed: {operation_name} [{duration:.2f}s]"
        )
        if node_info.node_rank is not None:
            success_msg += f" [Rank {node_info.node_rank}]"

        # Add resource summary
        if monitor_resources:
            manager.take_resource_snapshot(f"end_{operation_name}")
            resource_summary = manager.get_resource_summary()
            success_msg += f" [CPU: {resource_summary['current_cpu_percent']:.1f}%, Memory: {resource_summary['current_memory_percent']:.1f}%]"

        logger.info(success_msg)

    except Exception as e:
        # Error logging
        end_time = time.perf_counter()
        duration = end_time - start_time

        error_msg = (
            f"Distributed operation failed: {operation_name} [{duration:.2f}s]: {e}"
        )
        if node_info.node_rank is not None:
            error_msg += f" [Rank {node_info.node_rank}]"

        logger.error(error_msg)

        if monitor_resources:
            manager.take_resource_snapshot(f"error_{operation_name}")

        raise

    finally:
        # Stop resource monitoring
        if monitoring_thread:
            stop_monitoring.set()
            monitoring_thread.join(timeout=1.0)


def log_mpi_operation(logger: Optional = None, barrier_sync: bool = True):
    """Decorator for MPI operations with synchronization."""

    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_distributed_logger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"

            # MPI info
            rank = 0
            size = 1
            if HAS_MPI:
                try:
                    comm = MPI.COMM_WORLD
                    rank = comm.Get_rank()
                    size = comm.Get_size()
                except:
                    pass

            logger.debug(f"MPI operation: {func_name} [Rank {rank}/{size}]")

            # Barrier synchronization before operation
            if barrier_sync and HAS_MPI and size > 1:
                try:
                    comm = MPI.COMM_WORLD
                    comm.Barrier()
                    logger.debug(f"MPI barrier sync complete for {func_name}")
                except:
                    logger.warning(f"MPI barrier sync failed for {func_name}")

            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)

                duration = time.perf_counter() - start_time
                logger.info(
                    f"MPI operation completed: {func_name} [{duration:.3f}s] [Rank {rank}]"
                )

                return result

            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"MPI operation failed: {func_name} [{duration:.3f}s] [Rank {rank}]: {e}"
                )
                raise

        return wrapper

    return decorator


def aggregate_distributed_logs(
    job_log_dir: Path, output_file: Optional[Path] = None
) -> Path:
    """Aggregate logs from all nodes in a distributed job."""
    if output_file is None:
        output_file = job_log_dir / "aggregated_logs.txt"

    logger = get_distributed_logger(__name__)

    # Find all node log files
    node_log_files = []
    for node_dir in job_log_dir.glob("node_*"):
        if node_dir.is_dir():
            for log_file in node_dir.glob("*.log"):
                node_log_files.append((node_dir.name, log_file))

    # Sort by node name and timestamp
    node_log_files.sort()

    # Aggregate logs
    with open(output_file, "w") as outf:
        outf.write(f"Aggregated Homodyne Distributed Logs\n")
        outf.write(f"Generated: {datetime.now().isoformat()}\n")
        outf.write(f"Job Directory: {job_log_dir}\n")
        outf.write(f"Nodes Found: {len(set(node for node, _ in node_log_files))}\n")
        outf.write("=" * 80 + "\n\n")

        for node_name, log_file in node_log_files:
            outf.write(f"\n{'=' * 20} {node_name} - {log_file.name} {'=' * 20}\n")
            try:
                with open(log_file, "r") as inf:
                    outf.write(inf.read())
            except Exception as e:
                outf.write(f"Error reading {log_file}: {e}\n")
            outf.write(f"\n{'=' * 60}\n")

    logger.info(f"Distributed logs aggregated to: {output_file}")
    return output_file


def get_distributed_computing_stats() -> Dict[str, Any]:
    """Get comprehensive distributed computing statistics."""
    manager = get_distributed_logger_manager()

    stats = {
        "node_info": {
            "hostname": manager.node_info.hostname,
            "ip_address": manager.node_info.ip_address,
            "process_id": manager.node_info.process_id,
            "cpu_count": manager.node_info.cpu_count,
            "memory_total_gb": manager.node_info.memory_total_gb,
            "node_rank": manager.node_info.node_rank,
            "job_id": manager.node_info.job_id,
            "partition": manager.node_info.partition,
        },
        "resource_summary": manager.get_resource_summary(),
        "capabilities": {
            "has_mpi": HAS_MPI,
            "has_ray": HAS_RAY,
            "has_dask": HAS_DASK,
            "has_psutil": HAS_PSUTIL,
        },
        "log_directory": str(manager.node_log_dir),
    }

    # MPI-specific stats
    if HAS_MPI:
        try:
            comm = MPI.COMM_WORLD
            stats["mpi_info"] = {
                "rank": comm.Get_rank(),
                "size": comm.Get_size(),
                "processor_name": MPI.Get_processor_name(),
            }
        except:
            stats["mpi_info"] = {"error": "MPI not initialized"}

    return stats


# Example integration functions for common HPC schedulers
def get_slurm_job_info() -> Dict[str, Any]:
    """Get SLURM job information from environment."""
    slurm_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_SUBMIT_DIR",
        "SLURM_NODELIST",
    ]

    return {var: os.environ.get(var) for var in slurm_vars if var in os.environ}


def get_pbs_job_info() -> Dict[str, Any]:
    """Get PBS job information from environment."""
    pbs_vars = [
        "PBS_JOBID",
        "PBS_JOBNAME",
        "PBS_QUEUE",
        "PBS_NUM_NODES",
        "PBS_NP",
        "PBS_O_WORKDIR",
        "PBS_NODEFILE",
    ]

    return {var: os.environ.get(var) for var in pbs_vars if var in os.environ}
