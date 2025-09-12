"""
Advanced Validation Modules for Homodyne v2
==========================================

Specialized validation modules for complex scenarios including:
- HPC cluster configurations (PBS, SLURM)
- GPU hardware detection and validation
- Large dataset processing scenarios
- Batch processing workflows
- Complex phi angle filtering configurations

These validators extend the core validation system with production-ready
capabilities for scientific computing environments.
"""

import os
import re
import shutil
import subprocess
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import psutil
import numpy as np

from homodyne.utils.logging import get_logger

logger = get_logger(__name__)

# Optional GPU libraries
try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


@dataclass
class HPCValidationResult:
    """
    Result of HPC configuration validation.
    
    Attributes:
        is_valid: Whether HPC configuration is valid
        scheduler_type: Detected job scheduler (PBS, SLURM, etc.)
        estimated_walltime: Estimated job walltime
        resource_efficiency: Efficiency score (0-1)
        warnings: List of configuration warnings
        recommendations: List of optimization recommendations
    """
    is_valid: bool
    scheduler_type: Optional[str]
    estimated_walltime: Optional[str]
    resource_efficiency: float
    warnings: List[str]
    recommendations: List[str]


@dataclass
class GPUValidationResult:
    """
    Result of GPU hardware validation.
    
    Attributes:
        gpu_available: Whether GPUs are detected
        gpu_compatible: Whether GPUs are compatible with configuration
        total_memory_gb: Total GPU memory available
        recommended_memory_fraction: Recommended memory fraction
        warnings: List of GPU-related warnings
        optimization_suggestions: List of GPU optimization suggestions
    """
    gpu_available: bool
    gpu_compatible: bool
    total_memory_gb: float
    recommended_memory_fraction: float
    warnings: List[str]
    optimization_suggestions: List[str]


class HPCValidator:
    """
    Validator for HPC cluster configurations.
    
    Validates PBS Professional, SLURM, and other job scheduler configurations
    with focus on resource allocation efficiency and job success probability.
    """
    
    def __init__(self):
        """Initialize HPC validator."""
        self.scheduler_detection_commands = {
            'pbs': ['qstat', 'pbsnodes', 'qsub'],
            'slurm': ['squeue', 'sinfo', 'sbatch'],
            'lsf': ['bjobs', 'bqueues', 'bsub'],
            'sge': ['qstat', 'qhost', 'qsub']
        }
        
        # Common resource limits by scheduler
        self.typical_limits = {
            'pbs': {
                'max_walltime_hours': 168,  # 1 week
                'max_cores_per_job': 1024,
                'max_memory_per_core_gb': 8,
                'typical_queue_wait_hours': 2
            },
            'slurm': {
                'max_walltime_hours': 168,
                'max_cores_per_job': 2048,
                'max_memory_per_core_gb': 16,
                'typical_queue_wait_hours': 1
            }
        }
    
    def validate_hpc_configuration(self, hpc_config: Dict[str, Any]) -> HPCValidationResult:
        """
        Validate complete HPC configuration.
        
        Args:
            hpc_config: HPC configuration dictionary
            
        Returns:
            Comprehensive HPC validation result
        """
        result = HPCValidationResult(
            is_valid=True,
            scheduler_type=None,
            estimated_walltime=None,
            resource_efficiency=1.0,
            warnings=[],
            recommendations=[]
        )
        
        # Detect available job schedulers
        detected_scheduler = self._detect_job_scheduler()
        result.scheduler_type = detected_scheduler
        
        if not detected_scheduler:
            result.warnings.append("No job scheduler detected - running in local mode")
            return result
        
        # Validate scheduler-specific configuration
        if detected_scheduler in hpc_config:
            scheduler_config = hpc_config[detected_scheduler]
            
            if detected_scheduler == 'pbs':
                self._validate_pbs_configuration(scheduler_config, result)
            elif detected_scheduler == 'slurm':
                self._validate_slurm_configuration(scheduler_config, result)
            else:
                result.warnings.append(f"Scheduler '{detected_scheduler}' detected but not fully supported")
        else:
            result.warnings.append(f"Scheduler '{detected_scheduler}' detected but not configured")
        
        # Validate distributed computing settings
        if 'distributed' in hpc_config:
            self._validate_distributed_settings(hpc_config['distributed'], result)
        
        # Calculate resource efficiency
        self._calculate_resource_efficiency(hpc_config, result)
        
        return result
    
    def _detect_job_scheduler(self) -> Optional[str]:
        """
        Detect available job scheduler on the system.
        
        Returns:
            Scheduler name or None if not detected
        """
        for scheduler, commands in self.scheduler_detection_commands.items():
            for command in commands:
                if shutil.which(command):
                    logger.debug(f"Detected {scheduler.upper()} scheduler (command: {command})")
                    return scheduler
        return None
    
    def _validate_pbs_configuration(self, pbs_config: Dict[str, Any], result: HPCValidationResult) -> None:
        """
        Validate PBS Professional configuration.
        
        Args:
            pbs_config: PBS configuration
            result: Validation result to update
        """
        # Extract PBS parameters
        nodes = pbs_config.get('nodes', 1)
        ppn = pbs_config.get('ppn', 1)
        mem = pbs_config.get('mem', '4gb')
        walltime = pbs_config.get('walltime', '24:00:00')
        queue = pbs_config.get('queue', 'normal')
        
        total_cores = nodes * ppn
        
        # Validate resource requests
        limits = self.typical_limits.get('pbs', {})
        
        if total_cores > limits.get('max_cores_per_job', 1024):
            result.warnings.append(
                f"Requesting {total_cores} cores may exceed typical PBS job limits "
                f"({limits['max_cores_per_job']})"
            )
            result.resource_efficiency *= 0.8
        
        # Parse and validate memory
        try:
            mem_value, mem_unit = self._parse_memory_spec(mem)
            mem_gb_per_core = self._convert_to_gb(mem_value, mem_unit) / total_cores
            
            max_mem_per_core = limits.get('max_memory_per_core_gb', 8)
            if mem_gb_per_core > max_mem_per_core:
                result.warnings.append(
                    f"High memory per core ({mem_gb_per_core:.1f} GB) may exceed limits "
                    f"({max_mem_per_core} GB/core)"
                )
            
            if mem_gb_per_core < 1.0:
                result.recommendations.append(
                    f"Low memory per core ({mem_gb_per_core:.1f} GB) - consider increasing for better performance"
                )
        
        except ValueError as e:
            result.warnings.append(f"Invalid PBS memory specification: {e}")
            result.resource_efficiency *= 0.9
        
        # Validate walltime
        if not self._validate_time_format(walltime):
            result.warnings.append(f"Invalid PBS walltime format: {walltime}")
            result.is_valid = False
        else:
            walltime_hours = self._parse_walltime_to_hours(walltime)
            max_walltime = limits.get('max_walltime_hours', 168)
            
            if walltime_hours > max_walltime:
                result.warnings.append(
                    f"Walltime ({walltime_hours}h) may exceed queue limits ({max_walltime}h)"
                )
            
            result.estimated_walltime = walltime
        
        # Queue-specific recommendations
        if queue == 'debug' and total_cores > 64:
            result.recommendations.append(
                "Debug queue typically has core limits - consider normal queue for large jobs"
            )
    
    def _validate_slurm_configuration(self, slurm_config: Dict[str, Any], result: HPCValidationResult) -> None:
        """
        Validate SLURM configuration.
        
        Args:
            slurm_config: SLURM configuration
            result: Validation result to update
        """
        # Extract SLURM parameters
        ntasks = slurm_config.get('ntasks', 1)
        cpus_per_task = slurm_config.get('cpus-per-task', 1)
        mem_per_cpu = slurm_config.get('mem-per-cpu', '2G')
        time_limit = slurm_config.get('time', '24:00:00')
        partition = slurm_config.get('partition', 'normal')
        nodes = slurm_config.get('nodes', 1)
        
        total_cores = ntasks * cpus_per_task
        
        # Validate resource consistency
        if nodes > 1 and ntasks < nodes:
            result.warnings.append(
                f"Requesting {nodes} nodes with only {ntasks} tasks - resources may be underutilized"
            )
            result.resource_efficiency *= 0.7
        
        # Validate memory specification
        try:
            mem_value, mem_unit = self._parse_memory_spec(mem_per_cpu)
            mem_gb_per_cpu = self._convert_to_gb(mem_value, mem_unit)
            
            if mem_gb_per_cpu > 16:  # Typical SLURM limit
                result.warnings.append(
                    f"High memory per CPU ({mem_gb_per_cpu:.1f} GB) may exceed node capacity"
                )
        except ValueError as e:
            result.warnings.append(f"Invalid SLURM memory specification: {e}")
            result.resource_efficiency *= 0.9
        
        # Validate time format and limits
        if not self._validate_time_format(time_limit):
            result.warnings.append(f"Invalid SLURM time format: {time_limit}")
            result.is_valid = False
        
        # Partition-specific validation
        if partition == 'gpu' and 'gres' not in slurm_config:
            result.warnings.append("GPU partition selected but no GPU resources (gres) specified")
        
        result.estimated_walltime = time_limit
    
    def _validate_distributed_settings(self, dist_config: Dict[str, Any], result: HPCValidationResult) -> None:
        """
        Validate distributed computing settings.
        
        Args:
            dist_config: Distributed configuration
            result: Validation result to update
        """
        if 'mpi' in dist_config:
            mpi_config = dist_config['mpi']
            processes = mpi_config.get('processes', 1)
            
            # Check MPI availability
            if not self._check_mpi_availability():
                result.warnings.append("MPI configuration specified but MPI not available")
                result.is_valid = False
            else:
                result.recommendations.append("MPI detected - ensure proper module loading in job script")
            
            # Validate process count
            if processes > psutil.cpu_count():
                result.warnings.append(
                    f"MPI processes ({processes}) exceed local CPU count - ensure multi-node setup"
                )
    
    def _calculate_resource_efficiency(self, hpc_config: Dict[str, Any], result: HPCValidationResult) -> None:
        """
        Calculate resource allocation efficiency.
        
        Args:
            hpc_config: HPC configuration
            result: Validation result to update
        """
        efficiency_factors = []
        
        # Check for overprovisioning
        scheduler_type = result.scheduler_type
        if scheduler_type and scheduler_type in hpc_config:
            scheduler_config = hpc_config[scheduler_type]
            
            if scheduler_type == 'pbs':
                nodes = scheduler_config.get('nodes', 1)
                ppn = scheduler_config.get('ppn', 1)
                
                # Typical node has 24-48 cores
                if ppn < 12:  # Underutilizing nodes
                    efficiency_factors.append(0.7)
                elif ppn > 48:  # May exceed node capacity
                    efficiency_factors.append(0.8)
            
            elif scheduler_type == 'slurm':
                ntasks = scheduler_config.get('ntasks', 1)
                cpus_per_task = scheduler_config.get('cpus-per-task', 1)
                
                # Check for reasonable task distribution
                if cpus_per_task == 1 and ntasks > 100:
                    efficiency_factors.append(0.8)  # Many small tasks
                elif cpus_per_task > 24 and ntasks == 1:
                    efficiency_factors.append(0.9)  # Single large task
        
        # Calculate final efficiency
        if efficiency_factors:
            result.resource_efficiency = min(efficiency_factors)
        
        # Add efficiency recommendations
        if result.resource_efficiency < 0.8:
            result.recommendations.append("Consider optimizing resource allocation for better efficiency")
    
    def _parse_memory_spec(self, mem_spec: str) -> Tuple[float, str]:
        """Parse memory specification like '4gb' or '8GB'."""
        match = re.match(r'(\d+(?:\.\d+)?)([a-zA-Z]+)', mem_spec.lower())
        if match:
            value, unit = match.groups()
            return float(value), unit.lower()
        raise ValueError(f"Invalid memory specification: {mem_spec}")
    
    def _convert_to_gb(self, value: float, unit: str) -> float:
        """Convert memory value to GB."""
        conversions = {
            'b': 1e-9,
            'kb': 1e-6,
            'mb': 1e-3,
            'gb': 1,
            'tb': 1e3
        }
        return value * conversions.get(unit, 1)
    
    def _validate_time_format(self, time_str: str) -> bool:
        """Validate HPC time format."""
        patterns = [
            r'^\d{1,2}:\d{2}:\d{2}$',      # HH:MM:SS
            r'^\d{1,3}:\d{2}:\d{2}:\d{2}$'  # DD:HH:MM:SS
        ]
        return any(re.match(pattern, time_str) for pattern in patterns)
    
    def _parse_walltime_to_hours(self, walltime: str) -> float:
        """Parse walltime string to hours."""
        if ':' in walltime:
            parts = walltime.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(int, parts)
                return hours + minutes/60 + seconds/3600
            elif len(parts) == 4:  # DD:HH:MM:SS
                days, hours, minutes, seconds = map(int, parts)
                return days*24 + hours + minutes/60 + seconds/3600
        return float(walltime)
    
    def _check_mpi_availability(self) -> bool:
        """Check if MPI is available."""
        mpi_commands = ['mpirun', 'mpiexec', 'srun']
        return any(shutil.which(cmd) for cmd in mpi_commands)


class GPUValidator:
    """
    Validator for GPU hardware and configuration.
    
    Provides comprehensive GPU detection, memory validation, and 
    compatibility checking with JAX/CUDA configurations.
    """
    
    def __init__(self):
        """Initialize GPU validator."""
        self.gpu_cache = {}
        self.compatibility_cache = {}
    
    def validate_gpu_configuration(self, 
                                 hardware_config: Dict[str, Any],
                                 optimization_config: Dict[str, Any]) -> GPUValidationResult:
        """
        Validate GPU configuration against available hardware.
        
        Args:
            hardware_config: Hardware configuration
            optimization_config: Optimization configuration
            
        Returns:
            Comprehensive GPU validation result
        """
        result = GPUValidationResult(
            gpu_available=False,
            gpu_compatible=False,
            total_memory_gb=0.0,
            recommended_memory_fraction=0.8,
            warnings=[],
            optimization_suggestions=[]
        )
        
        # Detect GPU hardware
        gpu_info = self._detect_gpu_hardware()
        result.gpu_available = gpu_info['available']
        result.total_memory_gb = gpu_info['total_memory_gb']
        
        if not result.gpu_available:
            result.warnings.append("No GPU hardware detected")
            return result
        
        # Validate memory configuration
        self._validate_gpu_memory_config(hardware_config, gpu_info, result)
        
        # Check software compatibility
        self._validate_gpu_software_compatibility(optimization_config, result)
        
        # Generate optimization suggestions
        self._generate_gpu_optimization_suggestions(hardware_config, gpu_info, result)
        
        result.gpu_compatible = len(result.warnings) == 0
        
        return result
    
    def _detect_gpu_hardware(self) -> Dict[str, Any]:
        """
        Detect available GPU hardware using multiple methods.
        
        Returns:
            Dictionary with GPU hardware information
        """
        if 'hardware_info' in self.gpu_cache:
            return self.gpu_cache['hardware_info']
        
        gpu_info = {
            'available': False,
            'devices': [],
            'total_memory_gb': 0.0,
            'driver_version': None,
            'cuda_version': None
        }
        
        # Try different detection methods
        detection_methods = [
            self._detect_via_gputil,
            self._detect_via_pynvml, 
            self._detect_via_nvidia_smi,
            self._detect_via_jax
        ]
        
        for method in detection_methods:
            try:
                method_result = method()
                if method_result['available']:
                    gpu_info.update(method_result)
                    break
            except Exception as e:
                logger.debug(f"GPU detection method failed: {e}")
        
        self.gpu_cache['hardware_info'] = gpu_info
        return gpu_info
    
    def _detect_via_gputil(self) -> Dict[str, Any]:
        """Detect GPUs using GPUtil library."""
        if not HAS_GPUTIL:
            return {'available': False}
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return {'available': False}
            
            devices = []
            total_memory = 0
            
            for gpu in gpus:
                device_info = {
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_free_mb': gpu.memoryFree,
                    'utilization': gpu.load,
                    'temperature': gpu.temperature
                }
                devices.append(device_info)
                total_memory += gpu.memoryTotal
            
            return {
                'available': True,
                'devices': devices,
                'total_memory_gb': total_memory / 1024,
                'detection_method': 'gputil'
            }
        
        except Exception:
            return {'available': False}
    
    def _detect_via_pynvml(self) -> Dict[str, Any]:
        """Detect GPUs using pynvml library."""
        if not HAS_PYNVML:
            return {'available': False}
        
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count == 0:
                return {'available': False}
            
            devices = []
            total_memory = 0
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                device_info = {
                    'id': i,
                    'name': name,
                    'memory_total_mb': mem_info.total // (1024*1024),
                    'memory_used_mb': mem_info.used // (1024*1024),
                    'memory_free_mb': mem_info.free // (1024*1024)
                }
                devices.append(device_info)
                total_memory += mem_info.total
            
            # Get driver version
            try:
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode()
            except:
                driver_version = None
            
            pynvml.nvmlShutdown()
            
            return {
                'available': True,
                'devices': devices,
                'total_memory_gb': total_memory / (1024**3),
                'driver_version': driver_version,
                'detection_method': 'pynvml'
            }
        
        except Exception:
            return {'available': False}
    
    def _detect_via_nvidia_smi(self) -> Dict[str, Any]:
        """Detect GPUs using nvidia-smi command."""
        try:
            # Get basic GPU info
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return {'available': False}
            
            devices = []
            total_memory = 0
            
            for i, line in enumerate(result.stdout.strip().split('\n')):
                if line:
                    parts = line.split(', ')
                    if len(parts) >= 6:
                        name, mem_total, mem_used, mem_free, util, temp = parts
                        device_info = {
                            'id': i,
                            'name': name,
                            'memory_total_mb': int(mem_total),
                            'memory_used_mb': int(mem_used),
                            'memory_free_mb': int(mem_free),
                            'utilization': int(util) if util != '[N/A]' else 0,
                            'temperature': int(temp) if temp != '[N/A]' else 0
                        }
                        devices.append(device_info)
                        total_memory += int(mem_total)
            
            # Get driver and CUDA version
            driver_version = None
            cuda_version = None
            
            try:
                version_result = subprocess.run([
                    'nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'
                ], capture_output=True, text=True, timeout=5)
                if version_result.returncode == 0:
                    driver_version = version_result.stdout.strip()
            except:
                pass
            
            return {
                'available': len(devices) > 0,
                'devices': devices,
                'total_memory_gb': total_memory / 1024,
                'driver_version': driver_version,
                'cuda_version': cuda_version,
                'detection_method': 'nvidia-smi'
            }
        
        except Exception:
            return {'available': False}
    
    def _detect_via_jax(self) -> Dict[str, Any]:
        """Detect GPUs using JAX."""
        if not HAS_JAX:
            return {'available': False}
        
        try:
            gpu_devices = jax.devices('gpu')
            if not gpu_devices:
                return {'available': False}
            
            devices = []
            for i, device in enumerate(gpu_devices):
                device_info = {
                    'id': i,
                    'name': str(device),
                    'platform': device.platform,
                    'device_kind': device.device_kind
                }
                devices.append(device_info)
            
            return {
                'available': True,
                'devices': devices,
                'total_memory_gb': 0.0,  # JAX doesn't provide memory info easily
                'detection_method': 'jax'
            }
        
        except Exception:
            return {'available': False}
    
    def _validate_gpu_memory_config(self, 
                                  hardware_config: Dict[str, Any],
                                  gpu_info: Dict[str, Any],
                                  result: GPUValidationResult) -> None:
        """
        Validate GPU memory configuration.
        
        Args:
            hardware_config: Hardware configuration
            gpu_info: GPU information
            result: Validation result to update
        """
        memory_fraction = hardware_config.get('gpu_memory_fraction', 0.8)
        total_memory_gb = gpu_info.get('total_memory_gb', 0)
        
        if total_memory_gb == 0:
            result.warnings.append("Cannot determine GPU memory capacity")
            return
        
        # Calculate requested memory
        requested_memory_gb = total_memory_gb * memory_fraction
        
        # Check against available memory
        devices = gpu_info.get('devices', [])
        if devices:
            min_free_memory_mb = min(dev.get('memory_free_mb', 0) for dev in devices)
            min_free_memory_gb = min_free_memory_mb / 1024
            
            if requested_memory_gb > min_free_memory_gb:
                result.warnings.append(
                    f"Requested memory ({requested_memory_gb:.1f} GB) exceeds "
                    f"available memory ({min_free_memory_gb:.1f} GB) on GPU"
                )
            
            # Recommend optimal memory fraction
            if min_free_memory_gb > 0:
                safe_fraction = min(0.9, (min_free_memory_gb / total_memory_gb) * 0.8)
                result.recommended_memory_fraction = safe_fraction
                
                if memory_fraction > safe_fraction + 0.1:
                    result.optimization_suggestions.append(
                        f"Consider reducing gpu_memory_fraction to {safe_fraction:.2f} for safety"
                    )
    
    def _validate_gpu_software_compatibility(self, 
                                           optimization_config: Dict[str, Any],
                                           result: GPUValidationResult) -> None:
        """
        Validate GPU software compatibility.
        
        Args:
            optimization_config: Optimization configuration
            result: Validation result to update
        """
        # Check JAX availability for GPU backends
        mcmc_config = optimization_config.get('mcmc_sampling', {})
        if 'gpu_backend' in mcmc_config.get('backend_specific', {}):
            if not HAS_JAX:
                result.warnings.append(
                    "GPU MCMC backend requested but JAX not available"
                )
            else:
                try:
                    gpu_devices = jax.devices('gpu')
                    if not gpu_devices:
                        result.warnings.append(
                            "JAX available but no GPU devices detected by JAX"
                        )
                except:
                    result.warnings.append(
                        "JAX GPU backend initialization failed"
                    )
        
        # Check CUDA compatibility
        if platform.system() != 'Linux':
            result.optimization_suggestions.append(
                "GPU acceleration works best on Linux systems"
            )
    
    def _generate_gpu_optimization_suggestions(self, 
                                             hardware_config: Dict[str, Any],
                                             gpu_info: Dict[str, Any],
                                             result: GPUValidationResult) -> None:
        """
        Generate GPU optimization suggestions.
        
        Args:
            hardware_config: Hardware configuration  
            gpu_info: GPU information
            result: Validation result to update
        """
        devices = gpu_info.get('devices', [])
        
        # Multi-GPU suggestions
        if len(devices) > 1:
            result.optimization_suggestions.append(
                f"Multiple GPUs detected ({len(devices)}) - consider parallel processing"
            )
        
        # Memory optimization suggestions
        for device in devices:
            memory_total_gb = device.get('memory_total_mb', 0) / 1024
            memory_used_gb = device.get('memory_used_mb', 0) / 1024
            
            if memory_used_gb / memory_total_gb > 0.7:
                result.optimization_suggestions.append(
                    f"GPU {device['id']} has high memory usage - consider reducing memory fraction"
                )
            
            # Temperature warnings
            temp = device.get('temperature', 0)
            if temp > 80:
                result.warnings.append(
                    f"GPU {device['id']} temperature high ({temp}°C) - check cooling"
                )
        
        # Driver version suggestions
        driver_version = gpu_info.get('driver_version')
        if driver_version:
            result.optimization_suggestions.append(
                f"GPU driver version: {driver_version} - ensure compatibility with CUDA libraries"
            )


class AdvancedScenarioValidator:
    """
    Validator for advanced analysis scenarios.
    
    Handles validation of complex configurations including:
    - Large dataset processing
    - Batch processing workflows  
    - Complex phi angle filtering
    - Multi-configuration analyses
    """
    
    def __init__(self):
        """Initialize advanced scenario validator."""
        self.memory_estimation_factors = {
            'static_isotropic': 1.0,
            'static_anisotropic': 2.0,
            'laminar_flow': 3.5
        }
    
    def validate_large_dataset_scenario(self, 
                                      config: Dict[str, Any],
                                      estimated_data_size: int) -> Dict[str, Any]:
        """
        Validate configuration for large dataset processing.
        
        Args:
            config: Configuration dictionary
            estimated_data_size: Estimated dataset size in points
            
        Returns:
            Validation result with recommendations
        """
        result = {
            'is_suitable': True,
            'performance_warnings': [],
            'memory_warnings': [],
            'optimization_recommendations': [],
            'estimated_runtime_hours': None
        }
        
        # Define size thresholds
        large_threshold = 10_000_000    # 10M points
        very_large_threshold = 100_000_000  # 100M points
        extreme_threshold = 1_000_000_000   # 1B points
        
        if estimated_data_size < large_threshold:
            return result  # Not a large dataset scenario
        
        # Memory estimation
        mode = config.get('analysis_mode', 'static_isotropic')
        memory_factor = self.memory_estimation_factors.get(mode, 2.0)
        estimated_memory_gb = estimated_data_size * memory_factor * 8 / (1024**3)
        
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if estimated_memory_gb > available_memory_gb * 0.8:
            result['memory_warnings'].append(
                f"Estimated memory usage ({estimated_memory_gb:.1f} GB) may exceed "
                f"available memory ({available_memory_gb:.1f} GB)"
            )
            result['optimization_recommendations'].append("Enable low_memory_mode")
            result['optimization_recommendations'].append("Enable disk caching")
        
        # Performance analysis by data size
        if estimated_data_size >= extreme_threshold:
            result['performance_warnings'].append(
                f"Extreme dataset size ({estimated_data_size:,} points) - "
                "expect very long runtimes"
            )
            result['optimization_recommendations'].extend([
                "Consider data subsampling for initial analysis",
                "Use distributed computing if available",
                "Enable all performance optimizations"
            ])
            result['estimated_runtime_hours'] = self._estimate_runtime(
                estimated_data_size, mode, config
            )
        
        elif estimated_data_size >= very_large_threshold:
            result['performance_warnings'].append(
                f"Very large dataset ({estimated_data_size:,} points) - "
                "expect long runtimes"
            )
            result['optimization_recommendations'].extend([
                "Enable numba optimization",
                "Consider reducing MCMC samples for initial runs",
                "Use hybrid optimization method"
            ])
        
        elif estimated_data_size >= large_threshold:
            result['optimization_recommendations'].extend([
                "Enable performance optimizations",
                "Consider batch processing for multiple analyses"
            ])
        
        # Configuration-specific recommendations
        self._add_config_specific_recommendations(config, estimated_data_size, result)
        
        return result
    
    def validate_batch_processing_scenario(self, batch_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate batch processing configuration.
        
        Args:
            batch_config: Batch processing configuration
            
        Returns:
            Validation result
        """
        result = {
            'is_valid': True,
            'warnings': [],
            'optimization_suggestions': [],
            'resource_requirements': {}
        }
        
        batch_size = batch_config.get('batch_size', 10)
        parallel_jobs = batch_config.get('parallel_jobs', 2)
        output_directory = batch_config.get('output_directory', './batch_results')
        
        # Validate resource requirements
        total_parallel_load = batch_size * parallel_jobs
        available_cores = psutil.cpu_count(logical=True)
        
        if total_parallel_load > available_cores:
            result['warnings'].append(
                f"Total parallel load ({total_parallel_load}) exceeds CPU cores ({available_cores})"
            )
            result['optimization_suggestions'].append(
                f"Reduce parallel_jobs to {max(1, available_cores // batch_size)}"
            )
        
        # Validate output directory
        output_path = Path(output_directory)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            if not os.access(output_path, os.W_OK):
                result['warnings'].append(f"Output directory not writable: {output_directory}")
                result['is_valid'] = False
        except Exception as e:
            result['warnings'].append(f"Cannot create output directory: {e}")
            result['is_valid'] = False
        
        # Estimate storage requirements
        estimated_storage_per_job_gb = batch_config.get('estimated_output_size_gb', 0.1)
        total_storage_gb = batch_size * estimated_storage_per_job_gb
        
        if total_storage_gb > 100:  # More than 100 GB
            result['warnings'].append(
                f"Large storage requirement ({total_storage_gb:.1f} GB) for batch processing"
            )
            result['optimization_suggestions'].append("Consider output compression")
        
        result['resource_requirements'] = {
            'estimated_storage_gb': total_storage_gb,
            'recommended_cores': min(total_parallel_load, available_cores),
            'estimated_memory_gb': total_parallel_load * 2  # Rough estimate
        }
        
        return result
    
    def validate_complex_phi_filtering(self, angle_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complex phi angle filtering configuration.
        
        Args:
            angle_config: Angle filtering configuration
            
        Returns:
            Validation result
        """
        result = {
            'is_valid': True,
            'coverage_analysis': {},
            'warnings': [],
            'recommendations': []
        }
        
        if not angle_config.get('enabled', False):
            return result
        
        target_ranges = angle_config.get('target_ranges', [])
        if not target_ranges:
            result['warnings'].append("Angle filtering enabled but no ranges specified")
            result['is_valid'] = False
            return result
        
        # Analyze coverage
        total_coverage = 0
        valid_ranges = 0
        overlapping_ranges = []
        
        for i, range_spec in enumerate(target_ranges):
            if not isinstance(range_spec, dict):
                result['warnings'].append(f"Invalid range specification at index {i}")
                continue
            
            min_angle = range_spec.get('min_angle')
            max_angle = range_spec.get('max_angle')
            
            if min_angle is None or max_angle is None:
                result['warnings'].append(f"Missing angles in range {i}")
                continue
            
            if min_angle >= max_angle:
                result['warnings'].append(
                    f"Invalid range {i}: min ({min_angle}) >= max ({max_angle})"
                )
                continue
            
            range_width = max_angle - min_angle
            total_coverage += range_width
            valid_ranges += 1
            
            # Check for overlaps with previous ranges
            for j, other_range in enumerate(target_ranges[:i]):
                if isinstance(other_range, dict):
                    other_min = other_range.get('min_angle')
                    other_max = other_range.get('max_angle')
                    if other_min is not None and other_max is not None:
                        if not (max_angle <= other_min or min_angle >= other_max):
                            overlapping_ranges.append((i, j))
        
        result['coverage_analysis'] = {
            'total_coverage_degrees': total_coverage,
            'valid_ranges': valid_ranges,
            'total_ranges': len(target_ranges),
            'overlapping_ranges': overlapping_ranges
        }
        
        # Coverage assessment
        if total_coverage < 20:  # Less than 20 degrees
            result['warnings'].append(
                f"Limited angular coverage ({total_coverage:.1f}°) may reduce analysis quality"
            )
        elif total_coverage > 300:  # More than 300 degrees
            result['recommendations'].append(
                f"Very wide coverage ({total_coverage:.1f}°) - consider computational cost"
            )
        
        # Overlap warnings
        if overlapping_ranges:
            result['warnings'].append(
                f"Overlapping angle ranges detected: {overlapping_ranges}"
            )
        
        # Efficiency recommendations
        if len(target_ranges) > 10:
            result['recommendations'].append(
                f"Many ranges ({len(target_ranges)}) - consider consolidation for efficiency"
            )
        
        return result
    
    def _estimate_runtime(self, data_size: int, mode: str, config: Dict[str, Any]) -> float:
        """
        Estimate analysis runtime in hours.
        
        Args:
            data_size: Dataset size in points
            mode: Analysis mode
            config: Configuration
            
        Returns:
            Estimated runtime in hours
        """
        # Base runtime factors (hours per million points)
        base_factors = {
            'static_isotropic': 0.1,
            'static_anisotropic': 0.3,
            'laminar_flow': 1.0
        }
        
        base_factor = base_factors.get(mode, 0.5)
        base_hours = (data_size / 1_000_000) * base_factor
        
        # Method complexity multipliers
        opt_config = config.get('optimization_config', {})
        
        if opt_config.get('mcmc_sampling', {}).get('enabled', False):
            mcmc_draws = opt_config.get('mcmc_sampling', {}).get('draws', 3000)
            base_hours *= (mcmc_draws / 3000) * 2  # MCMC is expensive
        
        if opt_config.get('robust_optimization', {}).get('enabled', False):
            base_hours *= 1.5  # Robust methods add overhead
        
        return max(0.1, base_hours)  # Minimum 6 minutes
    
    def _add_config_specific_recommendations(self, 
                                           config: Dict[str, Any],
                                           data_size: int,
                                           result: Dict[str, Any]) -> None:
        """
        Add configuration-specific recommendations for large datasets.
        
        Args:
            config: Configuration dictionary
            data_size: Dataset size
            result: Result dictionary to update
        """
        # MCMC-specific recommendations
        mcmc_config = config.get('optimization_config', {}).get('mcmc_sampling', {})
        if mcmc_config.get('enabled', False):
            draws = mcmc_config.get('draws', 3000)
            if data_size > 50_000_000 and draws > 2000:
                result['optimization_recommendations'].append(
                    f"Consider reducing MCMC draws from {draws} to 2000-3000 for large datasets"
                )
        
        # Performance settings recommendations
        perf_config = config.get('performance_settings', {})
        if not perf_config.get('caching', {}).get('enable_disk_cache', False):
            result['optimization_recommendations'].append(
                "Enable disk caching for large dataset processing"
            )
        
        if not perf_config.get('numba_optimization', {}).get('enable_numba', False):
            result['optimization_recommendations'].append(
                "Enable Numba optimization for computational speedup"
            )