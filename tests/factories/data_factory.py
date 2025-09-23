"""
Test Data Factory for Homodyne v2
==================================

Factory functions for generating realistic test datasets:
- Synthetic XPCS correlation data with various characteristics
- Mock experimental data with realistic noise and artifacts
- Parameter sets for different physical scenarios
- HDF5 file generation with multiple formats
"""

import numpy as np
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

# Handle optional dependencies
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False
    h5py = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Handle JAX imports
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np


class XPCSDataFactory:
    """Factory for generating XPCS test data."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize factory with optional random seed.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducible data generation
        """
        if seed is not None:
            np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def create_synthetic_correlation_data(
        self,
        n_times: int = 50,
        n_angles: int = 36,
        true_parameters: Optional[Dict[str, float]] = None,
        noise_level: float = 0.01,
        q_value: float = 0.01,
        add_artifacts: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Create synthetic XPCS correlation data.

        Parameters
        ----------
        n_times : int
            Number of time points
        n_angles : int
            Number of angle points
        true_parameters : dict, optional
            True physical parameters
        noise_level : float
            Noise level (standard deviation)
        q_value : float
            Wavevector magnitude
        add_artifacts : bool
            Whether to add realistic experimental artifacts

        Returns
        -------
        dict
            Dictionary with XPCS data structure
        """
        if true_parameters is None:
            true_parameters = {
                'offset': 1.0,
                'contrast': 0.4,
                'diffusion_coefficient': 0.1,
                'shear_rate': 0.0,
                'L': 1.0
            }

        # Create time arrays
        t1, t2 = np.meshgrid(np.arange(n_times), np.arange(n_times), indexing='ij')

        # Create angle array
        phi = np.linspace(0, 2*np.pi, n_angles)

        # Generate correlation function
        c2_exp = self._generate_correlation_function(
            t1, t2, phi, true_parameters, q_value
        )

        # Add noise
        if noise_level > 0:
            noise = self.rng.normal(0, noise_level, c2_exp.shape)
            c2_exp += noise

        # Add experimental artifacts if requested
        if add_artifacts:
            c2_exp = self._add_experimental_artifacts(c2_exp, t1, t2)

        # Create uncertainty array
        sigma = np.full_like(c2_exp, noise_level)

        # Add some heteroscedastic noise (realistic)
        if add_artifacts:
            sigma *= (1 + 0.1 * self.rng.random(sigma.shape))

        return {
            't1': t1,
            't2': t2,
            'phi_angles_list': phi,
            'c2_exp': c2_exp,
            'wavevector_q_list': np.array([q_value]),
            'sigma': sigma,
            'true_parameters': true_parameters
        }

    def _generate_correlation_function(
        self,
        t1: np.ndarray,
        t2: np.ndarray,
        phi: np.ndarray,
        params: Dict[str, float],
        q: float
    ) -> np.ndarray:
        """Generate theoretical correlation function."""
        # Time difference
        tau = np.abs(t1 - t2)

        # Diffusion contribution
        g1_diff = np.exp(-params['diffusion_coefficient'] * q**2 * tau)

        # Shear contribution (simplified)
        if params['shear_rate'] > 0:
            # This is a simplified shear model
            shear_phase = params['shear_rate'] * q * params['L'] * tau
            g1_shear = np.sinc(shear_phase / np.pi)**2
        else:
            g1_shear = 1.0

        # Combine contributions
        g1 = g1_diff * g1_shear

        # Create g2 from g1
        c2_base = params['offset'] + params['contrast'] * g1**2

        # Expand to include angle dimension
        c2_exp = np.broadcast_to(c2_base[np.newaxis, :, :], (len(phi), t1.shape[0], t1.shape[1]))

        return c2_exp.copy()

    def _add_experimental_artifacts(
        self,
        c2_exp: np.ndarray,
        t1: np.ndarray,
        t2: np.ndarray
    ) -> np.ndarray:
        """Add realistic experimental artifacts."""
        # Diagonal suppression (common in real data)
        for angle_idx in range(c2_exp.shape[0]):
            for i in range(min(c2_exp.shape[1], c2_exp.shape[2])):
                # Reduce diagonal elements slightly
                c2_exp[angle_idx, i, i] *= 0.95

        # Add occasional outliers
        n_outliers = max(1, int(0.001 * c2_exp.size))
        outlier_indices = [
            self.rng.integers(0, s, n_outliers) for s in c2_exp.shape
        ]
        outlier_indices = tuple(outlier_indices)
        c2_exp[outlier_indices] *= (1 + 0.5 * self.rng.random(n_outliers))

        # Add systematic drift (very small)
        drift = 0.001 * np.linspace(0, 1, c2_exp.shape[-1])
        c2_exp += drift[np.newaxis, np.newaxis, :]

        return c2_exp

    def create_multi_q_dataset(
        self,
        q_values: List[float],
        n_times: int = 40,
        n_angles: int = 24,
        base_parameters: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create dataset with multiple q-values.

        Parameters
        ----------
        q_values : list
            List of q-values to generate
        n_times : int
            Number of time points
        n_angles : int
            Number of angles
        base_parameters : dict, optional
            Base physical parameters

        Returns
        -------
        dict
            Multi-q dataset
        """
        if base_parameters is None:
            base_parameters = {
                'offset': 1.0,
                'contrast': 0.4,
                'diffusion_coefficient': 0.12,
                'shear_rate': 0.0,
                'L': 1.0
            }

        datasets = []
        for q in q_values:
            data = self.create_synthetic_correlation_data(
                n_times=n_times,
                n_angles=n_angles,
                true_parameters=base_parameters,
                q_value=q,
                noise_level=0.005
            )
            datasets.append(data)

        # Combine into multi-q structure
        combined_data = {
            't1': datasets[0]['t1'],
            't2': datasets[0]['t2'],
            'phi_angles_list': datasets[0]['phi_angles_list'],
            'wavevector_q_list': np.array(q_values),
            'c2_exp_list': [d['c2_exp'] for d in datasets],
            'sigma_list': [d['sigma'] for d in datasets],
            'true_parameters': base_parameters
        }

        return combined_data

    def create_time_series_dataset(
        self,
        n_frames: int = 100,
        frame_spacing: float = 0.1,
        n_angles: int = 36,
        dynamic_parameters: bool = False
    ) -> Dict[str, Any]:
        """
        Create time-series XPCS dataset.

        Parameters
        ----------
        n_frames : int
            Number of time frames
        frame_spacing : float
            Time spacing between frames
        n_angles : int
            Number of angles
        dynamic_parameters : bool
            Whether parameters change over time

        Returns
        -------
        dict
            Time-series dataset
        """
        frame_times = np.arange(n_frames) * frame_spacing

        # Create correlation matrices for each frame
        correlation_frames = []
        parameter_evolution = []

        base_params = {
            'offset': 1.0,
            'contrast': 0.4,
            'diffusion_coefficient': 0.1,
            'shear_rate': 0.0,
            'L': 1.0
        }

        for i, frame_time in enumerate(frame_times):
            # Modify parameters if dynamic
            if dynamic_parameters:
                params = base_params.copy()
                # Slowly varying diffusion coefficient
                params['diffusion_coefficient'] += 0.02 * np.sin(frame_time * 0.1)
                # Slowly varying contrast
                params['contrast'] += 0.05 * np.cos(frame_time * 0.05)
            else:
                params = base_params

            # Create correlation data for this frame
            frame_data = self.create_synthetic_correlation_data(
                n_times=30,
                n_angles=n_angles,
                true_parameters=params,
                noise_level=0.008,
                add_artifacts=True
            )

            correlation_frames.append(frame_data['c2_exp'])
            parameter_evolution.append(params.copy())

        return {
            'frame_times': frame_times,
            'correlation_frames': correlation_frames,
            'parameter_evolution': parameter_evolution,
            'phi_angles_list': frame_data['phi_angles_list'],
            't1': frame_data['t1'],
            't2': frame_data['t2'],
            'wavevector_q_list': frame_data['wavevector_q_list']
        }

    def create_noisy_dataset(
        self,
        noise_type: str = 'gaussian',
        noise_level: float = 0.05,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Create dataset with specific noise characteristics.

        Parameters
        ----------
        noise_type : str
            Type of noise: 'gaussian', 'poisson', 'uniform'
        noise_level : float
            Noise intensity
        **kwargs
            Additional arguments for data generation

        Returns
        -------
        dict
            Noisy dataset
        """
        # Create base data
        data = self.create_synthetic_correlation_data(noise_level=0, **kwargs)
        c2_clean = data['c2_exp']

        # Add specific noise type
        if noise_type == 'gaussian':
            noise = self.rng.normal(0, noise_level, c2_clean.shape)
        elif noise_type == 'poisson':
            # Scale data to make Poisson noise reasonable
            scaled_data = c2_clean * 100  # Scale up
            noisy_scaled = self.rng.poisson(scaled_data)
            noise = (noisy_scaled - scaled_data) / 100  # Scale back
        elif noise_type == 'uniform':
            noise = self.rng.uniform(-noise_level, noise_level, c2_clean.shape)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        data['c2_exp'] = c2_clean + noise
        data['sigma'] = np.full_like(c2_clean, noise_level)
        data['noise_type'] = noise_type

        return data

    def create_edge_case_dataset(
        self,
        case_type: str = 'high_noise'
    ) -> Dict[str, np.ndarray]:
        """
        Create edge case datasets for testing robustness.

        Parameters
        ----------
        case_type : str
            Type of edge case: 'high_noise', 'low_contrast', 'fast_diffusion', 'slow_diffusion'

        Returns
        -------
        dict
            Edge case dataset
        """
        if case_type == 'high_noise':
            return self.create_synthetic_correlation_data(
                noise_level=0.2,  # Very high noise
                true_parameters={'offset': 1.0, 'contrast': 0.3, 'diffusion_coefficient': 0.1,
                               'shear_rate': 0.0, 'L': 1.0}
            )
        elif case_type == 'low_contrast':
            return self.create_synthetic_correlation_data(
                noise_level=0.01,
                true_parameters={'offset': 1.0, 'contrast': 0.05, 'diffusion_coefficient': 0.1,
                               'shear_rate': 0.0, 'L': 1.0}
            )
        elif case_type == 'fast_diffusion':
            return self.create_synthetic_correlation_data(
                noise_level=0.01,
                true_parameters={'offset': 1.0, 'contrast': 0.4, 'diffusion_coefficient': 1.0,
                               'shear_rate': 0.0, 'L': 1.0}
            )
        elif case_type == 'slow_diffusion':
            return self.create_synthetic_correlation_data(
                noise_level=0.01,
                true_parameters={'offset': 1.0, 'contrast': 0.4, 'diffusion_coefficient': 0.001,
                               'shear_rate': 0.0, 'L': 1.0}
            )
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")


class HDF5FileFactory:
    """Factory for creating mock HDF5 files."""

    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize HDF5 factory.

        Parameters
        ----------
        temp_dir : Path, optional
            Directory for temporary files
        """
        if not HAS_H5PY:
            raise ImportError("h5py is required for HDF5 file generation")

        self.temp_dir = temp_dir or Path(tempfile.gettempdir())
        self.created_files = []

    def create_aps_old_format(
        self,
        filename: Optional[str] = None,
        data: Optional[Dict[str, np.ndarray]] = None
    ) -> Path:
        """
        Create HDF5 file in APS old format.

        Parameters
        ----------
        filename : str, optional
            Output filename
        data : dict, optional
            Data to include (if None, uses synthetic data)

        Returns
        -------
        Path
            Path to created file
        """
        if filename is None:
            filename = f"aps_old_{len(self.created_files)}.h5"

        file_path = self.temp_dir / filename

        if data is None:
            factory = XPCSDataFactory(seed=42)
            data = factory.create_synthetic_correlation_data(n_times=40, n_angles=24)

        with h5py.File(file_path, 'w') as f:
            # APS old format structure
            exchange = f.create_group('exchange')

            # Main datasets
            exchange.create_dataset('correlation', data=data['c2_exp'])
            exchange.create_dataset('phi_angles', data=data['phi_angles_list'])
            exchange.create_dataset('wavevector_q', data=data['wavevector_q_list'])
            exchange.create_dataset('time_grid', data=np.arange(data['t1'].shape[0]))

            # Add metadata
            f.attrs['format'] = 'APS_old'
            f.attrs['version'] = '1.0'
            f.attrs['created_by'] = 'homodyne_test_factory'

            # Optional datasets
            if 'sigma' in data:
                exchange.create_dataset('sigma', data=data['sigma'])

        self.created_files.append(file_path)
        return file_path

    def create_aps_u_format(
        self,
        filename: Optional[str] = None,
        data: Optional[Dict[str, np.ndarray]] = None
    ) -> Path:
        """
        Create HDF5 file in APS-U format.

        Parameters
        ----------
        filename : str, optional
            Output filename
        data : dict, optional
            Data to include

        Returns
        -------
        Path
            Path to created file
        """
        if filename is None:
            filename = f"aps_u_{len(self.created_files)}.h5"

        file_path = self.temp_dir / filename

        if data is None:
            factory = XPCSDataFactory(seed=42)
            data = factory.create_synthetic_correlation_data(n_times=50, n_angles=36)

        with h5py.File(file_path, 'w') as f:
            # APS-U format structure (different organization)
            measurement = f.create_group('measurement')

            # Datasets with different names/organization
            measurement.create_dataset('correlation_data', data=data['c2_exp'])
            measurement.create_dataset('angle_list', data=data['phi_angles_list'])
            measurement.create_dataset('q_vector', data=data['wavevector_q_list'])
            measurement.create_dataset('time_stamps', data=np.arange(data['t1'].shape[0]))

            # Additional metadata structure
            metadata = f.create_group('metadata')
            metadata.attrs['instrument'] = 'APS-U'
            metadata.attrs['beamline'] = 'test_beamline'

            # Format identifier
            f.attrs['format'] = 'APS-U'
            f.attrs['version'] = '2.0'

        self.created_files.append(file_path)
        return file_path

    def create_custom_format(
        self,
        filename: Optional[str] = None,
        structure: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create HDF5 file with custom structure.

        Parameters
        ----------
        filename : str, optional
            Output filename
        structure : dict, optional
            Custom structure definition

        Returns
        -------
        Path
            Path to created file
        """
        if filename is None:
            filename = f"custom_{len(self.created_files)}.h5"

        file_path = self.temp_dir / filename

        if structure is None:
            # Default custom structure
            factory = XPCSDataFactory(seed=42)
            data = factory.create_synthetic_correlation_data()

            structure = {
                'data/correlations': data['c2_exp'],
                'data/angles': data['phi_angles_list'],
                'parameters/q_values': data['wavevector_q_list'],
                'metadata/time_info': np.arange(data['t1'].shape[0])
            }

        with h5py.File(file_path, 'w') as f:
            for path, dataset in structure.items():
                # Create groups as needed
                group_path = '/'.join(path.split('/')[:-1])
                if group_path and group_path not in f:
                    f.create_group(group_path)

                # Create dataset
                f.create_dataset(path, data=dataset)

            f.attrs['format'] = 'custom'
            f.attrs['version'] = '1.0'

        self.created_files.append(file_path)
        return file_path

    def cleanup(self):
        """Remove all created files."""
        for file_path in self.created_files:
            if file_path.exists():
                file_path.unlink()
        self.created_files.clear()


class ConfigFactory:
    """Factory for creating test configurations."""

    @staticmethod
    def create_basic_config(
        analysis_mode: str = 'static_isotropic',
        optimization_method: str = 'nlsq',
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create basic configuration."""
        config = {
            'analysis_mode': analysis_mode,
            'optimization': {
                'method': optimization_method,
                'lsq': {
                    'max_iterations': 100,
                    'tolerance': 1e-6
                }
            },
            'hardware': {
                'force_cpu': True,
                'gpu_memory_fraction': 0.8
            },
            'output': {
                'save_plots': False,
                'verbose': False
            }
        }

        if output_dir:
            config['output']['directory'] = output_dir

        return config

    @staticmethod
    def create_performance_config(
        max_iterations: int = 50,
        tolerance: float = 1e-6
    ) -> Dict[str, Any]:
        """Create configuration optimized for performance testing."""
        return {
            'analysis_mode': 'static_isotropic',
            'optimization': {
                'method': 'nlsq',
                'lsq': {
                    'max_iterations': max_iterations,
                    'tolerance': tolerance
                }
            },
            'hardware': {
                'force_cpu': True,
                'parallel_processing': True
            },
            'output': {
                'save_plots': False,
                'verbose': False,
                'save_intermediate': False
            }
        }

    @staticmethod
    def create_gpu_config(
        memory_fraction: float = 0.8,
        force_gpu: bool = False
    ) -> Dict[str, Any]:
        """Create configuration for GPU testing."""
        return {
            'analysis_mode': 'static_isotropic',
            'optimization': {
                'method': 'nlsq',
                'lsq': {
                    'max_iterations': 100,
                    'tolerance': 1e-6
                }
            },
            'hardware': {
                'force_cpu': False,
                'force_gpu': force_gpu,
                'gpu_memory_fraction': memory_fraction
            },
            'output': {
                'save_plots': False,
                'verbose': True
            }
        }

    @staticmethod
    def create_mcmc_config(
        num_samples: int = 1000,
        num_warmup: int = 500
    ) -> Dict[str, Any]:
        """Create configuration for MCMC testing."""
        return {
            'analysis_mode': 'static_isotropic',
            'optimization': {
                'method': 'mcmc',
                'mcmc': {
                    'num_samples': num_samples,
                    'num_warmup': num_warmup,
                    'chains': 1
                }
            },
            'hardware': {
                'force_cpu': True
            },
            'output': {
                'save_plots': False,
                'save_chains': False,
                'verbose': False
            }
        }

    @staticmethod
    def save_config_file(
        config: Dict[str, Any],
        file_path: Path,
        format_type: str = 'json'
    ) -> Path:
        """Save configuration to file."""
        if format_type == 'json':
            import json
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
        elif format_type == 'yaml' and HAS_YAML:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        return file_path


class ParameterFactory:
    """Factory for creating parameter sets for testing."""

    @staticmethod
    def create_realistic_parameters() -> Dict[str, float]:
        """Create realistic physical parameters."""
        return {
            'offset': 1.0,
            'contrast': 0.4,
            'diffusion_coefficient': 0.12,
            'shear_rate': 0.0,
            'L': 1.0
        }

    @staticmethod
    def create_parameter_sweep(
        param_name: str,
        values: List[float],
        base_params: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, float]]:
        """Create parameter sweep for testing."""
        if base_params is None:
            base_params = ParameterFactory.create_realistic_parameters()

        parameter_sets = []
        for value in values:
            params = base_params.copy()
            params[param_name] = value
            parameter_sets.append(params)

        return parameter_sets

    @staticmethod
    def create_edge_case_parameters() -> List[Dict[str, float]]:
        """Create edge case parameter sets."""
        base = ParameterFactory.create_realistic_parameters()

        edge_cases = [
            # Low contrast
            {**base, 'contrast': 0.01},
            # High contrast
            {**base, 'contrast': 0.95},
            # Fast diffusion
            {**base, 'diffusion_coefficient': 1.0},
            # Slow diffusion
            {**base, 'diffusion_coefficient': 0.001},
            # High shear
            {**base, 'shear_rate': 0.5},
            # Large L
            {**base, 'L': 10.0},
            # Small L
            {**base, 'L': 0.1}
        ]

        return edge_cases