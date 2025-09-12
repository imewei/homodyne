"""
XPCS Data Loader for Homodyne v2
================================

Enhanced XPCS data loader supporting both APS (old) and APS-U (new) HDF5 formats
with YAML-first configuration system, JAX compatibility, and modern architecture integration.

This module provides:
- YAML-first configuration with JSON support
- Smart NPZ caching to avoid reloading large HDF5 files  
- Auto-detection of APS vs APS-U format
- Half-matrix reconstruction for correlation matrices
- Optional diagonal correction
- JAX array output with numpy fallback
- Integration with v2 logging and physics validation

Key Features:
- Format Support: APS old format and APS-U new format
- Configuration: YAML primary, JSON via converter
- Caching: Intelligent NPZ caching with compression
- Output: JAX arrays when available, numpy fallback
- Validation: Optional physics-based data quality checks
"""

import os
import json
import time
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path

# Handle optional dependencies with graceful fallback
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

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
    yaml = None

# JAX integration
try:
    import jax.numpy as jnp
    from homodyne.core.jax_backend import jax_available
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax_available = False
    jnp = np

# V2 system integration
try:
    from homodyne.utils.logging import get_logger, log_performance, log_calls
    HAS_V2_LOGGING = True
except ImportError:
    # Fallback to standard logging if v2 logging not available
    import logging
    HAS_V2_LOGGING = False
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def log_calls(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Physics validation integration
try:
    from homodyne.core.physics import PhysicsConstants, validate_experimental_setup
    HAS_PHYSICS_VALIDATION = True
except ImportError:
    HAS_PHYSICS_VALIDATION = False
    PhysicsConstants = None
    validate_experimental_setup = None

# Performance engine integration
try:
    from homodyne.data.performance_engine import PerformanceEngine
    from homodyne.data.memory_manager import AdvancedMemoryManager
    from homodyne.data.optimization import AdvancedDatasetOptimizer
    HAS_PERFORMANCE_ENGINE = True
except ImportError:
    HAS_PERFORMANCE_ENGINE = False
    PerformanceEngine = None
    AdvancedMemoryManager = None
    AdvancedDatasetOptimizer = None

logger = get_logger(__name__)

class XPCSDataFormatError(Exception):
    """Raised when XPCS data format is not recognized or invalid."""
    pass

class XPCSDependencyError(Exception):
    """Raised when required dependencies are not available."""
    pass

class XPCSConfigurationError(Exception):
    """Raised when configuration is invalid or missing required parameters."""
    pass

def load_xpcs_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load XPCS configuration from YAML or JSON file.
    
    Primary format: YAML
    JSON support: Automatically converted to YAML format
    
    Args:
        config_path: Path to YAML or JSON configuration file
        
    Returns:
        Configuration dictionary with YAML-style structure
        
    Raises:
        XPCSConfigurationError: If configuration format is unsupported or invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise XPCSConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            if not HAS_YAML:
                raise XPCSDependencyError("PyYAML required for YAML configuration files")
            
            # Native YAML loading
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded YAML configuration: {config_path}")
            return config
        
        elif config_path.suffix.lower() == '.json':
            # JSON loading with structure conversion
            with open(config_path, 'r') as f:
                json_config = json.load(f)
            
            logger.info(f"Loaded JSON configuration (converted to YAML): {config_path}")
            logger.info("Consider migrating to YAML format for better readability")
            
            # Convert JSON structure to YAML-style (for now, keep identical structure)
            # In future, can add more sophisticated conversion via existing converter
            return json_config
        
        else:
            raise XPCSConfigurationError(
                f"Unsupported configuration format: {config_path.suffix}. "
                f"Supported formats: .yaml, .yml, .json"
            )
    
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise XPCSConfigurationError(f"Failed to parse configuration file {config_path}: {e}")

class XPCSDataLoader:
    """
    Enhanced XPCS data loader for Homodyne v2.
    
    Supports both APS (old) and APS-U (new) formats with YAML-first configuration,
    intelligent caching, and JAX integration.
    
    Features:
    - YAML-first configuration with JSON support
    - Auto-detection of HDF5 format (APS vs APS-U)
    - Smart NPZ caching with compression
    - Half-matrix reconstruction for correlation matrices
    - Optional diagonal correction
    - JAX array output when available
    - Integration with v2 physics validation
    """
    
    @log_calls(include_args=False)
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None, 
                 configure_logging: bool = True):
        """
        Initialize XPCS data loader with YAML-first configuration.
        
        Args:
            config_path: Path to YAML or JSON configuration file
            config_dict: Configuration dictionary (alternative to config_path)
            configure_logging: Whether to apply logging configuration from config
            
        Raises:
            XPCSDependencyError: If required dependencies are not available
            XPCSConfigurationError: If configuration is invalid
        """
        # Check for required dependencies
        self._check_dependencies()
        
        if config_path and config_dict:
            raise ValueError("Provide either config_path or config_dict, not both")
        
        if config_path:
            self.config = load_xpcs_config(config_path)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        # Process v2 configuration enhancements
        self._process_v2_config_enhancements()
        
        # Extract main configuration sections
        self.exp_config = self.config.get('experimental_data', {})
        self.analyzer_config = self.config.get('analyzer_parameters', {})
        self.v2_config = self.config.get('v2_features', {})
        
        # Initialize performance optimization components
        self._init_performance_components()
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info(f"XPCS data loader initialized with {len(self.config)} config sections")
        
    def _check_dependencies(self) -> None:
        """Check for required dependencies and raise error if missing."""
        missing_deps = []
        
        if not HAS_NUMPY:
            missing_deps.append("numpy")
        if not HAS_H5PY:
            missing_deps.append("h5py")
            
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}. "
            error_msg += "Please install them with: pip install " + " ".join(missing_deps)
            logger.error(error_msg)
            raise XPCSDependencyError(error_msg)
    
    def _process_v2_config_enhancements(self) -> None:
        """Process v2 configuration enhancements and set defaults."""
        if 'v2_features' not in self.config:
            self.config['v2_features'] = {}
        
        v2_defaults = {
            'output_format': 'auto',  # 'numpy', 'jax', 'auto'
            'validation_level': 'basic',  # 'none', 'basic', 'full'
            'performance_optimization': True,
            'physics_validation': False,
            'cache_strategy': 'intelligent'  # 'none', 'simple', 'intelligent'
        }
        
        for key, default_value in v2_defaults.items():
            if key not in self.config['v2_features']:
                self.config['v2_features'][key] = default_value
        
        # Add performance optimization defaults
        performance_defaults = {
            'performance_engine_enabled': True,
            'memory_mapped_io': True,
            'advanced_chunking': True,
            'multi_level_caching': True,
            'background_prefetching': True,
            'memory_pressure_monitoring': True
        }
        
        if 'performance' not in self.config:
            self.config['performance'] = {}
            
        for key, default_value in performance_defaults.items():
            if key not in self.config['performance']:
                self.config['performance'][key] = default_value
    
    def _init_performance_components(self) -> None:
        """Initialize performance optimization components."""
        self.performance_engine = None
        self.memory_manager = None
        self.advanced_optimizer = None
        
        # Check if performance optimization is enabled
        performance_config = self.config.get('performance', {})
        if not performance_config.get('performance_engine_enabled', True):
            logger.info("Performance engine disabled in configuration")
            return
        
        if not HAS_PERFORMANCE_ENGINE:
            logger.warning("Performance engine not available - falling back to basic optimization")
            return
        
        try:
            # Initialize performance engine
            if performance_config.get('performance_engine_enabled', True):
                self.performance_engine = PerformanceEngine(self.config)
                logger.info("Performance engine initialized")
            
            # Initialize memory manager
            if performance_config.get('memory_pressure_monitoring', True):
                self.memory_manager = AdvancedMemoryManager(self.config)
                logger.info("Advanced memory manager initialized")
            
            # Initialize advanced optimizer
            self.advanced_optimizer = AdvancedDatasetOptimizer(
                config=self.config,
                performance_engine=self.performance_engine,
                memory_manager=self.memory_manager
            )
            logger.info("Advanced dataset optimizer initialized")
            
        except Exception as e:
            logger.warning(f"Performance components initialization failed: {e}")
            logger.info("Falling back to basic optimization")
            self.performance_engine = None
            self.memory_manager = None
            self.advanced_optimizer = None
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters."""
        required_exp_data = ['data_folder_path', 'data_file_name']
        required_analyzer = ['dt', 'start_frame', 'end_frame']
        
        for key in required_exp_data:
            if key not in self.exp_config:
                raise XPCSConfigurationError(f"Missing required experimental_data parameter: {key}")
        
        for key in required_analyzer:
            if key not in self.analyzer_config:
                raise XPCSConfigurationError(f"Missing required analyzer_parameters parameter: {key}")
        
        # Validate file existence
        data_file_path = os.path.join(
            self.exp_config['data_folder_path'],
            self.exp_config['data_file_name']
        )
        
        if not os.path.exists(data_file_path):
            logger.warning(f"Data file not found: {data_file_path}")
            logger.info("File will be checked again during data loading")
    
    def _get_output_format(self) -> str:
        """Get output array format from configuration."""
        return self.v2_config.get('output_format', 'auto')
    
    def _should_perform_validation(self) -> Dict[str, bool]:
        """Get validation settings from configuration."""
        validation_level = self.v2_config.get('validation_level', 'basic')
        return {
            'physics_checks': self.v2_config.get('physics_validation', False) and HAS_PHYSICS_VALIDATION,
            'data_quality': validation_level != 'none',
            'comprehensive': validation_level == 'full'
        }
    
    def _convert_arrays_to_target_format(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Convert arrays to target format based on configuration.
        
        Args:
            data: Dictionary with numpy arrays
            
        Returns:
            Dictionary with arrays in target format (JAX or numpy)
        """
        output_format = self._get_output_format()
        
        if output_format == 'jax' and HAS_JAX and jax_available:
            logger.debug("Converting arrays to JAX format")
            return {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v 
                    for k, v in data.items()}
        
        elif output_format == 'auto' and HAS_JAX and jax_available:
            logger.debug("Auto-selecting JAX format (available)")
            return {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v 
                    for k, v in data.items()}
        
        elif output_format == 'auto':
            logger.debug("Auto-selecting numpy format (JAX not available)")
        
        return data  # Keep numpy format
    
    @log_performance(threshold=0.5)
    def load_experimental_data(self) -> Dict[str, Any]:
        """
        Load experimental data with priority: cache NPZ → raw HDF → error.
        
        Returns:
            Dictionary containing:
            - wavevector_q_list: Array of q values
            - phi_angles_list: Array of phi angles  
            - t1: Time array for first dimension
            - t2: Time array for second dimension
            - c2_exp: Experimental correlation data
        """
        # Construct file paths
        data_folder = self.exp_config.get('data_folder_path', './')
        data_file = self.exp_config.get('data_file_name', '')
        cache_folder = self.exp_config.get('cache_file_path', data_folder)
        
        # Get frame parameters
        start_frame = self.analyzer_config.get('start_frame', 1)
        end_frame = self.analyzer_config.get('end_frame', 8000)
        
        # Construct cache filename
        cache_template = self.exp_config.get('cache_filename_template', 
                                           'cached_c2_frames_{start_frame}_{end_frame}.npz')
        cache_filename = cache_template.format(start_frame=start_frame, end_frame=end_frame)
        cache_path = os.path.join(cache_folder, cache_filename)
        
        # Check cache first
        if os.path.exists(cache_path) and self.v2_config.get('cache_strategy', 'intelligent') != 'none':
            logger.info(f"Loading cached data from: {cache_path}")
            data = self._load_from_cache(cache_path)
        else:
            # Load from raw HDF file
            hdf_path = os.path.join(data_folder, data_file)
            if not os.path.exists(hdf_path):
                raise FileNotFoundError(f"Neither cache file {cache_path} nor HDF file {hdf_path} exists")
            
            logger.info(f"Loading raw data from: {hdf_path}")
            data = self._load_from_hdf(hdf_path)
            
            # Save to cache if caching enabled
            if self.v2_config.get('cache_strategy', 'intelligent') != 'none':
                logger.info(f"Saving processed data to cache: {cache_path}")
                self._save_to_cache(data, cache_path)
            
            # Generate text files
            self._save_text_files(data)
        
        # Initialize quality control if enabled
        quality_controller = self._initialize_quality_control()
        quality_results = []
        
        # Stage 1: Raw data validation
        if quality_controller:
            raw_validation_result = quality_controller.validate_data_stage(
                data, quality_controller.QualityControlStage.RAW_DATA
            )
            quality_results.append(raw_validation_result)
            
            # Apply auto-repair if data was modified
            if raw_validation_result.data_modified:
                logger.info("Raw data was modified by quality control auto-repair")
        
        # Apply filtering with quality control validation
        if quality_controller:
            filtered_validation_result = quality_controller.validate_data_stage(
                data, quality_controller.QualityControlStage.FILTERED_DATA, 
                previous_result=quality_results[-1] if quality_results else None
            )
            quality_results.append(filtered_validation_result)
        
        # Apply preprocessing pipeline if enabled with quality control
        data = self._apply_preprocessing_pipeline(data, quality_controller, quality_results)
        
        # Convert to target array format (JAX or numpy)
        data = self._convert_arrays_to_target_format(data)
        
        # Final quality control validation
        if quality_controller:
            final_validation_result = quality_controller.validate_data_stage(
                data, quality_controller.QualityControlStage.FINAL_DATA,
                previous_result=quality_results[-1] if quality_results else None
            )
            quality_results.append(final_validation_result)
            
            # Generate quality report if enabled
            if self.v2_config.get('quality_control', {}).get('generate_reports', True):
                quality_report = quality_controller.generate_quality_report(
                    quality_results, self._get_quality_report_path()
                )
                logger.info(f"Quality report generated with overall status: {quality_report['overall_summary']['status']}")
        
        # Perform legacy validation if enabled
        validation_settings = self._should_perform_validation()
        if any(validation_settings.values()) and not quality_controller:
            self._validate_loaded_data(data, validation_settings)
        
        logger.info(f"Data loaded successfully - shapes: q{data['wavevector_q_list'].shape}, "
                   f"phi{data['phi_angles_list'].shape}, c2{data['c2_exp'].shape}")
        
        return data
    
    @log_performance(threshold=0.2)
    def _load_from_cache(self, cache_path: str) -> Dict[str, Any]:
        """Load data from NPZ cache file."""
        with np.load(cache_path) as data:
            return {
                'wavevector_q_list': data['wavevector_q_list'],
                'phi_angles_list': data['phi_angles_list'], 
                't1': data['t1'],
                't2': data['t2'],
                'c2_exp': data['c2_exp']
            }
    
    @log_performance(threshold=1.0)
    def _load_from_hdf(self, hdf_path: str) -> Dict[str, Any]:
        """Load and process data from HDF5 file."""
        # Detect format
        logger.debug("Starting HDF5 format detection")
        format_type = self._detect_format(hdf_path)
        logger.info(f"Detected format: {format_type}")
        
        # Load based on format
        if format_type == 'aps_old':
            return self._load_aps_old_format(hdf_path)
        elif format_type == 'aps_u':
            return self._load_aps_u_format(hdf_path)
        else:
            raise XPCSDataFormatError(f"Unsupported format: {format_type}")
    
    @log_performance(threshold=0.1)
    def _detect_format(self, hdf_path: str) -> str:
        """Detect whether HDF5 file is APS old or APS-U new format."""
        with h5py.File(hdf_path, 'r') as f:
            # Check for APS-U format keys
            if ('xpcs' in f and 'qmap' in f['xpcs'] and 
                'dynamic_v_list_dim0' in f['xpcs/qmap'] and
                'twotime' in f['xpcs'] and 'correlation_map' in f['xpcs/twotime']):
                return 'aps_u'
            
            # Check for APS old format keys  
            elif ('xpcs' in f and 'dqlist' in f['xpcs'] and 'dphilist' in f['xpcs'] and
                  'exchange' in f and 'C2T_all' in f['exchange']):
                return 'aps_old'
            
            else:
                available_keys = list(f.keys())
                raise XPCSDataFormatError(
                    f"Cannot determine HDF5 format - missing expected keys. "
                    f"Available root keys: {available_keys}"
                )
    
    @log_performance(threshold=0.8)
    def _load_aps_old_format(self, hdf_path: str) -> Dict[str, Any]:
        """Load data from APS old format HDF5 file."""
        with h5py.File(hdf_path, 'r') as f:
            # Load q and phi lists
            dqlist = f['xpcs/dqlist'][0, :]  # Shape (1, N) -> (N,)
            dphilist = f['xpcs/dphilist'][0, :]  # Shape (1, N) -> (N,)
            
            # Load correlation data from exchange/C2T_all
            c2t_group = f['exchange/C2T_all']
            c2_keys = list(c2t_group.keys())
            
            # Load all correlation matrices first for potential quality filtering
            logger.debug(f"Loading {len(c2_keys)} correlation matrices for filtering")
            c2_matrices_for_filtering = []
            for key in c2_keys:
                c2_half = c2t_group[key][()]
                # Reconstruct full matrix from half matrix
                c2_full = self._reconstruct_full_matrix(c2_half)
                c2_matrices_for_filtering.append(c2_full)
            
            # Apply comprehensive data filtering
            logger.debug("Applying comprehensive data filtering")
            selected_indices = self._get_selected_indices(dqlist, dphilist, c2_matrices_for_filtering)
            
            # Select final data based on filtering (APS old format)
            logger.debug(f"Selecting data based on filtering results")
            if selected_indices is not None:
                # Filter all data arrays consistently
                filtered_dqlist = dqlist[selected_indices]
                filtered_dphilist = dphilist[selected_indices] 
                c2_matrices = [c2_matrices_for_filtering[i] for i in selected_indices]
                logger.debug(f"Selected {len(c2_matrices)} correlation matrices after filtering")
            else:
                # No filtering applied
                filtered_dqlist = dqlist
                filtered_dphilist = dphilist
                c2_matrices = c2_matrices_for_filtering
                logger.debug(f"No filtering applied - using all {len(c2_matrices)} correlation matrices")
            
            c2_exp = np.array(c2_matrices)
            
            # Calculate time arrays
            t1, t2 = self._calculate_time_arrays(c2_exp.shape[-1])
            
            return {
                'wavevector_q_list': filtered_dqlist,
                'phi_angles_list': filtered_dphilist,
                't1': t1,
                't2': t2,
                'c2_exp': c2_exp
            }
    
    @log_performance(threshold=0.8)
    def _load_aps_u_format(self, hdf_path: str) -> Dict[str, Any]:
        """Load data from APS-U new format HDF5 file."""
        with h5py.File(hdf_path, 'r') as f:
            # Load q and phi lists
            dqlist = f['xpcs/qmap/dynamic_v_list_dim0'][()]
            dphilist = f['xpcs/qmap/dynamic_v_list_dim1'][()]
            
            # Load correlation data from twotime/correlation_map
            corr_group = f['xpcs/twotime/correlation_map']
            c2_keys = list(corr_group.keys())
            
            # Load all correlation matrices first for potential quality filtering
            logger.debug(f"Loading {len(c2_keys)} correlation matrices for filtering")
            c2_matrices_for_filtering = []
            for key in c2_keys:
                c2_half = corr_group[key][()]
                # Reconstruct full matrix from half matrix
                c2_full = self._reconstruct_full_matrix(c2_half)
                c2_matrices_for_filtering.append(c2_full)
            
            # Apply comprehensive data filtering
            logger.debug("Applying comprehensive data filtering")
            selected_indices = self._get_selected_indices(dqlist, dphilist, c2_matrices_for_filtering)
            
            # Select final data based on filtering (APS-U format)
            logger.debug(f"Selecting data based on filtering results")
            if selected_indices is not None:
                # Filter all data arrays consistently
                filtered_dqlist = dqlist[selected_indices]
                filtered_dphilist = dphilist[selected_indices] 
                c2_matrices = [c2_matrices_for_filtering[i] for i in selected_indices]
                logger.debug(f"Selected {len(c2_matrices)} correlation matrices after filtering")
            else:
                # No filtering applied
                filtered_dqlist = dqlist
                filtered_dphilist = dphilist
                c2_matrices = c2_matrices_for_filtering
                logger.debug(f"No filtering applied - using all {len(c2_matrices)} correlation matrices")
            
            c2_exp = np.array(c2_matrices)
            
            # Calculate time arrays
            t1, t2 = self._calculate_time_arrays(c2_exp.shape[-1])
            
            return {
                'wavevector_q_list': filtered_dqlist,
                'phi_angles_list': filtered_dphilist,
                't1': t1,
                't2': t2,
                'c2_exp': c2_exp
            }
    
    def _reconstruct_full_matrix(self, c2_half: np.ndarray) -> np.ndarray:
        """
        Reconstruct full correlation matrix from half matrix (APS storage format).
        
        Based on pyXPCSViewer's approach:
        c2 = c2_half + c2_half.T
        c2[diag] /= 2
        """
        c2_full = c2_half + c2_half.T
        # Correct diagonal (was doubled in addition)
        diag_indices = np.diag_indices(c2_half.shape[0])
        c2_full[diag_indices] /= 2
        
        # Optional diagonal correction
        if self.exp_config.get('apply_diagonal_correction', True):
            c2_full = self._correct_diagonal(c2_full)
            
        return c2_full
    
    def _correct_diagonal(self, c2_mat: np.ndarray) -> np.ndarray:
        """
        Apply diagonal correction to correlation matrix.
        
        Based on pyXPCSViewer's correct_diagonal_c2 function.
        """
        size = c2_mat.shape[0]
        side_band = c2_mat[(np.arange(size - 1), np.arange(1, size))]
        diag_val = np.zeros(size)
        diag_val[:-1] += side_band
        diag_val[1:] += side_band
        norm = np.ones(size)
        norm[1:-1] = 2
        c2_mat[np.diag_indices(size)] = diag_val / norm
        return c2_mat
    
    def _get_selected_indices(self, dqlist: np.ndarray, dphilist: np.ndarray, 
                             correlation_matrices: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """
        Get indices for comprehensive data filtering based on configuration.
        
        Implements multi-criteria filtering including:
        - Q-range filtering based on wavevector values
        - Phi angle filtering (integrates with existing phi_filtering.py)
        - Quality-based filtering using correlation matrix properties
        - Frame-based filtering with configurable criteria
        - Combined filtering with AND/OR logic
        
        Args:
            dqlist: Array of q-values (wavevector magnitudes)
            dphilist: Array of phi angles in degrees
            correlation_matrices: Optional list of correlation matrices for quality filtering
            
        Returns:
            Array of selected indices, or None if no filtering is applied
        """
        try:
            # Import filtering utilities
            from homodyne.data.filtering_utils import XPCSDataFilter, DataFilteringError
            
            # Check if filtering is enabled
            filtering_config = self.config.get('data_filtering', {})
            if not filtering_config.get('enabled', False):
                logger.debug("Data filtering disabled in configuration")
                return None
            
            logger.info(f"Applying comprehensive data filtering to {len(dqlist)} data points")
            
            # Initialize data filter
            data_filter = XPCSDataFilter(self.config)
            
            # Apply comprehensive filtering
            filtering_result = data_filter.apply_filtering(dqlist, dphilist, correlation_matrices)
            
            # Log filtering statistics
            if filtering_result.filter_statistics:
                logger.info("Filtering statistics:")
                for filter_name, stats in filtering_result.filter_statistics.items():
                    if isinstance(stats, dict) and 'selected_count' in stats:
                        logger.info(f"  {filter_name}: {stats['selected_count']} selected "
                                   f"({stats.get('selection_fraction', 0.0):.2%})")
            
            # Handle warnings and errors
            if filtering_result.warnings:
                for warning in filtering_result.warnings:
                    logger.warning(f"Data filtering warning: {warning}")
            
            if filtering_result.errors:
                for error in filtering_result.errors:
                    logger.error(f"Data filtering error: {error}")
                if not filtering_result.fallback_used:
                    raise DataFilteringError(f"Data filtering failed: {filtering_result.errors}")
            
            # Log final result
            if filtering_result.selected_indices is not None:
                selected_count = len(filtering_result.selected_indices)
                total_count = len(dqlist)
                selection_fraction = selected_count / total_count if total_count > 0 else 0.0
                
                logger.info(f"Data filtering completed: {selected_count}/{total_count} "
                           f"data points selected ({selection_fraction:.2%})")
                
                if filtering_result.fallback_used:
                    logger.warning("Filtering used fallback - all data points included")
                
                # Additional integration with phi filtering for compatibility
                selected_indices = self._integrate_with_phi_filtering(
                    filtering_result.selected_indices, dphilist, filtering_result
                )
                
                return selected_indices
            else:
                logger.info("No data filtering applied - returning all indices")
                return None
                
        except ImportError as e:
            logger.warning(f"Filtering utilities not available: {e}. Skipping data filtering.")
            return None
        except Exception as e:
            logger.error(f"Data filtering failed with unexpected error: {e}")
            
            # Check if we should fallback or raise
            fallback_on_empty = filtering_config.get('fallback_on_empty', True)
            if fallback_on_empty:
                logger.warning("Falling back to no filtering due to error")
                return None
            else:
                raise XPCSDataFormatError(f"Data filtering failed: {e}")
    
    def _integrate_with_phi_filtering(self, selected_indices: np.ndarray, dphilist: np.ndarray, 
                                     filtering_result) -> np.ndarray:
        """
        Integrate with existing phi filtering system for backward compatibility.
        
        This method ensures that the new filtering system works well with
        existing phi angle filtering configurations and provides consistent results.
        """
        try:
            # Import existing phi filtering system
            from homodyne.data.phi_filtering import PhiAngleFilter
            
            # Check if phi filtering was already applied in the main filtering
            if 'phi_range' in filtering_result.filters_applied:
                logger.debug("Phi filtering already applied in main filtering system")
                return selected_indices
            
            # Check for legacy phi filtering configuration
            optimization_config = self.config.get('optimization_config', {})
            angle_filtering = optimization_config.get('angle_filtering', {})
            
            if not angle_filtering.get('enabled', False):
                logger.debug("Legacy phi filtering not enabled")
                return selected_indices
            
            # Apply legacy phi filtering to already filtered data
            selected_phi_angles = dphilist[selected_indices]
            
            phi_filter = PhiAngleFilter(self.config)
            phi_indices, filtered_angles = phi_filter.filter_angles_for_optimization(selected_phi_angles)
            
            # Map back to original indices
            final_selected_indices = selected_indices[phi_indices]
            
            logger.info(f"Legacy phi filtering applied: {len(final_selected_indices)} "
                       f"out of {len(selected_indices)} filtered indices selected")
            
            return final_selected_indices
            
        except ImportError:
            logger.debug("Phi filtering system not available - using original selection")
            return selected_indices
        except Exception as e:
            logger.warning(f"Phi filtering integration failed: {e} - using original selection")
            return selected_indices
    
    def _calculate_time_arrays(self, matrix_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate t1 and t2 time arrays based on configuration."""
        dt = self.analyzer_config.get('dt', 1.0)
        start_frame = self.analyzer_config.get('start_frame', 1)
        end_frame = self.analyzer_config.get('end_frame', matrix_size + start_frame - 1)
        
        # t1 = t2 = dt * (end_frame - start_frame)
        time_max = dt * (end_frame - start_frame)
        time_array = np.linspace(0, time_max, matrix_size)
        
        return time_array, time_array
    
    @log_performance(threshold=0.3)
    def _save_to_cache(self, data: Dict[str, Any], cache_path: str) -> None:
        """Save processed data to NPZ cache file."""
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Convert JAX arrays back to numpy for caching
        cache_data = {}
        for key, value in data.items():
            if HAS_JAX and hasattr(value, 'device'):  # JAX array
                cache_data[key] = np.array(value)
            else:
                cache_data[key] = value
        
        # Save with compression if specified
        if self.exp_config.get('cache_compression', True):
            np.savez_compressed(cache_path, **cache_data)
        else:
            np.savez(cache_path, **cache_data)
        
        logger.debug(f"Cached data saved to: {cache_path}")
    
    @log_performance(threshold=0.1)
    def _save_text_files(self, data: Dict[str, Any]) -> None:
        """Save phi_angles and wavevector_q lists to text files."""
        # Get output directory
        phi_folder = self.exp_config.get('phi_angles_path', './')
        data_folder = self.exp_config.get('data_folder_path', './')
        
        # Convert JAX arrays to numpy for text file saving
        phi_angles = np.array(data['phi_angles_list']) if HAS_JAX else data['phi_angles_list']
        q_values = np.array(data['wavevector_q_list']) if HAS_JAX else data['wavevector_q_list']
        
        # Save phi angles list
        phi_file = os.path.join(phi_folder, 'phi_angles_list.txt')
        os.makedirs(os.path.dirname(phi_file), exist_ok=True)
        np.savetxt(phi_file, phi_angles, fmt='%.6f', 
                   header='Phi angles (degrees)', comments='# ')
        
        # Save wavevector q list  
        q_file = os.path.join(data_folder, 'wavevector_q_list.txt')
        np.savetxt(q_file, q_values, fmt='%.8e',
                   header='Wavevector q (1/Angstrom)', comments='# ')
        
        logger.debug(f"Text files saved: {phi_file}, {q_file}")
    
    def _validate_loaded_data(self, data: Dict[str, Any], validation_settings: Dict[str, bool]) -> None:
        """
        Perform validation on loaded data.
        
        Args:
            data: Loaded data dictionary
            validation_settings: Validation configuration
        """
        if validation_settings.get('physics_checks', False):
            self._perform_physics_validation(data)
        
        if validation_settings.get('data_quality', False):
            self._perform_data_quality_checks(data, validation_settings.get('comprehensive', False))
    
    def _perform_physics_validation(self, data: Dict[str, Any]) -> None:
        """Perform physics-based validation using v2 PhysicsConstants."""
        if not HAS_PHYSICS_VALIDATION:
            logger.warning("Physics validation requested but v2 physics module not available")
            return
        
        # Validate q-range
        q_values = np.array(data['wavevector_q_list']) if HAS_JAX else data['wavevector_q_list']
        if np.any(q_values < PhysicsConstants.Q_MIN_TYPICAL):
            logger.warning(f"Some q-values below typical range: {PhysicsConstants.Q_MIN_TYPICAL}")
        if np.any(q_values > PhysicsConstants.Q_MAX_TYPICAL):
            logger.warning(f"Some q-values above typical range: {PhysicsConstants.Q_MAX_TYPICAL}")
        
        # Validate time parameters
        dt = self.analyzer_config.get('dt', 1.0)
        if dt < PhysicsConstants.TIME_MIN_XPCS:
            logger.warning(f"Time step dt={dt}s below typical XPCS minimum: {PhysicsConstants.TIME_MIN_XPCS}s")
        
        logger.info("Physics validation completed")
    
    def _perform_data_quality_checks(self, data: Dict[str, Any], comprehensive: bool = False) -> None:
        """Perform data quality validation."""
        c2_exp = np.array(data['c2_exp']) if HAS_JAX else data['c2_exp']
        
        # Basic checks
        if np.any(~np.isfinite(c2_exp)):
            logger.error("Correlation data contains non-finite values (NaN or Inf)")
        
        if np.any(c2_exp < 0):
            logger.warning("Correlation data contains negative values")
        
        # Check for reasonable correlation values (should be around 1.0 at t=0)
        diagonal_values = np.array([c2_exp[i].diagonal() for i in range(len(c2_exp))])
        mean_diagonal = np.mean(diagonal_values[:, 0])  # t=0 correlation
        if not (0.5 < mean_diagonal < 2.0):
            logger.warning(f"Unusual t=0 correlation value: {mean_diagonal:.3f} (expected ~1.0)")
        
        if comprehensive:
            # Additional comprehensive checks
            logger.info("Performing comprehensive data quality analysis...")
            
            # Check correlation decay
            decay_rates = []
            for i in range(len(c2_exp)):
                diag = c2_exp[i].diagonal()
                if len(diag) > 10:
                    decay_rate = (diag[0] - diag[10]) / diag[0]
                    decay_rates.append(decay_rate)
            
            if decay_rates:
                mean_decay = np.mean(decay_rates)
                logger.info(f"Mean correlation decay over 10 time steps: {mean_decay:.3f}")
        
        logger.info("Data quality validation completed")
    
    def _initialize_quality_control(self) -> Optional[Any]:
        """Initialize quality control system if enabled."""
        try:
            quality_config = self.config.get('quality_control', {})
            if not quality_config.get('enabled', False):
                logger.debug("Quality control disabled in configuration")
                return None
            
            # Import quality control system
            from homodyne.data.quality_controller import DataQualityController
            
            logger.info("Initializing data quality control system")
            controller = DataQualityController(self.config)
            
            # Store reference to stage enum for convenience
            from homodyne.data.quality_controller import QualityControlStage
            controller.QualityControlStage = QualityControlStage
            
            return controller
            
        except ImportError as e:
            logger.warning(f"Quality control system not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize quality control: {e}")
            return None
    
    def _get_quality_report_path(self) -> str:
        """Generate path for quality control report."""
        data_folder = self.exp_config.get('data_folder_path', './')
        data_file = self.exp_config.get('data_file_name', 'unknown')
        data_file_base = os.path.splitext(data_file)[0]
        
        # Create quality reports subdirectory
        quality_dir = os.path.join(data_folder, 'quality_reports')
        os.makedirs(quality_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        quality_filename = f"{data_file_base}_quality_report_{timestamp}.json"
        
        return os.path.join(quality_dir, quality_filename)
    
    @log_performance(threshold=0.5)
    def _apply_preprocessing_pipeline(self, data: Dict[str, Any], 
                                     quality_controller: Optional[Any] = None,
                                     quality_results: Optional[List] = None) -> Dict[str, Any]:
        """
        Apply preprocessing pipeline to loaded data if enabled.
        
        Args:
            data: Raw data loaded from HDF5 files
            
        Returns:
            Processed data after applying preprocessing pipeline
        """
        try:
            # Check if preprocessing is enabled
            preprocessing_config = self.config.get('preprocessing', {})
            if not preprocessing_config.get('enabled', False):
                logger.debug("Preprocessing pipeline disabled in configuration")
                return data
            
            logger.info("Applying preprocessing pipeline to loaded data")
            
            # Import preprocessing pipeline
            from homodyne.data.preprocessing import PreprocessingPipeline
            
            # Create and execute preprocessing pipeline
            pipeline = PreprocessingPipeline(self.config)
            result = pipeline.process(data)
            
            if result.success:
                logger.info(f"Preprocessing pipeline completed successfully")
                logger.info(f"Pipeline stages executed: {len(result.stage_results)}")
                
                # Log stage results
                successful_stages = sum(result.stage_results.values())
                total_stages = len(result.stage_results)
                logger.info(f"Successful stages: {successful_stages}/{total_stages}")
                
                # Quality control validation after preprocessing
                if quality_controller and quality_results:
                    preprocessing_validation_result = quality_controller.validate_data_stage(
                        result.data, quality_controller.QualityControlStage.PREPROCESSED_DATA,
                        previous_result=quality_results[-1] if quality_results else None
                    )
                    quality_results.append(preprocessing_validation_result)
                    
                    if not preprocessing_validation_result.passed:
                        logger.warning(f"Preprocessing quality validation failed: score={preprocessing_validation_result.metrics.overall_score:.1f}")
                
                # Save provenance if requested
                if preprocessing_config.get('save_provenance', False):
                    provenance_path = self._get_provenance_path()
                    pipeline.save_provenance(result.provenance, provenance_path)
                
                # Log warnings if any
                if result.provenance.warnings:
                    for warning in result.provenance.warnings:
                        logger.warning(f"Preprocessing warning: {warning}")
                
                return result.data
            else:
                logger.error("Preprocessing pipeline failed")
                
                # Log errors
                for error in result.provenance.errors:
                    logger.error(f"Preprocessing error: {error}")
                
                # Return original data if fallback is enabled
                if preprocessing_config.get('fallback_on_failure', True):
                    logger.warning("Falling back to original data after preprocessing failure")
                    return data
                else:
                    raise XPCSDataFormatError("Preprocessing pipeline failed and fallback disabled")
                    
        except ImportError as e:
            logger.warning(f"Preprocessing pipeline not available: {e}. Using original data.")
            return data
        except Exception as e:
            logger.error(f"Unexpected error in preprocessing pipeline: {e}")
            
            # Check fallback setting
            preprocessing_config = self.config.get('preprocessing', {})
            if preprocessing_config.get('fallback_on_failure', True):
                logger.warning("Falling back to original data after preprocessing error")
                return data
            else:
                raise XPCSDataFormatError(f"Preprocessing pipeline failed: {e}")
    
    def _get_provenance_path(self) -> str:
        """Generate path for saving preprocessing provenance."""
        # Use data folder as base
        data_folder = self.exp_config.get('data_folder_path', './')
        
        # Create provenance subdirectory
        provenance_dir = os.path.join(data_folder, 'preprocessing_provenance')
        os.makedirs(provenance_dir, exist_ok=True)
        
        # Generate filename based on data file and timestamp
        data_file = self.exp_config.get('data_file_name', 'unknown')
        data_file_base = os.path.splitext(data_file)[0]
        timestamp = int(time.time())
        
        provenance_filename = f"{data_file_base}_preprocessing_provenance_{timestamp}.json"
        return os.path.join(provenance_dir, provenance_filename)

# Convenience function for simple usage
@log_performance(threshold=1.0)
def load_xpcs_data(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load XPCS data from configuration file.
    
    Supports both YAML and JSON configuration files with auto-detection.
    
    Args:
        config_path: Path to YAML or JSON configuration file
        
    Returns:
        Dictionary containing loaded experimental data with JAX arrays when available
        
    Example:
        >>> data = load_xpcs_data("xpcs_config.yaml")
        >>> print(data.keys())
        dict_keys(['wavevector_q_list', 'phi_angles_list', 't1', 't2', 'c2_exp'])
    """
    loader = XPCSDataLoader(config_path=config_path)
    return loader.load_experimental_data()

# Export main classes and functions
__all__ = [
    'XPCSDataLoader',
    'load_xpcs_data', 
    'XPCSDataFormatError',
    'XPCSDependencyError',
    'XPCSConfigurationError',
    'load_xpcs_config'
]