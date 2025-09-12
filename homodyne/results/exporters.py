"""
Multi-Format Result Exporters for Homodyne v2
==============================================

Comprehensive export system supporting multiple formats with validation,
metadata embedding, and format-specific optimizations.

Supported Formats:
- YAML: Human-readable configuration-style output
- JSON: Web-compatible structured data
- NPZ: NumPy compressed binary format
- HDF5: Hierarchical scientific data format
- CSV: Tabular data for spreadsheet applications

Key Features:
- Format-specific validation and optimization
- Metadata embedding with provenance tracking
- Compression and size optimization
- Error handling and recovery
- Export progress tracking for large datasets
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set
from datetime import datetime
import numpy as np

from homodyne.utils.logging import get_logger
from homodyne.optimization.variational import VIResult
from homodyne.optimization.mcmc import MCMCResult
from homodyne.optimization.hybrid import HybridResult

logger = get_logger(__name__)

ResultType = Union[VIResult, MCMCResult, HybridResult]


class ResultExporter:
    """
    Base result exporter with common functionality.
    
    Provides foundation for format-specific exporters with
    validation, metadata handling, and error recovery.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize result exporter.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metadata
        self.export_timestamp = datetime.now()
        self.export_metadata = self._create_base_metadata()
    
    def _create_base_metadata(self) -> Dict[str, Any]:
        """
        Create base export metadata.
        
        Returns:
            Base metadata dictionary
        """
        return {
            'export_timestamp': self.export_timestamp.isoformat(),
            'homodyne_version': getattr(__import__('homodyne'), '__version__', 'unknown'),
            'exporter_class': self.__class__.__name__,
            'export_format_version': '2.0'
        }
    
    def _extract_result_data(self, result: ResultType) -> Dict[str, Any]:
        """
        Extract exportable data from analysis result.
        
        Args:
            result: Analysis result object
            
        Returns:
            Exportable data dictionary
        """
        data = {
            'method': result.__class__.__name__.replace('Result', '').lower(),
            'analysis_mode': getattr(result, 'analysis_mode', 'unknown'),
            'success': True,
            'metadata': self.export_metadata
        }
        
        # Common parameters
        if hasattr(result, 'mean_params'):
            data['parameters'] = {
                'mean': result.mean_params.tolist() if hasattr(result.mean_params, 'tolist') else result.mean_params,
                'count': len(result.mean_params) if hasattr(result.mean_params, '__len__') else 0
            }
        
        if hasattr(result, 'std_params'):
            data['parameters']['std'] = result.std_params.tolist() if hasattr(result.std_params, 'tolist') else result.std_params
        
        # Quality metrics
        if hasattr(result, 'chi_squared'):
            data['quality_metrics'] = {'chi_squared': float(result.chi_squared)}
        
        if hasattr(result, 'final_elbo'):
            if 'quality_metrics' not in data:
                data['quality_metrics'] = {}
            data['quality_metrics']['final_elbo'] = float(result.final_elbo)
        
        # Method-specific data
        if isinstance(result, VIResult):
            data.update(self._extract_vi_data(result))
        elif isinstance(result, MCMCResult):
            data.update(self._extract_mcmc_data(result))
        elif isinstance(result, HybridResult):
            data.update(self._extract_hybrid_data(result))
        
        return data
    
    def _extract_vi_data(self, result: VIResult) -> Dict[str, Any]:
        """
        Extract VI-specific data.
        
        Args:
            result: VI result object
            
        Returns:
            VI-specific data dictionary
        """
        return {
            'vi_specific': {
                'converged': getattr(result, 'converged', False),
                'n_iterations': getattr(result, 'n_iterations', 0),
                'final_elbo': float(getattr(result, 'final_elbo', 0.0)),
                'elbo_history': getattr(result, 'elbo_history', [])
            }
        }
    
    def _extract_mcmc_data(self, result: MCMCResult) -> Dict[str, Any]:
        """
        Extract MCMC-specific data.
        
        Args:
            result: MCMC result object
            
        Returns:
            MCMC-specific data dictionary
        """
        mcmc_data = {
            'mcmc_specific': {
                'n_samples': getattr(result, 'n_samples', 0),
                'n_chains': getattr(result, 'n_chains', 0)
            }
        }
        
        # Add convergence diagnostics if available
        if hasattr(result, 'r_hat'):
            mcmc_data['mcmc_specific']['r_hat'] = result.r_hat.tolist() if hasattr(result.r_hat, 'tolist') else result.r_hat
        
        if hasattr(result, 'ess'):
            mcmc_data['mcmc_specific']['ess'] = result.ess.tolist() if hasattr(result.ess, 'tolist') else result.ess
        
        return mcmc_data
    
    def _extract_hybrid_data(self, result: HybridResult) -> Dict[str, Any]:
        """
        Extract Hybrid-specific data.
        
        Args:
            result: Hybrid result object
            
        Returns:
            Hybrid-specific data dictionary
        """
        hybrid_data = {
            'hybrid_specific': {
                'recommended_method': getattr(result, 'recommended_method', 'unknown')
            }
        }
        
        # Include VI and MCMC sub-results if available
        if hasattr(result, 'vi_result'):
            hybrid_data['vi_phase'] = self._extract_vi_data(result.vi_result)
        
        if hasattr(result, 'mcmc_result'):
            hybrid_data['mcmc_phase'] = self._extract_mcmc_data(result.mcmc_result)
        
        return hybrid_data
    
    def _validate_export_data(self, data: Dict[str, Any]) -> bool:
        """
        Validate export data structure.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid for export
        """
        required_keys = ['method', 'success', 'metadata']
        
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required export key: {key}")
                return False
        
        # Check for NaN or infinite values
        def has_invalid_numbers(obj: Any) -> bool:
            if isinstance(obj, (int, float)):
                return not np.isfinite(obj)
            elif isinstance(obj, list):
                return any(has_invalid_numbers(item) for item in obj)
            elif isinstance(obj, dict):
                return any(has_invalid_numbers(value) for value in obj.values())
            return False
        
        if has_invalid_numbers(data):
            logger.error("Export data contains NaN or infinite values")
            return False
        
        return True


class YAMLExporter(ResultExporter):
    """YAML format exporter for human-readable output."""
    
    def export(self, result: ResultType, filename: Optional[str] = None) -> Path:
        """
        Export result to YAML format.
        
        Args:
            result: Analysis result to export
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML export")
        
        filename = filename or f"analysis_result_{self.export_timestamp.strftime('%Y%m%d_%H%M%S')}.yaml"
        output_path = self.output_dir / filename
        
        data = self._extract_result_data(result)
        
        if not self._validate_export_data(data):
            raise ValueError("Invalid data for YAML export")
        
        # Add YAML-specific formatting hints
        data['_format_info'] = {
            'format': 'yaml',
            'readable': True,
            'suggested_viewer': 'text editor or YAML viewer'
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)
        
        logger.info(f"✓ YAML export saved: {output_path}")
        return output_path


class JSONExporter(ResultExporter):
    """JSON format exporter for web compatibility."""
    
    def export(self, result: ResultType, filename: Optional[str] = None, pretty: bool = True) -> Path:
        """
        Export result to JSON format.
        
        Args:
            result: Analysis result to export
            filename: Custom filename (optional)
            pretty: Use pretty-printing (default True)
            
        Returns:
            Path to exported file
        """
        filename = filename or f"analysis_result_{self.export_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        output_path = self.output_dir / filename
        
        data = self._extract_result_data(result)
        
        if not self._validate_export_data(data):
            raise ValueError("Invalid data for JSON export")
        
        # Add JSON-specific metadata
        data['_format_info'] = {
            'format': 'json',
            'web_compatible': True,
            'suggested_viewer': 'web browser or JSON viewer'
        }
        
        with open(output_path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
            else:
                json.dump(data, f, separators=(',', ':'))
        
        logger.info(f"✓ JSON export saved: {output_path}")
        return output_path


class NPZExporter(ResultExporter):
    """NumPy NPZ format exporter for efficient binary storage."""
    
    def export(self, result: ResultType, filename: Optional[str] = None) -> Path:
        """
        Export result to NPZ format.
        
        Args:
            result: Analysis result to export
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        filename = filename or f"analysis_result_{self.export_timestamp.strftime('%Y%m%d_%H%M%S')}.npz"
        output_path = self.output_dir / filename
        
        data = self._extract_result_data(result)
        
        # Convert data to NumPy arrays where possible
        arrays_to_save = {}
        
        # Extract numerical data
        if 'parameters' in data and 'mean' in data['parameters']:
            arrays_to_save['mean_params'] = np.array(data['parameters']['mean'])
        
        if 'parameters' in data and 'std' in data['parameters']:
            arrays_to_save['std_params'] = np.array(data['parameters']['std'])
        
        # VI-specific arrays
        if 'vi_specific' in data and 'elbo_history' in data['vi_specific']:
            arrays_to_save['elbo_history'] = np.array(data['vi_specific']['elbo_history'])
        
        # MCMC-specific arrays
        if 'mcmc_specific' in data:
            mcmc_data = data['mcmc_specific']
            if 'r_hat' in mcmc_data:
                arrays_to_save['r_hat'] = np.array(mcmc_data['r_hat'])
            if 'ess' in mcmc_data:
                arrays_to_save['ess'] = np.array(mcmc_data['ess'])
        
        # Store metadata as structured array or string
        metadata_str = json.dumps(data['metadata'])
        arrays_to_save['metadata'] = np.array([metadata_str], dtype='U')
        
        # Save method and quality info as strings
        arrays_to_save['method'] = np.array([data['method']], dtype='U')
        if 'quality_metrics' in data:
            quality_str = json.dumps(data['quality_metrics'])
            arrays_to_save['quality_metrics'] = np.array([quality_str], dtype='U')
        
        np.savez_compressed(output_path, **arrays_to_save)
        
        logger.info(f"✓ NPZ export saved: {output_path}")
        return output_path


class HDF5Exporter(ResultExporter):
    """HDF5 format exporter for hierarchical scientific data."""
    
    def export(self, result: ResultType, filename: Optional[str] = None) -> Path:
        """
        Export result to HDF5 format.
        
        Args:
            result: Analysis result to export
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py is required for HDF5 export")
        
        filename = filename or f"analysis_result_{self.export_timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
        output_path = self.output_dir / filename
        
        data = self._extract_result_data(result)
        
        with h5py.File(output_path, 'w') as f:
            # Create hierarchical structure
            self._write_hdf5_group(f, data, '/')
        
        logger.info(f"✓ HDF5 export saved: {output_path}")
        return output_path
    
    def _write_hdf5_group(self, parent, data: Dict[str, Any], path: str) -> None:
        """
        Recursively write data to HDF5 group.
        
        Args:
            parent: HDF5 parent group
            data: Data to write
            path: Current path in hierarchy
        """
        for key, value in data.items():
            if isinstance(value, dict):
                # Create subgroup
                group = parent.create_group(key)
                self._write_hdf5_group(group, value, f"{path}/{key}")
            elif isinstance(value, (list, tuple)):
                # Convert to array
                try:
                    arr = np.array(value)
                    parent.create_dataset(key, data=arr)
                except (ValueError, TypeError):
                    # Store as string if conversion fails
                    parent.attrs[key] = str(value)
            elif isinstance(value, (int, float, str, bool)):
                # Store as attribute
                parent.attrs[key] = value
            elif isinstance(value, np.ndarray):
                # Store as dataset
                parent.create_dataset(key, data=value)
            else:
                # Convert to string
                parent.attrs[key] = str(value)


class CSVExporter(ResultExporter):
    """CSV format exporter for tabular data."""
    
    def export(self, result: ResultType, filename: Optional[str] = None) -> Path:
        """
        Export result to CSV format.
        
        Args:
            result: Analysis result to export
            filename: Custom filename (optional)
            
        Returns:
            Path to exported file
        """
        filename = filename or f"analysis_result_{self.export_timestamp.strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = self.output_dir / filename
        
        data = self._extract_result_data(result)
        
        # Create tabular representation
        rows = self._flatten_to_rows(data)
        
        with open(output_path, 'w', newline='') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"✓ CSV export saved: {output_path}")
        return output_path
    
    def _flatten_to_rows(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Flatten nested data to tabular rows.
        
        Args:
            data: Nested data dictionary
            
        Returns:
            List of flattened row dictionaries
        """
        rows = []
        
        # Create base row with non-array data
        base_row = {
            'method': data.get('method', ''),
            'analysis_mode': data.get('analysis_mode', ''),
            'success': data.get('success', False),
            'export_timestamp': data.get('metadata', {}).get('export_timestamp', '')
        }
        
        # Add quality metrics
        if 'quality_metrics' in data:
            for key, value in data['quality_metrics'].items():
                base_row[f'quality_{key}'] = value
        
        # Handle parameters
        if 'parameters' in data:
            params = data['parameters']
            if 'mean' in params:
                mean_params = params['mean']
                if isinstance(mean_params, list):
                    for i, param in enumerate(mean_params):
                        base_row[f'param_{i}_mean'] = param
                
                if 'std' in params:
                    std_params = params['std']
                    if isinstance(std_params, list):
                        for i, param in enumerate(std_params):
                            base_row[f'param_{i}_std'] = param
        
        rows.append(base_row)
        return rows


class MultiFormatExporter:
    """
    Multi-format exporter supporting simultaneous export to multiple formats.
    
    Coordinates export across different formats with unified error handling
    and progress tracking.
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize multi-format exporter.
        
        Args:
            output_dir: Output directory path
        """
        self.output_dir = Path(output_dir)
        self.exporters = {
            'yaml': YAMLExporter(self.output_dir),
            'json': JSONExporter(self.output_dir),
            'npz': NPZExporter(self.output_dir),
            'hdf5': HDF5Exporter(self.output_dir),
            'csv': CSVExporter(self.output_dir)
        }
    
    def export_all_formats(self, 
                          result: ResultType,
                          formats: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Export result to multiple formats.
        
        Args:
            result: Analysis result to export
            formats: List of formats to export (default: all supported)
            
        Returns:
            Dictionary mapping format names to output paths
        """
        if formats is None:
            formats = list(self.exporters.keys())
        
        logger.info(f"🔄 Exporting to {len(formats)} formats: {', '.join(formats)}")
        
        export_paths = {}
        errors = {}
        
        for format_name in formats:
            if format_name not in self.exporters:
                logger.warning(f"Unknown export format: {format_name}")
                continue
            
            try:
                exporter = self.exporters[format_name]
                output_path = exporter.export(result)
                export_paths[format_name] = output_path
                
            except Exception as e:
                logger.error(f"Export to {format_name} failed: {e}")
                errors[format_name] = str(e)
        
        # Log summary
        if export_paths:
            logger.info(f"✓ Successfully exported to {len(export_paths)} formats")
        
        if errors:
            logger.warning(f"❌ Failed to export to {len(errors)} formats")
            for format_name, error in errors.items():
                logger.warning(f"  {format_name}: {error}")
        
        return export_paths
    
    def export_selective(self,
                        result: ResultType,
                        format_preferences: Dict[str, bool]) -> Dict[str, Path]:
        """
        Export to formats based on preferences.
        
        Args:
            result: Analysis result to export
            format_preferences: Dictionary of format preferences
            
        Returns:
            Dictionary mapping format names to output paths
        """
        enabled_formats = [fmt for fmt, enabled in format_preferences.items() if enabled]
        return self.export_all_formats(result, enabled_formats)
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported export formats.
        
        Returns:
            List of supported format names
        """
        return list(self.exporters.keys())
    
    def estimate_export_sizes(self, result: ResultType) -> Dict[str, str]:
        """
        Estimate export file sizes for each format.
        
        Args:
            result: Analysis result to estimate
            
        Returns:
            Dictionary mapping formats to size estimates
        """
        # This is a rough estimation - actual sizes will vary
        base_data = self.exporters['yaml']._extract_result_data(result)
        
        # Estimate based on data complexity
        param_count = len(base_data.get('parameters', {}).get('mean', []))
        has_arrays = any('history' in str(key).lower() for key in str(base_data))
        
        estimates = {
            'yaml': '2-10 KB' if param_count < 10 else '10-50 KB',
            'json': '1-5 KB' if param_count < 10 else '5-25 KB',
            'npz': '1-2 KB' if not has_arrays else '5-20 KB',
            'hdf5': '2-5 KB' if not has_arrays else '10-50 KB',
            'csv': '500 B-2 KB'
        }
        
        return estimates