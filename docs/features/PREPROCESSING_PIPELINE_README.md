# Advanced Data Preprocessing Pipeline for Homodyne v2

## Overview

Homodyne v2 now includes an advanced data preprocessing pipeline that provides intelligent data transformation capabilities far beyond basic diagonal correction. This system builds on the config-based filtering implemented by Subagent 1 to deliver a comprehensive, production-ready preprocessing solution.

## Architecture

The preprocessing pipeline follows a multi-stage architecture:
```
Raw Data → Filter → Transform → Normalize → Validate → Processed Data
```

### Pipeline Stages

1. **load_raw**: Load raw data (handled by `xpcs_loader.py`)
2. **apply_filtering**: Use config-based filtering from `filtering_utils.py` 
3. **correct_diagonal**: Enhanced diagonal correction with statistical methods
4. **normalize_data**: Multiple normalization strategies
5. **reduce_noise**: Optional denoising algorithms
6. **standardize_format**: Ensure consistent data formats
7. **validate_output**: Final data integrity and physics validation

Each stage can be independently enabled/disabled and configured through YAML.

## Key Features

### 🎯 Enhanced Diagonal Correction
- **Statistical Methods**: Use robust estimators (median, trimmed mean) instead of simple interpolation
- **Interpolation-based**: Smooth transitions using linear/cubic interpolation
- **Configurable Windows**: Adjust neighborhood size for statistical methods

### 📊 Advanced Normalization
- **Baseline**: Physics-preserving normalization by t=0 correlation value
- **Statistical**: Z-score normalization (mean=0, std=1) 
- **MinMax**: Scale to [0, 1] range
- **Robust**: Percentile-based scaling resistant to outliers
- **Physics-based**: Constrained normalization preserving correlation properties

### 🔧 Noise Reduction
- **Median Filtering**: Effective for salt-and-pepper noise
- **Gaussian Smoothing**: Good for thermal noise
- **Wiener Filtering**: Adaptive noise reduction (requires scipy)
- **Savitzky-Golay**: Feature-preserving smoothing (requires scipy)

### 🏗️ Data Standardization
- Ensures consistent data types and formats
- Compatible with both APS old and APS-U formats
- Validates array consistency and shapes

### 📋 Complete Audit Trail
- **Provenance Tracking**: Full record of all transformations
- **Reproducibility**: Save/load processing parameters for exact reproduction
- **Performance Monitoring**: Track processing time and memory usage
- **Version Tracking**: Record which preprocessing version was used

## Configuration

### Basic Configuration

Add to your YAML config file:

```yaml
preprocessing:
  enabled: true
  progress_reporting: true
  fallback_on_failure: true
  stages:
    correct_diagonal:
      enabled: true
      method: statistical
      estimator: median
    normalize_data:
      enabled: true
      method: baseline
    reduce_noise:
      enabled: false
    standardize_format:
      enabled: true
    validate_output:
      enabled: true
```

### Advanced Configuration Examples

#### High-Quality Analysis
```yaml
preprocessing:
  enabled: true
  save_provenance: true
  stages:
    correct_diagonal:
      enabled: true
      method: statistical
      estimator: median
      window_size: 3
    normalize_data:
      enabled: true
      method: physics_based
    reduce_noise:
      enabled: true
      method: gaussian
      sigma: 0.8
    standardize_format:
      enabled: true
    validate_output:
      enabled: true
```

#### Noisy Data Processing
```yaml
preprocessing:
  enabled: true
  stages:
    correct_diagonal:
      enabled: true
      method: statistical
      estimator: trimmed_mean
      trim_fraction: 0.3
    normalize_data:
      enabled: true
      method: robust
      percentile_range: [10, 90]
    reduce_noise:
      enabled: true
      method: median
      kernel_size: 5
    standardize_format:
      enabled: true
    validate_output:
      enabled: true
```

#### Fast Preprocessing
```yaml
preprocessing:
  enabled: true
  progress_reporting: false
  stages:
    correct_diagonal:
      enabled: true
      method: basic
    normalize_data:
      enabled: true
      method: baseline
    reduce_noise:
      enabled: false
    standardize_format:
      enabled: true
    validate_output:
      enabled: true
```

## Usage Examples

### With XPCS Data Loader
The preprocessing pipeline integrates seamlessly with the existing XPCS data loader:

```python
from homodyne.data import XPCSDataLoader

# Load configuration with preprocessing enabled
loader = XPCSDataLoader(config_path="config_with_preprocessing.yaml")
data = loader.load_experimental_data()

# Data is automatically preprocessed according to config
print(f"Processed data shape: {data['c2_exp'].shape}")
```

### Direct Pipeline Usage
For more control, use the preprocessing pipeline directly:

```python
from homodyne.data.preprocessing import PreprocessingPipeline
from homodyne.data import load_xpcs_data

# Load raw data
raw_data = load_xpcs_data("config_without_preprocessing.yaml")

# Create and configure preprocessing pipeline
config = {
    'preprocessing': {
        'enabled': True,
        'stages': {
            'correct_diagonal': {'enabled': True, 'method': 'statistical'},
            'normalize_data': {'enabled': True, 'method': 'baseline'},
            'standardize_format': {'enabled': True},
            'validate_output': {'enabled': True}
        }
    }
}

pipeline = PreprocessingPipeline(config)
result = pipeline.process(raw_data)

if result.success:
    processed_data = result.data
    print(f"Pipeline completed in {result.provenance.total_duration:.3f}s")
    print(f"Successful stages: {sum(result.stage_results.values())}/{len(result.stage_results)}")
else:
    print("Pipeline failed:", result.provenance.errors)
```

### Convenience Function
For simple use cases:

```python
from homodyne.data.preprocessing import preprocess_xpcs_data

# Uses default preprocessing configuration
result = preprocess_xpcs_data(data)

# Or with custom config
result = preprocess_xpcs_data(data, custom_config)
```

### Provenance Tracking
Save and load complete processing history:

```python
# Save processing provenance
pipeline.save_provenance(result.provenance, "processing_history.json")

# Load provenance for analysis or reproduction
provenance = pipeline.load_provenance("processing_history.json")
print(f"Pipeline ID: {provenance.pipeline_id}")
print(f"Processing duration: {provenance.total_duration:.3f}s")
```

## Performance Features

### Memory Efficiency
- **In-place Operations**: Minimize memory copying where possible
- **Chunked Processing**: Handle large correlation matrices efficiently  
- **Smart Caching**: Optional intermediate result caching

### Speed Optimization
- **JAX Integration**: JIT compilation for performance-critical operations
- **Numpy Fallback**: Graceful degradation when JAX not available
- **Progress Reporting**: User feedback for long operations

### Resource Management
- **Memory Monitoring**: Track memory usage during processing
- **Error Handling**: Graceful failures with clear error messages
- **Fallback Strategies**: Continue processing on non-critical failures

## Integration with Existing Systems

### Config-Based Filtering
The preprocessing pipeline builds on Subagent 1's filtering system:
- Filtering applied before preprocessing transformations
- Consistent configuration interface
- Combined audit trail

### JAX Backend
Leverages Homodyne v2's JAX-first architecture:
- JIT compilation for statistical computations
- GPU acceleration when available
- Automatic fallback to numpy

### Logging System
Integrates with v2 logging infrastructure:
- Performance monitoring
- Detailed debug logging
- Stage-by-stage progress tracking

## Validation and Testing

The preprocessing pipeline includes comprehensive validation:

### Data Integrity Checks
- Non-finite value detection
- Physics constraint validation
- Array consistency verification
- Correlation matrix property checks

### Error Handling
- Stage-by-stage error isolation
- Configurable failure policies
- Detailed error reporting
- Automatic fallback options

### Performance Testing
Run the integration test script:
```bash
python test_preprocessing_integration.py
```

## Configuration Template Updates

The preprocessing configuration has been added to all YAML templates:

- **`config_default_template.yaml`**: Complete preprocessing configuration with examples
- **`config_default_static_isotropic.yaml`**: Basic preprocessing for simple analyses
- Additional templates include appropriate preprocessing defaults

## Advanced Features

### Custom Transformations
The pipeline architecture supports adding custom transformation stages:

```python
class CustomPreprocessingPipeline(PreprocessingPipeline):
    def _execute_custom_stage(self, data, config):
        # Implement custom transformation
        return processed_data, transform_record
```

### Batch Processing  
Process multiple datasets with consistent preprocessing:

```python
pipeline = PreprocessingPipeline(config)
for dataset in datasets:
    result = pipeline.process(dataset)
    # Process results...
```

### Configuration Validation
Automatic validation of preprocessing parameters:
- Method availability checking
- Parameter range validation  
- Dependency verification (scipy for advanced methods)

## Performance Benefits

Based on testing with synthetic data:

- **Enhanced Quality**: Statistical diagonal correction provides more robust results than linear interpolation
- **Flexible Normalization**: Multiple methods preserve physics while improving numerical stability
- **Noise Reduction**: Optional denoising improves signal-to-noise ratio for challenging datasets
- **Data Standardization**: Consistent processing regardless of APS vs APS-U source format
- **Reproducibility**: Complete audit trail enables exact reproduction and debugging
- **Memory Efficiency**: Chunked processing handles large datasets without memory issues

## Migration from Basic Processing

To upgrade from basic diagonal correction to the advanced preprocessing pipeline:

1. **Update Configuration**: Add `preprocessing` section to YAML config
2. **Enable Pipeline**: Set `preprocessing.enabled: true`
3. **Configure Stages**: Customize stages based on data quality needs
4. **Test with Data**: Run with your existing datasets
5. **Monitor Performance**: Check processing time and memory usage
6. **Tune Parameters**: Adjust methods based on results

## Future Extensions

The preprocessing pipeline architecture supports future enhancements:

- **Machine Learning**: Automated parameter tuning based on data characteristics
- **Advanced Denoising**: Deep learning-based noise reduction
- **Quality Scoring**: Automatic data quality assessment
- **Adaptive Processing**: Dynamic stage selection based on data properties
- **Parallel Processing**: Multi-core processing for large datasets

## Summary

The advanced preprocessing pipeline provides a significant enhancement to Homodyne v2's data processing capabilities:

✅ **Production Ready**: Comprehensive error handling and validation  
✅ **Highly Configurable**: Independent stage control with multiple algorithm options  
✅ **Performance Optimized**: JAX acceleration with memory-efficient processing  
✅ **Fully Integrated**: Seamless integration with existing XPCS loader and config system  
✅ **Reproducible**: Complete audit trail and provenance tracking  
✅ **Backwards Compatible**: Works with existing configurations and can be disabled  

This preprocessing pipeline transforms Homodyne v2 from basic data loading to intelligent data preparation, providing researchers with sophisticated tools for handling challenging XPCS datasets while maintaining the simplicity and reliability of the existing system.