# Homodyne v2 Enhanced Data Loading Testing Framework

This directory contains a comprehensive testing framework for the enhanced data loading system implemented in Homodyne v2. The testing framework validates all components and their interactions developed by Subagents 1-4.

## 📁 Testing Structure

```
homodyne/tests/
├── README.md                           # This file
├── __init__.py                         # Test package initialization
├── integration/                        # End-to-end integration tests
│   ├── __init__.py
│   └── test_data_loading_integration.py
├── performance/                        # Performance and scalability tests
│   ├── __init__.py
│   └── test_data_loading_performance.py
├── robustness/                         # Edge cases and error handling tests
│   ├── __init__.py
│   └── test_data_loading_robustness.py
├── config/                             # Configuration system tests
│   ├── __init__.py
│   └── test_enhanced_data_loading_config.py
└── data/                               # Test data and utilities
    ├── __init__.py
    └── synthetic_data_generator.py
```

## 🎯 Test Coverage

### Enhanced Components Tested
- **Filtering System** (Subagent 1): `filtering_utils.py`
- **Preprocessing Pipeline** (Subagent 2): `preprocessing.py`
- **Quality Control** (Subagent 3): `quality_controller.py`
- **Performance Engine** (Subagent 4): `performance_engine.py`, `memory_manager.py`

### Test Categories

#### 1. Integration Tests (`integration/`)
- **End-to-End Pipeline Testing**: Complete workflow validation from raw HDF5 → processed data
- **Cross-Component Integration**: Ensures all components work together seamlessly
- **Configuration Testing**: Validates YAML configurations and parameter combinations
- **Format Consistency**: Tests APS old vs APS-U format compatibility
- **Error Handling**: Validates graceful error handling across components

#### 2. Performance Tests (`performance/`)
- **Memory Usage Profiling**: Validates memory optimization for large datasets
- **Performance Benchmarking**: Confirms performance improvements vs baselines
- **Scalability Testing**: Tests behavior across different dataset sizes
- **Parallel Processing**: Validates multi-threaded operations
- **Cache Effectiveness**: Tests multi-level caching system performance

#### 3. Robustness Tests (`robustness/`)
- **Data Corruption Recovery**: Tests handling of corrupted/incomplete data
- **Resource Constraint Adaptation**: Tests behavior under memory/disk pressure
- **Concurrent Access Safety**: Thread safety and parallel operation validation
- **Configuration Robustness**: Invalid/edge case configuration handling
- **System Resource Limits**: Behavior at file descriptor/memory limits

#### 4. Configuration Tests (`config/`)
- **Schema Validation**: All YAML parameters and combinations
- **Template Validation**: Standard configuration templates
- **Migration Testing**: JSON to YAML configuration migration
- **Parameter Boundary Testing**: Edge values and constraints
- **Inheritance and Overrides**: Configuration merging and precedence

## 🚀 Quick Start

### Prerequisites

```bash
# Required dependencies
pip install pytest h5py numpy pyyaml psutil

# Optional dependencies for enhanced testing
pip install memory-profiler jax  # For performance testing
```

### Running Tests

#### Run All Tests
```bash
# From project root
pytest homodyne/tests/ -v

# Or from tests directory
cd homodyne/tests
pytest -v
```

#### Run Specific Test Suites
```bash
# Integration tests only
pytest homodyne/tests/integration/ -v

# Performance tests only  
pytest homodyne/tests/performance/ -v

# Robustness tests only
pytest homodyne/tests/robustness/ -v

# Configuration tests only
pytest homodyne/tests/config/ -v
```

#### Run Tests by Marker
```bash
# Quick tests only (fast-running tests)
pytest homodyne/tests/ -m "not slow" -v

# Performance tests only
pytest homodyne/tests/ -m "performance" -v

# Integration tests only
pytest homodyne/tests/ -m "integration" -v
```

### Generating Test Data

The testing framework includes a synthetic test data generator:

```python
from homodyne.tests.data.synthetic_data_generator import generate_test_dataset_suite

# Generate complete test dataset suite
test_datasets = generate_test_dataset_suite()

# Generate specific dataset
from homodyne.tests.data.synthetic_data_generator import (
    SyntheticDataGenerator, SyntheticDatasetConfig, 
    DatasetSize, DataQuality, DatasetFormat
)

config = SyntheticDatasetConfig(
    name="my_test_data",
    size=DatasetSize.SMALL,
    quality=DataQuality.GOOD,
    format=DatasetFormat.APS_U
)

generator = SyntheticDataGenerator(config)
dataset_path = generator.generate_dataset()
```

## 📊 Test Reporting

### Performance Reports
Performance tests automatically generate detailed reports:

```bash
# Run performance tests with reporting
pytest homodyne/tests/performance/ -v -s

# Performance metrics will be displayed including:
# - Memory usage profiling
# - Execution time benchmarks  
# - Throughput measurements
# - Scalability analysis
```

### Coverage Reports
Generate test coverage reports:

```bash
# Install coverage tool
pip install pytest-cov

# Run tests with coverage
pytest homodyne/tests/ --cov=homodyne.data --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Quality Reports
Quality control tests generate validation reports:

```bash
# Run robustness tests with detailed output
pytest homodyne/tests/robustness/ -v -s --tb=long
```

## 🔧 Configuration Examples

### Basic Test Configuration
```yaml
# homodyne/tests/config/examples/basic_test_config.yaml
data_loading:
  enhanced_features:
    enable_filtering: true
    enable_preprocessing: true
    enable_quality_control: false
    enable_performance_optimization: false
  
  filtering:
    quality_threshold: 0.8
    q_range:
      min: 1e-4
      max: 1e-2
  
  preprocessing:
    pipeline_stages:
      - diagonal_correction
      - normalization
```

### Performance Test Configuration  
```yaml
# homodyne/tests/config/examples/performance_test_config.yaml
data_loading:
  enhanced_features:
    enable_filtering: true
    enable_preprocessing: true
    enable_quality_control: true
    enable_performance_optimization: true
  
  performance:
    optimization_level: aggressive
    caching:
      enable_memory_cache: true
      enable_disk_cache: true
    parallel_processing:
      enable: true
      max_threads: auto
    memory_management:
      enable_memory_mapping: true
```

## 🧪 Test Development Guidelines

### Adding New Tests

1. **Choose the appropriate test category**:
   - Integration: End-to-end workflows and component interactions
   - Performance: Benchmarking and scalability
   - Robustness: Error handling and edge cases
   - Configuration: Parameter validation and schema testing

2. **Follow naming conventions**:
   - Test files: `test_<component>_<category>.py`
   - Test classes: `Test<Component><Category>`
   - Test methods: `test_<specific_behavior>`

3. **Use appropriate fixtures**:
   ```python
   import pytest
   from homodyne.tests.data.synthetic_data_generator import generate_test_dataset_suite
   
   @pytest.fixture(scope="session")
   def test_datasets():
       return generate_test_dataset_suite()
   
   def test_my_feature(test_datasets):
       dataset_path = test_datasets['integration_small']
       # Test implementation
   ```

4. **Include proper cleanup**:
   ```python
   import tempfile
   import shutil
   from pathlib import Path
   
   class TestMyFeature:
       @classmethod
       def setup_class(cls):
           cls.test_dir = Path(tempfile.mkdtemp(prefix="my_test_"))
       
       @classmethod  
       def teardown_class(cls):
           if cls.test_dir.exists():
               shutil.rmtree(cls.test_dir)
   ```

### Performance Test Guidelines

1. **Use performance monitoring**:
   ```python
   from homodyne.tests.performance.test_data_loading_performance import performance_monitor
   
   def test_my_performance():
       with performance_monitor("my_operation", dataset_size_mb=100):
           # Performance-critical code here
           pass
       
       # Metrics automatically recorded
       metrics = performance_monitor.results["my_operation"]
       assert metrics.execution_time < 5.0  # 5 second limit
   ```

2. **Include baseline comparisons**:
   ```python
   def test_performance_regression():
       baseline_time = 2.0  # seconds
       
       with performance_monitor("regression_test"):
           # Operation to test
           pass
       
       actual_time = performance_monitor.results["regression_test"].execution_time
       assert actual_time < baseline_time * 1.2  # Allow 20% regression
   ```

### Robustness Test Guidelines

1. **Test error conditions**:
   ```python
   def test_invalid_data_handling():
       with pytest.raises(ValueError) as exc_info:
           # Code that should raise ValueError
           pass
       
       # Verify error message is helpful
       assert "descriptive error" in str(exc_info.value).lower()
   ```

2. **Test resource constraints**:
   ```python
   from homodyne.tests.robustness.test_data_loading_robustness import memory_pressure_simulation
   
   def test_memory_pressure_handling():
       with memory_pressure_simulation(target_usage_mb=500):
           # Test under memory pressure
           result = my_memory_intensive_operation()
           assert result is not None  # Should still work
   ```

## 📈 Continuous Integration

### GitHub Actions Integration
```yaml
# .github/workflows/test-enhanced-data-loading.yml
name: Enhanced Data Loading Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov h5py numpy pyyaml psutil
    
    - name: Run integration tests
      run: pytest homodyne/tests/integration/ -v
    
    - name: Run performance tests
      run: pytest homodyne/tests/performance/ -v -m "not slow"
    
    - name: Run robustness tests  
      run: pytest homodyne/tests/robustness/ -v
    
    - name: Run configuration tests
      run: pytest homodyne/tests/config/ -v
    
    - name: Generate coverage report
      run: pytest homodyne/tests/ --cov=homodyne.data --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
```

### Local Pre-commit Hooks
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run tests before commit
pre-commit run --all-files
```

## 🐛 Debugging Tests

### Running Tests with Debug Output
```bash
# Verbose output with full traceback
pytest homodyne/tests/integration/ -v -s --tb=long

# Stop on first failure
pytest homodyne/tests/ -x

# Run specific test with debugging
pytest homodyne/tests/integration/test_data_loading_integration.py::TestDataLoadingIntegration::test_basic_integration_pipeline -v -s
```

### Common Issues and Solutions

1. **Test data generation fails**:
   ```bash
   # Ensure h5py is installed
   pip install h5py
   
   # Check disk space for test data generation
   df -h
   ```

2. **Memory-related test failures**:
   ```bash
   # Run with more memory or skip memory-intensive tests
   pytest homodyne/tests/ -m "not memory_intensive"
   ```

3. **Performance test variability**:
   ```bash
   # Run performance tests multiple times
   pytest homodyne/tests/performance/ --count=3
   ```

4. **Configuration test failures**:
   ```bash
   # Verify YAML syntax
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

## 📚 Additional Resources

### Test Data Specifications
- **Tiny**: ~10MB - Unit tests
- **Small**: ~100MB - Integration tests
- **Medium**: ~500MB - Performance tests
- **Large**: ~2GB - Scalability tests
- **Massive**: ~10GB - Stress tests

### Performance Baselines
Current performance baselines (updated with each release):
- Loading small dataset: < 1.0s, < 100MB memory
- Loading medium dataset: < 5.0s, < 500MB memory
- Filtering small dataset: < 0.5s, < 50MB memory overhead
- Full pipeline small dataset: < 5.0s, < 300MB memory

### Quality Metrics
Quality control thresholds:
- Signal-to-noise ratio: > 5.0
- Data completeness: > 0.9
- Baseline stability: > 0.95
- Physics consistency: > 0.9

## 🤝 Contributing

When adding new features to the enhanced data loading system:

1. **Add corresponding tests** in the appropriate test category
2. **Update synthetic data generator** if new data characteristics are needed
3. **Add configuration tests** for new parameters
4. **Update performance baselines** if performance characteristics change
5. **Document test requirements** in this README

For questions or issues with the testing framework, please refer to the main project documentation or create an issue in the project repository.

---

**Note**: This testing framework is designed to be comprehensive and may take significant time to run completely. For development purposes, use the quick test markers or run specific test categories as needed.