# Homodyne v2 Test Suite

Comprehensive test suite for the JAX-first homodyne package covering all aspects of
scientific computing, performance validation, and API compatibility.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                    # Pytest configuration
├── test_runner.py                 # Comprehensive test runner
├── unit/                         # Unit tests for individual components
│   ├── test_jax_backend.py       # JAX computational backend tests
│   ├── test_optimization_nlsq.py # NLSQ optimization tests
│   └── test_data_loader.py       # Data loading and preprocessing tests
├── integration/                  # End-to-end workflow tests
│   └── test_workflows.py         # Complete scientific workflows
├── performance/                  # Performance and benchmark tests
│   └── test_benchmarks.py        # Performance validation and regression
├── property/                     # Property-based mathematical tests
│   └── test_mathematical_properties.py # Hypothesis-driven property tests
├── factories/                    # Test data generation
│   └── data_factory.py          # Synthetic data and mock file generators
├── gpu/                         # GPU acceleration validation
│   └── test_gpu_validation.py   # GPU performance and consistency tests
├── mcmc/                        # MCMC statistical validation
│   └── test_statistical_validation.py # Bayesian sampling validation
└── api/                         # API compatibility tests
    └── test_compatibility.py    # Public API stability validation
```

## Test Categories

### 🧪 Unit Tests (`unit/`)

- **JAX Backend**: Mathematical functions, automatic differentiation, vectorization
- **NLSQ Optimization**: Parameter recovery, convergence, error handling
- **Data Loading**: HDF5 formats, YAML/JSON configs, caching, validation

### 🔗 Integration Tests (`integration/`)

- **End-to-End Workflows**: Complete data analysis pipelines
- **Module Interactions**: Cross-module compatibility
- **Configuration Integration**: YAML/JSON configuration workflows
- **Cross-Platform Compatibility**: Consistent behavior across platforms

### ⚡ Performance Tests (`performance/`)

- **Computational Benchmarks**: JAX operations, optimization performance
- **Memory Scaling**: Memory usage patterns and scaling behavior
- **Regression Testing**: Performance baseline validation
- **CPU vs GPU Comparison**: Acceleration validation

### 🎯 Property Tests (`property/`)

- **Mathematical Invariants**: Physical constraints, symmetries, bounds
- **Numerical Stability**: Extreme values, precision consistency
- **Parameter Scaling**: Linear relationships, monotonicity
- **Statistical Properties**: Residual behavior, chi-squared properties

### 🏭 Test Factories (`factories/`)

- **Synthetic Data Generation**: Realistic XPCS correlation data
- **Mock File Creation**: HDF5 files in APS and APS-U formats
- **Parameter Sets**: Edge cases, realistic scenarios, sweeps
- **Noise Models**: Various noise types and artifacts

### 🖥️ GPU Tests (`gpu/`)

- **GPU Detection**: Hardware detection and activation
- **Performance Validation**: Speedup verification, memory management
- **Numerical Consistency**: CPU vs GPU result validation
- **Error Handling**: Graceful fallbacks, invalid configurations

### 📊 MCMC Tests (`mcmc/`)

- **Convergence Diagnostics**: R-hat, effective sample size, mixing
- **Parameter Recovery**: Bayesian parameter estimation accuracy
- **Statistical Properties**: Posterior distributions, credible intervals
- **Chain Quality**: Autocorrelation, stationarity, reproducibility

### 🔧 API Tests (`api/`)

- **Public API Stability**: Function signatures, return types
- **Import Structure**: Module organization, backward compatibility
- **Error Handling**: Consistent error types and messages
- **Version Compatibility**: Dependency requirements, deprecations

## Running Tests

### Quick Development Tests

```bash
python tests/test_runner.py quick
```

### Complete Test Suite

```bash
python tests/test_runner.py full
```

### Specific Test Categories

```bash
python tests/test_runner.py unit          # Unit tests only
python tests/test_runner.py integration   # Integration tests
python tests/test_runner.py performance   # Performance benchmarks
python tests/test_runner.py gpu          # GPU acceleration tests
python tests/test_runner.py mcmc         # MCMC statistical tests
python tests/test_runner.py property     # Property-based tests
python tests/test_runner.py api          # API compatibility tests
```

### CI/CD Tests

```bash
python tests/test_runner.py ci           # CI-friendly test suite
```

### Environment Check

```bash
python tests/test_runner.py env          # Check test environment
```

### Direct Pytest Usage

```bash
pytest tests/unit/                       # Unit tests
pytest tests/ -m "not slow"             # Fast tests only
pytest tests/ -m "gpu and requires_gpu" # GPU tests only
pytest tests/ -k "test_optimization"    # Specific test patterns
```

## Test Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests for workflows
- `performance`: Performance and benchmark tests
- `gpu`: GPU acceleration tests
- `mcmc`: MCMC statistical tests
- `property`: Property-based tests with Hypothesis
- `slow`: Slow tests (> 5 seconds)
- `requires_jax`: Requires JAX installation
- `requires_gpu`: Requires GPU hardware
- `api`: API compatibility tests

## Dependencies

### Core Testing

- `pytest >= 7.4.0`
- `pytest-cov >= 4.1.0`
- `pytest-html >= 3.1.0`
- `pytest-xdist >= 3.3.0`

### Property Testing

- `hypothesis >= 6.82.0`

### Performance Testing

- `pytest-benchmark >= 4.0.0`
- `psutil >= 5.9.0`

### Statistical Testing

- `scipy >= 1.11.0`
- `arviz >= 0.15.0` (optional)

### Scientific Computing

- `numpy >= 1.25.0`
- `jax >= 0.7.2`
- `h5py >= 3.9.0`
- `PyYAML >= 6.0`

## Coverage Requirements

- **Minimum Coverage**: 80%
- **Critical Modules**: 90%+ (core, optimization, data)
- **Test Types**: All categories must pass
- **Performance**: No regressions > 20%

## Test Data

### Synthetic Data

- Realistic XPCS correlation functions
- Multiple q-values and experimental conditions
- Controlled noise levels and artifacts
- Known parameter sets for validation

### Mock Files

- APS old format HDF5 files
- APS-U new format HDF5 files
- YAML and JSON configuration files
- Various file sizes and structures

### Edge Cases

- High noise scenarios
- Low contrast data
- Fast/slow diffusion regimes
- Small datasets
- Large performance datasets

## Continuous Integration

The test suite is designed for automated CI/CD with:

- Parallel test execution
- GPU test skipping in environments without GPU
- Comprehensive reporting (HTML, XML, coverage)
- Performance regression detection
- Cross-platform validation

## Performance Baselines

### Computational Performance

- **Matrix Operations**: > 10 GFLOPS (CPU), > 100 GFLOPS (GPU)
- **NLSQ Optimization**: < 30s for standard datasets
- **Data Loading**: > 1 MB/s for HDF5 files
- **Memory Usage**: < 10x theoretical minimum

### Statistical Accuracy

- **Parameter Recovery**: < 5% error for low-noise synthetic data
- **MCMC Convergence**: R-hat < 1.1, ESS > 100
- **Physical Constraints**: All bounds respected
- **Numerical Precision**: < 1e-10 relative error for exact cases

## Contributing

When adding new features:

1. Add corresponding unit tests
1. Include integration test coverage
1. Add property tests for mathematical invariants
1. Update performance baselines if needed
1. Ensure API compatibility is maintained

## Troubleshooting

### Common Issues

- **JAX not available**: Install with `pip install "jax[cpu]"` or
  `pip install "jax[cuda12-local]"`
- **GPU tests failing**: Ensure CUDA 12.1+ and compatible drivers
- **MCMC tests slow**: Reduce sample sizes in test configuration
- **H5PY issues**: Install with `pip install h5py`
- **Hypothesis tests failing**: Update to latest hypothesis version

### Performance Issues

- **Slow tests**: Use `pytest -m "not slow"` for development
- **Memory errors**: Reduce dataset sizes in performance tests
- **GPU memory**: Lower `memory_fraction` in GPU tests
- **Parallel execution**: Use `pytest -n auto` for multicore testing

For detailed troubleshooting, see the main package documentation.
