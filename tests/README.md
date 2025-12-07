# Homodyne Test Suite

Comprehensive test suite for the JAX-first homodyne package covering all aspects of
scientific computing, performance validation, and API compatibility.

**Test Regeneration:** v2.4.1 (Dec 2025) - Complete reorganization following module
restructuring.

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_runner.py                 # Comprehensive test runner
├── unit/                          # Unit tests (47 files)
│   ├── test_jax_backend.py        # JAX computational backend
│   ├── test_nlsq_core.py          # NLSQ core optimization
│   ├── test_nlsq_wrapper.py       # NLSQ wrapper functions
│   ├── test_mcmc_core.py          # MCMC core sampling
│   ├── test_cmc_core.py           # CMC coordinator tests
│   ├── test_data_loader.py        # Data loading and preprocessing
│   ├── test_per_angle_scaling.py  # Per-angle scaling (mandatory v2.4.0)
│   ├── test_sharding.py           # Stratified sharding
│   ├── test_combination.py        # CMC result combination
│   ├── test_diagnostics.py        # Convergence diagnostics
│   └── ... (37 more files)
├── integration/                   # End-to-end workflows (7 files)
│   ├── test_workflows.py          # Complete analysis pipelines
│   ├── test_nlsq_integration.py   # NLSQ full workflow
│   ├── test_mcmc_integration.py   # MCMC full workflow
│   ├── test_cmc_integration.py    # CMC distributed workflow
│   ├── test_parameter_recovery.py # Parameter estimation accuracy
│   └── test_optimization_edge_cases.py
├── performance/                   # Benchmarks (6 files)
│   ├── test_benchmarks.py         # Core performance baselines
│   ├── test_nlsq_performance.py   # NLSQ optimization speed
│   ├── test_stratified_chunking_performance.py
│   ├── test_wrapper_overhead.py   # Wrapper vs raw overhead
│   └── test_angle_filtering_performance.py
├── regression/                    # Regression tests (2 files)
│   ├── test_nlsq_regression.py    # NLSQ result stability
│   └── test_save_results_compat.py
├── validation/                    # Scientific validation (4 files)
│   ├── test_scientific_validation.py
│   ├── test_cmc_accuracy.py       # CMC accuracy benchmarks
│   └── test_real_data_stratification.py
├── self_consistency/              # Self-consistency checks (1 file)
│   └── test_cmc_consistency.py    # CMC internal consistency
├── property/                      # Property-based tests (2 files)
│   └── test_mathematical_properties.py
├── mcmc/                          # MCMC statistical validation (2 files)
│   └── test_statistical_validation.py
├── api/                           # API compatibility (2 files)
│   └── test_compatibility.py
├── fixtures/                      # Reusable test fixtures (3 files)
│   ├── mcmc_fixtures.py           # MCMC-specific fixtures
│   └── physics_fixtures.py        # Physics parameter fixtures
└── factories/                     # Test data generation (6 files)
    ├── data_factory.py            # Synthetic XPCS data
    ├── synthetic_data.py          # Physics-based generators
    ├── config_factory.py          # Configuration builders
    ├── optimization_factory.py    # Optimization test cases
    └── large_dataset_factory.py   # Large-scale test data
```

## Test Categories

### Unit Tests (`unit/`)

47 files covering individual components:

- **JAX Backend**: Mathematical functions, automatic differentiation, vectorization
- **NLSQ Optimization**: Core algorithms, wrapper functions, Jacobian computation
- **MCMC Sampling**: Core sampling, log-space D0, priors, scaling
- **CMC Coordinator**: Sharding, combination, diagnostics, backends
- **Data Loading**: HDF5 formats, YAML/JSON configs, caching, validation
- **Per-Angle Scaling**: Mandatory per-angle mode (v2.4.0)
- **Stratified Chunking**: Large dataset handling, residual computation

### Integration Tests (`integration/`)

7 files for end-to-end workflows:

- **NLSQ Integration**: Complete NLSQ optimization pipelines
- **MCMC Integration**: Full Bayesian analysis workflows
- **CMC Integration**: Distributed consensus Monte Carlo
- **Parameter Recovery**: Accuracy of parameter estimation
- **Edge Cases**: Boundary conditions, error recovery

### Performance Tests (`performance/`)

6 files for benchmarking:

- **Core Benchmarks**: JAX operations, optimization speed
- **NLSQ Performance**: Trust-region convergence timing
- **Stratified Chunking**: Large dataset performance
- **Wrapper Overhead**: Abstraction layer costs
- **Angle Filtering**: Phi selection performance

### Regression Tests (`regression/`)

2 files for result stability:

- **NLSQ Regression**: Result consistency across versions
- **Save/Load Compatibility**: Result file format stability

### Validation Tests (`validation/`)

4 files for scientific correctness:

- **Scientific Validation**: Physical constraints, model accuracy
- **CMC Accuracy**: Distributed sampling correctness
- **Real Data Stratification**: Realistic data handling

### Self-Consistency Tests (`self_consistency/`)

1 file for internal consistency:

- **CMC Consistency**: Worker-coordinator agreement

### Property Tests (`property/`)

2 files using Hypothesis:

- **Mathematical Invariants**: Physical constraints, symmetries, bounds
- **Numerical Stability**: Extreme values, precision consistency

### MCMC Tests (`mcmc/`)

2 files for statistical validation:

- **Convergence Diagnostics**: R-hat, effective sample size, mixing
- **Parameter Recovery**: Bayesian parameter estimation accuracy

### API Tests (`api/`)

2 files for API stability:

- **Public API Stability**: Function signatures, return types
- **Import Structure**: Module organization, backward compatibility

### Fixtures (`fixtures/`)

3 files with reusable test data:

- **MCMC Fixtures**: Sampling configurations, chain data
- **Physics Fixtures**: Parameter sets, physical constraints

### Factories (`factories/`)

6 files for test data generation:

- **Data Factory**: Synthetic XPCS correlation data
- **Synthetic Data**: Physics-based test data generators
- **Config Factory**: Configuration builders
- **Optimization Factory**: Optimization test scenarios
- **Large Dataset Factory**: Large-scale test data

## Running Tests

### Makefile Commands (Recommended)

```bash
make test              # Core tests
make test-all          # All tests + coverage
make test-unit         # Unit tests only
make test-integration  # End-to-end tests
make test-nlsq         # NLSQ optimization tests
make test-mcmc         # MCMC validation tests
```

### Test Runner

```bash
python tests/test_runner.py quick        # Quick development tests
python tests/test_runner.py full         # Complete test suite
python tests/test_runner.py unit         # Unit tests only
python tests/test_runner.py integration  # Integration tests
python tests/test_runner.py performance  # Performance benchmarks
python tests/test_runner.py mcmc         # MCMC statistical tests
python tests/test_runner.py ci           # CI-friendly test suite
python tests/test_runner.py env          # Check test environment
```

### Direct Pytest Usage

```bash
pytest tests/unit/                       # Unit tests
pytest tests/ -m "not slow"              # Fast tests only
pytest tests/ -k "test_nlsq"             # NLSQ-related tests
pytest tests/ -k "test_cmc"              # CMC-related tests
pytest tests/regression/                 # Regression tests
pytest tests/validation/                 # Scientific validation
```

## Test Markers

- `unit`: Unit tests for individual components
- `integration`: Integration tests for workflows
- `performance`: Performance and benchmark tests
- `mcmc`: MCMC statistical tests
- `property`: Property-based tests with Hypothesis
- `slow`: Slow tests (> 5 seconds)
- `requires_jax`: Requires JAX installation
- `linux`: Requires Linux OS

## Dependencies

### Core Testing

- `pytest >= 8.3.0`
- `pytest-cov >= 4.1.0`
- `pytest-xdist >= 3.3.0`

### Property Testing

- `hypothesis >= 6.82.0`

### Performance Testing

- `pytest-benchmark >= 4.0.0`
- `psutil >= 6.0.0`

### Statistical Testing

- `scipy >= 1.14.0`
- `arviz >= 0.15.0` (optional)

### Scientific Computing

- `numpy >= 2.0.0`
- `jax == 0.8.0` (CPU-only, exact match required)
- `jaxlib == 0.8.0`
- `h5py >= 3.10.0`
- `PyYAML >= 6.0.2`
- `numpyro >= 0.18.0`
- `blackjax >= 1.2.0`

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
- Comprehensive reporting (HTML, XML, coverage)
- Performance regression detection
- Cross-platform validation

## Performance Baselines

### Computational Performance

- **Matrix Operations**: > 10 GFLOPS (CPU)
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

- **JAX version mismatch**: Install exact version with
  `pip install jax==0.8.0 jaxlib==0.8.0`
- **MCMC tests slow**: Reduce sample sizes in test configuration
- **H5PY issues**: Install with `pip install h5py`
- **Import errors after refactor**: Check `homodyne/optimization/{nlsq,mcmc}/` paths

### Performance Issues

- **Slow tests**: Use `pytest -m "not slow"` for development
- **Memory errors**: Reduce dataset sizes in performance tests
- **Parallel execution**: Use `pytest -n auto` for multicore testing

### Module Import Paths (v2.4.1)

After the optimization module reorganization (CMC):

```python
# NLSQ imports
from homodyne.optimization.nlsq.core import optimize_nlsq
from homodyne.optimization.nlsq.wrapper import NLSQWrapper

# CMC imports (MCMC inference)
from homodyne.optimization.cmc.core import fit_mcmc_jax
from homodyne.optimization.cmc.results import CMCResult
```

For detailed troubleshooting, see the main package documentation.
