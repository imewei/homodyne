# JAX Fallback Testing Suite for Homodyne v2

## Overview

This comprehensive testing suite validates all JAX fallback scenarios to ensure the entire homodyne system works correctly without JAX dependencies. The tests provide complete confidence in the NumPy fallback system for production deployment across diverse computational environments.

## Key Features

✅ **Complete JAX Independence**: No functionality lost without JAX  
✅ **Scientific Accuracy**: Maintains research-grade precision with NumPy fallbacks  
✅ **Performance Validation**: Benchmarks show acceptable NumPy fallback speeds  
✅ **Robust Error Handling**: Graceful degradation and informative user guidance  
✅ **End-to-End Workflows**: Complete XPCS analysis pipelines work without JAX  
✅ **Memory Efficiency**: Optimized for large parameter spaces and datasets  

## Test Suite Structure

### Core Test Files

1. **`homodyne/tests/test_jax_fallbacks.py`** - Main fallback testing suite
   - Accuracy validation (numerical vs analytical gradients)
   - Performance benchmarking (JAX vs NumPy timing)
   - Edge case testing (extreme parameters, conditioning issues)
   - Integration testing (end-to-end optimization workflows)
   - Regression testing (JAX/NumPy result consistency)

2. **`homodyne/tests/test_utils_fallback.py`** - Testing utilities and helpers
   - Realistic XPCS test data generation
   - Performance measurement utilities
   - Mock objects for testing scenarios
   - Scientific accuracy validation functions
   - Memory and resource monitoring

3. **`homodyne/tests/test_integration_fallback_workflows.py`** - Integration tests
   - Complete parameter estimation workflows
   - Static isotropic analysis (3 parameters)
   - Static anisotropic analysis (3 parameters with angles)
   - Laminar flow analysis (7 parameters with full physics)
   - Large dataset processing and memory management

### Configuration Files

- **`pytest-jax-fallback.ini`** - Pytest configuration for fallback testing
- **`run_fallback_tests.py`** - Comprehensive test runner script

## Test Categories

### 1. Accuracy Validation Tests
- **Numerical vs Analytical**: Compare numerical gradients with analytical solutions
- **JAX vs NumPy Consistency**: Ensure identical results between backends
- **XPCS Physics Validation**: Test with realistic experimental parameter ranges
- **Extreme Parameter Stability**: Numerical stability with challenging values

### 2. Performance Benchmarking Tests
- **Gradient Computation Speed**: JAX vs NumPy differentiation timing
- **Memory Usage**: Resource consumption during large computations
- **Batch Processing**: Vectorized operations performance
- **Large Parameter Spaces**: Scalability with 50+ parameters

### 3. Integration Workflow Tests
- **Static Isotropic Workflow**: Complete 3-parameter analysis pipeline
- **Laminar Flow Workflow**: Full 7-parameter physics analysis
- **Optimization Integration**: VI, MCMC, and Hybrid methods
- **Data Loading and Processing**: End-to-end experimental workflows

### 4. Robustness and Error Handling Tests
- **Graceful Degradation**: Behavior when numerical methods fail
- **User Guidance System**: Warning messages and recommendations
- **Recovery Scenarios**: Error handling and fallback mechanisms
- **Edge Case Handling**: Boundary conditions and singular cases

## Quick Start

### Running Basic Tests

```bash
# Run basic fallback functionality tests
python run_fallback_tests.py --mode basic

# Run with pytest (if available)
pytest -c pytest-jax-fallback.ini homodyne/tests/test_jax_fallbacks.py::TestJAXFallbackSystem::test_basic_math_functions_fallback
```

### Running Comprehensive Tests

```bash
# Run complete test suite
python run_fallback_tests.py --mode comprehensive

# Run all tests including benchmarks
python run_fallback_tests.py --mode all

# Run integration workflow tests
python run_fallback_tests.py --mode integration
```

### Running Performance Benchmarks

```bash
# Performance comparison (requires JAX for comparison)
python run_fallback_tests.py --mode performance
```

## Test Execution Modes

### Basic Mode (`--mode basic`)
Quick validation of core fallback functionality:
- Basic math functions (safe_divide, safe_exp, safe_sinc)
- Gradient computation accuracy
- Backend validation and diagnostics

**Runtime**: ~30 seconds  
**Use Case**: Development testing, CI/CD pipelines

### Comprehensive Mode (`--mode comprehensive`)
Full test suite covering all scenarios:
- All accuracy validation tests
- Integration tests with optimization
- Performance and memory tests
- Robustness and error handling tests

**Runtime**: ~5-15 minutes  
**Use Case**: Release validation, thorough testing

### Integration Mode (`--mode integration`)
End-to-end workflow validation:
- Complete XPCS analysis workflows
- Multiple analysis modes (3-param, 7-param)
- Large dataset processing
- Configuration management

**Runtime**: ~3-8 minutes  
**Use Case**: System integration testing

### Performance Mode (`--mode performance`)
Performance benchmarking and comparison:
- JAX vs NumPy speed comparison
- Memory usage analysis
- Scalability testing
- Performance optimization validation

**Runtime**: ~2-10 minutes  
**Use Case**: Performance validation, optimization

## Scientific Validation

### XPCS Physics Coverage

The tests validate the complete XPCS physics implementation:

**Static Isotropic Analysis (3 parameters)**:
- D₀: Reference diffusion coefficient [1e-3 to 1e6 Å²/s]
- α: Diffusion exponent [-2.0 to 2.0]
- D_offset: Baseline diffusion [0 to 1e4 Å²/s]

**Static Anisotropic Analysis (3 parameters with angles)**:
- Same parameters as isotropic
- Multiple scattering angles with filtering
- Angular correlation validation

**Laminar Flow Analysis (7 parameters)**:
- D₀, α, D_offset: Diffusion parameters
- γ̇₀: Reference shear rate [1e-4 to 1e3 s⁻¹]
- β: Shear exponent [-2.0 to 2.0]
- γ̇_offset: Baseline shear [0 to 1e2 s⁻¹]
- φ₀: Angular offset [-180 to 180 degrees]

### Accuracy Tolerances

- **Gradient Accuracy**: 1e-6 absolute tolerance
- **Function Values**: 1e-10 relative tolerance
- **Physics Consistency**: Parameter-dependent validation
- **Optimization Convergence**: Chi-squared improvement validation

## Performance Benchmarks

### Expected Performance Ratios (NumPy vs JAX)

| Operation | Typical Slowdown | Acceptable Range |
|-----------|-----------------|------------------|
| Forward Model | 5-20x | < 50x |
| Gradient Computation | 10-50x | < 100x |
| Hessian Computation | 50-200x | < 300x |
| Batch Processing | 20-100x | < 150x |

### Memory Usage

- **Small Problems** (3 params): < 50 MB additional
- **Medium Problems** (7 params): < 100 MB additional  
- **Large Problems** (50+ params): Chunked processing, < 200 MB

## Environment Setup

### Dependencies

**Required:**
```bash
numpy>=1.20.0
scipy>=1.7.0  # For numerical differentiation fallback
```

**Optional (for comparison testing):**
```bash
jax>=0.3.0     # For performance comparison
pytest>=6.0    # For automated test execution
psutil>=5.8.0  # For memory monitoring
```

### Installation

```bash
# Install homodyne with fallback support
pip install -e .[dev]

# Or install minimal dependencies
pip install numpy scipy
```

### Testing Without JAX

To test the pure fallback scenario:

```bash
# Uninstall JAX temporarily
pip uninstall jax jaxlib

# Run fallback tests
python run_fallback_tests.py --mode comprehensive

# Reinstall JAX
pip install jax
```

## Test Results and Reporting

### Automated Reports

Test results are automatically saved to `test_reports/` directory:
- **JSON Reports**: Machine-readable test results
- **Performance Metrics**: Timing and memory usage data
- **Accuracy Validation**: Scientific precision measurements
- **Error Logs**: Detailed failure information

### Sample Test Output

```
JAX FALLBACK TEST SUMMARY - COMPREHENSIVE MODE
============================================================
Duration: 127.45 seconds
Success Rate: 98.5%
Overall Status: PASS

Test Categories:
  accuracy_tests: 100.0% (PASS)
  integration_tests: 95.0% (PASS)
  performance_tests: 100.0% (PASS)
  robustness_tests: 100.0% (PASS)
============================================================
```

## Continuous Integration

### GitHub Actions Integration

```yaml
name: JAX Fallback Tests
on: [push, pull_request]

jobs:
  fallback-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    - name: Install dependencies (no JAX)
      run: |
        pip install numpy scipy pytest
        pip install -e .
    - name: Run fallback tests
      run: python run_fallback_tests.py --mode comprehensive
```

### Docker Testing

```dockerfile
# Test environment without JAX
FROM python:3.9-slim
RUN pip install numpy scipy
COPY . /app
WORKDIR /app
RUN pip install -e .
CMD ["python", "run_fallback_tests.py", "--mode", "all"]
```

## Troubleshooting

### Common Issues

**1. Import Errors**
```
ImportError: No module named 'jax'
```
*Solution*: This is expected in fallback testing. Ensure NumPy and SciPy are installed.

**2. Numerical Instabilities**
```
NumericalStabilityError: Gradient computation unstable
```
*Solution*: Check parameter bounds and function conditioning. Tests include stability validation.

**3. Performance Warnings**
```
WARNING: Using NumPy gradients - 10-50x slower than JAX
```
*Solution*: This is expected behavior. Install JAX for optimal performance.

### Debug Mode

```bash
# Run tests with detailed debugging
python run_fallback_tests.py --mode basic --verbose

# Run specific test with full output
pytest -c pytest-jax-fallback.ini -v -s homodyne/tests/test_jax_fallbacks.py::TestJAXFallbackSystem::test_gradient_fallback_accuracy
```

## Contributing

### Adding New Fallback Tests

1. **Test Structure**: Follow the pattern in `test_jax_fallbacks.py`
2. **Mock JAX Environment**: Use `MockJAXEnvironment` context manager
3. **Validation**: Include both accuracy and performance validation
4. **Documentation**: Add test descriptions and expected behavior

### Test Categories

- **@pytest.mark.fallback**: Core fallback functionality
- **@pytest.mark.accuracy**: Scientific accuracy validation
- **@pytest.mark.performance**: Performance benchmarking
- **@pytest.mark.integration**: End-to-end workflow testing
- **@pytest.mark.slow**: Long-running tests (>30 seconds)

### Example Test Addition

```python
def test_new_fallback_scenario(self):
    """Test description of new scenario."""
    with MockJAXEnvironment('jax_unavailable'):
        # Clear JAX modules
        jax_modules = [name for name in sys.modules.keys() if name.startswith('jax')]
        for module_name in jax_modules:
            if module_name in sys.modules:
                del sys.modules[module_name]
        
        # Import in fallback mode
        from homodyne.core.jax_backend import grad, jax_available
        assert not jax_available
        
        # Test implementation
        def test_func(x):
            return np.sum(x**2)
        
        grad_func = grad(test_func)
        result = grad_func(np.array([1.0, 2.0]))
        
        # Validation
        assert np.allclose(result, np.array([2.0, 4.0]))
```

## Support and Documentation

### Resources

- **Architecture Documentation**: `CLAUDE.md` - Core system architecture
- **API Reference**: Module docstrings and type hints
- **Examples**: `examples/` directory with usage patterns
- **Performance Guide**: `docs/performance.md`

### Getting Help

1. **Check Test Output**: Detailed error messages and recommendations
2. **Review Documentation**: Architecture and usage guides
3. **Examine Examples**: Working code patterns
4. **Performance Profiling**: Built-in monitoring and reporting

## Validation Summary

This testing suite provides comprehensive validation that:

✅ **Scientific Accuracy**: All XPCS physics computations maintain research-grade precision  
✅ **Complete Functionality**: Every feature works without JAX dependencies  
✅ **Performance Acceptability**: NumPy fallbacks provide reasonable computational speed  
✅ **Production Readiness**: Robust error handling and user guidance  
✅ **Deployment Flexibility**: Works across diverse computational environments  

The test suite ensures homodyne v2 delivers reliable, accurate XPCS analysis regardless of the available computational backend, making it suitable for deployment in environments where JAX may not be available or desired.