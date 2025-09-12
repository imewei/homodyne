# Homodyne v2 Scripts Directory

This directory contains utility scripts, test runners, and development tools for Homodyne v2.

## 📜 Available Scripts

### Testing Scripts
- **`run_fallback_tests.py`** - Comprehensive JAX fallback system test runner
- **`run_enhanced_data_loading_tests.py`** - Data loading enhancement test suite
- **`simple_fallback_test.py`** - Quick fallback functionality validation
- **`direct_backend_test.py`** - Direct backend testing utilities

### Validation Scripts
- **`validate_numpy_gradients.py`** - Numerical gradient accuracy validation

## 🚀 Usage Examples

### Running Fallback Tests
```bash
# From repository root
python scripts/run_fallback_tests.py --mode basic
python scripts/run_fallback_tests.py --mode comprehensive
```

### Running Data Loading Tests
```bash
# From repository root
python scripts/run_enhanced_data_loading_tests.py --all
python scripts/run_enhanced_data_loading_tests.py --quick
```

### Gradient Validation
```bash
# From repository root
python scripts/validate_numpy_gradients.py
```

## 📋 Script Categories

### Test Runners
Scripts that execute comprehensive test suites and generate reports.

### Validation Scripts  
Scripts that validate specific functionality or accuracy requirements.

### Development Utilities
Scripts that assist with development, debugging, and system validation.

## 🔧 Development Notes

These scripts are designed to be run from the repository root directory and may require the homodyne package to be installed in development mode:

```bash
pip install -e .
```

All scripts include comprehensive help documentation accessible via the `--help` flag.