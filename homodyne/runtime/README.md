# Homodyne Runtime System

**Advanced shell completion and comprehensive system validation for the homodyne
analysis package (CPU-only v2.3.0+).**

______________________________________________________________________

## System Architecture (v2.3.0+ CPU-Only)

```
runtime/
├── shell/              # Advanced shell completion system
│   └── completion.sh   # Context-aware completion with caching
├── utils/              # System validation and utilities
│   └── system_validator.py  # Comprehensive system testing
└── README.md          # This documentation
```

> **Note**: GPU support was removed in v2.3.0. This is now a CPU-only package with HPC
> multi-core optimization.

## Quick Setup

### One-Command Installation

```bash
# Complete setup with all features
homodyne-post-install --shell zsh --advanced

# Interactive setup (choose features)
homodyne-post-install --interactive

# Basic shell completion only
homodyne-post-install --shell zsh
```

### Installed Components

1. **Smart Shell Completion** - Context-aware completion with caching
1. **Unified Aliases** - CPU shortcuts (`hm`, `hc`, `hr`, `ha`)
1. **Advanced Tools** - System validation, benchmarking
1. **Environment Integration** - Auto-activation in conda/mamba/venv

## Shell Completion System

### Key Features

- **Context-Aware**: Suggests methods based on config file analysis
- **Intelligent Caching**: 5-minute TTL cache for file discovery
- **Smart Parameters**: Common values for angles, contrast, output directories
- **Interactive Builder**: `homodyne_build` for guided command creation
- **Cross-Shell Support**: Works with bash, zsh, and fish

### Usage Examples

```bash
# Auto-completes with recent JSON config files
homodyne --config <TAB>

# Smart method suggestions based on config mode
homodyne --config laminar_flow.json --method <TAB>
# → Shows: mcmc nlsq

# Interactive command builder
homodyne_build
# → Guided menu system for building commands
```

### Smart Completion Logic

| Config Mode | Suggested Methods | Reasoning |
|-------------|------------------|-----------| | `static` | `nlsq mcmc` | Fast parameter
estimation | | `laminar_flow` | `mcmc nlsq` | Uncertainty quantification |

## MCMC Backend System (CPU-Only)

### CMC-Only Architecture (v2.4.1+)

The homodyne package uses Consensus Monte Carlo (CMC) for all MCMC operations:

| Command | Backend | Implementation | Use Case |
|---------|---------|----------------|----------| | `homodyne` | **NumPyro CMC** |
`mcmc.py` | Production, all datasets |

### CMC Features

- **Multi-core parallelization**: Uses multiprocessing backend for parallel shard
  execution
- **Automatic sharding**: Dataset size-aware shard allocation
- **Physics-informed priors**: Domain-specific prior distributions
- **Robust convergence**: Auto-retry with up to 3 attempts

## System Validation

### Comprehensive Testing

```bash
# Complete system validation
homodyne-validate

# Verbose validation with timing
homodyne-validate --verbose

# Component-specific testing
homodyne-validate --test completion  # Shell completion only

# JSON output for automation
homodyne-validate --json
```

### Test Categories

1. **Environment Detection**: Platform, Python version, virtual environment, shell type
1. **Installation Verification**: Commands, help output, module imports, version
   consistency
1. **Shell Completion**: Files, activation scripts, aliases, cache system
1. **Integration Testing**: Cross-component functionality, environment propagation

### Sample Output

```
HOMODYNE SYSTEM VALIDATION REPORT
================================================================================

Summary: 5/5 tests passed
All systems operational!

Environment:
   Platform: Linux x86_64
   Python: 3.12.0
   Environment: conda (homodyne)
   Shell: zsh

Test Results:
PASS Environment Detection (0.003s)
PASS Homodyne Installation (0.152s)
PASS Shell Completion (0.089s)
PASS CPU Backend (0.234s)
PASS Integration (0.067s)

Recommendations:
   Your homodyne installation is optimized and ready!
   Run 'homodyne --help' to see command options
   CPU optimization active for HPC multi-core systems
```

## Configuration

### Essential Environment Variables

```bash
# JAX Configuration (CPU-only)
export JAX_ENABLE_X64=1              # Use float64 for numerical precision
export JAX_PLATFORMS=cpu             # Force CPU platform

# Shell Completion Settings
export HOMODYNE_COMPLETION_CACHE_TTL=300    # 5-minute cache
export HOMODYNE_COMPLETION_MAX_FILES=20     # Max cached files
export HOMODYNE_COMPLETION_DEBUG=1          # Debug mode

# Cache location
~/.cache/homodyne/completion_cache
```

## Enhanced Logging

### Logging Modes

```bash
# Normal: Console + file logging
homodyne --config config.json --method mcmc

# Quiet: File-only logging (no console output)
homodyne --config config.json --quiet

# Verbose: DEBUG level logging
homodyne --config config.json --verbose
```

**Log Location:** `./homodyne_results/run.log` (or `--output-dir/run.log`)

## Performance Monitoring (CPU)

### CPU Monitoring

```bash
# Real-time CPU utilization
htop

# Process-specific monitoring
top -H -p $(pgrep -f homodyne)

# Memory tracking
free -h
```

### JAX Debugging

```bash
export JAX_DEBUG_NANS=1     # Debug NaN values
export JAX_LOG_COMPILES=1   # Log JIT compilation
```

## Troubleshooting

### Common Issues

#### Shell Completion Not Working

```bash
# Check installation
ls -la $CONDA_PREFIX/etc/conda/activate.d/homodyne-*

# Manual activation
source $CONDA_PREFIX/etc/conda/activate.d/homodyne-completion.sh

# Reset cache
rm ~/.cache/homodyne/completion_cache
conda deactivate && conda activate your-env
```

#### MCMC Backend Problems

```bash
# Check JAX platform
python3 -c "import jax; print(jax.devices())"

# Verify NumPyro import
python3 -c "import numpyro; print(numpyro.__version__)"
```

#### Performance Issues

```bash
# Check CPU utilization
htop

# Verify multi-core usage
python3 -c "import os; print(f'CPU cores: {os.cpu_count()}')"

# Force single-threaded execution (debugging)
export JAX_NUM_THREADS=1
homodyne --method mcmc
```

## Integration Examples

### CI/CD Integration

```yaml
name: Homodyne System Tests
on: [push, pull_request]

jobs:
  test-homodyne:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install homodyne
        run: |
          pip install homodyne-analysis
          homodyne-post-install --shell bash --non-interactive
      - name: System validation
        run: homodyne-validate --json
      - name: Run tests
        run: |
          homodyne --config test_config.json --method nlsq
```

### Docker Integration

```dockerfile
FROM python:3.12-slim

RUN pip install homodyne-analysis
RUN homodyne-post-install --shell bash --non-interactive

RUN homodyne-validate
ENTRYPOINT ["homodyne"]
```

## Environment Support

| Environment | Shell Completion | Auto-Activation | Advanced Tools |
|-------------|------------------|-----------------|----------------| | **Conda** | Full
support | On activate | All features | | **Mamba** | Full support | On activate | All
features | | **venv** | Manual setup | Manual sourcing | All features | | **virtualenv**
| Manual setup | Manual sourcing | All features | | **System Python** | User-wide | Not
recommended | Limited |

## Uninstallation

### Complete Cleanup (Recommended)

```bash
# Step 1: Smart cleanup (use interactive mode)
homodyne-cleanup --interactive

# Step 2: Remove package
pip uninstall homodyne-analysis

# Step 3: Verify cleanup
homodyne-validate 2>/dev/null || echo "Successfully uninstalled"
```

### Post-Cleanup Verification

```bash
# Restart shell
conda deactivate && conda activate <your-env>

# Verify removal
which hm 2>/dev/null || echo "Aliases removed"
which homodyne-validate 2>/dev/null || echo "Tools removed"
find "$CONDA_PREFIX" -name "*homodyne*" 2>/dev/null || echo "Files cleaned"
```

______________________________________________________________________

*This runtime system provides intelligent automation, cross-platform compatibility, and
CPU optimization for homodyne analysis workflows (v2.3.0+ CPU-only architecture).*
