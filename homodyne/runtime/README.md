# Homodyne Runtime System

**Advanced shell completion, smart GPU acceleration, and comprehensive system validation for the homodyne analysis package.**

---

## 📁 System Architecture

```
runtime/
├── shell/              # Advanced shell completion system
│   └── completion.sh   # Context-aware completion with caching
├── gpu/                # Smart GPU acceleration system  
│   ├── activation.sh   # CUDA detection and NumPyro setup
│   ├── gpu_wrapper.py  # homodyne-gpu command wrapper
│   └── optimizer.py    # Hardware-specific optimization
├── utils/              # System validation and utilities
│   └── system_validator.py  # Comprehensive system testing
└── README.md          # This documentation
```

## 🚀 Quick Setup

### One-Command Installation

```bash
# Complete setup with all features
homodyne-post-install --shell zsh --gpu --advanced

# Interactive setup (choose features)
homodyne-post-install --interactive

# Basic shell completion only
homodyne-post-install --shell zsh
```

### Installed Components

1. **🔧 Smart Shell Completion** - Context-aware completion with caching
2. **⚡ Isolated GPU Backend** - Pure NumPyro/JAX backend completely separated from PyMC
3. **📝 Unified Aliases** - Isolated CPU shortcuts (`hm`, `hc`, `hr`, `ha`) + isolated GPU shortcuts (`hgm`, `hga`)
4. **🛠️ Advanced Tools** - GPU optimization, system validation, benchmarking
5. **📋 Environment Integration** - Auto-activation in conda/mamba/venv

## 🔧 Shell Completion System

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
# → Shows: mcmc robust all (optimized for laminar flow)

# Interactive command builder
homodyne_build
# → Guided menu system for building commands
```

### Smart Completion Logic

| Config Mode | Suggested Methods | Reasoning |
|-------------|------------------|-----------|
| `static_isotropic` | `classical robust` | Optimized for static analysis |
| `static_anisotropic` | `classical robust` | CPU-optimized methods |
| `laminar_flow` | `mcmc robust all` | Benefits from GPU acceleration |

## ⚡ Isolated MCMC Backend System

### Completely Isolated Backend Architecture

The homodyne package features **completely separated MCMC implementations** to prevent PyTensor/JAX conflicts:

| Command | Backend | Implementation | Isolation | Use Case |
|---------|---------|----------------|-----------|----------|
| `homodyne` | **Pure PyMC CPU** | `mcmc_cpu_backend.py` | No JAX imports | Cross-platform, reliable |
| `homodyne-gpu` | **Pure NumPyro GPU** | `mcmc_gpu_backend.py` | No PyMC imports | High-performance, Linux + CUDA |

### Smart GPU Detection

**Key Features:**
- **Platform Detection**: Linux-only with automatic CPU fallback
- **CUDA Discovery**: Searches system and conda installations
- **Hardware Analysis**: GPU memory and capability detection
- **NumPyro Configuration**: Automatic JAX environment setup

### Auto-Configuration Profiles

Memory-based optimization applied automatically:

| GPU Memory | Memory Fraction | Batch Size | XLA Optimizations |
|------------|-----------------|------------|-------------------|
| < 4GB | 70% | Conservative | Basic flags |
| 4-8GB | 80% | Standard | Standard optimizations |
| > 8GB | 90% | Aggressive | Advanced XLA features |

### GPU Commands

```bash
# Smart GPU activation
gpu-on                   # Manual activation

# Detailed status with hardware info
gpu-status               # GPU hardware, CUDA, JAX devices, memory

# Performance benchmarking
gpu-bench                # Matrix multiplication benchmarks

# Hardware optimization
homodyne-gpu-optimize    # Detect hardware, apply optimal settings
```

### Isolated Backend Selection Logic

**`homodyne` Command (Isolated PyMC CPU Backend):**
- Backend Wrapper: `mcmc_cpu_backend.py` - completely isolated from JAX
- Environment: `HOMODYNE_GPU_INTENT` not set or `false`
- Platform: Linux, Windows, macOS (identical behavior)
- PyTensor Configuration: CPU-only mode with dedicated compilation directory
- Use Cases: Development, testing, cross-platform compatibility, CPU-only systems

**`homodyne-gpu` Command (Isolated NumPyro GPU Backend):**
- Backend Wrapper: `mcmc_gpu_backend.py` - completely isolated from PyMC
- Environment: `HOMODYNE_GPU_INTENT=true` (auto-set by wrapper)
- Platform: Linux with CUDA (JAX CPU fallback if no GPU), macOS/Windows (JAX CPU mode)
- JAX Configuration: GPU detection with automatic CPU fallback within JAX ecosystem
- Use Cases: Production, large datasets, high-performance computing, GPU acceleration

## 🚀 GPU Optimization System

### Hardware Analysis & Optimization

```bash
# Generate hardware report
homodyne-gpu-optimize --report

# Run benchmark and apply optimal settings
homodyne-gpu-optimize --benchmark --apply

# Create custom optimization profile
homodyne-gpu-optimize --profile conservative
homodyne-gpu-optimize --profile aggressive
```

### Optimization Features

1. **Hardware Detection**: GPU memory, CUDA compatibility, driver validation
2. **Performance Benchmarking**: Matrix operations, memory bandwidth, JAX compilation
3. **Auto-Configuration**: XLA flags, memory tuning, batch size optimization

### Custom Configuration

Create profiles in `~/.config/homodyne/gpu_profiles.json`:

```json
{
  "profiles": {
    "conservative": {
      "memory_fraction": 0.6,
      "batch_size": 500,
      "enable_x64": false
    },
    "aggressive": {
      "memory_fraction": 0.95,
      "batch_size": 5000,
      "enable_x64": true,
      "xla_flags": [
        "--xla_gpu_enable_triton_gemm=true",
        "--xla_gpu_enable_latency_hiding_scheduler=true"
      ]
    }
  }
}
```

## 🧪 System Validation

### Comprehensive Testing

```bash
# Complete system validation
homodyne-validate

# Verbose validation with timing
homodyne-validate --verbose  

# Component-specific testing
homodyne-validate --test gpu         # GPU system only
homodyne-validate --test completion  # Shell completion only

# JSON output for automation
homodyne-validate --json
```

### Test Categories

1. **Environment Detection**: Platform, Python version, virtual environment, shell type
2. **Installation Verification**: Commands, help output, module imports, version consistency
3. **Shell Completion**: Files, activation scripts, aliases, cache system
4. **GPU Setup** (Linux): Hardware detection, CUDA, JAX devices, driver compatibility
5. **Integration Testing**: Cross-component functionality, environment propagation

### Sample Output

```
🔍 HOMODYNE SYSTEM VALIDATION REPORT
================================================================================

📊 Summary: 5/5 tests passed ✅
🎉 All systems operational!

🖥️  Environment:
   Platform: Linux x86_64
   Python: 3.12.0
   Environment: conda (homodyne)
   Shell: zsh

📋 Test Results:
✅ PASS Environment Detection (0.003s)
✅ PASS Homodyne Installation (0.152s)
✅ PASS Shell Completion (0.089s)  
✅ PASS GPU Setup (0.234s)
✅ PASS Integration (0.067s)

💡 Recommendations:
   🚀 Your homodyne installation is optimized and ready!
   📖 Run 'hm --help' to see unified command options
   ⚡ GPU acceleration active - use 'gpu-bench' to test performance
```

## 🛠️ Configuration

### Essential Environment Variables

```bash
# MCMC Backend Routing
export HOMODYNE_GPU_INTENT=true     # NumPyro GPU backend
export HOMODYNE_GPU_INTENT=false    # PyMC CPU backend

# JAX Configuration
export JAX_ENABLE_X64=0              # Use float32 for GPU performance  
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8  # GPU memory allocation

# Advanced XLA Optimization
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true \
                  --xla_gpu_triton_gemm_any=true \
                  --xla_gpu_enable_latency_hiding_scheduler=true"
```

### Shell Completion Settings

```bash
# Cache configuration
export HOMODYNE_COMPLETION_CACHE_TTL=300    # 5-minute cache
export HOMODYNE_COMPLETION_MAX_FILES=20     # Max cached files
export HOMODYNE_COMPLETION_DEBUG=1          # Debug mode

# Cache location
~/.cache/homodyne/completion_cache
```

## 📝 Enhanced Logging

### Logging Modes

```bash
# Normal: Console + file logging
homodyne --config config.json --method mcmc

# Quiet: File-only logging (no console output)
homodyne --config config.json --quiet
homodyne-gpu --config config.json --quiet

# Verbose: DEBUG level logging  
homodyne --config config.json --verbose
```

**Log Location:** `./homodyne_results/run.log` (or `--output-dir/run.log`)

## 📊 Performance Monitoring

### GPU Monitoring

```bash
# Real-time GPU utilization
nvidia-smi dmon -i 0 -s pucvmet -d 1 &
hm --config config.json

# Memory tracking
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1

# Performance benchmarking
gpu-bench                           # Quick test
homodyne-gpu-optimize --benchmark   # Comprehensive analysis
```

### JAX Debugging

```bash
export JAX_DEBUG_NANS=1     # Debug NaN values
export JAX_TRACE_LEVEL=2    # Detailed JAX tracing
```

## 🐛 Troubleshooting

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
# Check current backend
python3 -c "
import os
gpu_intent = os.environ.get('HOMODYNE_GPU_INTENT', 'false').lower() == 'true'
print(f'GPU Intent: {gpu_intent}')
"

# Force specific backend
HOMODYNE_GPU_INTENT=false homodyne --method mcmc    # PyMC CPU
homodyne-gpu --method mcmc                          # NumPyro GPU
```

#### GPU Activation Issues
```bash
# Comprehensive diagnostics
gpu-status
homodyne-validate --test gpu --verbose
homodyne-gpu-optimize --report

# Manual troubleshooting
nvidia-smi                          # Check GPU/driver
ls -la /usr/local/cuda              # Verify CUDA
echo $LD_LIBRARY_PATH               # Check library paths
```

#### Performance Issues
```bash
# GPU optimization
homodyne-gpu-optimize --benchmark --apply
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7  # Reduce memory
export JAX_ENABLE_X64=0                     # Use float32

# Force JAX CPU fallback
export JAX_PLATFORMS=cpu
homodyne-gpu --method mcmc  # Still uses NumPyro, but on CPU
```

## 🔄 Integration Examples

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
          pip install homodyne-analysis[jax]
          homodyne-post-install --shell bash --non-interactive
      - name: System validation
        run: homodyne-validate --json
      - name: Run tests
        run: |
          homodyne --config test_config.json --method classical
          hm test_config.json
```

### Docker Integration

```dockerfile
FROM nvidia/cuda:12.3-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install homodyne-analysis[jax]
RUN homodyne-post-install --shell bash --gpu --non-interactive

ENV HOMODYNE_GPU_AUTO=1
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

RUN homodyne-validate && gpu-bench
ENTRYPOINT ["homodyne"]
```

## 📦 Environment Support

| Environment | Shell Completion | GPU Setup | Auto-Activation | Advanced Tools |
|-------------|------------------|-----------|-----------------|----------------|
| **Conda** | ✅ Full support | ✅ Automatic | ✅ On activate | ✅ All features |
| **Mamba** | ✅ Full support | ✅ Automatic | ✅ On activate | ✅ All features |
| **venv** | ✅ Manual setup | ✅ Manual setup | 🔶 Manual sourcing | ✅ All features |
| **virtualenv** | ✅ Manual setup | ✅ Manual setup | 🔶 Manual sourcing | ✅ All features |
| **System Python** | 🔶 User-wide | 🔶 User-wide | 🔴 Not recommended | 🔶 Limited |

## 🧹 Uninstallation

### Complete Cleanup (Recommended)

```bash
# Step 1: Smart cleanup (use interactive mode)
homodyne-cleanup --interactive

# Step 2: Remove package
pip uninstall homodyne-analysis

# Step 3: Verify cleanup
homodyne-validate 2>/dev/null || echo "✅ Successfully uninstalled"
```

### Why Cleanup Order Matters

**Always run cleanup first** - the smart removal system is part of the homodyne package and provides:
- Intelligent component detection
- Safe file removal with dry-run preview
- Cross-platform compatibility
- Verification of complete cleanup

### Post-Cleanup Verification

```bash
# Restart shell
conda deactivate && conda activate <your-env>

# Verify removal
which hm 2>/dev/null || echo "✅ Aliases removed"
which homodyne-validate 2>/dev/null || echo "✅ Tools removed"
find "$CONDA_PREFIX" -name "*homodyne*" 2>/dev/null || echo "✅ Files cleaned"
```

---

*This runtime system provides intelligent automation, cross-platform compatibility, and performance optimization for homodyne analysis workflows.*