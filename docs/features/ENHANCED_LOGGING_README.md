# Homodyne v2 Enhanced Logging System

## Overview

The Homodyne v2 Enhanced Logging System is a comprehensive, production-ready logging infrastructure specifically designed for scientific computing workflows involving X-ray Photon Correlation Spectroscopy (XPCS) analysis. It extends the existing robust logging foundation with specialized capabilities for modern scientific computing environments.

## 🚀 Key Features

### **JAX-Specific Logging**
- JIT compilation monitoring with timing and memory tracking
- GPU/TPU memory usage monitoring with device placement logging
- Gradient computation logging with norm analysis and convergence tracking
- Automatic device optimization hints and compilation statistics

### **Scientific Computing Contexts**
- XPCS data validation with physics parameter bounds checking
- Correlation function computation monitoring with cache performance tracking
- Model fitting progress tracking with convergence analysis
- Numerical stability monitoring with condition number analysis

### **Distributed Computing Support**
- Multi-node logging coordination for HPC environments
- MPI-aware logging with rank identification and synchronization
- SLURM/PBS job integration with hierarchical log organization
- Resource monitoring across compute nodes with aggregation capabilities

### **Advanced Debugging Features**
- Intelligent error recovery with exponential backoff and retry strategies
- Numerical stability analysis with automatic issue detection
- Performance anomaly detection using statistical analysis
- Memory leak detection with process monitoring

### **Production Monitoring**
- Comprehensive health checks with configurable thresholds
- Multi-channel alerting (email, webhook, logging) with severity-based routing
- Performance baseline tracking with drift detection
- Real-time metrics collection with dashboard export capabilities

## 📁 System Architecture

```
Enhanced Logging System
├── Core Infrastructure (homodyne/utils/)
│   ├── logging.py              # Base logging system (existing)
│   ├── jax_logging.py          # JAX-specific logging utilities (NEW)
│   ├── scientific_logging.py   # XPCS scientific contexts (NEW)
│   ├── distributed_logging.py  # HPC/distributed support (NEW)
│   ├── advanced_debugging.py   # Error recovery & analysis (NEW)
│   └── production_monitoring.py # Health checks & alerting (NEW)
├── Configuration (homodyne/config/)
│   ├── enhanced_logging_manager.py    # Enhanced config management (NEW)
│   └── templates/v2_yaml/
│       └── config_enhanced_logging_template.yaml (NEW)
└── Documentation (docs/)
    └── enhanced_logging_guide.md      # Comprehensive guide (NEW)
```

## 🛠 Installation & Setup

### Prerequisites
```bash
# Core dependencies (required)
pip install pyyaml numpy

# Enhanced features (optional)
pip install jax[gpu] psutil requests  # JAX, system monitoring, webhooks
pip install mpi4py                    # Distributed computing support
pip install scipy                     # Statistical analysis for anomaly detection
```

### Basic Configuration
Create a configuration file (`config.yaml`):

```yaml
logging:
  enabled: true
  level: INFO
  
  console:
    enabled: true
    colors: true
    
  file:
    enabled: true
    path: ~/.homodyne/logs/
    
  # Enhanced features
  jax:
    enabled: true
    compilation:
      enabled: true
      track_memory_usage: true
      
  scientific:
    enabled: true
    data_loading:
      validate_data_quality: true
    physics_validation:
      validate_parameter_bounds: true
      
  production:
    enabled: false  # Enable in production environments
    health_checks:
      enabled: true
```

### Quick Start
```python
from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager
from homodyne.utils.jax_logging import get_logger

# Initialize enhanced logging
config_manager = create_enhanced_config_manager("config.yaml")
logger = get_logger(__name__)

logger.info("Enhanced logging system active!")
```

## 📊 Usage Examples

### JAX Operations with Comprehensive Logging
```python
from homodyne.utils.jax_logging import jax_operation_context, log_jit_compilation
import jax.numpy as jnp

@log_jit_compilation(track_memory=True)
def compute_correlation_jax(data):
    return jnp.correlate(data, data, mode='full')

with jax_operation_context("correlation_computation", track_memory=True):
    result = compute_correlation_jax(experimental_data)
    # JIT compilation timing, memory usage, and device placement logged automatically
```

### Scientific Computing with XPCS Validation
```python
from homodyne.utils.scientific_logging import (
    xpcs_data_loading_context, 
    model_fitting_context
)

# Data loading with automatic validation
with xpcs_data_loading_context("experiment.hdf", "HDF5") as data_info:
    data = load_experimental_data("experiment.hdf")
    data_info.data_shape = data.shape
    # Data quality issues logged automatically

# Model fitting with progress tracking
with model_fitting_context("DiffusionModel", "VI", initial_params) as tracker:
    for iteration in optimization_loop():
        # Progress automatically tracked and convergence monitored
        tracker.record_iteration(snapshot)
```

### Production Monitoring with Health Checks
```python
from homodyne.utils.production_monitoring import (
    production_monitoring_context,
    run_health_checks
)

# Critical operations with monitoring
with production_monitoring_context("analysis_pipeline", critical=True):
    results = run_complete_analysis(data)
    # Performance tracked, alerts raised on issues

# System health monitoring
health_status = run_health_checks()
# CPU, memory, disk, JAX devices checked automatically
```

### Distributed Computing on HPC Systems
```python
from homodyne.utils.distributed_logging import distributed_operation_context

with distributed_operation_context("parallel_analysis", monitor_resources=True):
    # Multi-node coordination with resource monitoring
    local_results = analyze_data_chunk(data_chunk)
    # Node-specific logs organized hierarchically
```

### Advanced Debugging with Error Recovery
```python
from homodyne.utils.advanced_debugging import auto_recover, numerical_stability_context

@auto_recover(max_retries=3, backoff_factor=2.0)
def unstable_computation(data):
    return potentially_failing_operation(data)
    # Automatic retry with intelligent recovery strategies

with numerical_stability_context("matrix_operations") as stability:
    result = compute_correlation_matrix(data)
    stability.add_array(result, "correlation_matrix")
    # Numerical issues detected and reported automatically
```

## 🔧 Configuration Reference

### Environment Variable Overrides
```bash
export HOMODYNE_LOG_LEVEL=DEBUG
export HOMODYNE_LOG_JAX_ENABLED=true
export HOMODYNE_LOG_PRODUCTION_ENABLED=true
export HOMODYNE_LOG_DISTRIBUTED_ENABLED=false
```

### Complete Configuration Example
See [`config_enhanced_logging_template.yaml`](homodyne/config/templates/v2_yaml/config_enhanced_logging_template.yaml) for a comprehensive configuration template with all options documented.

## 🏁 Running Examples

The system includes comprehensive examples demonstrating all features:

```bash
# Enable enhanced logging in your analysis
python -m homodyne --method vi --config your_config.yaml --verbose

# Enable file logging
python -m homodyne --method vi --config your_config.yaml --log-file

# Quiet mode (file logging only)
python -m homodyne --method vi --config your_config.yaml --quiet --log-file

# Distributed analysis with enhanced logging
mpirun -n 4 python -m homodyne --method mcmc --config distributed_config.yaml
```

## 📈 Monitoring & Dashboards

### Health Check Dashboard
```python
from homodyne.utils.production_monitoring import export_monitoring_dashboard

# Export comprehensive monitoring data
dashboard_file = export_monitoring_dashboard()
# Import into Grafana, Prometheus, or custom dashboards
```

### Log Analysis
```bash
# View recent logs
tail -f ~/.homodyne/logs/homodyne.log

# JAX compilation logs
tail -f ~/.homodyne/logs/jax_compilation.log

# Performance logs
tail -f ~/.homodyne/logs/performance.log

# Distributed logs (HPC environments)
ls ~/.homodyne/distributed_logs/current/
```

## 🔄 Integration with Existing Code

The enhanced logging system is designed for seamless integration:

### Minimal Integration
```python
# Replace existing logger imports
# from homodyne.utils.logging import get_logger  # Old
from homodyne.utils.jax_logging import get_logger  # Enhanced

# No other changes required - enhanced features available as decorators/contexts
```

### Full Integration
```python
# Use enhanced configuration management
from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager

# Replace standard config manager
config_manager = create_enhanced_config_manager("config.yaml")

# Enhanced logging features now available throughout application
```

## 🚨 Production Deployment

### Health Monitoring Setup
```yaml
logging:
  production:
    enabled: true
    health_checks:
      enabled: true
      check_interval_minutes: 15
      
    alerting:
      enabled: true
      email_notifications:
        enabled: true
        recipients: ["admin@lab.edu", "scientist@lab.edu"]
        min_level: ERROR
      webhook_notifications:
        enabled: true
        webhook_url: "https://monitoring.lab.edu/webhooks/homodyne"
        min_level: WARNING
```

### HPC Deployment
```bash
# SLURM job script
#!/bin/bash
#SBATCH --job-name=homodyne_analysis
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

export HOMODYNE_LOG_DISTRIBUTED_ENABLED=true
export HOMODYNE_LOG_LEVEL=INFO

mpirun python analysis_pipeline.py
# Logs automatically organized by job ID and node
```

## 📖 Documentation

- **[Complete User Guide](docs/enhanced_logging_guide.md)** - Comprehensive documentation with examples
- **[Configuration Reference](homodyne/config/templates/v2_yaml/config_enhanced_logging_template.yaml)** - Full configuration template
- **[CLI Reference](../../README.md)** - Command-line usage examples

## 🤝 Contributing

The enhanced logging system is designed for extensibility:

### Adding Custom Health Checks
```python
from homodyne.utils.production_monitoring import HealthChecker

def custom_health_check():
    # Your custom health check logic
    return HealthCheckResult(
        check_name="custom_check",
        status="healthy",
        message="Custom check passed"
    )

health_checker = HealthChecker()
health_checker.register_check("custom", custom_health_check)
```

### Custom Alert Handlers
```python
from homodyne.utils.production_monitoring import AlertManager, AlertLevel

alert_manager = AlertManager()
alert_manager.configure_webhook_notifications(
    webhook_url="your-monitoring-system.com/webhooks",
    headers={"Authorization": "Bearer your-token"}
)
```

## 🏷 Version Compatibility

- **Homodyne v2.x**: Full enhanced logging support
- **Homodyne v1.x**: Basic logging compatibility (enhanced features disabled)
- **JAX**: Optional dependency - features gracefully disabled if unavailable
- **MPI**: Optional dependency for distributed computing features

## 📄 License

This enhanced logging system extends the existing Homodyne package and follows the same licensing terms.

## 🆘 Support & Troubleshooting

### Common Issues

1. **JAX logging not working**: Verify JAX installation and enable in config
2. **Distributed logs not organized**: Enable hierarchical logging in distributed config
3. **Production alerts not sent**: Check notification configuration and network access
4. **Large log files**: Configure appropriate rotation and thresholds

### Debug Information
```python
# Export comprehensive debugging information
from homodyne.utils.advanced_debugging import dump_debugging_report
report_file = dump_debugging_report(include_full_traceback=True)

# Check system capabilities
from homodyne.utils.production_monitoring import get_production_monitoring_stats
capabilities = get_production_monitoring_stats()['system_capabilities']
```

### Performance Tuning
```yaml
# Optimize for high-throughput environments
logging:
  performance:
    threshold_seconds: 1.0    # Higher threshold
  jax:
    memory:
      snapshot_interval_seconds: 60  # Less frequent monitoring
  distributed:
    resource_monitoring:
      snapshot_interval_seconds: 120
```

---

**The Enhanced Logging System transforms scientific computing workflows with comprehensive observability, intelligent debugging, and production-ready monitoring - all while maintaining the simplicity and performance of the original Homodyne logging infrastructure.**