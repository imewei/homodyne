# Homodyne v2 Enhanced Logging System Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Configuration](#configuration)
4. [JAX-Specific Logging](#jax-specific-logging)
5. [Scientific Computing Contexts](#scientific-computing-contexts)
6. [Distributed Computing Support](#distributed-computing-support)
7. [Advanced Debugging Features](#advanced-debugging-features)
8. [Production Monitoring](#production-monitoring)
9. [Examples and Usage](#examples-and-usage)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Overview

The Homodyne v2 Enhanced Logging System provides comprehensive logging capabilities specifically designed for scientific computing workflows involving X-ray Photon Correlation Spectroscopy (XPCS) analysis. The system extends the existing robust logging infrastructure with specialized features for:

- **JAX-specific operations**: JIT compilation monitoring, GPU memory tracking, gradient logging
- **Scientific computing contexts**: XPCS data validation, physics parameter checking, model fitting tracking
- **Distributed computing**: Multi-node logging coordination, HPC integration, resource monitoring
- **Advanced debugging**: Error recovery, numerical stability analysis, performance anomaly detection
- **Production monitoring**: Health checks, alerting system, performance baseline tracking

### Key Benefits

- **Domain-specific insights**: Tailored logging for XPCS analysis workflows
- **Performance optimization**: JIT compilation and GPU memory monitoring
- **Reliability**: Intelligent error recovery and numerical stability checks
- **Scalability**: Distributed computing support for HPC environments
- **Production readiness**: Comprehensive monitoring and alerting capabilities

## Architecture

### Component Overview

```
Enhanced Logging System
├── Core Logging Infrastructure (existing)
│   ├── LoggerManager
│   ├── HomodyneFormatter
│   └── ContextFilter
├── JAX-Specific Logging (new)
│   ├── JITCompilationTracker
│   ├── JAXMemoryTracker
│   └── JAX operation contexts
├── Scientific Computing Logging (new)
│   ├── XPCSDataValidator
│   ├── FittingProgressTracker
│   └── Physics validation contexts
├── Distributed Computing Support (new)
│   ├── DistributedLoggerManager
│   ├── NodeInfo management
│   └── Resource monitoring
├── Advanced Debugging (new)
│   ├── ErrorRecoveryManager
│   ├── NumericalStabilityAnalyzer
│   └── PerformanceAnomalyDetector
└── Production Monitoring (new)
    ├── HealthChecker
    ├── AlertManager
    └── MetricsCollector
```

### Integration Points

The enhanced logging system integrates seamlessly with:
- Existing homodyne configuration management
- JAX computational backend
- XPCS data loading and analysis pipelines
- HPC job schedulers (SLURM, PBS)
- Monitoring platforms (Prometheus, Grafana)

## Configuration

### Basic Configuration

The enhanced logging system uses YAML configuration with backward compatibility for JSON:

```yaml
logging:
  enabled: true
  level: INFO
  
  console:
    enabled: true
    level: INFO
    format: detailed
    colors: true
    
  file:
    enabled: true
    level: DEBUG
    path: ~/.homodyne/logs/
    filename: homodyne.log
    max_size_mb: 50
    backup_count: 10
```

### Enhanced Features Configuration

```yaml
logging:
  # JAX-specific logging
  jax:
    enabled: true
    compilation:
      enabled: true
      log_threshold_seconds: 0.5
      track_memory_usage: true
    memory:
      enabled: true
      snapshot_interval_seconds: 30
      
  # Scientific computing logging  
  scientific:
    enabled: true
    data_loading:
      validate_data_quality: true
    physics_validation:
      validate_parameter_bounds: true
      
  # Distributed computing
  distributed:
    enabled: false  # Enable for HPC environments
    resource_monitoring:
      enabled: true
      snapshot_interval_seconds: 60
      
  # Advanced debugging
  advanced_debugging:
    enabled: true
    error_recovery:
      enabled: true
      max_retry_attempts: 3
      
  # Production monitoring
  production:
    enabled: false  # Enable in production
    health_checks:
      enabled: true
      check_interval_minutes: 15
```

### Environment Variable Overrides

Configuration can be overridden using environment variables:

```bash
export HOMODYNE_LOG_LEVEL=DEBUG
export HOMODYNE_LOG_JAX_ENABLED=true
export HOMODYNE_LOG_PRODUCTION_ENABLED=true
```

## JAX-Specific Logging

### JIT Compilation Monitoring

Automatically track JAX JIT compilation events:

```python
from homodyne.utils.jax_logging import log_jit_compilation

@log_jit_compilation(track_memory=True, log_threshold_seconds=0.5)
def compute_correlation_jax(data):
    # JAX computations here
    return jnp.correlate(data, data, mode='full')
```

### GPU Memory Tracking

Monitor GPU memory usage throughout operations:

```python
from homodyne.utils.jax_logging import jax_operation_context

with jax_operation_context("correlation_computation", track_memory=True):
    result = compute_large_correlation_matrix(data)
```

### Gradient Logging

Track gradient computation and detect numerical issues:

```python
from homodyne.utils.jax_logging import log_gradient_computation

@log_gradient_computation(include_grad_norm=True, norm_threshold=1e-6)
def compute_gradients(params, data):
    return jax.grad(loss_function)(params, data)
```

## Scientific Computing Contexts

### XPCS Data Loading and Validation

```python
from homodyne.utils.scientific_logging import xpcs_data_loading_context

with xpcs_data_loading_context("experimental_data.hdf", "HDF5") as data_info:
    # Load XPCS data
    correlation_data = load_correlation_data(filepath)
    
    # Update data info for logging
    data_info.data_shape = correlation_data.shape
    data_info.q_vectors = len(q_vector_list)
    
    # Validation warnings will be logged automatically
    if has_nan_values(correlation_data):
        data_info.validation_warnings.append("NaN values detected in correlation data")
```

### Physics Parameter Validation

```python
from homodyne.utils.scientific_logging import log_physics_validation

@log_physics_validation()
def optimize_parameters(initial_params):
    # Returns dictionary with parameter names and values
    return {
        'D0': 1.5e-12,      # Diffusion coefficient
        'alpha': 0.85,      # Anomalous diffusion exponent
        'D_offset': 1e-14   # Diffusion offset
    }
    # Validation results logged automatically
```

### Model Fitting Progress

```python
from homodyne.utils.scientific_logging import model_fitting_context

with model_fitting_context("DiffusionModel", "variational_inference", initial_params) as tracker:
    for iteration in range(max_iterations):
        # Update fitting progress
        snapshot = FittingProgressSnapshot(
            iteration=iteration,
            loss_value=current_loss,
            parameter_values=current_params
        )
        tracker.record_iteration(snapshot)
        
        # Check convergence
        converged, message = tracker.check_convergence()
        if converged:
            break
```

## Distributed Computing Support

### Multi-Node Logging Coordination

```python
from homodyne.utils.distributed_logging import distributed_operation_context

with distributed_operation_context("parallel_correlation_computation", 
                                  monitor_resources=True) as logger:
    # Distributed computation
    results = mpi_parallel_correlate(data_chunks)
    
    # Node-specific information automatically logged
    # Resource usage monitored across all nodes
```

### HPC Job Integration

The system automatically detects SLURM/PBS environments and organizes logs hierarchically:

```
~/.homodyne/distributed_logs/
├── job_12345/
│   ├── node_compute001/
│   │   ├── rank_0/
│   │   │   └── homodyne_pid_1234.log
│   │   └── rank_1/
│   │       └── homodyne_pid_1235.log
│   └── node_compute002/
│       └── rank_2/
│           └── homodyne_pid_1236.log
└── current -> job_12345/  # Symlink to current session
```

### Resource Monitoring

```python
from homodyne.utils.distributed_logging import get_distributed_computing_stats

# Get comprehensive distributed computing statistics
stats = get_distributed_computing_stats()
print(f"Node: {stats['node_info']['hostname']}")
print(f"MPI Rank: {stats['node_info']['node_rank']}")
print(f"Current CPU: {stats['resource_summary']['current_cpu_percent']:.1f}%")
print(f"Current Memory: {stats['resource_summary']['current_memory_percent']:.1f}%")
```

## Advanced Debugging Features

### Automatic Error Recovery

```python
from homodyne.utils.advanced_debugging import auto_recover

@auto_recover(max_retries=3, backoff_factor=2.0)
def unstable_computation(data):
    # Computation that might fail due to numerical issues
    return compute_correlation_with_potential_instability(data)

# Automatic retry with intelligent recovery strategies:
# 1. Garbage collection
# 2. JAX cache clearing  
# 3. Numerical stabilization suggestions
```

### Numerical Stability Monitoring

```python
from homodyne.utils.advanced_debugging import numerical_stability_context

with numerical_stability_context("correlation_matrix_inversion", 
                                check_inputs=True, 
                                check_outputs=True) as interceptor:
    
    # Add arrays to be checked
    interceptor.add_array(correlation_matrix, "correlation_matrix")
    
    # Perform potentially unstable computation
    inverse_matrix = jnp.linalg.inv(correlation_matrix)
    
    interceptor.add_array(inverse_matrix, "inverse_matrix")
    
    # Stability analysis logged automatically
    # Issues like high condition numbers, NaN values detected
```

### Correlation Matrix Debugging

```python
from homodyne.utils.advanced_debugging import debug_correlation_matrix

# Comprehensive XPCS correlation matrix validation
debug_info = debug_correlation_matrix(
    correlation_matrix, 
    q_vectors=q_array, 
    time_points=time_array
)

# Checks performed:
# - Proper g2 behavior (values >= 1 at t=0)
# - Monotonic decay patterns
# - Q-vector and time point consistency
# - Numerical stability issues
```

### Performance Anomaly Detection

```python
from homodyne.utils.advanced_debugging import get_advanced_debugging_stats

# Get comprehensive debugging statistics
debug_stats = get_advanced_debugging_stats()

# Check for performance anomalies
anomalies = debug_stats['performance_anomalies']
if anomalies['total_anomalies'] > 0:
    print(f"Performance anomalies detected: {anomalies['total_anomalies']}")
    print(f"Most anomalous operation: {anomalies['most_anomalous_operation']}")
```

## Production Monitoring

### Health Checks

```python
from homodyne.utils.production_monitoring import run_health_checks

# Run comprehensive health checks
health_results = run_health_checks()

for check_name, result in health_results.items():
    status = result.status  # 'healthy', 'degraded', 'unhealthy'
    message = result.message
    print(f"{check_name}: {status} - {message}")
    
    if result.recommendations:
        for rec in result.recommendations:
            print(f"  Recommendation: {rec}")
```

### Performance Monitoring

```python
from homodyne.utils.production_monitoring import monitor_performance

@monitor_performance("correlation_computation", baseline_duration=2.5, alert_on_anomaly=True)
def compute_correlations(data):
    return expensive_correlation_computation(data)

# Automatically:
# - Records performance metrics
# - Compares against baseline
# - Raises alerts for significant deviations
```

### Production Context Manager

```python
from homodyne.utils.production_monitoring import production_monitoring_context

with production_monitoring_context("critical_analysis_pipeline", 
                                  critical=True, 
                                  expected_duration=300.0):
    # Critical production operation
    results = run_complete_analysis_pipeline(experiment_data)
    
    # Automatic monitoring:
    # - Performance tracking
    # - Resource monitoring  
    # - Alert generation for failures
```

### Dashboard Export

```python
from homodyne.utils.production_monitoring import export_monitoring_dashboard

# Export comprehensive monitoring data for visualization
dashboard_file = export_monitoring_dashboard()
print(f"Monitoring dashboard exported to: {dashboard_file}")

# Can be imported into Grafana, Prometheus, or custom dashboards
```

## Examples and Usage

### Example 1: Basic Enhanced Logging Setup

```python
# config.yaml
logging:
  enabled: true
  level: INFO
  console:
    enabled: true
    colors: true
  file:
    enabled: true
    path: ~/.homodyne/logs/
  jax:
    enabled: true
    compilation:
      enabled: true
  scientific:
    enabled: true

# Python code
from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager

# Create and configure enhanced logging
config_manager = create_enhanced_config_manager("config.yaml")

# Use enhanced logging features
from homodyne.utils.jax_logging import get_logger
logger = get_logger(__name__)

logger.info("Enhanced logging system active")
```

### Example 2: Complete XPCS Analysis with Enhanced Logging

```python
from homodyne.utils.scientific_logging import (
    xpcs_data_loading_context, 
    correlation_computation_context,
    model_fitting_context
)
from homodyne.utils.jax_logging import jax_operation_context
from homodyne.utils.advanced_debugging import numerical_stability_context

def analyze_xpcs_data(data_file, q_vectors, analysis_params):
    """Complete XPCS analysis with comprehensive logging."""
    
    # Step 1: Load and validate data
    with xpcs_data_loading_context(data_file, "HDF5") as data_info:
        raw_data = load_experimental_data(data_file)
        data_info.data_shape = raw_data.shape
        data_info.q_vectors = len(q_vectors)
        
        # Preprocessing with logging
        if np.any(np.isnan(raw_data)):
            data_info.validation_warnings.append("NaN values in raw data")
        
        preprocessed_data = preprocess_data(raw_data)
        data_info.preprocessing_applied.append("baseline_correction")
    
    # Step 2: Compute correlations with JAX and stability monitoring
    with correlation_computation_context("jax_vectorized", raw_data.shape):
        with jax_operation_context("correlation_matrix_computation", track_memory=True):
            with numerical_stability_context("correlation_computation") as stability:
                
                correlation_matrix = compute_correlation_jax(preprocessed_data, q_vectors)
                stability.add_array(correlation_matrix, "correlation_matrix")
    
    # Step 3: Model fitting with progress tracking
    with model_fitting_context("DiffusionModel", "variational_inference", analysis_params) as tracker:
        
        optimizer = create_optimizer(analysis_params)
        
        for iteration in range(1000):
            loss, gradients = compute_loss_and_gradients(correlation_matrix, analysis_params)
            analysis_params = optimizer.update(analysis_params, gradients)
            
            # Record fitting progress
            tracker.record_iteration(FittingProgressSnapshot(
                iteration=iteration,
                loss_value=loss,
                parameter_values=dict(zip(['D0', 'alpha', 'D_offset'], analysis_params)),
                parameter_gradients=dict(zip(['D0', 'alpha', 'D_offset'], gradients))
            ))
            
            # Check convergence
            converged, message = tracker.check_convergence()
            if converged:
                break
    
    return analysis_params

# Usage
results = analyze_xpcs_data("experiment_data.hdf", q_vectors, initial_params)
```

### Example 3: Distributed HPC Analysis

```python
from homodyne.utils.distributed_logging import (
    distributed_operation_context,
    get_distributed_logger
)
from mpi4py import MPI

def distributed_xpcs_analysis(data_chunks):
    """Distributed XPCS analysis across HPC nodes."""
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    logger = get_distributed_logger(__name__)
    
    with distributed_operation_context(f"distributed_analysis_rank_{rank}", 
                                      monitor_resources=True,
                                      resource_snapshot_interval=30.0) as dist_logger:
        
        dist_logger.info(f"Processing data chunk {rank}/{size}")
        
        # Process local data chunk
        local_results = analyze_local_chunk(data_chunks[rank])
        
        # Synchronize results across nodes
        comm.Barrier()  # Logged automatically by MPI integration
        
        all_results = comm.gather(local_results, root=0)
        
        if rank == 0:
            # Aggregate results on master node
            final_results = aggregate_distributed_results(all_results)
            dist_logger.info("Distributed analysis completed successfully")
            return final_results
    
    return None

# Run with: mpirun -n 4 python distributed_analysis.py
```

### Example 4: Production Deployment with Monitoring

```python
from homodyne.utils.production_monitoring import (
    production_monitoring_context,
    run_health_checks,
    export_monitoring_dashboard
)
from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager

def production_analysis_service():
    """Production XPCS analysis service with full monitoring."""
    
    # Configure enhanced logging for production
    config_manager = create_enhanced_config_manager("production_config.yaml")
    
    # Initial health check
    health_results = run_health_checks()
    overall_health = all(result.status == 'healthy' for result in health_results.values())
    
    if not overall_health:
        logger.critical("System health check failed - aborting analysis")
        return
    
    # Main analysis loop with monitoring
    while True:
        try:
            with production_monitoring_context("analysis_pipeline", 
                                              critical=True,
                                              expected_duration=600.0):
                
                # Get next analysis request
                request = get_analysis_request()
                if not request:
                    time.sleep(10)
                    continue
                
                # Process request
                results = process_analysis_request(request)
                
                # Store results
                store_analysis_results(results)
                
        except Exception as e:
            logger.critical(f"Analysis pipeline failed: {e}")
            # Error recovery and alerting handled automatically
            
        # Periodic health checks
        if time.time() % 900 == 0:  # Every 15 minutes
            run_health_checks()
            
        # Export monitoring dashboard hourly
        if time.time() % 3600 == 0:  # Every hour
            export_monitoring_dashboard()

# Production configuration (production_config.yaml)
"""
logging:
  enabled: true
  level: INFO
  production:
    enabled: true
    health_checks:
      enabled: true
      check_interval_minutes: 15
    alerting:
      enabled: true
      email_notifications:
        enabled: true
        recipients: ["admin@lab.edu"]
      webhook_notifications:
        enabled: true
        webhook_url: "https://monitoring.lab.edu/webhooks/homodyne"
    metrics_collection:
      enabled: true
    performance_baselines:
      enabled: true
"""
```

## Best Practices

### Configuration Best Practices

1. **Use YAML configuration** for better readability and comments
2. **Enable appropriate logging levels** based on environment:
   - Development: `DEBUG` for detailed information
   - Production: `INFO` for operational visibility
   - Critical systems: `WARNING` for minimal noise
3. **Configure log rotation** to prevent disk space issues
4. **Use environment variables** for deployment-specific overrides

### Performance Best Practices

1. **Use appropriate log thresholds**:
   ```yaml
   logging:
     performance:
       threshold_seconds: 0.1  # Only log operations > 100ms
     jax:
       compilation:
         log_threshold_seconds: 0.5  # Only log slow compilations
   ```

2. **Enable memory tracking selectively**:
   ```yaml
   logging:
     jax:
       memory:
         enabled: true
         snapshot_interval_seconds: 30  # Not too frequent
   ```

3. **Monitor resource usage in distributed environments**:
   ```yaml
   logging:
     distributed:
       resource_monitoring:
         snapshot_interval_seconds: 60  # Reasonable for HPC
   ```

### Scientific Computing Best Practices

1. **Always validate physics parameters**:
   ```python
   @log_physics_validation()
   def fitting_function(params):
       # Parameter validation logged automatically
       return optimized_parameters
   ```

2. **Monitor numerical stability**:
   ```python
   with numerical_stability_context("matrix_operations"):
       result = potentially_unstable_computation()
   ```

3. **Track model fitting progress**:
   ```python
   with model_fitting_context("model_name", "method", params) as tracker:
       # Convergence monitoring automatic
   ```

### Production Best Practices

1. **Enable health checks in production**:
   ```yaml
   logging:
     production:
       enabled: true
       health_checks:
         enabled: true
         check_interval_minutes: 15
   ```

2. **Configure appropriate alerting**:
   ```yaml
   logging:
     production:
       alerting:
         enabled: true
         email_notifications:
           enabled: true
           min_level: ERROR
         webhook_notifications:
           enabled: true
           min_level: WARNING
   ```

3. **Use performance baselines**:
   ```yaml
   logging:
     production:
       performance_baselines:
         enabled: true
         auto_create_baselines: true
         deviation_alert_threshold: 3.0
   ```

## Troubleshooting

### Common Issues

#### 1. JAX Compilation Logging Not Working

**Problem**: JAX compilation events not being logged

**Solution**:
```yaml
# Ensure JAX logging is enabled
logging:
  jax:
    enabled: true
    compilation:
      enabled: true
      log_threshold_seconds: 0.1  # Lower threshold
```

**Check**: Verify JAX is available:
```python
from homodyne.utils.jax_logging import HAS_JAX
print(f"JAX available: {HAS_JAX}")
```

#### 2. Distributed Logging Not Creating Node-Specific Directories

**Problem**: All processes writing to same log file

**Solution**:
```yaml
logging:
  distributed:
    enabled: true
    hierarchical_logging:
      enabled: true
      create_node_specific_logs: true
```

**Check**: Verify distributed manager is active:
```python
from homodyne.utils.distributed_logging import get_distributed_computing_stats
stats = get_distributed_computing_stats()
print(f"Node: {stats['node_info']['hostname']}")
print(f"Process ID: {stats['node_info']['process_id']}")
```

#### 3. Production Alerts Not Being Sent

**Problem**: Health check failures not triggering alerts

**Solution**: Verify notification configuration:
```yaml
logging:
  production:
    enabled: true
    alerting:
      enabled: true
      email_notifications:
        enabled: true
        smtp_host: "your-smtp-server"
        username: "your-username"
        # Use environment variables for sensitive data
```

**Check**: Test notification settings:
```python
from homodyne.utils.production_monitoring import get_production_monitoring_stats
stats = get_production_monitoring_stats()
print(f"Active alerts: {stats['alert_summary']['active_alert_count']}")
```

#### 4. Log Files Growing Too Large

**Problem**: Log files consuming too much disk space

**Solution**: Configure appropriate rotation:
```yaml
logging:
  file:
    max_size_mb: 50     # Smaller files
    backup_count: 5     # Fewer backups
  performance:
    threshold_seconds: 1.0  # Higher threshold
```

#### 5. Memory Tracking Causing Performance Issues

**Problem**: Memory monitoring slowing down computation

**Solution**: Optimize monitoring intervals:
```yaml
logging:
  jax:
    memory:
      snapshot_interval_seconds: 60  # Less frequent
  distributed:
    resource_monitoring:
      snapshot_interval_seconds: 120  # Even less frequent for distributed
```

### Debug Commands

#### Check Enhanced Logging Status

```python
from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager

config_manager = create_enhanced_config_manager("config.yaml")
effective_config = config_manager.get_effective_config()

print("Enhanced Logging Status:")
for category, settings in effective_config.items():
    if isinstance(settings, dict) and 'enabled' in settings:
        print(f"  {category}: {'enabled' if settings['enabled'] else 'disabled'}")
```

#### Validate Configuration

```python
validation_issues = config_manager.validate_enhanced_configuration()
if validation_issues:
    print("Configuration issues found:")
    for issue in validation_issues:
        print(f"  - {issue}")
else:
    print("Configuration validation passed")
```

#### Export Debugging Information

```python
from homodyne.utils.advanced_debugging import dump_debugging_report

# Export comprehensive debugging information
report_file = dump_debugging_report(include_full_traceback=True)
print(f"Debug report exported to: {report_file}")
```

#### Monitor System Resources

```python
from homodyne.utils.production_monitoring import run_health_checks

health_results = run_health_checks()
for check_name, result in health_results.items():
    print(f"{check_name}: {result.status}")
    if result.status != 'healthy':
        print(f"  Issue: {result.message}")
        for rec in result.recommendations:
            print(f"  Recommendation: {rec}")
```

This comprehensive guide provides everything needed to effectively use the Homodyne v2 Enhanced Logging System for scientific computing workflows, from basic setup to advanced production deployment scenarios.