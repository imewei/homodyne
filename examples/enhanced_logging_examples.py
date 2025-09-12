#!/usr/bin/env python3
"""
Enhanced Logging System Examples for Homodyne v2
================================================

Practical examples demonstrating the enhanced logging capabilities for
scientific computing workflows. These examples can be run directly to
explore the various logging features.

Usage:
    python enhanced_logging_examples.py [example_number]

Examples:
    1. Basic enhanced logging setup
    2. JAX-specific logging features
    3. Scientific computing contexts
    4. Advanced debugging capabilities
    5. Production monitoring setup
    6. Distributed computing logging
    7. Complete XPCS analysis pipeline

Author: Claude (Anthropic)
Based on: Homodyne v2 Enhanced Logging System
"""

import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Import enhanced logging components
try:
    from homodyne.utils.logging import get_logger
    from homodyne.utils.jax_logging import (
        log_jit_compilation, 
        jax_operation_context,
        log_gradient_computation,
        get_jax_compilation_stats
    )
    from homodyne.utils.scientific_logging import (
        xpcs_data_loading_context,
        correlation_computation_context,
        model_fitting_context,
        log_physics_validation,
        FittingProgressSnapshot
    )
    from homodyne.utils.advanced_debugging import (
        auto_recover,
        numerical_stability_context,
        debug_correlation_matrix,
        get_advanced_debugging_stats
    )
    from homodyne.utils.production_monitoring import (
        monitor_performance,
        production_monitoring_context,
        run_health_checks,
        get_production_monitoring_stats
    )
    from homodyne.utils.distributed_logging import (
        distributed_operation_context,
        get_distributed_computing_stats
    )
    from homodyne.config.enhanced_logging_manager import create_enhanced_config_manager
    
    ENHANCED_LOGGING_AVAILABLE = True
    
except ImportError as e:
    print(f"Enhanced logging not available: {e}")
    print("Please ensure homodyne package is installed with enhanced logging support")
    ENHANCED_LOGGING_AVAILABLE = False

# Try to import JAX for JAX-specific examples
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    print("JAX not available - JAX examples will be skipped")
    HAS_JAX = False


def example1_basic_setup():
    """Example 1: Basic enhanced logging setup."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Enhanced Logging Setup")
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    # Create a simple configuration for demonstration
    config_dict = {
        'logging': {
            'enabled': True,
            'level': 'INFO',
            'console': {
                'enabled': True,
                'level': 'INFO',
                'colors': True,
                'format': 'detailed'
            },
            'file': {
                'enabled': True,
                'level': 'DEBUG',
                'path': '~/.homodyne/examples/',
                'filename': 'example_basic.log'
            },
            'jax': {
                'enabled': True,
                'compilation': {'enabled': True}
            },
            'scientific': {
                'enabled': True,
                'data_loading': {'validate_data_quality': True}
            }
        }
    }
    
    print("Setting up enhanced logging configuration...")
    
    # Get basic logger
    logger = get_logger(__name__)
    
    print("Enhanced logging system configured successfully!")
    
    # Demonstrate basic logging
    logger.debug("This is a debug message")
    logger.info("This is an info message") 
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("Check the log file at ~/.homodyne/examples/example_basic.log")
    print("Note the enhanced formatting with timestamps, levels, and context information")


def example2_jax_logging():
    """Example 2: JAX-specific logging features."""
    print("\n" + "="*60)
    print("EXAMPLE 2: JAX-Specific Logging Features")
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    if not HAS_JAX:
        print("JAX not available - skipping JAX examples")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating JAX-specific logging features")
    
    # Example: JIT compilation logging
    @log_jit_compilation(track_memory=True, log_threshold_seconds=0.1)
    def jax_matrix_multiply(a, b):
        """Example JAX function with JIT compilation logging."""
        return jnp.dot(a, b)
    
    # Example: JAX operation context
    def demonstrate_jax_logging():
        """Demonstrate various JAX logging features."""
        
        # Create test matrices
        a = jnp.ones((1000, 1000))
        b = jnp.ones((1000, 1000))
        
        logger.info("Starting JAX operations demonstration")
        
        # JAX operation with comprehensive logging
        with jax_operation_context("matrix_operations", 
                                  track_memory=True, 
                                  log_device_placement=True) as jax_logger:
            
            jax_logger.info("Performing matrix multiplication")
            result1 = jax_matrix_multiply(a, b)
            
            jax_logger.info("Performing element-wise operations")
            result2 = jnp.exp(result1 * 0.001)  # Scaled to avoid overflow
            
            jax_logger.info("Computing reduction")
            final_result = jnp.sum(result2)
        
        logger.info(f"Final result: {final_result}")
        
        # Get compilation statistics
        jax_stats = get_jax_compilation_stats()
        logger.info(f"JAX compilation events: {len(jax_stats['recent_compilations'])}")
        
        return final_result
    
    # Example: Gradient computation logging
    @log_gradient_computation(include_grad_norm=True, norm_threshold=1e-6)
    def compute_gradients_example(x):
        """Example gradient computation with logging."""
        def loss_fn(params):
            return jnp.sum((params - 1.0) ** 2)
        
        return jax.grad(loss_fn)(x)
    
    # Run demonstrations
    result = demonstrate_jax_logging()
    
    # Gradient example
    x = jnp.array([2.0, 3.0, 4.0])
    gradients = compute_gradients_example(x)
    logger.info(f"Computed gradients: {gradients}")
    
    print("JAX logging features demonstrated!")
    print("Check logs for JIT compilation timing, memory usage, and device placement info")


def example3_scientific_computing():
    """Example 3: Scientific computing contexts."""
    print("\n" + "="*60)  
    print("EXAMPLE 3: Scientific Computing Contexts")
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating scientific computing logging contexts")
    
    # Simulate XPCS data loading
    def simulate_data_loading():
        """Simulate XPCS data loading with validation."""
        
        # Create a dummy data file path
        dummy_filepath = Path("/tmp/simulated_xpcs_data.hdf")
        
        with xpcs_data_loading_context(str(dummy_filepath), "HDF5") as data_info:
            logger.info("Simulating XPCS data loading...")
            
            # Simulate data loading (normally would read from HDF5)
            simulated_data = np.random.rand(100, 50, 200)  # q_vectors x phi_angles x time_points
            
            # Update data info
            data_info.data_shape = simulated_data.shape
            data_info.q_vectors = simulated_data.shape[0]
            data_info.phi_angles = simulated_data.shape[1]
            data_info.file_size_mb = simulated_data.nbytes / 1024**2
            
            # Add some validation warnings
            if np.any(simulated_data < 0):
                data_info.validation_warnings.append("Negative values detected")
            
            # Simulate preprocessing steps
            processed_data = simulated_data / np.mean(simulated_data)
            data_info.preprocessing_applied.append("normalization")
            
            logger.info(f"Loaded data shape: {data_info.data_shape}")
            
            return processed_data
    
    # Physics parameter validation example
    @log_physics_validation()
    def analyze_diffusion_parameters():
        """Example physics parameter analysis with validation."""
        
        # Simulate parameter optimization results
        results = {
            'D0': 1.5e-12,          # Diffusion coefficient (m²/s)
            'alpha': 0.85,          # Anomalous diffusion exponent  
            'D_offset': 1e-14,      # Diffusion offset
            'contrast': 0.95,       # Correlation contrast
            'offset': 1.05          # Baseline offset
        }
        
        logger.info(f"Optimized parameters: {results}")
        return results
    
    # Correlation computation example
    def simulate_correlation_computation():
        """Simulate correlation computation with monitoring."""
        
        input_shape = (100, 50, 200)
        
        with correlation_computation_context("vectorized", input_shape) as metrics:
            logger.info("Computing correlation functions...")
            
            # Simulate correlation computation
            time.sleep(0.5)  # Simulate computation time
            
            correlation_matrix = np.random.rand(100, 200)  # q_vectors x time_points
            
            # Update metrics
            metrics.output_shape = correlation_matrix.shape
            metrics.computation_method = "vectorized"
            metrics.cache_hit = False
            
            logger.info(f"Correlation matrix computed: {correlation_matrix.shape}")
            
            return correlation_matrix
    
    # Model fitting progress tracking
    def simulate_model_fitting():
        """Simulate model fitting with progress tracking."""
        
        initial_params = {'D0': 1e-12, 'alpha': 1.0, 'D_offset': 0.0}
        
        with model_fitting_context("DiffusionModel", "variational_inference", initial_params) as tracker:
            
            logger.info("Starting model fitting simulation...")
            
            # Simulate optimization iterations
            current_params = initial_params.copy()
            
            for iteration in range(20):
                # Simulate parameter updates
                current_params['D0'] *= (1 + np.random.normal(0, 0.01))
                current_params['alpha'] += np.random.normal(0, 0.001)
                current_params['D_offset'] += np.random.normal(0, 1e-15)
                
                # Simulate loss decrease
                loss = 100.0 * np.exp(-iteration * 0.1) + np.random.normal(0, 0.1)
                
                # Record fitting progress
                snapshot = FittingProgressSnapshot(
                    iteration=iteration,
                    loss_value=loss,
                    parameter_values=current_params.copy(),
                    optimization_method="variational_inference"
                )
                
                tracker.record_iteration(snapshot)
                
                # Check convergence
                if iteration > 5:
                    converged, message = tracker.check_convergence(window_size=5)
                    if converged:
                        logger.info(f"Converged after {iteration} iterations: {message}")
                        break
                
                time.sleep(0.1)  # Simulate iteration time
            
            # Get final progress summary
            progress = tracker.get_progress_summary()
            logger.info(f"Fitting completed: {progress['total_iterations']} iterations")
            logger.info(f"Final loss: {progress['current_loss']:.4f}")
            logger.info(f"Improvement: {progress['improvement']*100:.2f}%")
    
    # Run scientific computing demonstrations
    data = simulate_data_loading()
    parameters = analyze_diffusion_parameters()
    correlation_matrix = simulate_correlation_computation()
    simulate_model_fitting()
    
    print("Scientific computing contexts demonstrated!")
    print("Check logs for data validation, physics parameter validation, and fitting progress")


def example4_advanced_debugging():
    """Example 4: Advanced debugging capabilities."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Advanced Debugging Capabilities") 
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating advanced debugging features")
    
    # Example: Automatic error recovery
    @auto_recover(max_retries=3, backoff_factor=1.5)
    def unstable_computation(failure_probability=0.7):
        """Simulate an unstable computation that sometimes fails."""
        
        if np.random.rand() < failure_probability:
            raise ValueError(f"Simulated numerical instability (probability: {failure_probability})")
        
        return np.random.rand(10, 10)
    
    # Example: Numerical stability monitoring
    def demonstrate_stability_monitoring():
        """Demonstrate numerical stability monitoring."""
        
        with numerical_stability_context("matrix_operations", 
                                       check_inputs=True, 
                                       check_outputs=True) as interceptor:
            
            logger.info("Creating potentially unstable matrices...")
            
            # Create a nearly singular matrix
            matrix = np.random.rand(50, 50)
            matrix[-1, :] = matrix[0, :] + 1e-15  # Make nearly singular
            
            interceptor.add_array(matrix, "input_matrix")
            
            # Attempt matrix inversion (potentially unstable)
            try:
                inverse = np.linalg.inv(matrix)
                interceptor.add_array(inverse, "inverse_matrix")
                logger.info("Matrix inversion successful")
            except np.linalg.LinAlgError as e:
                logger.error(f"Matrix inversion failed: {e}")
    
    # Example: Correlation matrix debugging
    def demonstrate_correlation_debugging():
        """Demonstrate XPCS correlation matrix debugging."""
        
        logger.info("Creating simulated correlation matrix for debugging...")
        
        # Create a realistic-looking correlation matrix
        time_points = np.linspace(0, 10, 100)
        q_vectors = np.logspace(-3, -1, 20)
        
        correlation_matrix = np.zeros((len(q_vectors), len(time_points)))
        
        for i, q in enumerate(q_vectors):
            # Simulate g2 correlation function: g2 = 1 + contrast * exp(-2*Gamma*t)
            gamma = 0.1 * q**2  # Diffusive decay rate
            contrast = 0.9
            g2 = 1.0 + contrast * np.exp(-2 * gamma * time_points)
            
            # Add some noise and potential issues
            g2 += np.random.normal(0, 0.01, len(time_points))
            
            # Introduce some potential issues for debugging
            if i == 5:  # One q-vector has unusual behavior
                g2[50:] += 0.1  # Non-monotonic behavior
            
            correlation_matrix[i, :] = g2
        
        # Debug the correlation matrix
        debug_info = debug_correlation_matrix(
            correlation_matrix,
            q_vectors=q_vectors, 
            time_points=time_points
        )
        
        logger.info(f"Correlation matrix debugging completed")
        logger.info(f"Issues found: {len(debug_info['issues_found'])}")
        for issue in debug_info['issues_found']:
            logger.warning(f"Debug issue: {issue}")
    
    # Run debugging demonstrations
    logger.info("Testing automatic error recovery...")
    
    try:
        # This should succeed after retries
        result = unstable_computation(failure_probability=0.5)
        logger.info("Unstable computation succeeded with error recovery")
    except Exception as e:
        logger.error(f"Error recovery failed: {e}")
    
    demonstrate_stability_monitoring()
    demonstrate_correlation_debugging()
    
    # Get debugging statistics
    debug_stats = get_advanced_debugging_stats()
    
    logger.info("Advanced debugging statistics:")
    logger.info(f"  Error recovery events: {debug_stats['error_recovery'].get('total_errors', 0)}")
    logger.info(f"  Numerical stability analyses: {debug_stats['numerical_stability'].get('total_analyses', 0)}")
    logger.info(f"  Performance anomalies detected: {debug_stats['performance_anomalies'].get('total_anomalies', 0)}")
    
    print("Advanced debugging features demonstrated!")
    print("Check logs for error recovery attempts, numerical stability issues, and debugging insights")


def example5_production_monitoring():
    """Example 5: Production monitoring setup."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Production Monitoring Setup")
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating production monitoring capabilities")
    
    # Example: Performance monitoring decorator
    @monitor_performance("data_analysis", baseline_duration=2.0, alert_on_anomaly=True)
    def analyze_data_batch(data_size):
        """Simulate data analysis with performance monitoring."""
        
        # Simulate variable processing time based on data size
        processing_time = 0.001 * data_size + np.random.normal(0, 0.2)
        processing_time = max(0.1, processing_time)  # Minimum processing time
        
        logger.info(f"Processing {data_size} data points...")
        time.sleep(processing_time)
        
        return np.random.rand(data_size)
    
    # Example: Production context manager
    def demonstrate_production_context():
        """Demonstrate production monitoring context."""
        
        with production_monitoring_context("critical_analysis_pipeline",
                                          critical=True,
                                          expected_duration=3.0):
            
            logger.info("Starting critical analysis pipeline...")
            
            # Simulate multi-step analysis
            step1_data = analyze_data_batch(1000)
            logger.info("Step 1 completed")
            
            step2_data = analyze_data_batch(500)  
            logger.info("Step 2 completed")
            
            # Final processing
            final_result = np.mean(step1_data) + np.std(step2_data)
            logger.info(f"Pipeline completed with result: {final_result:.4f}")
            
            return final_result
    
    # Example: Health checks
    def demonstrate_health_checks():
        """Demonstrate system health monitoring."""
        
        logger.info("Running comprehensive health checks...")
        
        health_results = run_health_checks()
        
        for check_name, result in health_results.items():
            status_color = {
                'healthy': '✓',
                'degraded': '⚠', 
                'unhealthy': '✗'
            }.get(result.status, '?')
            
            logger.info(f"Health check {check_name}: {status_color} {result.status}")
            logger.info(f"  Message: {result.message}")
            
            if result.recommendations:
                for rec in result.recommendations:
                    logger.info(f"  Recommendation: {rec}")
    
    # Run production monitoring demonstrations
    logger.info("Testing performance monitoring...")
    
    # Test with different data sizes to trigger performance monitoring
    for size in [500, 1000, 5000]:  # Increasing sizes
        result = analyze_data_batch(size)
        logger.debug(f"Processed batch of size {size}")
    
    # Test production context
    logger.info("Testing production monitoring context...")
    production_result = demonstrate_production_context()
    
    # Run health checks
    demonstrate_health_checks()
    
    # Get production monitoring statistics
    prod_stats = get_production_monitoring_stats()
    
    logger.info("Production monitoring statistics:")
    logger.info(f"  Health checks run: {prod_stats['monitoring_status']['health_checks_run']}")
    logger.info(f"  Active alerts: {prod_stats['alert_summary']['active_alert_count']}")
    logger.info(f"  Metrics collected: {prod_stats['monitoring_status']['metrics_collected']}")
    
    print("Production monitoring demonstrated!")
    print("Check logs for performance metrics, health check results, and monitoring data")


def example6_distributed_computing():
    """Example 6: Distributed computing logging."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Distributed Computing Logging") 
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating distributed computing logging")
    
    # Example: Distributed operation context
    def simulate_distributed_operation():
        """Simulate distributed computing with logging coordination."""
        
        with distributed_operation_context("distributed_correlation_computation",
                                          monitor_resources=True,
                                          resource_snapshot_interval=5.0) as dist_logger:
            
            dist_logger.info("Starting distributed correlation computation simulation")
            
            # Simulate parallel processing
            for chunk_id in range(4):  # Simulate 4 parallel chunks
                dist_logger.info(f"Processing chunk {chunk_id + 1}/4")
                
                # Simulate chunk processing time
                chunk_time = 1.0 + np.random.normal(0, 0.2)
                time.sleep(max(0.5, chunk_time))
                
                dist_logger.info(f"Chunk {chunk_id + 1} completed")
            
            # Simulate result aggregation
            dist_logger.info("Aggregating results from all chunks...")
            time.sleep(0.5)
            
            dist_logger.info("Distributed computation completed successfully")
            
            return np.random.rand(100, 200)  # Simulated correlation matrix
    
    # Run distributed computing demonstration
    result = simulate_distributed_operation()
    
    # Get distributed computing statistics
    dist_stats = get_distributed_computing_stats()
    
    logger.info("Distributed computing statistics:")
    logger.info(f"  Node hostname: {dist_stats['node_info']['hostname']}")
    logger.info(f"  Process ID: {dist_stats['node_info']['process_id']}")
    logger.info(f"  MPI available: {dist_stats['capabilities']['has_mpi']}")
    
    if 'resource_summary' in dist_stats:
        resource_summary = dist_stats['resource_summary']
        logger.info(f"  CPU usage: {resource_summary.get('current_cpu_percent', 0):.1f}%")
        logger.info(f"  Memory usage: {resource_summary.get('current_memory_percent', 0):.1f}%")
    
    print("Distributed computing logging demonstrated!")
    print("Check logs for node-specific information, resource monitoring, and distributed coordination")
    print("For true distributed operation, run with MPI: mpirun -n 4 python enhanced_logging_examples.py 6")


def example7_complete_pipeline():
    """Example 7: Complete XPCS analysis pipeline with enhanced logging."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Complete XPCS Analysis Pipeline")
    print("="*60)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Enhanced logging not available - skipping example")
        return
    
    logger = get_logger(__name__)
    logger.info("Demonstrating complete XPCS analysis pipeline with enhanced logging")
    
    def complete_xpcs_pipeline():
        """Complete XPCS analysis demonstration with all logging features."""
        
        # Overall pipeline monitoring
        with production_monitoring_context("complete_xpcs_analysis",
                                          critical=False,
                                          expected_duration=10.0):
            
            logger.info("Starting complete XPCS analysis pipeline")
            
            # Step 1: Data loading with validation
            with xpcs_data_loading_context("simulated_experiment.hdf", "HDF5") as data_info:
                
                logger.info("Loading experimental data...")
                
                # Simulate realistic XPCS data
                n_q = 50
                n_phi = 32  
                n_time = 200
                
                raw_data = np.random.rand(n_q, n_phi, n_time) + 1.0
                
                # Update data info
                data_info.data_shape = raw_data.shape
                data_info.q_vectors = n_q
                data_info.phi_angles = n_phi
                data_info.file_size_mb = raw_data.nbytes / 1024**2
                
                # Simulate data validation
                if np.any(raw_data < 0.5):
                    data_info.validation_warnings.append("Low signal values detected")
                
                data_info.preprocessing_applied.append("baseline_correction")
                data_info.preprocessing_applied.append("normalization")
                
                logger.info(f"Data loaded: {data_info.data_shape}")
            
            # Step 2: Correlation computation with monitoring
            with correlation_computation_context("jax_vectorized", raw_data.shape) as corr_metrics:
                
                if HAS_JAX:
                    # JAX-accelerated correlation computation
                    with jax_operation_context("correlation_matrix_computation", 
                                              track_memory=True) as jax_logger:
                        
                        jax_logger.info("Computing correlation matrix with JAX")
                        
                        # Simulate correlation computation
                        correlation_matrix = jnp.mean(raw_data, axis=1)  # Simplified
                        correlation_matrix = jnp.abs(correlation_matrix) + 1.0
                        
                        # Add exponential decay for realism
                        time_points = jnp.linspace(0, 10, n_time)
                        for i in range(n_q):
                            decay = jnp.exp(-0.1 * (i+1) * time_points)
                            correlation_matrix = correlation_matrix.at[i, :].multiply(decay)
                
                else:
                    # NumPy fallback
                    logger.info("Computing correlation matrix with NumPy")
                    correlation_matrix = np.mean(raw_data, axis=1)
                    correlation_matrix = np.abs(correlation_matrix) + 1.0
                    
                    time_points = np.linspace(0, 10, n_time)
                    for i in range(n_q):
                        decay = np.exp(-0.1 * (i+1) * time_points)
                        correlation_matrix[i, :] *= decay
                
                # Update correlation metrics
                corr_metrics.output_shape = correlation_matrix.shape
                corr_metrics.computation_method = "jax_vectorized" if HAS_JAX else "numpy_vectorized"
                corr_metrics.cache_hit = False
                
                logger.info(f"Correlation matrix computed: {correlation_matrix.shape}")
            
            # Step 3: Numerical stability check
            with numerical_stability_context("correlation_analysis") as stability:
                
                stability.add_array(correlation_matrix, "correlation_matrix")
                
                # Check correlation properties
                debug_info = debug_correlation_matrix(
                    np.array(correlation_matrix),
                    q_vectors=np.logspace(-3, -1, n_q),
                    time_points=np.linspace(0, 10, n_time)
                )
                
                if debug_info['issues_found']:
                    for issue in debug_info['issues_found']:
                        logger.warning(f"Correlation matrix issue: {issue}")
            
            # Step 4: Physics parameter optimization with tracking
            @log_physics_validation()
            def optimize_diffusion_parameters():
                return {
                    'D0': 1.2e-12,      # Diffusion coefficient
                    'alpha': 0.9,       # Anomalous diffusion exponent
                    'D_offset': 5e-15,  # Diffusion offset
                    'contrast': 0.85,   # Correlation contrast
                    'offset': 1.02      # Baseline offset
                }
            
            with model_fitting_context("AnomalousDiffusionModel", 
                                      "variational_inference",
                                      {'D0': 1e-12, 'alpha': 1.0, 'D_offset': 0.0}) as fitting_tracker:
                
                logger.info("Starting parameter optimization...")
                
                # Simulate optimization iterations with error recovery
                @auto_recover(max_retries=2, backoff_factor=1.5)
                def optimization_step(iteration, params):
                    
                    # Simulate occasional numerical issues
                    if iteration == 5 and np.random.rand() < 0.3:
                        raise ValueError("Numerical instability in optimization")
                    
                    # Simulate parameter updates
                    new_params = params.copy()
                    new_params['D0'] *= (1 + np.random.normal(0, 0.05))
                    new_params['alpha'] += np.random.normal(0, 0.01)
                    new_params['D_offset'] += np.random.normal(0, 1e-15)
                    
                    # Simulate decreasing loss
                    loss = 50.0 * np.exp(-iteration * 0.08) + np.random.normal(0, 0.5)
                    
                    return new_params, max(0.1, loss)
                
                current_params = {'D0': 1e-12, 'alpha': 1.0, 'D_offset': 0.0}
                
                for iteration in range(15):
                    try:
                        current_params, loss = optimization_step(iteration, current_params)
                        
                        # Record progress
                        snapshot = FittingProgressSnapshot(
                            iteration=iteration,
                            loss_value=loss,
                            parameter_values=current_params.copy(),
                            optimization_method="variational_inference"
                        )
                        
                        fitting_tracker.record_iteration(snapshot)
                        
                        # Check convergence
                        if iteration > 3:
                            converged, message = fitting_tracker.check_convergence(window_size=3)
                            if converged:
                                logger.info(f"Optimization converged: {message}")
                                break
                        
                        time.sleep(0.2)  # Simulate iteration time
                        
                    except Exception as e:
                        logger.warning(f"Optimization step {iteration} failed: {e}")
                
                # Final parameter validation
                final_params = optimize_diffusion_parameters()
                
                # Get fitting summary
                fitting_summary = fitting_tracker.get_progress_summary()
                logger.info(f"Optimization completed in {fitting_summary['total_iterations']} iterations")
                logger.info(f"Final loss: {fitting_summary['current_loss']:.4f}")
                logger.info(f"Improvement: {fitting_summary['improvement']*100:.1f}%")
            
            # Step 5: Results validation and export
            logger.info("Validating and exporting results...")
            
            results = {
                'correlation_matrix': correlation_matrix,
                'optimized_parameters': final_params,
                'data_info': {
                    'shape': raw_data.shape,
                    'q_vectors': n_q,
                    'preprocessing': data_info.preprocessing_applied
                },
                'analysis_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'method': 'enhanced_logging_pipeline',
                    'jax_enabled': HAS_JAX
                }
            }
            
            logger.info("Complete XPCS analysis pipeline finished successfully!")
            
            return results
    
    # Run complete pipeline
    results = complete_xpcs_pipeline()
    
    # Get comprehensive statistics
    if HAS_JAX:
        jax_stats = get_jax_compilation_stats()
        logger.info(f"JAX compilations: {len(jax_stats['recent_compilations'])}")
    
    debug_stats = get_advanced_debugging_stats()
    prod_stats = get_production_monitoring_stats()
    
    logger.info("Pipeline statistics:")
    logger.info(f"  Error recovery events: {debug_stats['error_recovery'].get('total_errors', 0)}")
    logger.info(f"  Numerical analyses: {debug_stats['numerical_stability'].get('total_analyses', 0)}")
    logger.info(f"  Health checks: {prod_stats['monitoring_status']['health_checks_run']}")
    
    print("Complete XPCS analysis pipeline demonstrated!")
    print("This example shows integration of all enhanced logging features in a realistic workflow")
    print("Check logs for comprehensive analysis tracking, error recovery, and performance monitoring")


def main():
    """Main function to run examples."""
    
    print("Enhanced Logging System Examples for Homodyne v2")
    print("="*50)
    
    if not ENHANCED_LOGGING_AVAILABLE:
        print("Error: Enhanced logging system not available")
        print("Please ensure the homodyne package is properly installed")
        return 1
    
    # Example selection
    examples = {
        1: ("Basic enhanced logging setup", example1_basic_setup),
        2: ("JAX-specific logging features", example2_jax_logging),
        3: ("Scientific computing contexts", example3_scientific_computing),
        4: ("Advanced debugging capabilities", example4_advanced_debugging),
        5: ("Production monitoring setup", example5_production_monitoring),
        6: ("Distributed computing logging", example6_distributed_computing),
        7: ("Complete XPCS analysis pipeline", example7_complete_pipeline)
    }
    
    if len(sys.argv) > 1:
        try:
            example_num = int(sys.argv[1])
            if example_num in examples:
                print(f"Running Example {example_num}: {examples[example_num][0]}")
                examples[example_num][1]()
            else:
                print(f"Example {example_num} not found")
                return 1
        except ValueError:
            print(f"Invalid example number: {sys.argv[1]}")
            return 1
    else:
        # Run all examples
        print("Running all examples...")
        
        for example_num, (description, example_func) in examples.items():
            try:
                print(f"\n{'='*60}")
                print(f"Running Example {example_num}: {description}")
                print('='*60)
                example_func()
                print("✓ Example completed successfully")
            except Exception as e:
                print(f"✗ Example failed: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("Enhanced Logging Examples Complete!")
    print("="*60)
    print("Check the following locations for generated logs:")
    print("  ~/.homodyne/examples/")
    print("  ~/.homodyne/logs/")
    print("  ~/.homodyne/distributed_logs/")
    print("\nTo run individual examples:")
    for num, (desc, _) in examples.items():
        print(f"  python enhanced_logging_examples.py {num}  # {desc}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())