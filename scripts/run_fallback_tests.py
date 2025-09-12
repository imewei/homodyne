#!/usr/bin/env python3
"""
JAX Fallback Test Runner for Homodyne v2
========================================

Comprehensive test runner for validating JAX fallback scenarios across
the entire homodyne codebase. Provides various test execution modes
and detailed reporting.

Usage:
    python run_fallback_tests.py --mode basic
    python run_fallback_tests.py --mode comprehensive 
    python run_fallback_tests.py --mode performance
    python run_fallback_tests.py --mode integration
    python run_fallback_tests.py --mode all

Features:
- Multiple test execution modes
- Detailed performance reporting
- Scientific accuracy validation
- Memory usage monitoring  
- JAX vs NumPy comparison benchmarking
- User-friendly progress reporting
"""

import sys
import os
import argparse
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add homodyne to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

class FallbackTestRunner:
    """Comprehensive test runner for JAX fallback scenarios."""
    
    def __init__(self, verbose: bool = True, save_reports: bool = True):
        self.verbose = verbose
        self.save_reports = save_reports
        self.base_dir = Path(__file__).parent
        self.test_dir = self.base_dir / 'homodyne' / 'tests'
        self.reports_dir = self.base_dir / 'test_reports'
        
        # Ensure reports directory exists
        if self.save_reports:
            self.reports_dir.mkdir(exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp."""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    def run_basic_tests(self) -> Dict[str, Any]:
        """Run basic JAX fallback functionality tests."""
        self.log("Running basic JAX fallback tests...")
        
        test_results = {
            'mode': 'basic',
            'start_time': time.time(),
            'tests_run': [],
            'success': False
        }
        
        # Basic test modules to run
        basic_test_files = [
            'test_jax_fallbacks.py::TestJAXFallbackSystem::test_basic_math_functions_fallback',
            'test_jax_fallbacks.py::TestJAXFallbackSystem::test_gradient_fallback_accuracy',
            'test_jax_fallbacks.py::TestJAXFallbackSystem::test_backend_validation_and_diagnostics'
        ]
        
        if PYTEST_AVAILABLE:
            for test_file in basic_test_files:
                test_path = self.test_dir / test_file.split('::')[0]
                if test_path.exists():
                    try:
                        result = pytest.main([
                            str(test_path), 
                            '-v', 
                            '-k', 
                            test_file.split('::')[-1] if '::' in test_file else '',
                            '--tb=short'
                        ])
                        
                        test_results['tests_run'].append({
                            'test': test_file,
                            'success': result == 0,
                            'return_code': result
                        })
                        
                    except Exception as e:
                        test_results['tests_run'].append({
                            'test': test_file,
                            'success': False,
                            'error': str(e)
                        })
        else:
            # Manual test execution
            self.log("Pytest not available - running manual basic tests")
            try:
                # Import and run key tests manually
                from homodyne.tests.test_jax_fallbacks import TestJAXFallbackSystem
                
                test_instance = TestJAXFallbackSystem()
                test_instance.setup_test_environment()
                
                # Run critical tests
                test_methods = [
                    'test_basic_math_functions_fallback',
                    'test_gradient_fallback_accuracy', 
                    'test_backend_validation_and_diagnostics'
                ]
                
                for method_name in test_methods:
                    try:
                        method = getattr(test_instance, method_name)
                        method()
                        
                        test_results['tests_run'].append({
                            'test': method_name,
                            'success': True
                        })
                        
                    except Exception as e:
                        test_results['tests_run'].append({
                            'test': method_name,
                            'success': False,
                            'error': str(e)
                        })
                        
            except Exception as e:
                self.log(f"Error running manual tests: {e}", "ERROR")
        
        # Determine overall success
        successful_tests = sum(1 for t in test_results['tests_run'] if t['success'])
        test_results['success'] = successful_tests == len(test_results['tests_run'])
        test_results['success_rate'] = successful_tests / len(test_results['tests_run']) if test_results['tests_run'] else 0
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        self.log(f"Basic tests completed: {successful_tests}/{len(test_results['tests_run'])} passed")
        
        return test_results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive JAX fallback test suite."""
        self.log("Running comprehensive JAX fallback test suite...")
        
        test_results = {
            'mode': 'comprehensive',
            'start_time': time.time(),
            'test_categories': {},
            'success': False
        }
        
        # Comprehensive test categories
        test_categories = {
            'accuracy_tests': [
                'test_gradient_fallback_accuracy',
                'test_xpcs_physics_functions_fallback',
                'test_extreme_parameter_values_stability'
            ],
            'integration_tests': [
                'test_optimization_gradient_integration',
                'test_end_to_end_analysis_workflow'
            ],
            'performance_tests': [
                'test_large_parameter_space_memory_management',
                'test_batch_processing_fallback'
            ],
            'robustness_tests': [
                'test_error_recovery_and_graceful_degradation',
                'test_warning_system_and_user_guidance',
                'test_hessian_computation_fallback'
            ]
        }
        
        for category, test_methods in test_categories.items():
            self.log(f"Running {category}...")
            
            category_results = {
                'tests_run': [],
                'start_time': time.time()
            }
            
            if PYTEST_AVAILABLE:
                # Run tests with pytest
                for test_method in test_methods:
                    test_file = self.test_dir / 'test_jax_fallbacks.py'
                    if test_file.exists():
                        try:
                            result = pytest.main([
                                str(test_file),
                                '-v',
                                '-k', test_method,
                                '--tb=short'
                            ])
                            
                            category_results['tests_run'].append({
                                'test': test_method,
                                'success': result == 0,
                                'return_code': result
                            })
                            
                        except Exception as e:
                            category_results['tests_run'].append({
                                'test': test_method,
                                'success': False,
                                'error': str(e)
                            })
            else:
                # Manual execution
                try:
                    from homodyne.tests.test_jax_fallbacks import TestJAXFallbackSystem
                    
                    test_instance = TestJAXFallbackSystem()
                    test_instance.setup_test_environment()
                    
                    for test_method in test_methods:
                        try:
                            if hasattr(test_instance, test_method):
                                method = getattr(test_instance, test_method)
                                method()
                                
                                category_results['tests_run'].append({
                                    'test': test_method,
                                    'success': True
                                })
                            else:
                                category_results['tests_run'].append({
                                    'test': test_method,
                                    'success': False,
                                    'error': 'Method not found'
                                })
                                
                        except Exception as e:
                            category_results['tests_run'].append({
                                'test': test_method,
                                'success': False,
                                'error': str(e)
                            })
                            
                except Exception as e:
                    self.log(f"Error setting up test instance: {e}", "ERROR")
            
            # Calculate category results
            successful_tests = sum(1 for t in category_results['tests_run'] if t['success'])
            category_results['success'] = successful_tests == len(category_results['tests_run'])
            category_results['success_rate'] = successful_tests / len(category_results['tests_run']) if category_results['tests_run'] else 0
            category_results['end_time'] = time.time()
            category_results['duration'] = category_results['end_time'] - category_results['start_time']
            
            test_results['test_categories'][category] = category_results
            
            self.log(f"{category} completed: {successful_tests}/{len(category_results['tests_run'])} passed")
        
        # Overall results
        total_tests = sum(len(cat['tests_run']) for cat in test_results['test_categories'].values())
        total_successful = sum(sum(1 for t in cat['tests_run'] if t['success']) for cat in test_results['test_categories'].values())
        
        test_results['success'] = total_successful == total_tests
        test_results['success_rate'] = total_successful / total_tests if total_tests > 0 else 0
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        self.log(f"Comprehensive tests completed: {total_successful}/{total_tests} passed")
        
        return test_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration workflow tests."""
        self.log("Running integration workflow tests...")
        
        test_results = {
            'mode': 'integration',
            'start_time': time.time(),
            'workflows_tested': [],
            'success': False
        }
        
        # Integration test file
        integration_test_file = self.test_dir / 'test_integration_fallback_workflows.py'
        
        if integration_test_file.exists():
            if PYTEST_AVAILABLE:
                try:
                    result = pytest.main([
                        str(integration_test_file),
                        '-v',
                        '--tb=short'
                    ])
                    
                    test_results['workflows_tested'].append({
                        'workflow': 'all_integration_workflows',
                        'success': result == 0,
                        'return_code': result
                    })
                    
                except Exception as e:
                    test_results['workflows_tested'].append({
                        'workflow': 'all_integration_workflows',
                        'success': False,
                        'error': str(e)
                    })
            else:
                # Manual execution
                try:
                    from homodyne.tests.test_integration_fallback_workflows import test_comprehensive_integration_suite
                    
                    test_comprehensive_integration_suite()
                    
                    test_results['workflows_tested'].append({
                        'workflow': 'comprehensive_integration_suite',
                        'success': True
                    })
                    
                except Exception as e:
                    test_results['workflows_tested'].append({
                        'workflow': 'comprehensive_integration_suite', 
                        'success': False,
                        'error': str(e)
                    })
        else:
            self.log("Integration test file not found", "WARNING")
        
        # Results
        successful_workflows = sum(1 for w in test_results['workflows_tested'] if w['success'])
        test_results['success'] = successful_workflows == len(test_results['workflows_tested'])
        test_results['success_rate'] = successful_workflows / len(test_results['workflows_tested']) if test_results['workflows_tested'] else 0
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        self.log(f"Integration tests completed: {successful_workflows}/{len(test_results['workflows_tested'])} passed")
        
        return test_results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance benchmarking tests."""
        self.log("Running performance benchmarking tests...")
        
        test_results = {
            'mode': 'performance',
            'start_time': time.time(),
            'benchmarks': {},
            'success': False
        }
        
        try:
            # Import performance test function
            from homodyne.tests.test_jax_fallbacks import test_performance_benchmarking
            
            # Run performance benchmarks
            test_performance_benchmarking()
            
            test_results['benchmarks']['performance_benchmarking'] = {
                'success': True,
                'description': 'JAX vs NumPy performance comparison completed'
            }
            
        except Exception as e:
            test_results['benchmarks']['performance_benchmarking'] = {
                'success': False,
                'error': str(e),
                'description': 'Performance benchmarking failed'
            }
        
        # Results
        successful_benchmarks = sum(1 for b in test_results['benchmarks'].values() if b['success'])
        test_results['success'] = successful_benchmarks == len(test_results['benchmarks'])
        test_results['success_rate'] = successful_benchmarks / len(test_results['benchmarks']) if test_results['benchmarks'] else 0
        test_results['end_time'] = time.time()
        test_results['duration'] = test_results['end_time'] - test_results['start_time']
        
        self.log(f"Performance tests completed: {successful_benchmarks}/{len(test_results['benchmarks'])} passed")
        
        return test_results
    
    def save_report(self, results: Dict[str, Any], filename: str):
        """Save test results to JSON file."""
        if not self.save_reports:
            return
        
        report_file = self.reports_dir / f"{filename}_{int(time.time())}.json"
        
        try:
            with open(report_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.log(f"Report saved to: {report_file}")
            
        except Exception as e:
            self.log(f"Failed to save report: {e}", "ERROR")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print test results summary."""
        print("\n" + "=" * 60)
        print(f"JAX FALLBACK TEST SUMMARY - {results['mode'].upper()} MODE")
        print("=" * 60)
        
        print(f"Duration: {results['duration']:.2f} seconds")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Overall Status: {'PASS' if results['success'] else 'FAIL'}")
        
        if 'test_categories' in results:
            print(f"\nTest Categories:")
            for category, cat_results in results['test_categories'].items():
                status = "PASS" if cat_results['success'] else "FAIL"
                print(f"  {category}: {cat_results['success_rate']:.1%} ({status})")
        
        if 'workflows_tested' in results:
            print(f"\nWorkflows Tested:")
            for workflow in results['workflows_tested']:
                status = "PASS" if workflow['success'] else "FAIL"
                print(f"  {workflow['workflow']}: {status}")
        
        if 'benchmarks' in results:
            print(f"\nBenchmarks:")
            for benchmark, bench_results in results['benchmarks'].items():
                status = "PASS" if bench_results['success'] else "FAIL"
                print(f"  {benchmark}: {status}")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="JAX Fallback Test Runner for Homodyne v2",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--mode', 
        choices=['basic', 'comprehensive', 'integration', 'performance', 'all'],
        default='basic',
        help='Test execution mode (default: basic)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    parser.add_argument(
        '--save-reports',
        action='store_true', 
        default=True,
        help='Save detailed reports to files (default: True)'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Disable report saving'
    )
    
    args = parser.parse_args()
    
    if args.no_reports:
        args.save_reports = False
    
    # Initialize test runner
    runner = FallbackTestRunner(verbose=args.verbose, save_reports=args.save_reports)
    
    print("Homodyne v2 JAX Fallback Test Runner")
    print("=" * 50)
    
    # Run tests based on mode
    if args.mode == 'basic':
        results = runner.run_basic_tests()
        runner.save_report(results, 'basic_fallback_tests')
        runner.print_summary(results)
        
    elif args.mode == 'comprehensive':
        results = runner.run_comprehensive_tests()
        runner.save_report(results, 'comprehensive_fallback_tests')
        runner.print_summary(results)
        
    elif args.mode == 'integration':
        results = runner.run_integration_tests()
        runner.save_report(results, 'integration_fallback_tests')
        runner.print_summary(results)
        
    elif args.mode == 'performance':
        results = runner.run_performance_tests()
        runner.save_report(results, 'performance_fallback_tests')
        runner.print_summary(results)
        
    elif args.mode == 'all':
        print("Running ALL test modes...")
        all_results = {}
        
        # Run each mode
        for mode in ['basic', 'comprehensive', 'integration', 'performance']:
            print(f"\n{'='*20} {mode.upper()} TESTS {'='*20}")
            
            if mode == 'basic':
                results = runner.run_basic_tests()
            elif mode == 'comprehensive':
                results = runner.run_comprehensive_tests()
            elif mode == 'integration':
                results = runner.run_integration_tests()
            elif mode == 'performance':
                results = runner.run_performance_tests()
            
            all_results[mode] = results
            runner.save_report(results, f'{mode}_fallback_tests')
            runner.print_summary(results)
        
        # Overall summary
        print(f"\n{'='*20} OVERALL SUMMARY {'='*20}")
        
        for mode, results in all_results.items():
            status = "PASS" if results['success'] else "FAIL"
            print(f"{mode.capitalize()}: {results['success_rate']:.1%} ({status})")
        
        overall_success_rate = sum(r['success_rate'] for r in all_results.values()) / len(all_results)
        overall_status = "PASS" if all(r['success'] for r in all_results.values()) else "FAIL"
        
        print(f"\nOverall Success Rate: {overall_success_rate:.1%}")
        print(f"Overall Status: {overall_status}")
    
    # Exit with appropriate code
    if 'results' in locals():
        sys.exit(0 if results['success'] else 1)
    elif 'all_results' in locals():
        sys.exit(0 if all(r['success'] for r in all_results.values()) else 1)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()