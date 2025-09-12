#!/usr/bin/env python3
"""
Enhanced Data Loading Test Runner for Homodyne v2
=================================================

Comprehensive test runner for the enhanced data loading system with intelligent
test selection, performance monitoring, and detailed reporting.

Usage:
    python run_enhanced_data_loading_tests.py [options]

Examples:
    # Run all tests
    python run_enhanced_data_loading_tests.py

    # Run only integration tests
    python run_enhanced_data_loading_tests.py --integration

    # Run quick tests (exclude slow tests)
    python run_enhanced_data_loading_tests.py --quick

    # Run with performance monitoring
    python run_enhanced_data_loading_tests.py --performance --monitor

    # Generate test data and run tests
    python run_enhanced_data_loading_tests.py --generate-data --all
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Optional dependencies
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False


class EnhancedTestRunner:
    """Enhanced test runner with intelligent test selection and monitoring"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "homodyne" / "tests"
        self.test_data_dir = None
        self.performance_results = {}
        
    def setup_test_environment(self):
        """Set up test environment and dependencies"""
        print("Setting up test environment...")
        
        # Check required dependencies
        required_packages = ['pytest', 'h5py', 'numpy', 'pyyaml']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check optional dependencies
        optional_packages = ['psutil', 'memory_profiler', 'jax']
        available_optional = []
        
        for package in optional_packages:
            try:
                __import__(package)
                available_optional.append(package)
            except ImportError:
                pass
        
        if available_optional:
            print(f"✓ Optional packages available: {', '.join(available_optional)}")
        
        return True
    
    def generate_test_data(self, force: bool = False):
        """Generate synthetic test data"""
        print("Generating synthetic test data...")
        
        try:
            from homodyne.tests.data.synthetic_data_generator import generate_test_dataset_suite
            
            # Create temporary test data directory
            self.test_data_dir = Path(tempfile.mkdtemp(prefix="homodyne_test_data_"))
            
            print(f"Test data directory: {self.test_data_dir}")
            
            # Generate test datasets
            test_datasets = generate_test_dataset_suite(self.test_data_dir / "datasets")
            
            print(f"✓ Generated {len(test_datasets)} test datasets")
            for name, path in test_datasets.items():
                size_mb = path.stat().st_size / 1024**2
                print(f"  - {name}: {size_mb:.1f} MB")
            
            return test_datasets
            
        except Exception as e:
            print(f"❌ Failed to generate test data: {e}")
            return None
    
    def run_test_category(self, category: str, extra_args: List[str] = None) -> bool:
        """Run tests for a specific category"""
        category_dirs = {
            'integration': 'integration',
            'performance': 'performance', 
            'robustness': 'robustness',
            'config': 'config'
        }
        
        if category not in category_dirs:
            print(f"❌ Unknown test category: {category}")
            return False
        
        test_path = self.test_dir / category_dirs[category]
        if not test_path.exists():
            print(f"❌ Test directory not found: {test_path}")
            return False
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', str(test_path), '-v']
        
        # Add extra arguments
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"\n{'='*60}")
        print(f"Running {category.upper()} Tests")
        print(f"{'='*60}")
        print(f"Command: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=False)
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            status = "✓ PASSED" if success else "❌ FAILED"
            
            print(f"\n{status} - {category.upper()} tests completed in {duration:.1f}s")
            
            # Store performance results
            self.performance_results[category] = {
                'duration': duration,
                'success': success,
                'return_code': result.returncode
            }
            
            return success
            
        except Exception as e:
            print(f"❌ Error running {category} tests: {e}")
            return False
    
    def run_quick_tests(self) -> bool:
        """Run quick tests (excluding slow/memory-intensive tests)"""
        print("\n🚀 Running Quick Tests (excluding slow tests)")
        
        cmd = [
            'python', '-m', 'pytest', 
            str(self.test_dir),
            '-v',
            '-m', 'not slow and not memory_intensive',
            '--tb=short'
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, cwd=self.project_root)
        duration = time.time() - start_time
        
        success = result.returncode == 0
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{status} - Quick tests completed in {duration:.1f}s")
        
        return success
    
    def run_comprehensive_tests(self, monitor_performance: bool = False) -> Dict[str, bool]:
        """Run comprehensive test suite"""
        print("\n🧪 Running Comprehensive Test Suite")
        
        categories = ['integration', 'performance', 'robustness', 'config']
        results = {}
        
        for category in categories:
            extra_args = []
            if monitor_performance and category == 'performance':
                extra_args.extend(['-s', '--tb=long'])
            elif category == 'robustness':
                extra_args.extend(['--tb=short'])
            
            success = self.run_test_category(category, extra_args)
            results[category] = success
        
        return results
    
    def generate_test_report(self, results: Dict[str, bool]):
        """Generate comprehensive test report"""
        print(f"\n{'='*60}")
        print("ENHANCED DATA LOADING TEST REPORT")
        print(f"{'='*60}")
        
        # Overall status
        all_passed = all(results.values())
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        print(f"\nOverall Status: {overall_status}")
        
        # Category breakdown
        print(f"\nTest Category Results:")
        for category, success in results.items():
            status = "✓" if success else "❌"
            duration = self.performance_results.get(category, {}).get('duration', 0)
            print(f"  {status} {category.title():15} - {duration:.1f}s")
        
        # Performance summary
        if self.performance_results:
            total_time = sum(r.get('duration', 0) for r in self.performance_results.values())
            print(f"\nTotal Test Time: {total_time:.1f}s")
        
        # System information
        if HAS_PSUTIL:
            print(f"\nSystem Information:")
            print(f"  CPU Count: {psutil.cpu_count()}")
            print(f"  Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
            print(f"  Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")
        
        # Test data information
        if self.test_data_dir and self.test_data_dir.exists():
            try:
                total_size = sum(f.stat().st_size for f in self.test_data_dir.rglob('*') if f.is_file())
                print(f"\nTest Data: {total_size / 1024**2:.1f} MB in {self.test_data_dir}")
            except:
                pass
        
        print(f"\n{'='*60}")
        
        return all_passed
    
    def cleanup(self):
        """Clean up temporary test data"""
        if self.test_data_dir and self.test_data_dir.exists():
            print(f"Cleaning up test data: {self.test_data_dir}")
            shutil.rmtree(self.test_data_dir)


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Enhanced Data Loading Test Runner for Homodyne v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all tests
  %(prog)s --quick                   # Run quick tests only
  %(prog)s --integration             # Run integration tests only
  %(prog)s --performance --monitor   # Run performance tests with monitoring
  %(prog)s --generate-data --all     # Generate test data and run all tests
  %(prog)s --config --verbose        # Run config tests with verbose output
        """
    )
    
    # Test category options
    parser.add_argument('--all', action='store_true', 
                       help='Run comprehensive test suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick tests (exclude slow tests)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run performance tests')
    parser.add_argument('--robustness', action='store_true',
                       help='Run robustness tests')
    parser.add_argument('--config', action='store_true',
                       help='Run configuration tests')
    
    # Options
    parser.add_argument('--generate-data', action='store_true',
                       help='Generate synthetic test data before running tests')
    parser.add_argument('--monitor', action='store_true',
                       help='Enable performance monitoring')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Do not clean up test data after completion')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no specific category is selected, default to all
    if not any([args.all, args.quick, args.integration, args.performance, 
               args.robustness, args.config]):
        args.all = True
    
    # Create test runner
    runner = EnhancedTestRunner()
    
    try:
        # Set up environment
        if not runner.setup_test_environment():
            return 1
        
        # Generate test data if requested
        if args.generate_data:
            test_data = runner.generate_test_data()
            if test_data is None:
                print("❌ Failed to generate test data, continuing with existing data")
        
        # Run tests
        results = {}
        
        if args.quick:
            success = runner.run_quick_tests()
            results['quick'] = success
        
        elif args.all:
            results = runner.run_comprehensive_tests(args.monitor)
        
        else:
            # Run specific categories
            if args.integration:
                results['integration'] = runner.run_test_category('integration')
            if args.performance:
                extra_args = ['-s', '--tb=long'] if args.monitor else []
                results['performance'] = runner.run_test_category('performance', extra_args)
            if args.robustness:
                results['robustness'] = runner.run_test_category('robustness')
            if args.config:
                extra_args = ['-v', '--tb=long'] if args.verbose else []
                results['config'] = runner.run_test_category('config', extra_args)
        
        # Generate report
        all_passed = runner.generate_test_report(results)
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        return 130
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    finally:
        # Cleanup unless requested not to
        if not args.no_cleanup:
            runner.cleanup()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)