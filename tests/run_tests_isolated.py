#!/usr/bin/env python
# tests/run_tests_isolated.py
"""
Run tests in isolated groups to prevent GPU memory issues.
This script runs test files separately to ensure proper cleanup between them.
"""

import subprocess
import sys
import os


def run_test_group(test_files, group_name):
    """Run a group of test files."""
    print(f"\n{'=' * 60}")
    print(f"Running {group_name}")
    print('=' * 60)
    
    cmd = ["pytest", "-v"] + test_files
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    return result.returncode


def main():
    """Run tests in isolated groups."""
    # Group 1: Unit tests (low memory usage)
    unit_tests = [
        "test_memory_config.py",
        "test_memory_profiler.py", 
        "test_device_manager.py",
        "test_block_swap_manager.py",
        "test_model_partitioner.py"
    ]
    
    # Group 2: Module tests (moderate memory usage)
    module_tests = [
        "test_dynamic_memory_module.py"
    ]
    
    # Group 3: Integration tests (higher memory usage)
    integration_tests = [
        "test_integration.py"
    ]
    
    # Group 4: Hardware scenario tests (highest memory usage)
    hardware_tests = [
        "test_hardware_scenarios.py"
    ]
    
    # Group 5: Performance benchmarks (optional)
    benchmark_tests = [
        "test_performance_benchmarks.py"
    ]
    
    # Run each group separately
    exit_code = 0
    
    exit_code |= run_test_group(unit_tests, "Unit Tests")
    exit_code |= run_test_group(module_tests, "Module Tests")
    exit_code |= run_test_group(integration_tests, "Integration Tests")
    exit_code |= run_test_group(hardware_tests, "Hardware Scenario Tests")
    
    # Only run benchmarks if requested
    if "--benchmark" in sys.argv:
        exit_code |= run_test_group(benchmark_tests + ["--benchmark-only"], "Performance Benchmarks")
    else:
        print("\nSkipping benchmarks (use --benchmark to include)")
    
    # Summary
    print(f"\n{'=' * 60}")
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Some tests failed (exit code: {exit_code})")
    print('=' * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())