#!/usr/bin/env python
# tests/run_tests.py
"""
Convenient test runner for the Overflow test suite.
Usage: python tests/run_tests.py [category]
"""

import sys
import subprocess
import argparse


def run_command(cmd):
    """Run a command and print output."""
    print(f"\n🚀 Running: {' '.join(cmd)}")
    print("-" * 60)
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run Overflow tests by category")
    parser.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=[
            "all", "unit", "integration", "benchmarks", "coverage",
            "profiler", "device", "swap", "partitioner", "config", 
            "module", "hardware", "quick"
        ],
        help="Test category to run"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-x", "--exitfirst", action="store_true", help="Exit on first failure")
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["pytest"]
    if args.verbose:
        base_cmd.append("-v")
    if args.exitfirst:
        base_cmd.append("-x")
    
    # Define test categories
    categories = {
        "all": ["tests/"],
        "unit": [
            "tests/test_memory_profiler.py",
            "tests/test_device_manager.py",
            "tests/test_block_swap_manager.py",
            "tests/test_model_partitioner.py",
            "tests/test_memory_config.py",
            "tests/test_dynamic_memory_module.py"
        ],
        "integration": [
            "tests/test_integration.py",
            "tests/test_hardware_scenarios.py"
        ],
        "benchmarks": ["tests/test_performance_benchmarks.py", "--benchmark-only"],
        "coverage": ["tests/", "--cov=overflow", "--cov-report=html", "--cov-report=term"],
        "profiler": ["tests/test_memory_profiler.py"],
        "device": ["tests/test_device_manager.py"],
        "swap": ["tests/test_block_swap_manager.py"],
        "partitioner": ["tests/test_model_partitioner.py"],
        "config": ["tests/test_memory_config.py"],
        "module": ["tests/test_dynamic_memory_module.py"],
        "hardware": ["tests/test_hardware_scenarios.py"],
        "quick": [  # Quick smoke tests
            "tests/test_memory_config.py",
            "tests/test_device_manager.py",
            "-k", "test_initialization or test_default"
        ]
    }
    
    # Get the test files/options for the category
    test_targets = categories.get(args.category, [])
    
    # Build and run the command
    cmd = base_cmd + test_targets
    
    print(f"\n🧪 Overflow Test Runner")
    print(f"📁 Category: {args.category}")
    
    exit_code = run_command(cmd)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())