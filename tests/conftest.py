# tests/conftest.py
"""
Pytest configuration and fixtures for Overflow tests.
"""

import pytest
import torch
import gc


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Automatically cleanup GPU memory before and after each test."""
    # Cleanup before test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    yield
    
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that take significant time"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "multi_gpu: Tests that require multiple GPUs"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )