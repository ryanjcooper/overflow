# src/overflow/config.py
"""
Configuration and data structures for the Overflow framework.
"""

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    """Hardware device information."""
    device_id: int
    device_type: str  # 'cuda' or 'cpu'
    total_memory: int
    available_memory: int
    device_name: str = ""


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    enable_profiling: bool = True
    checkpoint_threshold: float = 0.8  # Enable checkpointing at 80% memory usage
    offload_threshold: float = 0.9   # Start offloading at 90% memory usage
    prefetch_size: int = 2  # Number of blocks to prefetch
    min_gpu_memory_mb: int = 1024  # Minimum GPU memory to keep free
    profile_interval: int = 10  # Profile every N forward passes