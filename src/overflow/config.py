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
    prefer_data_parallel: bool = False  # Force data parallel for small models on multi-GPU
    data_parallel_threshold: float = 0.5  # Model must be < this fraction of GPU memory for data parallel
    
    # CPU Offload Optimization Settings
    cpu_offload_strategy: str = "sequential"  # "sequential" or "chunked"
    cpu_offload_chunk_size: int = 0  # 0 = auto-calculate, >0 = fixed chunk size
    cpu_offload_gpu_percent: float = 0.8  # Use this % of GPU memory for chunks
    cpu_offload_overlap: bool = True  # Overlap CPU-GPU transfers when possible
    
    # Debug/Verbose mode
    verbose: bool = False  # Enable detailed logging for debugging