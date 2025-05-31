# src/overflow/__init__.py
"""
Overflow: When your model overflows the GPU
A hardware abstraction layer for running large models on limited memory systems
"""

from .enums import ExecutionStrategy
from .config import DeviceInfo, MemoryConfig
from .profiler import MemoryProfiler
from .device_manager import DeviceManager
from .swap_manager import BlockSwapManager
from .partitioner import ModelPartitioner
from .module import DynamicMemoryModule

__version__ = "0.1.0"
__all__ = [
    "DynamicMemoryModule", 
    "MemoryConfig", 
    "ExecutionStrategy",
    "MemoryProfiler",
    "DeviceManager",
    "BlockSwapManager",
    "ModelPartitioner",
    "DeviceInfo"
]