# src/overflow/profiler.py
"""
Memory profiling functionality for the Overflow framework.
"""

import torch
import psutil
import threading
from typing import Dict

from .config import MemoryConfig


class MemoryProfiler:
    """Memory profiling and statistics collection."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_stats = {}
        self.peak_memory = 0
        self.profile_count = 0
        self._lock = threading.Lock()
    
    def profile_memory(self, device: torch.device) -> Dict[str, float]:
        """Profile current memory usage."""
        if device.type == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated(device) / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved(device) / 1024**2,
                'free': (torch.cuda.mem_get_info(device)[0]) / 1024**2,
                'total': (torch.cuda.mem_get_info(device)[1]) / 1024**2
            }
        else:
            # CPU memory profiling
            vm = psutil.virtual_memory()
            return {
                'allocated': (vm.total - vm.available) / 1024**2,
                'reserved': vm.total / 1024**2,
                'free': vm.available / 1024**2,
                'total': vm.total / 1024**2
            }
    
    def update_stats(self, module_name: str, memory_delta: float):
        """Update memory statistics for a module."""
        with self._lock:
            if module_name not in self.memory_stats:
                self.memory_stats[module_name] = {
                    'count': 0,
                    'total_memory': 0,
                    'peak_memory': 0
                }
            
            stats = self.memory_stats[module_name]
            stats['count'] += 1
            stats['total_memory'] += memory_delta
            stats['peak_memory'] = max(stats['peak_memory'], memory_delta)
            
            self.peak_memory = max(self.peak_memory, memory_delta)
    
    def get_memory_pressure(self, device: torch.device) -> float:
        """Get current memory pressure (0.0 to 1.0)."""
        stats = self.profile_memory(device)
        return 1.0 - (stats['free'] / stats['total'])