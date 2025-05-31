# src/overflow/swap_manager.py
"""
Block swapping functionality between GPU and CPU memory.
"""

import torch
import threading
import queue
from collections import OrderedDict
from typing import List, Optional

from .config import MemoryConfig


class BlockSwapManager:
    """Manages block swapping between GPU and CPU memory."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.cpu_cache = OrderedDict()
        self.swap_queue = queue.Queue()
        self.prefetch_thread = None
        self._running = False
        self._lock = threading.Lock()
        self.swap_stats = {'swaps_in': 0, 'swaps_out': 0, 'total_bytes': 0}
    
    def start(self):
        """Start the prefetch thread."""
        if not self._running:
            self._running = True
            self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
            self.prefetch_thread.daemon = True
            self.prefetch_thread.start()
    
    def stop(self):
        """Stop the prefetch thread."""
        self._running = False
        if self.prefetch_thread:
            self.prefetch_thread.join()
    
    def _prefetch_worker(self):
        """Worker thread for asynchronous prefetching."""
        while self._running:
            try:
                # Get prefetch request (block for up to 0.1s)
                param_name = self.swap_queue.get(timeout=0.1)
                self._prefetch_from_cpu(param_name)
            except queue.Empty:
                continue
    
    def swap_to_cpu(self, name: str, tensor: torch.Tensor) -> None:
        """Swap a tensor to CPU memory."""
        with self._lock:
            # Move to CPU and pin memory for faster transfers
            cpu_tensor = tensor.cpu().pin_memory()
            self.cpu_cache[name] = cpu_tensor
            self.swap_stats['swaps_out'] += 1
            self.swap_stats['total_bytes'] += tensor.numel() * tensor.element_size()
            
            # Limit cache size
            if len(self.cpu_cache) > 100:  # Configurable
                # Remove oldest entry
                self.cpu_cache.popitem(last=False)
    
    def swap_from_cpu(self, name: str, device: torch.device) -> Optional[torch.Tensor]:
        """Swap a tensor back from CPU to GPU."""
        with self._lock:
            if name in self.cpu_cache:
                cpu_tensor = self.cpu_cache[name]
                gpu_tensor = cpu_tensor.to(device, non_blocking=True)
                self.swap_stats['swaps_in'] += 1
                return gpu_tensor
        return None
    
    def _prefetch_from_cpu(self, name: str):
        """Prefetch a tensor from CPU (called by worker thread)."""
        # This is a placeholder for more sophisticated prefetching
        pass
    
    def schedule_prefetch(self, names: List[str]):
        """Schedule tensors for prefetching."""
        for name in names[:self.config.prefetch_size]:
            self.swap_queue.put(name)