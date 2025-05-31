# src/overflow/partitioner.py
"""
Automatic model partitioning for multi-GPU execution.
"""

import torch
import torch.nn as nn
from typing import List


class ModelPartitioner:
    """Automatic model partitioning for multi-GPU execution."""
    
    def __init__(self, model: nn.Module, devices: List[torch.device]):
        self.model = model
        self.devices = devices
        self.partition_map = {}
        self._analyze_model()
    
    def _analyze_model(self):
        """Analyze model structure and memory requirements."""
        # Calculate memory per module
        module_sizes = {}
        for name, module in self.model.named_modules():
            # Skip the root module and only process leaf modules
            if name and len(list(module.children())) == 0:  # Leaf module
                param_size = sum(p.numel() * p.element_size() for p in module.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
                module_sizes[name] = param_size + buffer_size
        
        # If no modules found, return empty partition map
        if not module_sizes:
            return
        
        # Simple partitioning: distribute modules evenly
        sorted_modules = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)
        for i, (name, size) in enumerate(sorted_modules):
            device_idx = i % len(self.devices)
            self.partition_map[name] = self.devices[device_idx]
    
    def get_device_for_module(self, module_name: str) -> torch.device:
        """Get the device assignment for a module."""
        return self.partition_map.get(module_name, self.devices[0])