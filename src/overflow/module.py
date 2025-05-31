# src/overflow/module.py
"""
Main DynamicMemoryModule implementation for the Overflow framework.
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, Optional, Any

from .enums import ExecutionStrategy
from .config import MemoryConfig
from .profiler import MemoryProfiler
from .device_manager import DeviceManager
from .swap_manager import BlockSwapManager
from .partitioner import ModelPartitioner


class DynamicMemoryModule(nn.Module):
    """Enhanced nn.Module with dynamic memory management."""
    
    def __init__(self, module: nn.Module, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.wrapped_module = module
        self.config = config or MemoryConfig()
        
        # Initialize components
        self.memory_profiler = MemoryProfiler(self.config)
        self.device_manager = DeviceManager()
        self.swap_manager = BlockSwapManager(self.config)
        
        # Determine execution strategy
        self.strategy = self._determine_strategy()
        
        # Setup based on strategy
        self._setup_execution()
        
        # Register hooks if profiling enabled
        if self.config.enable_profiling:
            self._register_memory_hooks()
    
    def _determine_strategy(self) -> ExecutionStrategy:
        """Automatically determine the best execution strategy."""
        model_size = self._estimate_model_size()
        total_gpu_memory = self.device_manager.get_total_gpu_memory()
        gpu_count = len([d for d in self.device_manager.devices if d.device_type == 'cuda'])
        
        # Decision logic
        if model_size < total_gpu_memory * 0.7:  # 70% to leave room for activations
            return ExecutionStrategy.STANDARD
        elif model_size < total_gpu_memory * 0.9:
            return ExecutionStrategy.GRADIENT_CHECKPOINT
        elif gpu_count > 1 and model_size < total_gpu_memory * gpu_count * 0.8:
            return ExecutionStrategy.MODEL_PARALLEL
        else:
            return ExecutionStrategy.CPU_OFFLOAD
    
    def _estimate_model_size(self) -> int:
        """Estimate total model size in bytes."""
        total_size = 0
        for param in self.wrapped_module.parameters():
            total_size += param.numel() * param.element_size()
        for buffer in self.wrapped_module.buffers():
            total_size += buffer.numel() * buffer.element_size()
        return total_size
    
    def _setup_execution(self):
        """Setup execution based on selected strategy."""
        if self.strategy == ExecutionStrategy.GRADIENT_CHECKPOINT:
            print(f"Using gradient checkpointing strategy")
            self._setup_gradient_checkpointing()
        elif self.strategy == ExecutionStrategy.MODEL_PARALLEL:
            print(f"Using model parallel strategy across {len(self.device_manager.devices)} devices")
            self._setup_model_parallel()
        elif self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            print(f"Using CPU offload strategy")
            self.swap_manager.start()
            self._setup_cpu_offload()
        else:
            print(f"Using standard execution strategy")
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory-intensive layers."""
        # This is a simplified example - in practice, you'd identify memory-intensive layers
        def checkpoint_forward(module, input):
            return torch.utils.checkpoint.checkpoint(module._forward_impl, input, use_reentrant=False)
        
        # Apply to specific layer types known to be memory-intensive
        for module in self.wrapped_module.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                module.forward = lambda x, m=module: checkpoint_forward(m, x)
    
    def _setup_model_parallel(self):
        """Setup model parallelism across available GPUs."""
        devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
        if len(devices) > 1:
            device_list = [torch.device(f'cuda:{d.device_id}') for d in devices]
            self.partitioner = ModelPartitioner(self.wrapped_module, device_list)
    
    def _setup_cpu_offload(self):
        """Setup CPU offloading for large models."""
        # Move less frequently used parameters to CPU
        # This is a placeholder - real implementation would use access patterns
        pass
    
    def _register_memory_hooks(self):
        """Register hooks for memory profiling."""
        def create_forward_hook(name):
            def hook(module, input, output):
                if self.memory_profiler.profile_count % self.config.profile_interval == 0:
                    device = next(module.parameters()).device if len(list(module.parameters())) > 0 else self.device_manager.primary_device
                    pre_mem = self.memory_profiler.profile_memory(device)['allocated']
                    # Note: actual memory allocation happens during forward pass
                    # This is a simplified measurement
                    torch.cuda.synchronize() if device.type == 'cuda' else None
                    post_mem = self.memory_profiler.profile_memory(device)['allocated']
                    self.memory_profiler.update_stats(name, post_mem - pre_mem)
                
                self.memory_profiler.profile_count += 1
            return hook
        
        # Register hooks for all submodules
        for name, module in self.wrapped_module.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module.register_forward_hook(create_forward_hook(name))
    
    def forward(self, *args, **kwargs):
        """Forward pass with memory management."""
        # Check memory pressure and adapt strategy if needed
        if self.config.enable_profiling:
            device = self.device_manager.primary_device
            pressure = self.memory_profiler.get_memory_pressure(device)
            
            if pressure > self.config.offload_threshold and self.strategy != ExecutionStrategy.CPU_OFFLOAD:
                warnings.warn(f"High memory pressure detected ({pressure:.1%}). Consider enabling CPU offload.")
        
        # Execute forward pass based on strategy
        if self.strategy == ExecutionStrategy.MODEL_PARALLEL and hasattr(self, 'partitioner'):
            return self._forward_model_parallel(*args, **kwargs)
        else:
            return self.wrapped_module(*args, **kwargs)
    
    def _forward_model_parallel(self, *args, **kwargs):
        """Forward pass with model parallelism."""
        # Simplified model parallel forward - real implementation would be more sophisticated
        # This is a placeholder showing the concept
        return self.wrapped_module(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_module, name)
    
    def __setattr__(self, name, value):
        """Delegate attribute setting appropriately."""
        if name in ['wrapped_module', 'config', 'memory_profiler', 'device_manager', 
                    'swap_manager', 'strategy', 'partitioner']:
            super().__setattr__(name, value)
        else:
            setattr(self.wrapped_module, name, value)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        return {
            'strategy': self.strategy.value,
            'peak_memory_mb': self.memory_profiler.peak_memory,
            'module_stats': self.memory_profiler.memory_stats,
            'swap_stats': self.swap_manager.swap_stats if hasattr(self.swap_manager, 'swap_stats') else {},
            'devices': [
                {
                    'type': d.device_type,
                    'id': d.device_id,
                    'name': d.device_name,
                    'total_memory_mb': d.total_memory / 1024**2,
                    'available_memory_mb': d.available_memory / 1024**2
                }
                for d in self.device_manager.devices
            ]
        }