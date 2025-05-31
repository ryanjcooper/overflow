# src/overflow/module.py
"""
Main DynamicMemoryModule implementation for the Overflow framework.
"""

import torch
import torch.nn as nn
import warnings
from typing import Dict, Optional, Any, Tuple

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
        
        # Move model to appropriate device
        self._move_to_device()
        
        # Setup based on strategy
        self._setup_execution()
        
        # Memory tracking
        self._pre_forward_memory = {}
        
        # Register hooks if profiling enabled
        if self.config.enable_profiling:
            self._register_memory_hooks()
    
    def _move_to_device(self):
        """Move model to appropriate device based on strategy."""
        if self.strategy in [ExecutionStrategy.STANDARD, ExecutionStrategy.GRADIENT_CHECKPOINT]:
            if self.device_manager.primary_device.type == 'cuda':
                self.wrapped_module = self.wrapped_module.to(self.device_manager.primary_device)
        elif self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            # Keep model on CPU for offloading strategy
            self.wrapped_module = self.wrapped_module.to('cpu')
    
    def _determine_strategy(self) -> ExecutionStrategy:
        """Automatically determine the best execution strategy."""
        model_size = self._estimate_model_size()
        total_gpu_memory = self.device_manager.get_total_gpu_memory()
        gpu_count = len([d for d in self.device_manager.devices if d.device_type == 'cuda'])
        
        # Print diagnostic info
        print(f"Model size estimate: {model_size / 1024**3:.2f} GB")
        print(f"Total GPU memory: {total_gpu_memory / 1024**3:.2f} GB")
        
        # Decision logic - adjusted for better thresholds
        if total_gpu_memory == 0:
            # No GPU available
            return ExecutionStrategy.CPU_OFFLOAD
        
        # Account for activation memory (roughly 2-3x model size during training)
        activation_multiplier = 3.0
        total_memory_needed = model_size * activation_multiplier
        
        if total_memory_needed < total_gpu_memory * 0.8:  # 80% to leave room
            return ExecutionStrategy.STANDARD
        elif model_size < total_gpu_memory * 0.8:  # Model fits but activations might not
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
        import functools
        
        # Apply checkpointing to transformer layers
        for module in self.wrapped_module.modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Store the original forward method
                module._forward = module.forward
                
                # Create a wrapper that properly handles checkpointing
                def make_checkpoint_forward(m):
                    @functools.wraps(m._forward)
                    def checkpointed_forward(*args, **kwargs):
                        # TransformerEncoderLayer expects (src, src_mask, src_key_padding_mask)
                        # We need to handle both positional and keyword arguments
                        def run_function(*args, **kwargs):
                            return m._forward(*args, **kwargs)
                        
                        # Use checkpoint only during training
                        if m.training:
                            return torch.utils.checkpoint.checkpoint(
                                run_function,
                                *args,
                                use_reentrant=False,
                                **kwargs
                            )
                        else:
                            return run_function(*args, **kwargs)
                    
                    return checkpointed_forward
                
                # Replace the forward method
                module.forward = make_checkpoint_forward(module)
    
    def _setup_model_parallel(self):
        """Setup model parallelism across available GPUs."""
        devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
        if len(devices) > 1:
            device_list = [torch.device(f'cuda:{d.device_id}') for d in devices]
            self.partitioner = ModelPartitioner(self.wrapped_module, device_list)
    
    def _setup_cpu_offload(self):
        """Setup CPU offloading for large models."""
        # For CPU offloading, we'll implement a simple layer-by-layer execution
        # This is a basic implementation - a production version would be more sophisticated
        
        # Store original forward method
        self._original_forward = self.wrapped_module.forward
        
        # Replace with offloading forward
        def offload_forward(*args, **kwargs):
            # Special handling for TransformerEncoder
            if isinstance(self.wrapped_module, nn.TransformerEncoder):
                src = args[0]
                mask = kwargs.get('mask', None)
                src_key_padding_mask = kwargs.get('src_key_padding_mask', None)
                
                # Move input to GPU
                device = self.device_manager.primary_device
                src = src.to(device)
                if mask is not None:
                    mask = mask.to(device)
                if src_key_padding_mask is not None:
                    src_key_padding_mask = src_key_padding_mask.to(device)
                
                output = src
                
                # Process each transformer layer
                for i, mod in enumerate(self.wrapped_module.layers):
                    # Move layer to GPU
                    mod = mod.to(device)
                    self.swap_manager.swap_stats['swaps_in'] += 1
                    
                    # Forward pass through this layer
                    output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    
                    # Move layer back to CPU
                    mod = mod.to('cpu')
                    self.swap_manager.swap_stats['swaps_out'] += 1
                    
                    # Clear cache periodically
                    if i % 4 == 0:
                        torch.cuda.empty_cache()
                
                # Apply final layer norm if it exists
                if self.wrapped_module.norm is not None:
                    self.wrapped_module.norm = self.wrapped_module.norm.to(device)
                    self.swap_manager.swap_stats['swaps_in'] += 1
                    output = self.wrapped_module.norm(output)
                    self.wrapped_module.norm = self.wrapped_module.norm.to('cpu')
                    self.swap_manager.swap_stats['swaps_out'] += 1
                
                return output
            else:
                # General fallback for other model types
                device = self.device_manager.primary_device
                
                # Move inputs to GPU
                gpu_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        gpu_args.append(arg.to(device))
                    else:
                        gpu_args.append(arg)
                
                gpu_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        gpu_kwargs[k] = v.to(device)
                    else:
                        gpu_kwargs[k] = v
                
                # Temporarily move model to GPU
                self.wrapped_module = self.wrapped_module.to(device)
                output = self._original_forward(*gpu_args, **gpu_kwargs)
                
                # Move model back to CPU
                self.wrapped_module = self.wrapped_module.to('cpu')
                torch.cuda.empty_cache()
                
                return output
        
        self.wrapped_module.forward = offload_forward
    
    def _register_memory_hooks(self):
        """Register hooks for memory profiling."""
        # Register pre-forward hooks to capture memory before computation
        def create_pre_forward_hook(name):
            def pre_hook(module, input):
                if self.memory_profiler.profile_count % self.config.profile_interval == 0:
                    device = self._get_module_device(module)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    mem_stats = self.memory_profiler.profile_memory(device)
                    self._pre_forward_memory[name] = mem_stats['allocated']
            return pre_hook
        
        # Register post-forward hooks to capture memory after computation
        def create_post_forward_hook(name):
            def post_hook(module, input, output):
                if self.memory_profiler.profile_count % self.config.profile_interval == 0:
                    device = self._get_module_device(module)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    mem_stats = self.memory_profiler.profile_memory(device)
                    post_mem = mem_stats['allocated']
                    
                    # Calculate memory used by this module
                    pre_mem = self._pre_forward_memory.get(name, 0)
                    memory_used = max(0, post_mem - pre_mem)
                    
                    # Update stats
                    self.memory_profiler.update_stats(name, memory_used)
                    
                    # Update peak memory
                    self.memory_profiler.peak_memory = max(
                        self.memory_profiler.peak_memory, 
                        mem_stats['allocated']
                    )
                
                self.memory_profiler.profile_count += 1
            return post_hook
        
        # Register hooks for all submodules
        for name, module in self.wrapped_module.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module.register_forward_pre_hook(create_pre_forward_hook(name))
                module.register_forward_hook(create_post_forward_hook(name))
    
    def _get_module_device(self, module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            # Try to get device from parameters
            return next(module.parameters()).device
        except StopIteration:
            # No parameters, try buffers
            try:
                return next(module.buffers()).device
            except StopIteration:
                # No parameters or buffers, use primary device
                return self.device_manager.primary_device
    
    def forward(self, *args, **kwargs):
        """Forward pass with memory management."""
        # For CPU offloading, don't move inputs here - let the offload forward handle it
        if self.strategy != ExecutionStrategy.CPU_OFFLOAD:
            # Move inputs to appropriate device if needed
            device = self._get_module_device(self.wrapped_module)
            args = self._move_tensors_to_device(args, device)
            kwargs = self._move_tensors_to_device(kwargs, device)
        
        # Get device for profiling
        device = self.device_manager.primary_device if self.strategy == ExecutionStrategy.CPU_OFFLOAD else self._get_module_device(self.wrapped_module)
        
        # Profile memory before forward pass
        if self.config.enable_profiling:
            pre_forward_stats = self.memory_profiler.profile_memory(device)
            pre_forward_memory = pre_forward_stats['allocated']
        
        # Check memory pressure and adapt strategy if needed
        if self.config.enable_profiling and device.type == 'cuda':
            pressure = self.memory_profiler.get_memory_pressure(device)
            
            if pressure > self.config.offload_threshold and self.strategy != ExecutionStrategy.CPU_OFFLOAD:
                warnings.warn(f"High memory pressure detected ({pressure:.1%}). Consider enabling CPU offload.")
        
        # Execute forward pass based on strategy
        if self.strategy == ExecutionStrategy.MODEL_PARALLEL and hasattr(self, 'partitioner'):
            output = self._forward_model_parallel(*args, **kwargs)
        else:
            output = self.wrapped_module(*args, **kwargs)
        
        # Profile memory after forward pass
        if self.config.enable_profiling:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            post_forward_stats = self.memory_profiler.profile_memory(device)
            post_forward_memory = post_forward_stats['allocated']
            
            # Update overall peak memory
            self.memory_profiler.peak_memory = max(
                self.memory_profiler.peak_memory,
                post_forward_memory
            )
            
            # If no module hooks captured memory, at least capture overall forward pass
            if not self.memory_profiler.memory_stats:
                memory_used = max(0, post_forward_memory - pre_forward_memory)
                self.memory_profiler.update_stats("forward_pass", memory_used)
        
        return output
    
    def _move_tensors_to_device(self, obj: Any, device: torch.device) -> Any:
        """Recursively move tensors to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device) if obj.device != device else obj
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tensors_to_device(item, device) for item in obj)
        elif isinstance(obj, dict):
            return {k: self._move_tensors_to_device(v, device) for k, v in obj.items()}
        else:
            return obj
    
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
                    'swap_manager', 'strategy', 'partitioner', '_pre_forward_memory', '_original_forward']:
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