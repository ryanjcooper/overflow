# src/overflow/__init__.py
"""
Overflow: When your model overflows the GPU
A hardware abstraction layer for running large models on limited memory systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import psutil
import gc
from dataclasses import dataclass
from enum import Enum
import warnings
from collections import OrderedDict
import threading
import queue
import time

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


class ExecutionStrategy(Enum):
    """Execution strategies based on hardware capabilities."""
    STANDARD = "standard"
    GRADIENT_CHECKPOINT = "gradient_checkpoint"
    MODEL_PARALLEL = "model_parallel"
    CPU_OFFLOAD = "cpu_offload"
    HYBRID = "hybrid"


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


class DeviceManager:
    """Manages available compute devices."""
    
    def __init__(self):
        self.devices = self._detect_devices()
        self.primary_device = self._select_primary_device()
    
    def _detect_devices(self) -> List[DeviceInfo]:
        """Detect all available devices."""
        devices = []
        
        # Check CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory
                free_mem, _ = torch.cuda.mem_get_info(i)
                
                devices.append(DeviceInfo(
                    device_id=i,
                    device_type='cuda',
                    total_memory=total_mem,
                    available_memory=free_mem,
                    device_name=props.name
                ))
        
        # Add CPU
        vm = psutil.virtual_memory()
        devices.append(DeviceInfo(
            device_id=0,
            device_type='cpu',
            total_memory=vm.total,
            available_memory=vm.available,
            device_name='CPU'
        ))
        
        return devices
    
    def _select_primary_device(self) -> torch.device:
        """Select the primary compute device."""
        cuda_devices = [d for d in self.devices if d.device_type == 'cuda']
        if cuda_devices:
            # Select GPU with most available memory
            best_device = max(cuda_devices, key=lambda d: d.available_memory)
            return torch.device(f'cuda:{best_device.device_id}')
        return torch.device('cpu')
    
    def get_total_gpu_memory(self) -> int:
        """Get total GPU memory across all devices."""
        return sum(d.total_memory for d in self.devices if d.device_type == 'cuda')
    
    def get_available_gpu_memory(self) -> int:
        """Get available GPU memory across all devices."""
        return sum(d.available_memory for d in self.devices if d.device_type == 'cuda')


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
            if len(list(module.children())) == 0:  # Leaf module
                param_size = sum(p.numel() * p.element_size() for p in module.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())
                module_sizes[name] = param_size + buffer_size
        
        # Simple partitioning: distribute modules evenly
        sorted_modules = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)
        for i, (name, size) in enumerate(sorted_modules):
            device_idx = i % len(self.devices)
            self.partition_map[name] = self.devices[device_idx]
    
    def get_device_for_module(self, module_name: str) -> torch.device:
        """Get the device assignment for a module."""
        return self.partition_map.get(module_name, self.devices[0])


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