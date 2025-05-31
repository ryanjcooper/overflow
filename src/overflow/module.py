# src/overflow/module.py
"""
Main DynamicMemoryModule implementation for the Overflow framework.
"""

import torch
import torch.nn as nn
import warnings
import logging
from typing import Dict, Optional, Any, Tuple

from .enums import ExecutionStrategy
import math
from .config import MemoryConfig
from .profiler import MemoryProfiler
from .device_manager import DeviceManager
from .swap_manager import BlockSwapManager
from .partitioner import ModelPartitioner


# Set up logging
logger = logging.getLogger(__name__)


class DynamicMemoryModule(nn.Module):
    """Enhanced nn.Module with dynamic memory management."""
    
    def __init__(self, module: nn.Module, config: Optional[MemoryConfig] = None):
        super().__init__()
        
        # Initialize tracking attributes FIRST before anything else
        self._pre_forward_memory = {}
        self._last_chunk_calculation = None
        
        self.wrapped_module = module
        self.config = config or MemoryConfig()
        
        # Set up logging based on config
        if self.config.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
        
        # Initialize components
        self.memory_profiler = MemoryProfiler(self.config)
        self.device_manager = DeviceManager()
        self.swap_manager = BlockSwapManager(self.config)
        
        # Preserve training mode from wrapped module
        self.training = module.training
        
        # Determine execution strategy
        self.strategy = self._determine_strategy()
        
        # Move model to appropriate device
        self._move_to_device()
        
        # Setup based on strategy
        self._setup_execution()
        
        # Register hooks if profiling enabled
        
        # Register hooks if profiling enabled
        if self.config.enable_profiling:
            self._register_memory_hooks()
        
        # Clear GPU cache after setup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Ensure we maintain the same training mode as the wrapped module
        if module.training:
            self.train()
        else:
            self.eval()
    
    def _move_to_device(self):
        """Move model to appropriate device based on strategy."""
        if self.strategy in [ExecutionStrategy.STANDARD, ExecutionStrategy.GRADIENT_CHECKPOINT]:
            if self.device_manager.primary_device.type == 'cuda':
                self.wrapped_module = self.wrapped_module.to(self.device_manager.primary_device)
        elif self.strategy == ExecutionStrategy.DATA_PARALLEL:
            # For data parallel, move to primary GPU (DataParallel will handle the rest)
            if self.device_manager.primary_device.type == 'cuda':
                self.wrapped_module = self.wrapped_module.to(self.device_manager.primary_device)
        elif self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            # Keep model on CPU for offloading strategy
            self.wrapped_module = self.wrapped_module.to('cpu')
        elif self.strategy == ExecutionStrategy.MODEL_PARALLEL:
            # For model parallel, we'll handle device placement in setup
            pass
    
    def _determine_strategy(self) -> ExecutionStrategy:
        """Automatically determine the best execution strategy."""
        model_size = self._estimate_model_size()
        total_gpu_memory = self.device_manager.get_total_gpu_memory()
        gpu_count = len([d for d in self.device_manager.devices if d.device_type == 'cuda'])
        
        # Get single GPU memory (memory of the primary device)
        single_gpu_memory = 0
        if gpu_count > 0:
            primary_device_id = int(self.device_manager.primary_device.index) if self.device_manager.primary_device.type == 'cuda' else 0
            for d in self.device_manager.devices:
                if d.device_type == 'cuda' and d.device_id == primary_device_id:
                    single_gpu_memory = d.total_memory
                    break
        
        # Log diagnostic info if verbose
        if self.config.verbose:
            model_size_gb = model_size / 1024**3
            logger.info(f"Model size estimate: {model_size_gb:.2f} GB")
            logger.info(f"Total GPU memory: {total_gpu_memory / 1024**3:.2f} GB")
            if gpu_count > 1:
                logger.info(f"Single GPU memory: {single_gpu_memory / 1024**3:.2f} GB")
                logger.info(f"Number of GPUs: {gpu_count}")
        
        # Decision logic
        if total_gpu_memory == 0:
            # No GPU available
            if self.config.verbose:
                logger.info("→ No GPU detected, using CPU execution")
            return ExecutionStrategy.CPU_OFFLOAD
        
        # Account for activation memory (roughly 2-3x model size during training)
        # For inference, we need less memory
        activation_multiplier = 1.5 if not self.training else 3.0
        total_memory_needed = model_size * activation_multiplier
        
        # Single-GPU scenarios
        if gpu_count == 1:
            if total_memory_needed < single_gpu_memory * 0.8:
                if self.config.verbose:
                    logger.info("→ Model fits comfortably on single GPU")
                return ExecutionStrategy.STANDARD
            elif model_size < single_gpu_memory * 0.8:
                if self.config.verbose:
                    logger.info("→ Model fits but activations might not, using gradient checkpointing")
                return ExecutionStrategy.GRADIENT_CHECKPOINT
            else:
                if self.config.verbose:
                    logger.info("→ Model too large for GPU, using CPU offloading")
                return ExecutionStrategy.CPU_OFFLOAD
        
        # Multi-GPU scenarios
        else:
            # For data parallel to be beneficial, we need:
            # 1. Model to fit very comfortably on single GPU (< 50% for room)
            # 2. Reasonable batch size (will be split across GPUs)
            # 3. Not training (training typically benefits more from larger batch on single GPU)
            
            # Case 1: Very small model - consider data parallel only if not training
            data_parallel_threshold = self.config.data_parallel_threshold if hasattr(self.config, 'data_parallel_threshold') else 0.5
            prefer_data_parallel = self.config.prefer_data_parallel if hasattr(self.config, 'prefer_data_parallel') else False
            
            if model_size < single_gpu_memory * data_parallel_threshold and (not self.training or prefer_data_parallel):
                if self.config.verbose:
                    logger.info("→ Small model on multi-GPU, using data parallelism for potential speedup")
                return ExecutionStrategy.DATA_PARALLEL
            
            # Case 2: Model fits comfortably on single GPU
            elif total_memory_needed < single_gpu_memory * 0.8:
                if self.config.verbose:
                    logger.info("→ Model fits comfortably on single GPU with standard execution")
                return ExecutionStrategy.STANDARD
            
            # Case 3: Model fits on single GPU with checkpointing
            elif model_size < single_gpu_memory * 0.8:
                if self.config.verbose:
                    logger.info("→ Model fits on single GPU with gradient checkpointing")
                return ExecutionStrategy.GRADIENT_CHECKPOINT
            
            # Case 4: Model doesn't fit on single GPU but fits across all GPUs
            elif model_size < total_gpu_memory * 0.8:
                if self.config.verbose:
                    logger.info("→ Model too large for single GPU, using model parallelism")
                return ExecutionStrategy.MODEL_PARALLEL
            
            # Case 5: Model too large even for all GPUs - CPU offload with multi-GPU support
            else:
                if self.config.verbose:
                    logger.info("→ Model too large for all GPUs, using CPU offloading with multi-GPU acceleration")
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
            self._setup_gradient_checkpointing()
        elif self.strategy == ExecutionStrategy.DATA_PARALLEL:
            self._setup_data_parallel()
        elif self.strategy == ExecutionStrategy.MODEL_PARALLEL:
            self._setup_model_parallel()
        elif self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            self.swap_manager.start()
            self._setup_cpu_offload()
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for memory-intensive layers."""
        import functools
        # Fix: correct import for checkpoint
        from torch.utils.checkpoint import checkpoint
        
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
                            return checkpoint(
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
    
    def _setup_data_parallel(self):
        """Setup data parallelism across multiple GPUs."""
        if torch.cuda.device_count() > 1:
            # Get list of GPU IDs
            device_ids = list(range(torch.cuda.device_count()))
            
            # IMPORTANT: Ensure model is on the primary device (cuda:0) before DataParallel
            # DataParallel requires all parameters to be on device_ids[0]
            primary_device = torch.device(f'cuda:{device_ids[0]}')
            self.wrapped_module = self.wrapped_module.to(primary_device)
            
            # Now wrap with DataParallel
            self.wrapped_module = nn.DataParallel(
                self.wrapped_module,
                device_ids=device_ids,
                output_device=device_ids[0]
            )
    
    def _setup_model_parallel(self):
        """Setup model parallelism across available GPUs."""
        devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
        if len(devices) > 1:
            device_list = [torch.device(f'cuda:{d.device_id}') for d in devices]
            
            # For TransformerEncoder, distribute layers across GPUs
            if isinstance(self.wrapped_module, nn.TransformerEncoder):
                num_layers = len(self.wrapped_module.layers)
                layers_per_device = num_layers // len(device_list)
                extra_layers = num_layers % len(device_list)
                
                # Store device assignments
                self.layer_devices = []
                layer_idx = 0
                
                for i, device in enumerate(device_list):
                    # Distribute layers evenly, with extra layers going to first devices
                    layers_on_this_device = layers_per_device + (1 if i < extra_layers else 0)
                    
                    for j in range(layers_on_this_device):
                        if layer_idx < num_layers:
                            self.wrapped_module.layers[layer_idx] = self.wrapped_module.layers[layer_idx].to(device)
                            self.layer_devices.append(device)
                            layer_idx += 1
                
                # Move norm layer to last device if it exists
                if hasattr(self.wrapped_module, 'norm') and self.wrapped_module.norm is not None:
                    self.wrapped_module.norm = self.wrapped_module.norm.to(device_list[-1])
                
                # Store original forward for model parallel execution
                self._original_forward = self.wrapped_module.forward
                
                # Replace forward with model parallel version
                def model_parallel_forward(src, mask=None, src_key_padding_mask=None):
                    output = src
                    
                    # Process each layer on its assigned device
                    for i, (layer, device) in enumerate(zip(self.wrapped_module.layers, self.layer_devices)):
                        # Move data to the layer's device
                        output = output.to(device)
                        if mask is not None and mask.device != device:
                            mask = mask.to(device)
                        if src_key_padding_mask is not None and src_key_padding_mask.device != device:
                            src_key_padding_mask = src_key_padding_mask.to(device)
                        
                        # Forward through layer
                        output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
                    
                    # Apply final norm if exists
                    if hasattr(self.wrapped_module, 'norm') and self.wrapped_module.norm is not None:
                        output = self.wrapped_module.norm(output)
                    
                    return output
                
                self.wrapped_module.forward = model_parallel_forward
            else:
                # For other model types, use the partitioner
                self.partitioner = ModelPartitioner(self.wrapped_module, device_list)
    
    def _get_available_gpu_memory(self, device_id: int = 0) -> int:
        """Get available GPU memory in bytes."""
        try:
            free_memory, _ = torch.cuda.mem_get_info(device_id)
            return free_memory
        except:
            return 0
    
    def _estimate_layer_memory(self, layer: nn.Module, batch_size: int = 1, seq_length: int = 128) -> int:
        """Estimate memory required for a layer in bytes."""
        # Count parameters and buffers
        param_size = 0
        for param in layer.parameters():
            param_size += param.numel() * param.element_size()
        for buffer in layer.buffers():
            param_size += buffer.numel() * buffer.element_size()
        
        # Estimate activation memory based on layer type
        if isinstance(layer, nn.TransformerEncoderLayer):
            # Get layer dimensions
            d_model = layer.self_attn.embed_dim
            nhead = layer.self_attn.num_heads
            ff_dim = layer.linear1.out_features
            
            # Calculate activation memory for batch_size and seq_length
            # Input/output: batch_size * seq_length * d_model
            io_memory = batch_size * seq_length * d_model * 4  # float32
            
            # Self-attention intermediates (Q, K, V): 3 * batch_size * seq_length * d_model
            attention_memory = 3 * batch_size * seq_length * d_model * 4
            
            # Attention scores: batch_size * nhead * seq_length * seq_length
            scores_memory = batch_size * nhead * seq_length * seq_length * 4
            
            # FFN intermediate: batch_size * seq_length * ff_dim
            ffn_memory = batch_size * seq_length * ff_dim * 4
            
            # Total activation memory
            activation_memory = io_memory + attention_memory + scores_memory + ffn_memory
            
            # For inference, we need parameters + activations
            total_memory = param_size + activation_memory
            
            # Add 20% overhead for temporary buffers
            return int(total_memory * 1.2)
        else:
            # For other layers, use parameter size + reasonable activation estimate
            # Assume activations are proportional to output size
            return int(param_size * 1.5)
    
    def _setup_cpu_offload(self):
        """Setup CPU offloading for large models."""
        # Store original forward method
        self._original_forward = self.wrapped_module.forward
        
        # Get available GPUs for round-robin offloading
        cuda_devices = [torch.device(f'cuda:{d.device_id}') for d in self.device_manager.devices if d.device_type == 'cuda']
        num_gpus = len(cuda_devices)
        
        # Check if we should use chunked offloading
        use_chunked = self._should_use_chunked_offloading()
        
        if use_chunked:
            if self.config.verbose:
                logger.info("  → Using optimized chunked CPU offloading for better performance")
                logger.info("  → Chunk size will be optimized based on input dimensions")
            self._setup_chunked_cpu_offload()
        else:
            if self.config.verbose:
                logger.info("  → Using sequential CPU offloading")
            self._setup_sequential_cpu_offload()
    
    def _should_use_chunked_offloading(self) -> bool:
        """Determine if chunked offloading would be beneficial."""
        # Check if model has layers (transformer-style)
        if not hasattr(self.wrapped_module, 'layers'):
            return False
        
        # Get available GPU memory
        total_gpu_memory = self.device_manager.get_available_gpu_memory()
        
        # Estimate memory per layer
        if len(self.wrapped_module.layers) > 0:
            sample_layer = self.wrapped_module.layers[0]
            # Use conservative estimates for deciding if chunking is beneficial
            layer_memory = self._estimate_layer_memory(sample_layer, batch_size=1, seq_length=128)
            
            # Check if we can fit multiple layers
            layers_that_fit = int(total_gpu_memory * 0.8 / layer_memory)
            
            # Use chunked if we can fit at least 2 layers
            return layers_that_fit >= 2
        
        return False
    
    def _calculate_optimal_chunk_size(self, batch_size: int = 1, seq_length: int = 128) -> int:
        """Calculate optimal chunk size for CPU offloading."""
        # Get available GPU memory for ALL GPUs
        cuda_devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
        num_gpus = len(cuda_devices)
        
        if num_gpus == 0:
            return 1
        
        # For multi-GPU, we can be more aggressive since we can use both simultaneously
        total_available_memory = 0
        min_available_memory = float('inf')
        for device in cuda_devices:
            # Get fresh available memory for each GPU
            free_memory = self._get_available_gpu_memory(device.device_id)
            total_available_memory += free_memory
            min_available_memory = min(min_available_memory, free_memory)
        
        # Use 95% of the minimum available GPU memory (very aggressive)
        # We use the minimum to ensure we don't OOM on any GPU
        usable_memory_per_gpu = min_available_memory * 0.95
        
        # Estimate memory per layer more accurately with actual batch/seq dimensions
        if hasattr(self.wrapped_module, 'layers') and len(self.wrapped_module.layers) > 0:
            sample_layer = self.wrapped_module.layers[0]
            layer_memory = self._estimate_layer_memory(sample_layer, batch_size, seq_length)
            
            # Calculate chunk size based on available memory
            chunk_size = int(usable_memory_per_gpu / layer_memory)
            
            # For multi-GPU, we want chunks that allow efficient parallel processing
            if num_gpus > 1:
                total_layers = len(self.wrapped_module.layers)
                # Try to make the number of chunks divisible by num_gpus for even distribution
                ideal_num_chunks = max(1, math.ceil(total_layers / chunk_size))
                if ideal_num_chunks % num_gpus != 0:
                    # Adjust to nearest multiple of num_gpus
                    ideal_num_chunks = math.ceil(ideal_num_chunks / num_gpus) * num_gpus
                    chunk_size = math.ceil(total_layers / ideal_num_chunks)
            
            # Ensure chunk size is valid
            chunk_size = max(1, min(chunk_size, len(self.wrapped_module.layers)))
            
            # Log detailed information if verbose
            if self.config.verbose:
                # Calculate parameter size for logging
                param_size = 0
                for param in sample_layer.parameters():
                    param_size += param.numel() * param.element_size()
                for buffer in sample_layer.buffers():
                    param_size += buffer.numel() * buffer.element_size()
                
                logger.info("  → Memory calculation details:")
                logger.info(f"    - Available GPU memory: {total_available_memory / 1024**3:.1f} GB total ({num_gpus} GPUs)")
                logger.info(f"    - Min available per GPU: {min_available_memory / 1024**3:.1f} GB")
                logger.info(f"    - Usable memory per GPU: {usable_memory_per_gpu / 1024**3:.1f} GB (95% of available)")
                logger.info(f"    - Layer parameter size: {param_size / 1024**2:.1f} MB")
                logger.info(f"    - Layer total memory (batch={batch_size}, seq={seq_length}): {layer_memory / 1024**2:.1f} MB")
                logger.info(f"  → Optimal chunk size: {chunk_size} layers")
                logger.info(f"  → Will process model in {math.ceil(len(self.wrapped_module.layers) / chunk_size)} chunks")
                logger.info(f"  → Expected GPU usage: ~{(chunk_size * layer_memory) / 1024**3:.1f} GB per GPU")
            
            # Store memory info for later use if needed
            self._last_chunk_calculation = {
                'total_available_memory': total_available_memory,
                'min_available_memory': min_available_memory,
                'usable_memory_per_gpu': usable_memory_per_gpu,
                'layer_memory': layer_memory,
                'chunk_size': chunk_size,
                'num_chunks': math.ceil(len(self.wrapped_module.layers) / chunk_size)
            }
            
            return chunk_size
        
        return 1
    
    def _setup_chunked_cpu_offload(self):
        """Setup CPU offloading with chunked layer transfers."""
        cuda_devices = [torch.device(f'cuda:{d.device_id}') for d in self.device_manager.devices if d.device_type == 'cuda']
        num_gpus = len(cuda_devices)
        
        def chunked_offload_forward(*args, **kwargs):
            if isinstance(self.wrapped_module, nn.TransformerEncoder):
                src = args[0]
                mask = kwargs.get('mask', None)
                src_key_padding_mask = kwargs.get('src_key_padding_mask', None)
                
                # Get actual batch size and sequence length from input
                batch_size = src.size(0)
                seq_length = src.size(1)
                
                # Calculate optimal chunk size based on actual input dimensions
                chunk_size = self._calculate_optimal_chunk_size(batch_size, seq_length)
                
                num_layers = len(self.wrapped_module.layers)
                num_chunks = math.ceil(num_layers / chunk_size)
                
                # For multi-GPU, we can process multiple chunks simultaneously
                if num_gpus > 1 and batch_size == 1:  # Single batch, can't split batch
                    if self.config.verbose:
                        logger.info(f"\n🔄 Processing {num_layers} layers in {num_chunks} chunks using {num_gpus} GPUs...")
                    
                    output = src
                    for chunk_base_idx in range(0, num_chunks, num_gpus):
                        # Process up to num_gpus chunks in parallel
                        parallel_chunks = []
                        
                        for gpu_offset in range(num_gpus):
                            chunk_idx = chunk_base_idx + gpu_offset
                            if chunk_idx >= num_chunks:
                                break
                            
                            start_idx = chunk_idx * chunk_size
                            end_idx = min(start_idx + chunk_size, num_layers)
                            device = cuda_devices[gpu_offset]
                            
                            parallel_chunks.append({
                                'start': start_idx,
                                'end': end_idx,
                                'device': device,
                                'chunk_idx': chunk_idx
                            })
                        
                        # Move layers to GPUs in parallel
                        for chunk_info in parallel_chunks:
                            if self.config.verbose:
                                logger.info(f"  Loading chunk {chunk_info['chunk_idx'] + 1}/{num_chunks} on GPU {chunk_info['device'].index}")
                            for i in range(chunk_info['start'], chunk_info['end']):
                                self.wrapped_module.layers[i] = self.wrapped_module.layers[i].to(chunk_info['device'])
                                self.swap_manager.swap_stats['swaps_in'] += 1
                        
                        # Process each chunk sequentially (can't parallelize single batch)
                        for chunk_info in parallel_chunks:
                            device = chunk_info['device']
                            
                            # Move data to device
                            output = output.to(device)
                            if mask is not None:
                                mask = mask.to(device)
                            if src_key_padding_mask is not None:
                                src_key_padding_mask = src_key_padding_mask.to(device)
                            
                            # Process layers in this chunk
                            for i in range(chunk_info['start'], chunk_info['end']):
                                output = self.wrapped_module.layers[i](
                                    output,
                                    src_mask=mask,
                                    src_key_padding_mask=src_key_padding_mask
                                )
                        
                        # Move layers back to CPU
                        for chunk_info in parallel_chunks:
                            for i in range(chunk_info['start'], chunk_info['end']):
                                self.wrapped_module.layers[i] = self.wrapped_module.layers[i].to('cpu')
                                self.swap_manager.swap_stats['swaps_out'] += 1
                        
                        # Clear cache on all GPUs
                        for device in cuda_devices:
                            with torch.cuda.device(device):
                                torch.cuda.empty_cache()
                
                else:
                    # Single GPU or batch size > 1 (can split batch)
                    output = src
                    
                    for chunk_idx in range(num_chunks):
                        start_idx = chunk_idx * chunk_size
                        end_idx = min(start_idx + chunk_size, num_layers)
                        
                        # Select GPU (round-robin for single GPU path)
                        device = cuda_devices[chunk_idx % num_gpus] if cuda_devices else torch.device('cpu')
                        
                        # Move data to device
                        output = output.to(device)
                        if mask is not None:
                            mask = mask.to(device)
                        if src_key_padding_mask is not None:
                            src_key_padding_mask = src_key_padding_mask.to(device)
                        
                        # Move chunk of layers to GPU
                        for i in range(start_idx, end_idx):
                            self.wrapped_module.layers[i] = self.wrapped_module.layers[i].to(device)
                            self.swap_manager.swap_stats['swaps_in'] += 1
                        
                        # Process all layers in chunk
                        for i in range(start_idx, end_idx):
                            output = self.wrapped_module.layers[i](
                                output, 
                                src_mask=mask, 
                                src_key_padding_mask=src_key_padding_mask
                            )
                        
                        # Move layers back to CPU
                        for i in range(start_idx, end_idx):
                            self.wrapped_module.layers[i] = self.wrapped_module.layers[i].to('cpu')
                            self.swap_manager.swap_stats['swaps_out'] += 1
                        
                        # Clear cache between chunks
                        if chunk_idx < num_chunks - 1:
                            torch.cuda.empty_cache()
                
                # Apply final norm
                if self.wrapped_module.norm is not None:
                    # Use the device where output currently is
                    device = output.device
                    self.wrapped_module.norm = self.wrapped_module.norm.to(device)
                    output = self.wrapped_module.norm(output)
                    self.wrapped_module.norm = self.wrapped_module.norm.to('cpu')
                
                return output
            else:
                # Fallback for other models
                return self._create_sequential_offload_forward(cuda_devices, num_gpus)(*args, **kwargs)
        
        self.wrapped_module.forward = chunked_offload_forward
    
    def _setup_sequential_cpu_offload(self):
        """Setup traditional sequential CPU offloading."""
        # Get available GPUs for round-robin offloading
        cuda_devices = [torch.device(f'cuda:{d.device_id}') for d in self.device_manager.devices if d.device_type == 'cuda']
        num_gpus = len(cuda_devices)
        
        # Create the forward function
        self.wrapped_module.forward = self._create_sequential_offload_forward(cuda_devices, num_gpus)
    
    def _create_sequential_offload_forward(self, cuda_devices, num_gpus):
        """Create the sequential offload forward function."""
        # Replace with offloading forward
        def offload_forward(*args, **kwargs):
            # Special handling for TransformerEncoder
            if isinstance(self.wrapped_module, nn.TransformerEncoder):
                src = args[0]
                mask = kwargs.get('mask', None)
                src_key_padding_mask = kwargs.get('src_key_padding_mask', None)
                
                # Initialize outputs list for accumulating results from different GPUs
                outputs = []
                
                # For multi-GPU: Split batch across GPUs for parallel processing
                if num_gpus > 1:
                    batch_size = src.size(0) if src.dim() > 1 else 1
                    
                    # Calculate splits
                    splits = []
                    base_size = batch_size // num_gpus
                    remainder = batch_size % num_gpus
                    
                    start = 0
                    for i in range(num_gpus):
                        size = base_size + (1 if i < remainder else 0)
                        if size > 0:
                            splits.append((start, start + size))
                            start += size
                    
                    # Process each split on a different GPU
                    for gpu_idx, (start_idx, end_idx) in enumerate(splits):
                        device = cuda_devices[gpu_idx % num_gpus]
                        
                        # Get batch slice
                        src_slice = src[start_idx:end_idx].to(device)
                        mask_slice = mask[start_idx:end_idx] if mask is not None else None
                        padding_mask_slice = src_key_padding_mask[start_idx:end_idx] if src_key_padding_mask is not None else None
                        
                        if mask_slice is not None:
                            mask_slice = mask_slice.to(device)
                        if padding_mask_slice is not None:
                            padding_mask_slice = padding_mask_slice.to(device)
                        
                        output = src_slice
                        
                        # Process each layer
                        for i, mod in enumerate(self.wrapped_module.layers):
                            # Check if we have enough memory to move this layer
                            required_memory = self._estimate_layer_memory(mod)
                            available_memory = self._get_available_gpu_memory(device.index)
                            
                            if available_memory < required_memory * 1.5:  # Safety margin
                                # Clear cache and try again
                                torch.cuda.empty_cache()
                                available_memory = self._get_available_gpu_memory(device.index)
                                
                                if available_memory < required_memory:
                                    # Skip to next GPU if available
                                    if num_gpus > 1:
                                        device = cuda_devices[(gpu_idx + 1) % num_gpus]
                                        output = output.to(device)
                                        torch.cuda.empty_cache()
                            
                            # Move layer to this GPU
                            mod = mod.to(device)
                            self.swap_manager.swap_stats['swaps_in'] += 1
                            
                            # Forward pass
                            output = mod(output, src_mask=mask_slice, src_key_padding_mask=padding_mask_slice)
                            
                            # Move layer back to CPU
                            mod = mod.to('cpu')
                            self.swap_manager.swap_stats['swaps_out'] += 1
                            
                            # Clear cache periodically
                            if i % 2 == 0:
                                torch.cuda.empty_cache()
                        
                        # Apply final layer norm if exists
                        if self.wrapped_module.norm is not None:
                            self.wrapped_module.norm = self.wrapped_module.norm.to(device)
                            output = self.wrapped_module.norm(output)
                            self.wrapped_module.norm = self.wrapped_module.norm.to('cpu')
                        
                        outputs.append(output.cpu())
                    
                    # Concatenate results
                    final_output = torch.cat(outputs, dim=0)
                    
                    # Move to first GPU if needed
                    if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].is_cuda:
                        final_output = final_output.to(args[0].device)
                    
                    return final_output
                
                else:
                    # Single GPU path
                    device = cuda_devices[0] if cuda_devices else torch.device('cpu')
                    src = src.to(device)
                    if mask is not None:
                        mask = mask.to(device)
                    if src_key_padding_mask is not None:
                        src_key_padding_mask = src_key_padding_mask.to(device)
                    
                    output = src
                    
                    # Process each transformer layer
                    for i, mod in enumerate(self.wrapped_module.layers):
                        # Check available memory before moving layer
                        required_memory = self._estimate_layer_memory(mod)
                        available_memory = self._get_available_gpu_memory(device.index if device.type == 'cuda' else 0)
                        
                        if device.type == 'cuda' and available_memory < required_memory * 1.5:
                            torch.cuda.empty_cache()
                            available_memory = self._get_available_gpu_memory(device.index)
                            
                            if available_memory < required_memory:
                                # If we can't fit even a single layer, try aggressive cleanup
                                import gc
                                gc.collect()
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                                
                                available_memory = self._get_available_gpu_memory(device.index)
                                if available_memory < required_memory:
                                    # Last resort: process on CPU if GPU is too fragmented
                                    # Process this layer on CPU
                                    output = mod(output.cpu(), src_mask=mask.cpu() if mask is not None else None, 
                                               src_key_padding_mask=src_key_padding_mask.cpu() if src_key_padding_mask is not None else None)
                                    # Move output back to GPU for next layer
                                    if device.type == 'cuda' and i < len(self.wrapped_module.layers) - 1:
                                        output = output.to(device)
                                    # Skip the GPU swap for this layer
                                    continue
                        
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
        
        return offload_forward
    
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
        # Handle DataParallel wrapped modules
        if isinstance(module, nn.DataParallel):
            return torch.device(f'cuda:{module.device_ids[0]}')
        
        try:
            # Try to get device from parameters
            return next(module.parameters()).device
        except StopIteration:
            # No parameters, try buffers
            try:
                return next(module.buffers()).device
            except StopIteration:
                # No parameters or buffers
                # For model parallel, return the first GPU device
                if self.strategy == ExecutionStrategy.MODEL_PARALLEL:
                    cuda_devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
                    if cuda_devices:
                        return torch.device(f'cuda:{cuda_devices[0].device_id}')
                # Otherwise use primary device
                return self.device_manager.primary_device
    
    def forward(self, *args, **kwargs):
        """Forward pass with memory management."""
        # For CPU offloading and model parallel, don't move inputs here - let the strategy handle it
        if self.strategy not in [ExecutionStrategy.CPU_OFFLOAD, ExecutionStrategy.MODEL_PARALLEL]:
            # Move inputs to appropriate device if needed
            device = self._get_module_device(self.wrapped_module)
            args = self._move_tensors_to_device(args, device)
            kwargs = self._move_tensors_to_device(kwargs, device)
        
        # Get device for profiling
        if self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            device = self.device_manager.primary_device
        elif self.strategy == ExecutionStrategy.MODEL_PARALLEL:
            # For model parallel, use the first GPU for profiling
            cuda_devices = [d for d in self.device_manager.devices if d.device_type == 'cuda']
            device = torch.device(f'cuda:{cuda_devices[0].device_id}') if cuda_devices else self.device_manager.primary_device
        elif self.strategy == ExecutionStrategy.DATA_PARALLEL:
            # For data parallel, use the primary device
            device = self.device_manager.primary_device
        else:
            device = self._get_module_device(self.wrapped_module)
        
        # Profile memory before forward pass
        if self.config.enable_profiling:
            pre_forward_stats = self.memory_profiler.profile_memory(device)
            pre_forward_memory = pre_forward_stats['allocated']
        
        # Check memory pressure and adapt strategy if needed
        if self.config.enable_profiling and device.type == 'cuda':
            pressure = self.memory_profiler.get_memory_pressure(device)
            
            if pressure > self.config.offload_threshold and self.strategy not in [ExecutionStrategy.CPU_OFFLOAD, ExecutionStrategy.MODEL_PARALLEL]:
                warnings.warn(f"High memory pressure detected ({pressure:.1%}). Consider enabling CPU offload or model parallelism.")
        
        # Execute forward pass based on strategy
        if self.strategy == ExecutionStrategy.MODEL_PARALLEL and hasattr(self, 'layer_devices'):
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
        # For TransformerEncoder with custom forward, it's already handled
        if isinstance(self.wrapped_module, nn.TransformerEncoder) and hasattr(self, 'layer_devices'):
            return self.wrapped_module(*args, **kwargs)
        
        # For other models using partitioner
        if hasattr(self, 'partitioner'):
            # Basic implementation - would need more sophisticated handling for complex models
            return self.wrapped_module(*args, **kwargs)
        
        # Fallback
        return self.wrapped_module(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """Set the module in training mode."""
        self.training = mode
        self.wrapped_module.train(mode)
        return self
    
    def eval(self):
        """Set the module in evaluation mode."""
        self.training = False
        self.wrapped_module.eval()
        return self
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped module."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped_module, name)
    
    def __setattr__(self, name, value):
        """Delegate attribute setting appropriately."""
        if name in ['wrapped_module', 'config', 'memory_profiler', 'device_manager', 
                    'swap_manager', 'strategy', 'partitioner', '_pre_forward_memory', '_original_forward',
                    'training', 'layer_devices', '_last_chunk_calculation']:
            super().__setattr__(name, value)
        else:
            setattr(self.wrapped_module, name, value)
    

    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy selection and configuration."""
        model_size = self._estimate_model_size()
        total_gpu_memory = self.device_manager.get_total_gpu_memory()
        gpu_count = len([d for d in self.device_manager.devices if d.device_type == 'cuda'])
        
        info = {
            'strategy': self.strategy.value,
            'model_size_gb': model_size / 1024**3,
            'total_gpu_memory_gb': total_gpu_memory / 1024**3,
            'gpu_count': gpu_count
        }
        
        if gpu_count > 0:
            primary_device_id = int(self.device_manager.primary_device.index) if self.device_manager.primary_device.type == 'cuda' else 0
            for d in self.device_manager.devices:
                if d.device_type == 'cuda' and d.device_id == primary_device_id:
                    info['single_gpu_memory_gb'] = d.total_memory / 1024**3
                    break
        
        # Add strategy-specific information
        if self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            info['uses_chunked_offloading'] = self._should_use_chunked_offloading()
        
        return info
        """Get current memory statistics."""
        stats = {
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
        
        # Include chunk calculation info if available  
        if hasattr(self, '_last_chunk_calculation') and self._last_chunk_calculation is not None:
            stats['chunk_info'] = {
                'chunk_size': self._last_chunk_calculation['chunk_size'],
                'num_chunks': self._last_chunk_calculation['num_chunks'],
                'layer_memory_mb': self._last_chunk_calculation['layer_memory'] / 1024**2,
                'expected_gpu_usage_gb': (self._last_chunk_calculation['chunk_size'] * 
                                         self._last_chunk_calculation['layer_memory']) / 1024**3
            }
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {
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
        
        # Include chunk calculation info if available  
        if hasattr(self, '_last_chunk_calculation') and self._last_chunk_calculation is not None:
            stats['chunk_info'] = {
                'chunk_size': self._last_chunk_calculation['chunk_size'],
                'num_chunks': self._last_chunk_calculation['num_chunks'],
                'layer_memory_mb': self._last_chunk_calculation['layer_memory'] / 1024**2,
                'expected_gpu_usage_gb': (self._last_chunk_calculation['chunk_size'] * 
                                         self._last_chunk_calculation['layer_memory']) / 1024**3
            }
        
        return stats
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy selection and configuration."""
        model_size = self._estimate_model_size()
        total_gpu_memory = self.device_manager.get_total_gpu_memory()
        gpu_count = len([d for d in self.device_manager.devices if d.device_type == 'cuda'])
        
        info = {
            'strategy': self.strategy.value,
            'model_size_gb': model_size / 1024**3,
            'total_gpu_memory_gb': total_gpu_memory / 1024**3,
            'gpu_count': gpu_count
        }
        
        if gpu_count > 0:
            primary_device_id = int(self.device_manager.primary_device.index) if self.device_manager.primary_device.type == 'cuda' else 0
            for d in self.device_manager.devices:
                if d.device_type == 'cuda' and d.device_id == primary_device_id:
                    info['single_gpu_memory_gb'] = d.total_memory / 1024**3
                    break
        
        # Add strategy-specific information
        if self.strategy == ExecutionStrategy.CPU_OFFLOAD:
            info['uses_chunked_offloading'] = self._should_use_chunked_offloading()
        
        return info