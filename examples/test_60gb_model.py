# examples/test_60gb_model.py
"""
Test running a 60GB model with Overflow.
This demonstrates both sequential and chunked CPU offloading strategies.

New in this version:
- Automatic optimization (default): Detects when chunked offloading would help
- Manual strategies still available for testing and comparison

Usage:
    # Automatic optimization (recommended - just wrap and run!)
    python test_60gb_model.py
    
    # Compare different strategies
    python test_60gb_model.py --strategy compare
    
    # Force specific strategy
    python test_60gb_model.py --strategy chunked --chunk-size 25
"""

import torch
import torch.nn as nn
import psutil
import time
import gc
import math
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


def format_bytes(bytes_value):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def create_60gb_model():
    """
    Create a model that's approximately 60GB in size.
    
    Calculation:
    - Each parameter is 4 bytes (float32)
    - 60GB = 60 * 1024^3 bytes = 64,424,509,440 bytes
    - Number of parameters = 64,424,509,440 / 4 = 16,106,127,360 parameters (~16B)
    
    For a transformer with d_model=4096, ff_dim=16384:
    - Attention: 4096^2 * 4 = 67M parameters
    - FFN: 4096 * 16384 * 2 = 134M parameters  
    - Total per layer ≈ 201M parameters
    - For 16B parameters: ~80 layers
    """
    d_model = 4096
    nhead = 32
    ff_dim = 16384
    num_layers = 80
    
    print(f"Creating transformer model:")
    print(f"  - d_model: {d_model}")
    print(f"  - nhead: {nhead}")
    print(f"  - ff_dim: {ff_dim}")
    print(f"  - num_layers: {num_layers}")
    
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.0  # Disable dropout for inference
        ),
        num_layers=num_layers
    )
    
    # Calculate actual model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_bytes = total_params * 4  # float32
    
    print(f"\nModel statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Model size: {format_bytes(model_size_bytes)}")
    
    return model, model_size_bytes


def check_system_requirements():
    """Check if system has enough resources for the test."""
    # Check RAM
    vm = psutil.virtual_memory()
    available_ram_gb = vm.available / 1024**3
    total_ram_gb = vm.total / 1024**3
    
    print("System Resources:")
    print(f"  - Total RAM: {total_ram_gb:.1f} GB")
    print(f"  - Available RAM: {available_ram_gb:.1f} GB")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_gpu_memory = 0
        
        print(f"  - Number of GPUs: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = props.total_memory / 1024**3
            total_gpu_memory += props.total_memory
            print(f"  - GPU {i} ({props.name}): {gpu_memory_gb:.1f} GB")
        
        total_gpu_memory_gb = total_gpu_memory / 1024**3
        print(f"  - Total GPU memory: {total_gpu_memory_gb:.1f} GB")
    else:
        print("  - No GPUs detected")
        total_gpu_memory_gb = 0
    
    # We need at least 70GB RAM for a 60GB model (with some overhead)
    required_ram = 70
    if available_ram_gb < required_ram:
        print(f"\n⚠️  Warning: This test requires at least {required_ram}GB of available RAM.")
        print(f"   You currently have {available_ram_gb:.1f}GB available.")
        print("   The test will likely fail or cause system instability.")
        return False
    
    print(f"\n✓ System has sufficient RAM for the test")
    
    # Note about GPU memory
    if total_gpu_memory_gb < 60:
        print(f"ℹ️  Note: Total GPU memory ({total_gpu_memory_gb:.1f}GB) is less than model size.")
        print("   Overflow will use CPU offloading to run the model.")
    
    return True


def create_auto_optimized_wrapper(model, base_config=None):
    """
    Create an automatically optimized wrapper that uses chunked offloading when beneficial.
    This demonstrates what would be built into the framework.
    """
    # Create base config if not provided
    if base_config is None:
        base_config = MemoryConfig(
            checkpoint_threshold=0.5,
            offload_threshold=0.6,
            prefetch_size=1,
            min_gpu_memory_mb=4096,
            enable_profiling=True
        )
    
    # Wrap with standard Overflow
    wrapped = DynamicMemoryModule(model, base_config)
    
    # If CPU offloading was selected, check if chunked would be beneficial
    if wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD:
        # Check if model supports chunking and if we have enough GPU memory
        if hasattr(wrapped.wrapped_module, 'layers') and len(wrapped.wrapped_module.layers) > 0:
            # Calculate if chunking would help
            total_gpu_memory = wrapped.device_manager.get_available_gpu_memory()
            sample_layer = wrapped.wrapped_module.layers[0]
            layer_memory = wrapped._estimate_layer_memory(sample_layer)
            memory_per_layer = layer_memory * 3  # With activations
            
            layers_that_fit = int(total_gpu_memory * 0.8 / memory_per_layer)
            
            if layers_that_fit >= 2:  # Can fit at least 2 layers
                print("\n🚀 Auto-Optimization: Detected that chunked offloading will improve performance!")
                print(f"  → Can fit {layers_that_fit} layers at once instead of 1")
                print(f"  → Expected speedup: {min(layers_that_fit, len(wrapped.wrapped_module.layers))}x")
                
                # Apply chunked offloading
                wrapped, chunk_size = add_chunked_offloading(wrapped)
                wrapped._optimized_mode = "chunked"
                wrapped._chunk_size = chunk_size
            else:
                wrapped._optimized_mode = "sequential"
                wrapped._chunk_size = 1
        else:
            wrapped._optimized_mode = "sequential"
            wrapped._chunk_size = 1
    else:
        wrapped._optimized_mode = "none"
        wrapped._chunk_size = 0
    
    return wrapped
    """
    Modify the wrapped model to use chunked CPU offloading.
    This is a demonstration - in production, this would be built into the framework.
    """
    if not hasattr(wrapped_model.wrapped_module, 'layers'):
        print("Model doesn't have layers attribute, can't apply chunking")
        return wrapped_model, 1
    
    # Get layer memory estimate first
    sample_layer = wrapped_model.wrapped_module.layers[0]
    layer_memory = wrapped_model._estimate_layer_memory(sample_layer)
    
    # Calculate optimal chunk size if not provided
    if chunk_size is None:
        # Get available GPU memory
        total_gpu_memory = wrapped_model.device_manager.get_available_gpu_memory()
        usable_memory = total_gpu_memory * 0.8  # Use 80% of GPU memory
        
        memory_per_layer = layer_memory * 3  # Account for activations
        
        chunk_size = int(usable_memory / memory_per_layer)
        chunk_size = max(1, min(chunk_size, len(wrapped_model.wrapped_module.layers)))
    
    print(f"\n📦 Chunked Offloading Configuration:")
    print(f"  - Chunk size: {chunk_size} layers")
    print(f"  - Number of chunks: {math.ceil(len(wrapped_model.wrapped_module.layers) / chunk_size)}")
    print(f"  - Expected GPU usage: ~{(chunk_size * layer_memory * 3) / 1024**3:.1f} GB")
    
    # Store original forward
    original_forward = wrapped_model.wrapped_module.forward
    
    def chunked_forward(*args, **kwargs):
        """Forward pass with chunked layer loading."""
        src = args[0]
        mask = kwargs.get('mask', None)
        src_key_padding_mask = kwargs.get('src_key_padding_mask', None)
        
        # Get device
        device = wrapped_model.device_manager.primary_device
        
        # Process layers in chunks
        num_layers = len(wrapped_model.wrapped_module.layers)
        num_chunks = math.ceil(num_layers / chunk_size)
        output = src
        
        print(f"\n🔄 Processing {num_layers} layers in {num_chunks} chunks...")
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_layers)
            chunk_layers = range(start_idx, end_idx)
            
            print(f"  Chunk {chunk_idx + 1}/{num_chunks}: layers {start_idx}-{end_idx-1}", end="", flush=True)
            
            # Move chunk to GPU
            chunk_start_time = time.time()
            for i in chunk_layers:
                wrapped_model.wrapped_module.layers[i] = wrapped_model.wrapped_module.layers[i].to(device)
                wrapped_model.swap_manager.swap_stats['swaps_in'] += 1
            
            # Move data to device if needed
            if output.device != device:
                output = output.to(device)
            if mask is not None and mask.device != device:
                mask = mask.to(device)
            if src_key_padding_mask is not None and src_key_padding_mask.device != device:
                src_key_padding_mask = src_key_padding_mask.to(device)
            
            # Process all layers in chunk
            for i in chunk_layers:
                output = wrapped_model.wrapped_module.layers[i](
                    output, 
                    src_mask=mask, 
                    src_key_padding_mask=src_key_padding_mask
                )
            
            # Move chunk back to CPU
            for i in chunk_layers:
                wrapped_model.wrapped_module.layers[i] = wrapped_model.wrapped_module.layers[i].to('cpu')
                wrapped_model.swap_manager.swap_stats['swaps_out'] += 1
            
            chunk_time = time.time() - chunk_start_time
            print(f" - {chunk_time:.1f}s")
            
            # Clear cache between chunks
            if chunk_idx < num_chunks - 1:
                torch.cuda.empty_cache()
        
        # Apply final norm
        if wrapped_model.wrapped_module.norm is not None:
            wrapped_model.wrapped_module.norm = wrapped_model.wrapped_module.norm.to(device)
            output = wrapped_model.wrapped_module.norm(output)
            wrapped_model.wrapped_module.norm = wrapped_model.wrapped_module.norm.to('cpu')
        
        return output
    
    # Replace forward method
    wrapped_model.wrapped_module.forward = chunked_forward



def run_60gb_model_test(strategy="sequential", chunk_size=None):
    """Run the 60GB model test with specified strategy."""
    print("="*70)
    if strategy == "auto":
        print("60GB Model Test - Automatic Optimization")
    else:
        print(f"60GB Model Test - {strategy.capitalize()} CPU Offloading")
    print("="*70)
    print()
    
    # Check system requirements
    if not check_system_requirements():
        print("\nExiting due to insufficient system resources.")
        return None
    
    print("\n" + "-"*70)
    
    # Create the large model
    print("\nStep 1: Creating 60GB model...")
    start_time = time.time()
    
    try:
        model, model_size = create_60gb_model()
        creation_time = time.time() - start_time
        print(f"✓ Model created in {creation_time:.1f} seconds")
    except Exception as e:
        print(f"✗ Failed to create model: {str(e)}")
        print("\nThis might be due to:")
        print("  - Insufficient RAM to create the model")
        print("  - PyTorch version compatibility issues")
        print("\nTry running with --quick for a smaller model test")
        return None
    
    # Configure Overflow
    print("\nStep 2: Configuring Overflow...")
    config = MemoryConfig(
        checkpoint_threshold=0.5,
        offload_threshold=0.6,
        prefetch_size=1,
        min_gpu_memory_mb=4096,
        enable_profiling=True
    )
    
    # Wrap with Overflow
    print("\nStep 3: Wrapping model with Overflow...")
    start_time = time.time()
    
    # Initialize strategy tracking
    actual_strategy = strategy
    used_chunk_size = 1
    
    try:
        if strategy == "auto":
            # Use automatic optimization
            wrapped_model = create_auto_optimized_wrapper(model, config)
            used_chunk_size = getattr(wrapped_model, '_chunk_size', 1)
            actual_strategy = getattr(wrapped_model, '_optimized_mode', 'unknown')
            print(f"✓ Auto-selected strategy: {actual_strategy}")
        else:
            # Manual strategy selection
            wrapped_model = DynamicMemoryModule(model, config)
            
            if wrapped_model.strategy != ExecutionStrategy.CPU_OFFLOAD:
                print(f"\n⚠️  Warning: Expected CPU_OFFLOAD strategy but got {wrapped_model.strategy.value}")
            
            # Apply chunked offloading if requested
            if strategy == "chunked":
                print("\nStep 3.5: Applying chunked offloading optimization...")
                wrapped_model, used_chunk_size = add_chunked_offloading(wrapped_model, chunk_size)
            else:
                print(f"\n📋 Sequential Offloading (Default):")
                print(f"  - Processing 1 layer at a time")
                print(f"  - Expected swaps: {len(model.layers) * 2}")
                used_chunk_size = 1
                actual_strategy = "sequential"
        
        wrap_time = time.time() - start_time
        print(f"✓ Model wrapped in {wrap_time:.1f} seconds")
        
    except Exception as e:
        print(f"✗ Failed to wrap model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Prepare for inference
    print("\nStep 4: Preparing test input...")
    batch_size = 1  # Very small batch for 60GB model
    seq_length = 128  # Reasonable sequence length
    d_model = 4096
    
    # Create input on CPU first
    test_input = torch.randn(batch_size, seq_length, d_model)
    print(f"✓ Input shape: {test_input.shape}")
    
    # Run inference
    print("\nStep 5: Running inference...")
    print("This will take several minutes due to CPU-GPU swapping...")
    
    wrapped_model.eval()
    
    # Clear GPU cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    initial_stats = wrapped_model.get_memory_stats()
    swap_start_count_out = initial_stats['swap_stats'].get('swaps_out', 0)
    swap_start_count_in = initial_stats['swap_stats'].get('swaps_in', 0)
    
    print("\nStarting inference... (check GPU memory usage with nvidia-smi)")
    
    try:
        with torch.no_grad():
            output = wrapped_model(test_input)
        
        inference_time = time.time() - start_time
        
        # Get final statistics
        final_stats = wrapped_model.get_memory_stats()
        total_swaps_out = final_stats['swap_stats']['swaps_out'] - swap_start_count_out
        total_swaps_in = final_stats['swap_stats']['swaps_in'] - swap_start_count_in
        
        print(f"\n✓ Inference completed successfully!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Inference time: {inference_time:.1f} seconds ({inference_time/60:.1f} minutes)")
        print(f"  - Total layer swaps out: {total_swaps_out}")
        print(f"  - Total layer swaps in: {total_swaps_in}")
        if total_swaps_out > 0:
            print(f"  - Average time per swap: {inference_time/(total_swaps_out + total_swaps_in):.2f} seconds")
        
        # Memory statistics
        print(f"\nMemory Statistics:")
        print(f"  - Peak GPU memory used: {final_stats['peak_memory_mb']:.1f} MB")
        print(f"  - Total swaps out: {final_stats['swap_stats']['swaps_out']}")
        print(f"  - Total swaps in: {final_stats['swap_stats']['swaps_in']}")
        
        # Show per-device memory usage
        print(f"\nDevice Memory Usage:")
        for device in final_stats['devices']:
            if device['type'] == 'cuda':
                used_mb = device['total_memory_mb'] - device['available_memory_mb']
                usage_pct = (used_mb / device['total_memory_mb']) * 100
                print(f"  - GPU {device['id']}: {used_mb:.1f}/{device['total_memory_mb']:.1f} MB ({usage_pct:.1f}%)")
        
        # Performance analysis
        model_gb = model_size / 1024**3
        throughput_gb_per_sec = model_gb / inference_time
        
        print(f"\nPerformance Analysis:")
        print(f"  - Model size: {model_gb:.1f} GB")
        print(f"  - Inference throughput: {throughput_gb_per_sec:.3f} GB/s")
        print(f"  - Effective memory multiplication: {model_gb / (final_stats['peak_memory_mb']/1024):.1f}x")
        
        # Return results for comparison
        return {
            'strategy': actual_strategy,
            'chunk_size': used_chunk_size,
            'inference_time': inference_time,
            'total_swaps': total_swaps_out + total_swaps_in,
            'peak_gpu_mb': final_stats['peak_memory_mb'],
            'throughput_gb_s': throughput_gb_per_sec
        }
        
    except Exception as e:
        print(f"\n✗ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Cleanup
        print("\nStep 6: Cleanup...")
        del wrapped_model
        del model
        if 'output' in locals():
            del output
        del test_input
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def compare_strategies():
    """Run both strategies and compare results."""
    print("\n" + "="*70)
    print("Comparing Sequential vs Chunked CPU Offloading")
    print("="*70)
    
    results = []
    
    # Run sequential strategy
    print("\n🔵 Running Sequential Strategy...")
    seq_result = run_60gb_model_test(strategy="sequential")
    if seq_result:
        results.append(seq_result)
    
    # Clean up between runs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(5)  # Give system time to recover
    
    # Run chunked strategy
    print("\n\n🟢 Running Chunked Strategy...")
    chunk_result = run_60gb_model_test(strategy="chunked")
    if chunk_result:
        results.append(chunk_result)
    
    # Compare results
    if len(results) == 2:
        print("\n\n" + "="*70)
        print("Performance Comparison")
        print("="*70)
        
        seq, chunk = results[0], results[1]
        speedup = seq['inference_time'] / chunk['inference_time']
        
        print(f"\n{'Metric':<30} {'Sequential':<20} {'Chunked':<20} {'Improvement':<20}")
        print("-"*90)
        print(f"{'Inference Time (s)':<30} {seq['inference_time']:<20.1f} {chunk['inference_time']:<20.1f} {speedup:<20.1f}x faster")
        print(f"{'Total Swaps':<30} {seq['total_swaps']:<20} {chunk['total_swaps']:<20} {seq['total_swaps']/chunk['total_swaps']:<20.1f}x fewer")
        print(f"{'Peak GPU Memory (MB)':<30} {seq['peak_gpu_mb']:<20.1f} {chunk['peak_gpu_mb']:<20.1f} {chunk['peak_gpu_mb']/seq['peak_gpu_mb']:<20.1f}x more")
        print(f"{'Throughput (GB/s)':<30} {seq['throughput_gb_s']:<20.3f} {chunk['throughput_gb_s']:<20.3f} {speedup:<20.1f}x higher")
        
        print("\n🎯 Key Insights:")
        print(f"  - Chunked offloading is {speedup:.1f}x faster")
        print(f"  - Uses {chunk['peak_gpu_mb']/seq['peak_gpu_mb']:.0f}x more GPU memory (still safe)")
        print(f"  - Reduces swaps by {(1 - chunk['total_swaps']/seq['total_swaps'])*100:.0f}%")
        print(f"  - Better GPU utilization: {chunk['peak_gpu_mb']/1024:.1f}GB vs {seq['peak_gpu_mb']/1024:.1f}GB")


def quick_test_smaller_model():
    """Quick test with a smaller model for systems with limited RAM."""
    print("\n" + "="*70)
    print("Quick Test with Smaller Model (Auto-Optimized)")
    print("="*70)
    
    print("\nCreating a 1GB model for quick testing...")
    
    # Create a ~1GB model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=12
    )
    
    # Calculate size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = (total_params * 4) / 1024**3
    print(f"Model size: {model_size_gb:.2f} GB ({total_params:,} parameters)")
    
    # Wrap with Overflow using automatic optimization
    print("\nWrapping with Overflow (automatic optimization)...")
    wrapped = create_auto_optimized_wrapper(model)
    print(f"✓ Strategy selected: {wrapped.strategy.value}")
    
    # Show optimization details
    if hasattr(wrapped, '_optimized_mode'):
        if wrapped._optimized_mode == "chunked":
            print(f"✓ Auto-optimization: Using chunked offloading with {wrapped._chunk_size} layers/chunk")
        elif wrapped._optimized_mode == "sequential":
            print("✓ Auto-optimization: Using sequential offloading (optimal for this config)")
    
    # Explain the strategy
    if wrapped.strategy == ExecutionStrategy.STANDARD:
        print("  → Model fits comfortably in GPU memory")
    elif wrapped.strategy == ExecutionStrategy.GRADIENT_CHECKPOINT:
        print("  → Using gradient checkpointing to save memory")
    elif wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD:
        print("  → Using CPU offloading (model larger than GPU)")
    
    # Run inference
    print("\nRunning quick inference test...")
    x = torch.randn(4, 128, 1024)
    
    start_time = time.time()
    with torch.no_grad():
        output = wrapped(x)
    inference_time = time.time() - start_time
    
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Inference time: {inference_time:.2f}s")
    
    stats = wrapped.get_memory_stats()
    if stats['swap_stats']['swaps_out'] > 0:
        print(f"✓ CPU offloading used: {stats['swap_stats']['swaps_out']} swaps")
    else:
        print(f"✓ Peak GPU memory: {stats['peak_memory_mb'] / 1024:.1f} GB")
    
    print("\nThis demonstrates that Overflow automatically optimizes for your hardware!")
    print("The framework automatically chose the best execution strategy.")
    print("\nTry the full 60GB test: python examples/test_60gb_model.py")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test running a 60GB model with Overflow"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test with smaller model (1GB instead of 60GB)"
    )
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Force run even with insufficient RAM (not recommended)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sequential", "chunked", "compare", "auto"],
        default="auto",
        help="CPU offloading strategy to use (default: auto)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for chunked strategy (default: auto-calculate)"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test_smaller_model()
    else:
        # Check available RAM
        available_ram_gb = psutil.virtual_memory().available / 1024**3
        
        if available_ram_gb < 70 and not args.force:
            print(f"⚠️  This test requires ~70GB of available RAM for safety.")
            print(f"   You have {available_ram_gb:.1f}GB available.")
            print("\nOptions:")
            print("  1. Free up more RAM and try again")
            print("  2. Run with --quick for a smaller model test")
            print("  3. Run with --force to attempt anyway (may crash)")
        else:
            if args.strategy == "compare":
                compare_strategies()
            else:
                run_60gb_model_test(strategy=args.strategy, chunk_size=args.chunk_size)
            
            if args.strategy == "sequential":
                print("\n💡 Tips:")
                print("   1. Try automatic optimization (recommended):")
                print("      python examples/test_60gb_model.py --strategy auto")
                print("   2. Try chunked strategy for better performance:")
                print("      python examples/test_60gb_model.py --strategy chunked")
                print("   3. Compare all strategies:")
                print("      python examples/test_60gb_model.py --strategy compare")
            elif args.strategy == "chunked":
                print("\n💡 Tip: Try automatic optimization:")
                print("   python examples/test_60gb_model.py --strategy auto")


if __name__ == "__main__":
    main()