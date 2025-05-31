# examples/test_60gb_model.py
"""
Test running a 60GB model with Overflow.
This demonstrates how Overflow automatically optimizes CPU offloading for large models.

Usage:
    # Run with automatic optimization (recommended)
    python test_60gb_model.py
    
    # Run quick test with smaller model
    python test_60gb_model.py --quick
    
    # Force run even with low RAM (not recommended)
    python test_60gb_model.py --force
"""

import torch
import torch.nn as nn
import psutil
import time
import gc
import logging
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


# Configure logging for the example
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


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


def run_60gb_model_test():
    """Run the 60GB model test with automatic optimization."""
    print("="*70)
    print("60GB Model Test - Automatic Optimization")
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
        enable_profiling=True,
        verbose=True  # Enable verbose logging to see chunk details
    )
    
    # Wrap with Overflow
    print("\nStep 3: Wrapping model with Overflow...")
    start_time = time.time()
    
    try:
        # Let Overflow handle everything automatically
        wrapped_model = DynamicMemoryModule(model, config)
        
        wrap_time = time.time() - start_time
        print(f"✓ Model wrapped in {wrap_time:.1f} seconds")
        print(f"✓ Strategy selected: {wrapped_model.strategy.value}")
        
        # Get strategy information
        strategy_info = wrapped_model.get_strategy_info()
        print(f"\n📊 Strategy Details:")
        print(f"  - Model size: {strategy_info['model_size_gb']:.1f} GB")
        print(f"  - Total GPU memory: {strategy_info['total_gpu_memory_gb']:.1f} GB")
        if 'single_gpu_memory_gb' in strategy_info:
            print(f"  - Single GPU memory: {strategy_info['single_gpu_memory_gb']:.1f} GB")
        print(f"  - Number of GPUs: {strategy_info['gpu_count']}")
        
        # Check if chunked offloading was automatically applied
        if wrapped_model.strategy == ExecutionStrategy.CPU_OFFLOAD:
            # The framework will have automatically applied chunked offloading if beneficial
            print("\n📋 CPU Offloading Details:")
            if strategy_info.get('uses_chunked_offloading', False):
                print("  → Using optimized chunked CPU offloading")
                print("  → The framework will optimize chunk size based on input dimensions")
            else:
                print("  → Using sequential CPU offloading")
                print("  → Processing one layer at a time")
        
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
        
        # Show chunk information if available
        if 'chunk_info' in final_stats:
            print(f"\n📊 Chunk Optimization Details:")
            print(f"  - Chunk size: {final_stats['chunk_info']['chunk_size']} layers")
            print(f"  - Number of chunks: {final_stats['chunk_info']['num_chunks']}")
            print(f"  - Layer memory: {final_stats['chunk_info']['layer_memory_mb']:.1f} MB")
            print(f"  - Expected GPU usage: {final_stats['chunk_info']['expected_gpu_usage_gb']:.1f} GB per GPU")
        
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
        
        # Tips based on performance
        if wrapped_model.strategy == ExecutionStrategy.CPU_OFFLOAD and inference_time > 60:
            print("\n💡 Performance Tips:")
            print("  - The framework automatically optimizes CPU offloading when possible")
            print("  - Consider using a smaller model or adding more GPUs for faster inference")
            print("  - Ensure you're using pinned memory (automatic with Overflow)")
        
        return {
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


def quick_test_smaller_model():
    """Quick test with a smaller model for systems with limited RAM."""
    print("\n" + "="*70)
    print("Quick Test with Smaller Model")
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
    
    # Wrap with Overflow
    print("\nWrapping with Overflow...")
    wrapped = DynamicMemoryModule(model)
    print(f"✓ Strategy selected: {wrapped.strategy.value}")
    
    # Explain the strategy
    if wrapped.strategy == ExecutionStrategy.STANDARD:
        print("  → Model fits comfortably in GPU memory")
    elif wrapped.strategy == ExecutionStrategy.GRADIENT_CHECKPOINT:
        print("  → Using gradient checkpointing to save memory")
    elif wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD:
        print("  → Using CPU offloading (model larger than GPU)")
        print("  → The framework will automatically optimize the offloading strategy")
    
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
            # Run the 60GB model test
            result = run_60gb_model_test()
            
            if result:
                print("\n" + "="*70)
                print("Summary")
                print("="*70)
                print("✅ Successfully ran a 60GB model on your hardware!")
                print("\nKey achievements:")
                print(f"  - Model size: 60GB")
                print(f"  - Peak GPU memory used: {result['peak_gpu_mb']/1024:.1f}GB")
                print(f"  - Time taken: {result['inference_time']:.1f}s")
                if result['total_swaps'] > 0:
                    print(f"  - CPU-GPU swaps: {result['total_swaps']}")
                    print("\nThe framework automatically optimized the CPU offloading strategy")
                    print("to maximize performance while working within your hardware limits.")


if __name__ == "__main__":
    main()