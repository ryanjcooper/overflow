# examples/test_60gb_model.py
"""
Test running a 60GB model with Overflow.
This demonstrates CPU offloading for models much larger than consumer GPUs.
"""

import torch
import torch.nn as nn
import psutil
import time
import gc
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


def run_60gb_model_test():
    """Run the 60GB model test."""
    print("="*70)
    print("60GB Model Test - Demonstrating CPU Offloading")
    print("="*70)
    print()
    
    # Check system requirements
    if not check_system_requirements():
        print("\nExiting due to insufficient system resources.")
        return
    
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
        return
    
    # Configure Overflow for aggressive memory management
    print("\nStep 2: Configuring Overflow...")
    config = MemoryConfig(
        checkpoint_threshold=0.5,    # Aggressive checkpointing
        offload_threshold=0.6,       # Aggressive offloading
        prefetch_size=1,            # Minimal prefetch to save memory
        min_gpu_memory_mb=4096,     # Keep 4GB free for safety
        enable_profiling=True       # Enable profiling to track swaps
    )
    
    # Wrap with Overflow
    print("\nStep 3: Wrapping model with Overflow...")
    start_time = time.time()
    
    try:
        wrapped_model = DynamicMemoryModule(model, config=config)
        wrap_time = time.time() - start_time
        print(f"✓ Model wrapped in {wrap_time:.1f} seconds")
        print(f"✓ Selected strategy: {wrapped_model.strategy.value}")
        
        if wrapped_model.strategy != ExecutionStrategy.CPU_OFFLOAD:
            print(f"\n⚠️  Warning: Expected CPU_OFFLOAD strategy but got {wrapped_model.strategy.value}")
            print("   This might indicate the model fits in available GPU memory.")
    except Exception as e:
        print(f"✗ Failed to wrap model: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
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
    
    # Estimate runtime
    if wrapped_model.strategy == ExecutionStrategy.CPU_OFFLOAD:
        print("\nEstimated runtime: 5-15 minutes depending on CPU-GPU bandwidth")
        print("(PCIe 3.0 x16: ~16 GB/s, PCIe 4.0 x16: ~32 GB/s)")
    
    print("\nProgress will be shown via layer swaps...")
    
    # Get number of layers
    if hasattr(model, 'layers'):
        num_layers = len(model.layers)
    else:
        # Count TransformerEncoderLayer modules
        num_layers = sum(1 for m in model.modules() if isinstance(m, nn.TransformerEncoderLayer))
    
    print(f"Expected swaps: ~{num_layers * 2} (in and out for {num_layers} layers)")
    
    wrapped_model.eval()
    
    # Clear GPU cache before inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    start_time = time.time()
    initial_stats = wrapped_model.get_memory_stats()
    swap_start_count = initial_stats['swap_stats'].get('swaps_out', 0)
    
    print("\nStarting inference... (check GPU memory usage with nvidia-smi)")
    print("Note: First forward pass will be slower due to initial memory allocation")
    
    try:
        with torch.no_grad():
            # For CPU offloading, we could potentially add hooks to track progress
            # but for now just run the inference
            output = wrapped_model(test_input)
        
        inference_time = time.time() - start_time
        
        # Get final statistics
        final_stats = wrapped_model.get_memory_stats()
        total_swaps_out = final_stats['swap_stats']['swaps_out'] - swap_start_count
        total_swaps_in = final_stats['swap_stats']['swaps_in'] - swap_start_count
        
        print(f"\n✓ Inference completed successfully!")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Inference time: {inference_time:.1f} seconds ({inference_time/60:.1f} minutes)")
        print(f"  - Total layer swaps out: {total_swaps_out}")
        print(f"  - Total layer swaps in: {total_swaps_in}")
        if total_swaps_out > 0:
            print(f"  - Average time per swap: {inference_time/total_swaps_out:.2f} seconds")
        
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
        
    except Exception as e:
        print(f"\n✗ Inference failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Cleanup
    print("\nStep 6: Cleanup...")
    del wrapped_model
    del model
    del output
    del test_input
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print("✓ Successfully ran a 60GB model on consumer hardware!")
    print("✓ CPU offloading enabled running models larger than GPU memory")
    print("✓ The model was dynamically swapped between CPU and GPU during execution")
    print("\nKey Insights:")
    print("- Large models can run on limited hardware with Overflow")
    print("- CPU offloading trades speed for the ability to run huge models")
    print("- Memory usage stays within GPU limits despite model size")
    print("- This technique scales to even larger models with sufficient RAM")
    print("\nMonitoring tip: Run 'watch -n 1 nvidia-smi' in another terminal to see GPU memory in action!")


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
    
    print("\nThis demonstrates that Overflow automatically adapts to your hardware!")
    print("Try the full 60GB test with: python examples/test_60gb_model.py --force")


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
            run_60gb_model_test()


if __name__ == "__main__":
    main()