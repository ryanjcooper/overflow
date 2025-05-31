# examples/test_all_scenarios.py
"""
Test all GPU scenarios for the Overflow framework
"""

import torch
import torch.nn as nn
import time
import psutil
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


def create_model_of_size(target_gb: float) -> nn.Module:
    """Create a model of approximately the target size in GB."""
    # Calculate parameters needed for target size
    # Each parameter is 4 bytes (float32)
    target_params = int((target_gb * 1024**3) / 4)
    
    # Create a simple model with the target number of parameters
    # Use a transformer as it's memory intensive
    d_model = 1024
    nhead = 16
    
    # Estimate layers needed
    params_per_layer = d_model * d_model * 12  # Rough estimate for transformer layer
    num_layers = max(1, int(target_params / params_per_layer))
    
    print(f"Creating model with {num_layers} transformer layers (target: {target_gb:.1f} GB)")
    
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=num_layers
    )


def test_scenario(name: str, model_size_gb: float, force_strategy: ExecutionStrategy = None):
    """Test a specific scenario."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"Model size: {model_size_gb} GB")
    print(f"{'='*60}")
    
    # Create model
    model = create_model_of_size(model_size_gb)
    
    # Force strategy if specified
    if force_strategy:
        original_determine = DynamicMemoryModule._determine_strategy
        DynamicMemoryModule._determine_strategy = lambda self: force_strategy
    
    try:
        # Wrap with Overflow
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        print(f"\nSelected strategy: {wrapped.strategy.value}")
        
        # Create input
        batch_size = 8
        seq_length = 128
        d_model = 1024
        x = torch.randn(batch_size, seq_length, d_model)
        
        # Measure inference time
        print("\nRunning inference...")
        start_time = time.time()
        
        with torch.no_grad():
            output = wrapped(x)
        
        # Ensure computation completes
        if output.is_cuda:
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        
        print(f"✓ Success!")
        print(f"  Output shape: {output.shape}")
        print(f"  Inference time: {inference_time:.2f}s")
        print(f"  Throughput: {batch_size/inference_time:.2f} samples/sec")
        
        # Get memory stats
        stats = wrapped.get_memory_stats()
        print(f"\nMemory Statistics:")
        print(f"  Peak memory: {stats['peak_memory_mb']:.1f} MB")
        if stats['swap_stats']:
            total_swaps = stats['swap_stats'].get('swaps_in', 0) + stats['swap_stats'].get('swaps_out', 0)
            if total_swaps > 0:
                print(f"  CPU←→GPU swaps: {total_swaps}")
        
        # Clear memory
        del wrapped, model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    finally:
        if force_strategy:
            DynamicMemoryModule._determine_strategy = original_determine


def main():
    """Test all scenarios based on available hardware."""
    print("Overflow Framework - Comprehensive Hardware Test")
    print("=" * 60)
    
    # Get system info
    cpu_memory_gb = psutil.virtual_memory().total / 1024**3
    print(f"System RAM: {cpu_memory_gb:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPUs detected: {gpu_count}")
        
        total_gpu_memory = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = props.total_memory / 1024**3
            total_gpu_memory += gpu_memory_gb
            print(f"  GPU {i}: {props.name} ({gpu_memory_gb:.1f} GB)")
        
        single_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if gpu_count == 1:
            print("\n--- Single GPU Scenarios ---")
            
            # Scenario 1: Model fits comfortably
            test_scenario(
                name="Small model (fits comfortably on GPU)",
                model_size_gb=single_gpu_memory * 0.3
            )
            
            # Scenario 2: Model barely fits
            test_scenario(
                name="Medium model (fits with gradient checkpointing)",
                model_size_gb=single_gpu_memory * 0.7
            )
            
            # Scenario 3: Model too large
            test_scenario(
                name="Large model (requires CPU offloading)",
                model_size_gb=single_gpu_memory * 1.5
            )
        
        else:  # Multi-GPU
            print(f"\nTotal GPU memory: {total_gpu_memory:.1f} GB")
            print("\n--- Multi-GPU Scenarios ---")
            
            # Scenario 1: Model fits on single GPU - use data parallel
            test_scenario(
                name="Small model (use data parallelism for speed)",
                model_size_gb=single_gpu_memory * 0.3
            )
            
            # Scenario 2: Model barely fits on single GPU
            test_scenario(
                name="Medium model (fits on single GPU with checkpointing)",
                model_size_gb=single_gpu_memory * 0.7
            )
            
            # Scenario 3: Model too large for single GPU but fits across all
            test_scenario(
                name="Large model (requires model parallelism)",
                model_size_gb=single_gpu_memory * 1.5
            )
            
            # Scenario 4: Model too large for all GPUs
            test_scenario(
                name="Huge model (requires CPU offloading with multi-GPU)",
                model_size_gb=total_gpu_memory * 1.2
            )
            
            # Test forced data parallel
            print("\n--- Testing Forced Strategies ---")
            test_scenario(
                name="Force data parallel on small model",
                model_size_gb=single_gpu_memory * 0.2,
                force_strategy=ExecutionStrategy.DATA_PARALLEL
            )
    
    else:
        print("\n--- CPU-Only Scenarios ---")
        
        # Small model
        test_scenario(
            name="Small model on CPU",
            model_size_gb=1.0
        )
        
        # Large model
        test_scenario(
            name="Large model on CPU",
            model_size_gb=5.0
        )
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("\nSummary of strategies:")
    print("- STANDARD: Model fits comfortably in GPU memory")
    print("- GRADIENT_CHECKPOINT: Model fits but activations don't")
    print("- DATA_PARALLEL: Model fits on one GPU, use multiple for speed")
    print("- MODEL_PARALLEL: Model too large for one GPU but fits across many")
    print("- CPU_OFFLOAD: Model too large for all GPUs, swap layers dynamically")


if __name__ == "__main__":
    main()