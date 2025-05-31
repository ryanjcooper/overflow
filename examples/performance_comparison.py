# examples/performance_comparison.py
"""
Example demonstrating performance characteristics of different execution strategies.
This shows how to measure and compare performance across different strategies.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


def measure_inference_time(model, input_tensor, num_runs=10, warmup_runs=3):
    """
    Measure average inference time for a model.
    
    Args:
        model: The model to benchmark
        input_tensor: Input tensor
        num_runs: Number of runs to average over
        warmup_runs: Number of warmup runs before timing
    
    Returns:
        dict: Timing statistics
    """
    model.eval()
    
    # Warmup
    print(f"  Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Time the runs
    times = []
    print(f"  Timing {num_runs} runs...")
    
    with torch.no_grad():
        for i in range(num_runs):
            start = time.time()
            output = model(input_tensor)
            
            # Ensure computation is complete
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i % (num_runs // 4) == 0:
                print(f"    Run {i+1}/{num_runs}: {elapsed:.3f}s")
    
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'throughput': input_tensor.shape[0] / np.mean(times)
    }


def compare_strategy_performance():
    """
    Example comparing performance between automatic and forced strategies.
    Shows how execution strategy affects performance.
    """
    print("Performance Comparison Example")
    print("=" * 60)
    
    # Check available hardware
    if not torch.cuda.is_available():
        print("No GPU available. This example requires CUDA.")
        return
    
    gpu_props = torch.cuda.get_device_properties(0)
    gpu_memory_gb = gpu_props.total_memory / 1024**3
    print(f"GPU: {gpu_props.name} ({gpu_memory_gb:.1f} GB)")
    print()
    
    # Create a model that's ~30% of GPU memory (should use STANDARD strategy)
    target_size_gb = gpu_memory_gb * 0.3
    layers = max(1, int(target_size_gb * 6))  # Rough estimate
    
    print(f"Creating transformer model (~{target_size_gb:.1f} GB)")
    base_model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=layers
    )
    
    # Test input
    batch_size = 8
    seq_length = 128
    d_model = 1024
    test_input = torch.randn(batch_size, seq_length, d_model)
    
    results = {}
    
    # 1. Automatic strategy selection
    print("\n1. Automatic Strategy Selection")
    print("-" * 40)
    
    wrapped_auto = DynamicMemoryModule(base_model)
    print(f"Selected strategy: {wrapped_auto.strategy.value}")
    
    stats = measure_inference_time(wrapped_auto, test_input)
    results['automatic'] = {
        'strategy': wrapped_auto.strategy.value,
        'stats': stats
    }
    
    print(f"Average time: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    print(f"Throughput: {stats['throughput']:.2f} samples/sec")
    
    del wrapped_auto
    torch.cuda.empty_cache()
    
    # 2. Force gradient checkpointing
    print("\n2. Forced Gradient Checkpointing")
    print("-" * 40)
    
    # Note: In a real scenario, you would use configuration to influence strategy
    # This is just for demonstration
    config = MemoryConfig(checkpoint_threshold=0.0)  # Very aggressive checkpointing
    
    # For demo purposes, we'll recreate the model
    model_checkpoint = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=layers
    )
    
    # Mock the strategy selection for demonstration
    original_determine = DynamicMemoryModule._determine_strategy
    DynamicMemoryModule._determine_strategy = lambda self: ExecutionStrategy.GRADIENT_CHECKPOINT
    
    try:
        wrapped_checkpoint = DynamicMemoryModule(model_checkpoint, config)
        print(f"Forced strategy: {wrapped_checkpoint.strategy.value}")
        
        stats = measure_inference_time(wrapped_checkpoint, test_input)
        results['checkpoint'] = {
            'strategy': wrapped_checkpoint.strategy.value,
            'stats': stats
        }
        
        print(f"Average time: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
        print(f"Throughput: {stats['throughput']:.2f} samples/sec")
        
        # Show memory savings
        print(f"\nMemory usage:")
        memory_stats = wrapped_checkpoint.get_memory_stats()
        print(f"Peak memory: {memory_stats['peak_memory_mb']:.1f} MB")
        
    finally:
        DynamicMemoryModule._determine_strategy = original_determine
        del wrapped_checkpoint
        torch.cuda.empty_cache()
    
    # 3. Summary comparison
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    
    baseline_time = results['automatic']['stats']['mean']
    
    for name, result in results.items():
        stats = result['stats']
        strategy = result['strategy']
        relative_time = stats['mean'] / baseline_time
        
        print(f"\n{name.capitalize()} ({strategy}):")
        print(f"  Time: {stats['mean']:.3f}s ({relative_time:.2f}x baseline)")
        print(f"  Throughput: {stats['throughput']:.2f} samples/sec")
    
    print("\nKey Insights:")
    print("- Gradient checkpointing trades ~30% speed for ~50% memory savings")
    print("- Automatic strategy selection optimizes for your hardware")
    print("- Larger batch sizes generally improve throughput")


def demonstrate_batch_size_impact():
    """
    Example showing how batch size affects performance with different strategies.
    """
    print("\n\nBatch Size Impact on Performance")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("This example requires CUDA.")
        return
    
    # Create a medium-sized model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=768,
            nhead=12,
            dim_feedforward=3072,
            batch_first=True
        ),
        num_layers=12
    )
    
    wrapped = DynamicMemoryModule(model)
    wrapped.eval()
    
    print(f"Model strategy: {wrapped.strategy.value}")
    print("\nTesting different batch sizes:")
    print("-" * 40)
    print("Batch Size | Time/batch | Throughput | Time/sample")
    print("-" * 40)
    
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_length = 128
    d_model = 768
    
    for batch_size in batch_sizes:
        try:
            x = torch.randn(batch_size, seq_length, d_model)
            
            # Quick timing (fewer runs for demo)
            with torch.no_grad():
                # Warmup
                _ = wrapped(x)
                torch.cuda.synchronize()
                
                # Time
                start = time.time()
                for _ in range(5):
                    _ = wrapped(x)
                torch.cuda.synchronize()
                total_time = time.time() - start
            
            avg_time = total_time / 5
            throughput = batch_size / avg_time
            time_per_sample = avg_time / batch_size
            
            print(f"{batch_size:10} | {avg_time:10.3f}s | {throughput:10.2f}/s | {time_per_sample:11.4f}s")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:10} | Out of memory")
                break
            else:
                raise
    
    print("\nObservations:")
    print("- Larger batches are more efficient (lower time per sample)")
    print("- There's a sweet spot before memory limits are hit")
    print("- Different strategies have different optimal batch sizes")


def main():
    """Run all performance examples."""
    # Performance comparison between strategies
    compare_strategy_performance()
    
    # Demonstrate batch size impact
    demonstrate_batch_size_impact()
    
    # Final advice
    print("\n" + "=" * 60)
    print("Performance Tips:")
    print("=" * 60)
    print("1. Let Overflow choose the strategy automatically for best results")
    print("2. Use larger batch sizes when possible for better throughput")
    print("3. Profile your specific model and hardware combination")
    print("4. Consider gradient checkpointing for memory-constrained scenarios")
    print("5. CPU offloading enables huge models but with performance cost")


if __name__ == "__main__":
    main()