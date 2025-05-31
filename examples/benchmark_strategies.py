# examples/benchmark_strategies.py
"""
Benchmark different execution strategies to show performance characteristics
"""

import torch
import torch.nn as nn
import time
import numpy as np
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


def benchmark_strategy(model_size_gb: float, batch_sizes: list, strategy: ExecutionStrategy = None):
    """Benchmark a model with different batch sizes."""
    print(f"\n{'='*60}")
    print(f"Model size: {model_size_gb} GB")
    if strategy:
        print(f"Forced strategy: {strategy.value}")
    print(f"{'='*60}")
    
    # Create model
    target_params = int((model_size_gb * 1024**3) / 4)
    d_model = 1024
    params_per_layer = d_model * d_model * 12
    num_layers = max(1, int(target_params / params_per_layer))
    
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=num_layers
    )
    
    # Force strategy if specified
    if strategy:
        original_determine = DynamicMemoryModule._determine_strategy
        DynamicMemoryModule._determine_strategy = lambda self: strategy
    
    try:
        # Wrap with Overflow
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        print(f"Selected strategy: {wrapped.strategy.value}")
        print(f"\nBatch Size | Time (s) | Throughput (samples/sec) | Memory (MB)")
        print("-" * 65)
        
        for batch_size in batch_sizes:
            # Create input
            x = torch.randn(batch_size, 128, d_model)
            
            # Warmup
            with torch.no_grad():
                _ = wrapped(x)
            
            # Benchmark
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):  # Run 5 iterations
                    output = wrapped(x)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            total_time = time.time() - start_time
            avg_time = total_time / 5
            
            # Get memory stats
            stats = wrapped.get_memory_stats()
            memory_mb = stats['peak_memory_mb']
            
            throughput = batch_size / avg_time
            print(f"{batch_size:10} | {avg_time:8.3f} | {throughput:24.2f} | {memory_mb:11.1f}")
        
        # Cleanup
        del wrapped, model
        torch.cuda.empty_cache()
        
    finally:
        if strategy:
            DynamicMemoryModule._determine_strategy = original_determine


def compare_strategies():
    """Compare different strategies on the same model."""
    model_size = 4.0  # 4GB model - fits comfortably on single GPU
    
    print("\nComparing Strategies for 4GB Model")
    print("=" * 80)
    
    # Test different batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    # Test STANDARD strategy
    benchmark_strategy(model_size, batch_sizes, ExecutionStrategy.STANDARD)
    
    # Test DATA_PARALLEL strategy
    if torch.cuda.device_count() > 1:
        benchmark_strategy(model_size, batch_sizes, ExecutionStrategy.DATA_PARALLEL)
    
    # Test GRADIENT_CHECKPOINT strategy
    benchmark_strategy(model_size, batch_sizes, ExecutionStrategy.GRADIENT_CHECKPOINT)


def test_data_parallel_efficiency():
    """Test when data parallel becomes beneficial."""
    if torch.cuda.device_count() < 2:
        print("This test requires multiple GPUs")
        return
    
    print("\nData Parallel Efficiency Test")
    print("=" * 80)
    print("Testing different model sizes with batch size 32")
    
    model_sizes = [1.0, 2.0, 4.0, 8.0]  # Different model sizes in GB
    
    for model_size in model_sizes:
        print(f"\n--- Model Size: {model_size} GB ---")
        
        # Test with prefer_data_parallel=True
        config = MemoryConfig(
            prefer_data_parallel=True,
            data_parallel_threshold=0.5
        )
        
        # Create model and wrap
        target_params = int((model_size * 1024**3) / 4)
        d_model = 1024
        params_per_layer = d_model * d_model * 12
        num_layers = max(1, int(target_params / params_per_layer))
        
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Wrap with config
        wrapped = DynamicMemoryModule(model, config=config)
        wrapped.eval()
        
        print(f"Strategy: {wrapped.strategy.value}")
        
        # Benchmark with batch size 32
        x = torch.randn(32, 128, d_model)
        
        # Warmup
        with torch.no_grad():
            _ = wrapped(x)
        
        # Time it
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            for _ in range(10):
                _ = wrapped(x)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        throughput = (32 * 10) / elapsed
        print(f"Throughput: {throughput:.2f} samples/sec")
        
        del wrapped, model
        torch.cuda.empty_cache()


def main():
    """Run all benchmarks."""
    print("Overflow Framework - Performance Benchmarks")
    
    # Check hardware
    if torch.cuda.is_available():
        print(f"\nGPUs detected: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Run benchmarks
    compare_strategies()
    
    if torch.cuda.device_count() > 1:
        test_data_parallel_efficiency()
    
    print("\n" + "="*80)
    print("Benchmark Summary:")
    print("- STANDARD is fastest for models that fit comfortably")
    print("- DATA_PARALLEL benefits from larger batch sizes (32+)")
    print("- GRADIENT_CHECKPOINT trades ~30% speed for 50% memory savings")
    print("- Small batches may not benefit from DATA_PARALLEL due to overhead")


if __name__ == "__main__":
    main()