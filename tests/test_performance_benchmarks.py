# tests/test_performance_benchmarks.py
"""
Performance benchmarks for the Overflow framework.
Run with: pytest tests/test_performance_benchmarks.py -v --benchmark-only
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import gc
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


class BenchmarkModels:
    """Helper class to create models for benchmarking."""
    
    @staticmethod
    def create_transformer(layers: int, d_model: int = 1024) -> nn.Module:
        """Create a transformer model with specified layers."""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=16,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=layers
        )
    
    @staticmethod
    def estimate_model_size_gb(model: nn.Module) -> float:
        """Estimate model size in GB."""
        total_params = sum(p.numel() for p in model.parameters())
        return (total_params * 4) / 1024**3  # float32


@pytest.mark.benchmark(group="strategies")
class TestStrategyPerformance:
    """Benchmark different execution strategies."""
    
    def setup_method(self):
        """Setup for each benchmark."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Cleanup after each benchmark."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    def test_standard_strategy_throughput(self, benchmark, batch_size):
        """Benchmark standard execution strategy with varying batch sizes."""
        # Create small model that fits comfortably in GPU
        model = BenchmarkModels.create_transformer(layers=6)
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        # Skip if not using standard strategy (hardware dependent)
        if wrapped.strategy != ExecutionStrategy.STANDARD:
            pytest.skip(f"Model uses {wrapped.strategy.value}, not STANDARD")
        
        # Prepare input
        device = next(wrapped.parameters()).device
        x = torch.randn(batch_size, 128, 1024).to(device)
        
        def run_inference():
            with torch.no_grad():
                output = wrapped(x)
                torch.cuda.synchronize()
                return output
        
        # Warmup
        for _ in range(3):
            run_inference()
        
        # Benchmark
        result = benchmark(run_inference)
        assert result.shape == (batch_size, 128, 1024)
        
        # Store extra info
        benchmark.extra_info['batch_size'] = batch_size
        benchmark.extra_info['model_size_gb'] = BenchmarkModels.estimate_model_size_gb(model)
        benchmark.extra_info['strategy'] = wrapped.strategy.value
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_checkpoint_vs_standard_memory(self, benchmark):
        """Compare memory usage between standard and checkpoint strategies."""
        # Create model that uses significant memory
        model = BenchmarkModels.create_transformer(layers=12)
        model_size = BenchmarkModels.estimate_model_size_gb(model)
        
        # Force gradient checkpointing
        original_determine = DynamicMemoryModule._determine_strategy
        DynamicMemoryModule._determine_strategy = lambda self: ExecutionStrategy.GRADIENT_CHECKPOINT
        
        try:
            wrapped = DynamicMemoryModule(model)
            wrapped.eval()
            
            device = next(wrapped.parameters()).device
            x = torch.randn(8, 128, 1024).to(device)
            
            # Measure memory and time
            torch.cuda.reset_peak_memory_stats()
            
            def run_with_checkpointing():
                with torch.no_grad():
                    output = wrapped(x)
                    torch.cuda.synchronize()
                    return output
            
            result = benchmark(run_with_checkpointing)
            peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
            
            # Store metrics
            benchmark.extra_info['model_size_gb'] = model_size
            benchmark.extra_info['peak_memory_mb'] = peak_memory_mb
            benchmark.extra_info['strategy'] = 'gradient_checkpoint'
            
        finally:
            DynamicMemoryModule._determine_strategy = original_determine
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    @pytest.mark.parametrize("batch_size", [32, 64, 128])
    def test_data_parallel_scaling(self, benchmark, batch_size):
        """Benchmark data parallel scaling with batch size."""
        # Create model that benefits from data parallelism
        model = BenchmarkModels.create_transformer(layers=8)
        
        # Force data parallel with proper configuration
        config = MemoryConfig(prefer_data_parallel=True, data_parallel_threshold=0.7)
        
        # Clear any existing model from GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        wrapped = DynamicMemoryModule(model, config=config)
        wrapped.eval()
        
        if wrapped.strategy != ExecutionStrategy.DATA_PARALLEL:
            pytest.skip("Could not force data parallel strategy")
        
        x = torch.randn(batch_size, 128, 1024)
        
        def run_data_parallel():
            with torch.no_grad():
                output = wrapped(x)
                torch.cuda.synchronize()
                return output
        
        # Warmup
        try:
            for _ in range(3):
                run_data_parallel()
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip(f"Batch size {batch_size} too large for available memory")
            raise
        
        # Benchmark
        try:
            result = benchmark(run_data_parallel)
            
            # Calculate throughput
            benchmark.extra_info['batch_size'] = batch_size
            benchmark.extra_info['num_gpus'] = torch.cuda.device_count()
            benchmark.extra_info['samples_per_second'] = batch_size / benchmark.stats['mean']
        except RuntimeError as e:
            if "out of memory" in str(e):
                pytest.skip(f"Batch size {batch_size} too large for available memory")
            raise


@pytest.mark.benchmark(group="memory")
class TestMemoryEfficiency:
    """Benchmark memory efficiency of different strategies."""
    
    def setup_method(self):
        """Setup for each benchmark."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Cleanup after each benchmark."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_cpu_offload_memory_efficiency(self, benchmark):
        """Measure memory efficiency of CPU offloading."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Create model larger than GPU memory but not too large
        # to avoid excessive test time
        target_layers = int(gpu_memory_gb * 10)  # Conservative estimate
        model = BenchmarkModels.create_transformer(layers=min(target_layers, 24))
        model_size = BenchmarkModels.estimate_model_size_gb(model)
        
        if model_size < gpu_memory_gb * 0.9:
            pytest.skip("Model too small to require CPU offloading")
        
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        if wrapped.strategy != ExecutionStrategy.CPU_OFFLOAD:
            pytest.skip(f"Expected CPU_OFFLOAD but got {wrapped.strategy.value}")
        
        x = torch.randn(1, 32, 1024)  # Small batch for large model
        
        def run_with_offload():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                output = wrapped(x)
                torch.cuda.synchronize()
                peak_mb = torch.cuda.max_memory_allocated() / 1024**2
                return output, peak_mb
        
        result, peak_memory = benchmark(run_with_offload)
        
        # Calculate memory efficiency
        memory_efficiency = model_size * 1024 / peak_memory  # How much larger is model than peak GPU usage
        
        benchmark.extra_info['model_size_gb'] = model_size
        benchmark.extra_info['peak_gpu_memory_mb'] = peak_memory
        benchmark.extra_info['memory_efficiency'] = memory_efficiency
        benchmark.extra_info['swaps'] = wrapped.get_memory_stats()['swap_stats']['swaps_out']


@pytest.mark.benchmark(group="realworld")
class TestRealWorldScenarios:
    """Benchmark real-world usage patterns."""
    
    def setup_method(self):
        """Setup for each benchmark."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Cleanup after each benchmark."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_training_step_performance(self, benchmark):
        """Benchmark a complete training step."""
        model = BenchmarkModels.create_transformer(layers=12)
        wrapped = DynamicMemoryModule(model)
        wrapped.train()
        
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=1e-4)
        device = next(wrapped.parameters()).device
        x = torch.randn(16, 128, 1024).to(device)
        
        def training_step():
            # Forward pass
            output = wrapped(x)
            loss = output.mean()  # Dummy loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            return loss.item()
        
        # Warmup
        for _ in range(2):
            training_step()
        
        loss = benchmark(training_step)
        
        benchmark.extra_info['strategy'] = wrapped.strategy.value
        benchmark.extra_info['model_size_gb'] = BenchmarkModels.estimate_model_size_gb(model)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_inference_latency(self, benchmark):
        """Benchmark inference latency for different model sizes."""
        # Test with medium-sized model
        model = BenchmarkModels.create_transformer(layers=8, d_model=768)
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        device = next(wrapped.parameters()).device
        
        # Single sample for latency measurement
        x = torch.randn(1, 64, 768).to(device)
        
        def single_inference():
            with torch.no_grad():
                output = wrapped(x)
                torch.cuda.synchronize()
                return output
        
        # Warmup
        for _ in range(10):
            single_inference()
        
        # Benchmark
        result = benchmark(single_inference)
        
        benchmark.extra_info['strategy'] = wrapped.strategy.value
        benchmark.extra_info['model_size_gb'] = BenchmarkModels.estimate_model_size_gb(model)
        benchmark.extra_info['sequence_length'] = 64
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    @pytest.mark.parametrize("sequence_length", [32, 64, 128, 256])
    def test_sequence_length_scaling(self, benchmark, sequence_length):
        """Benchmark performance scaling with sequence length."""
        model = BenchmarkModels.create_transformer(layers=6, d_model=512)
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        device = next(wrapped.parameters()).device
        
        # Fixed batch size, varying sequence length
        x = torch.randn(8, sequence_length, 512).to(device)
        
        def run_inference():
            with torch.no_grad():
                output = wrapped(x)
                torch.cuda.synchronize()
                return output
        
        # Warmup
        for _ in range(3):
            run_inference()
        
        # Benchmark
        result = benchmark(run_inference)
        
        benchmark.extra_info['sequence_length'] = sequence_length
        benchmark.extra_info['strategy'] = wrapped.strategy.value