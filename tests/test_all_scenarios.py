# tests/test_all_scenarios.py
"""
Integration tests for all GPU scenarios in the Overflow framework.
These tests verify that the framework correctly handles different hardware configurations.
"""

import pytest
import torch
import torch.nn as nn
import time
import psutil
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


class TestHardwareScenarios:
    """Test suite for different hardware scenarios."""
    
    @staticmethod
    def create_model_of_relative_size(fraction_of_memory: float, 
                                     reference_memory_gb: float) -> nn.Module:
        """
        Create a model that's a fraction of the reference memory size.
        This ensures tests adapt to available hardware.
        """
        target_gb = fraction_of_memory * reference_memory_gb
        # Each parameter is 4 bytes (float32)
        target_params = int((target_gb * 1024**3) / 4)
        
        # Create a simple model with the target number of parameters
        d_model = 1024
        nhead = 16
        
        # Estimate layers needed
        params_per_layer = d_model * d_model * 12  # Rough estimate for transformer layer
        num_layers = max(1, int(target_params / params_per_layer))
        
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=4096,
                batch_first=True
            ),
            num_layers=num_layers
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_small_model(self):
        """Test small model that fits comfortably on single GPU."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        model = self.create_model_of_relative_size(0.3, gpu_memory_gb)
        
        wrapped = DynamicMemoryModule(model)
        assert wrapped.strategy == ExecutionStrategy.STANDARD
        
        # Test forward pass
        x = torch.randn(4, 128, 1024)
        output = wrapped(x)
        assert output.shape == (4, 128, 1024)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_medium_model(self):
        """Test model that requires gradient checkpointing on single GPU."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Create model that's 70% of GPU memory (should trigger checkpointing)
        model = self.create_model_of_relative_size(0.7, gpu_memory_gb)
        
        wrapped = DynamicMemoryModule(model)
        # Should use gradient checkpointing
        assert wrapped.strategy in [ExecutionStrategy.GRADIENT_CHECKPOINT, ExecutionStrategy.STANDARD]
        
        # Test forward pass
        x = torch.randn(2, 64, 1024)  # Smaller batch due to larger model
        output = wrapped(x)
        assert output.shape == (2, 64, 1024)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_large_model(self):
        """Test model that requires CPU offloading on single GPU."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Skip if system doesn't have enough RAM
        cpu_memory_gb = psutil.virtual_memory().total / 1024**3
        required_memory = gpu_memory_gb * 1.5
        if cpu_memory_gb < required_memory * 1.2:  # Need some headroom
            pytest.skip(f"Insufficient RAM for CPU offloading test (need {required_memory * 1.2:.1f} GB)")
        
        # Create model larger than GPU memory
        model = self.create_model_of_relative_size(1.5, gpu_memory_gb)
        
        wrapped = DynamicMemoryModule(model)
        assert wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD
        
        # Test forward pass with small batch
        x = torch.randn(1, 32, 1024)  # Very small batch for large model
        output = wrapped(x)
        assert output.shape == (1, 32, 1024)
        
        # Verify CPU offloading happened
        stats = wrapped.get_memory_stats()
        assert stats['swap_stats']['swaps_out'] > 0
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    def test_multi_gpu_data_parallel(self):
        """Test data parallel strategy on multiple GPUs."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Small model that fits easily on one GPU
        model = self.create_model_of_relative_size(0.3, gpu_memory_gb)
        
        # Force data parallel
        config = MemoryConfig(prefer_data_parallel=True)
        wrapped = DynamicMemoryModule(model, config=config)
        
        # Should use data parallel for small model on multi-GPU
        assert wrapped.strategy == ExecutionStrategy.DATA_PARALLEL
        
        # Test with larger batch that benefits from data parallel
        x = torch.randn(32, 128, 1024)
        output = wrapped(x)
        assert output.shape == (32, 128, 1024)
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    def test_multi_gpu_model_parallel(self):
        """Test model parallel strategy on multiple GPUs."""
        single_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Model larger than single GPU but fits across all GPUs
        model = self.create_model_of_relative_size(1.5, single_gpu_memory_gb)
        
        wrapped = DynamicMemoryModule(model)
        # Should use model parallel
        assert wrapped.strategy in [ExecutionStrategy.MODEL_PARALLEL, ExecutionStrategy.CPU_OFFLOAD]
        
        # Test forward pass
        x = torch.randn(2, 64, 1024)
        output = wrapped(x)
        assert output.shape == (2, 64, 1024)
    
    def test_cpu_only_execution(self):
        """Test execution on CPU-only systems."""
        # Force CPU-only mode for this test
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            # Temporarily disable CUDA
            torch.cuda.is_available = lambda: False
        
        try:
            model = nn.Linear(1000, 1000)
            wrapped = DynamicMemoryModule(model)
            
            assert wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD
            
            # Test forward pass
            x = torch.randn(10, 1000)
            output = wrapped(x)
            assert output.shape == (10, 1000)
        finally:
            if cuda_available:
                # Restore CUDA availability
                torch.cuda.is_available = lambda: True
    
    @pytest.mark.parametrize("strategy", [
        ExecutionStrategy.STANDARD,
        ExecutionStrategy.GRADIENT_CHECKPOINT,
        ExecutionStrategy.CPU_OFFLOAD
    ])
    def test_forced_strategies(self, strategy):
        """Test forcing specific execution strategies."""
        model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        
        # Mock strategy selection to force specific strategy
        original_determine = DynamicMemoryModule._determine_strategy
        DynamicMemoryModule._determine_strategy = lambda self: strategy
        
        try:
            wrapped = DynamicMemoryModule(model)
            assert wrapped.strategy == strategy
            
            # Test forward pass works with forced strategy
            x = torch.randn(8, 512)
            output = wrapped(x)
            assert output.shape == (8, 512)
        finally:
            DynamicMemoryModule._determine_strategy = original_determine


@pytest.mark.benchmark
class TestPerformanceScenarios:
    """Performance benchmarks for different scenarios."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_strategy_performance_comparison(self, benchmark):
        """Benchmark different strategies on same model."""
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Create model that's 40% of GPU memory
        model = TestHardwareScenarios.create_model_of_relative_size(0.4, gpu_memory_gb)
        wrapped = DynamicMemoryModule(model)
        wrapped.eval()
        
        x = torch.randn(8, 128, 1024)
        
        # Benchmark the forward pass
        def run_inference():
            with torch.no_grad():
                output = wrapped(x)
                if output.is_cuda:
                    torch.cuda.synchronize()
                return output
        
        result = benchmark(run_inference)
        
        # Verify output
        assert result.shape == (8, 128, 1024)