# tests/test_overflow.py
"""
Unit tests for Overflow package
Run with: python -m pytest tests/test_overflow.py -v
"""

import pytest
import torch
import torch.nn as nn
import gc
from unittest.mock import Mock, patch
import psutil

# Import framework components
from overflow import (
    DynamicMemoryModule, 
    MemoryConfig, 
    ExecutionStrategy,
    MemoryProfiler,
    DeviceManager,
    BlockSwapManager,
    ModelPartitioner
)


class TestMemoryProfiler:
    """Test memory profiling functionality."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        assert profiler.config == config
        assert profiler.memory_stats == {}
        assert profiler.peak_memory == 0
        assert profiler.profile_count == 0
    
    def test_profile_memory_cuda(self):
        """Test CUDA memory profiling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        device = torch.device("cuda:0")
        
        stats = profiler.profile_memory(device)
        
        assert "allocated" in stats
        assert "reserved" in stats
        assert "free" in stats
        assert "total" in stats
        assert all(v >= 0 for v in stats.values())
    
    def test_profile_memory_cpu(self):
        """Test CPU memory profiling."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        device = torch.device("cpu")
        
        stats = profiler.profile_memory(device)
        
        assert "allocated" in stats
        assert "reserved" in stats
        assert "free" in stats
        assert "total" in stats
        assert all(v >= 0 for v in stats.values())
    
    def test_update_stats(self):
        """Test statistics update."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        profiler.update_stats("layer1", 100.5)
        profiler.update_stats("layer1", 150.3)
        profiler.update_stats("layer2", 200.0)
        
        assert "layer1" in profiler.memory_stats
        assert profiler.memory_stats["layer1"]["count"] == 2
        assert profiler.memory_stats["layer1"]["total_memory"] == 250.8
        assert profiler.memory_stats["layer1"]["peak_memory"] == 150.3
        assert profiler.peak_memory == 200.0
    
    def test_memory_pressure(self):
        """Test memory pressure calculation."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        # Mock the profile_memory method
        with patch.object(profiler, 'profile_memory') as mock_profile:
            mock_profile.return_value = {
                'free': 2000,  # 2GB free
                'total': 8000  # 8GB total
            }
            
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            pressure = profiler.get_memory_pressure(device)
            
            assert pressure == 0.75  # 75% memory used


class TestDeviceManager:
    """Test device detection and management."""
    
    def test_device_detection(self):
        """Test device detection."""
        manager = DeviceManager()
        
        assert len(manager.devices) > 0
        assert any(d.device_type == "cpu" for d in manager.devices)
        
        if torch.cuda.is_available():
            assert any(d.device_type == "cuda" for d in manager.devices)
            cuda_devices = [d for d in manager.devices if d.device_type == "cuda"]
            assert len(cuda_devices) == torch.cuda.device_count()
    
    def test_primary_device_selection(self):
        """Test primary device selection."""
        manager = DeviceManager()
        
        assert manager.primary_device is not None
        assert isinstance(manager.primary_device, torch.device)
        
        if torch.cuda.is_available():
            assert manager.primary_device.type == "cuda"
        else:
            assert manager.primary_device.type == "cpu"
    
    def test_memory_calculations(self):
        """Test memory calculation methods."""
        manager = DeviceManager()
        
        total_gpu = manager.get_total_gpu_memory()
        available_gpu = manager.get_available_gpu_memory()
        
        if torch.cuda.is_available():
            assert total_gpu > 0
            assert available_gpu > 0
            assert available_gpu <= total_gpu
        else:
            assert total_gpu == 0
            assert available_gpu == 0


class TestBlockSwapManager:
    """Test block swapping functionality."""
    
    def test_initialization(self):
        """Test swap manager initialization."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        assert manager.config == config
        assert len(manager.cpu_cache) == 0
        assert manager.swap_stats["swaps_in"] == 0
        assert manager.swap_stats["swaps_out"] == 0
    
    def test_swap_to_cpu(self):
        """Test swapping tensor to CPU."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        # Create test tensor
        tensor = torch.randn(100, 100)
        if torch.cuda.is_available():
            tensor = tensor.cuda()
        
        # Swap to CPU
        manager.swap_to_cpu("test_tensor", tensor)
        
        assert "test_tensor" in manager.cpu_cache
        assert manager.cpu_cache["test_tensor"].device.type == "cpu"
        assert manager.swap_stats["swaps_out"] == 1
        assert manager.swap_stats["total_bytes"] > 0
    
    def test_swap_from_cpu(self):
        """Test swapping tensor back from CPU."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        # Create and swap tensor
        original_tensor = torch.randn(50, 50)
        manager.swap_to_cpu("test_tensor", original_tensor)
        
        # Swap back
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        restored_tensor = manager.swap_from_cpu("test_tensor", device)
        
        assert restored_tensor is not None
        assert restored_tensor.device == device
        assert torch.allclose(original_tensor.cpu(), restored_tensor.cpu())
        assert manager.swap_stats["swaps_in"] == 1
    
    def test_cache_limit(self):
        """Test CPU cache size limiting."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        # Add many tensors to trigger cache cleanup
        for i in range(110):
            tensor = torch.randn(10, 10)
            manager.swap_to_cpu(f"tensor_{i}", tensor)
        
        # Cache should be limited to 100 entries
        assert len(manager.cpu_cache) <= 100
        assert "tensor_0" not in manager.cpu_cache  # Oldest should be removed
        assert "tensor_109" in manager.cpu_cache  # Newest should be present


class TestModelPartitioner:
    """Test automatic model partitioning."""
    
    def test_model_analysis(self):
        """Test model analysis for partitioning."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU partitioning requires 2+ GPUs")
        
        # Create test model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 100)
        )
        
        devices = [torch.device(f"cuda:{i}") for i in range(2)]
        partitioner = ModelPartitioner(model, devices)
        
        # Check partition map is created
        assert len(partitioner.partition_map) > 0
        
        # Check all modules are assigned
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                device = partitioner.get_device_for_module(name)
                assert device in devices


class TestDynamicMemoryModule:
    """Test the main wrapper module."""
    
    def test_initialization(self):
        """Test wrapper initialization."""
        model = nn.Linear(100, 100)
        wrapped = DynamicMemoryModule(model)
        
        assert wrapped.wrapped_module is model
        assert wrapped.config is not None
        assert wrapped.strategy in ExecutionStrategy
        assert wrapped.memory_profiler is not None
        assert wrapped.device_manager is not None
    
    def test_strategy_selection(self):
        """Test automatic strategy selection."""
        # Small model - should use standard strategy
        small_model = nn.Linear(10, 10)
        wrapped_small = DynamicMemoryModule(small_model)
        assert wrapped_small.strategy == ExecutionStrategy.STANDARD
        
        # Large model - should use different strategy
        large_model = nn.Sequential(*[nn.Linear(5000, 5000) for _ in range(20)])
        wrapped_large = DynamicMemoryModule(large_model)
        assert wrapped_large.strategy != ExecutionStrategy.STANDARD
    
    def test_forward_pass(self):
        """Test forward pass through wrapper."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        wrapped = DynamicMemoryModule(model)
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = wrapped(x)
        
        assert output.shape == (5, 10)
        assert not torch.isnan(output).any()
    
    def test_attribute_delegation(self):
        """Test attribute access delegation."""
        model = nn.Linear(10, 20)
        model.custom_attribute = "test"
        wrapped = DynamicMemoryModule(model)
        
        # Should delegate to wrapped module
        assert wrapped.in_features == 10
        assert wrapped.out_features == 20
        assert wrapped.custom_attribute == "test"
        
        # Should be able to set attributes
        wrapped.new_attribute = "new"
        assert model.new_attribute == "new"
    
    def test_memory_stats(self):
        """Test memory statistics collection."""
        model = nn.Linear(100, 100)
        wrapped = DynamicMemoryModule(model)
        
        # Run forward pass
        x = torch.randn(10, 100)
        _ = wrapped(x)
        
        # Get stats
        stats = wrapped.get_memory_stats()
        
        assert "strategy" in stats
        assert "peak_memory_mb" in stats
        assert "module_stats" in stats
        assert "devices" in stats
        assert len(stats["devices"]) > 0
    
    @pytest.mark.parametrize("strategy", [
        ExecutionStrategy.STANDARD,
        ExecutionStrategy.GRADIENT_CHECKPOINT,
        ExecutionStrategy.CPU_OFFLOAD
    ])
    def test_forced_strategy(self, strategy):
        """Test forcing specific execution strategies."""
        model = nn.Linear(100, 100)
        
        # Mock the strategy selection
        with patch.object(DynamicMemoryModule, '_determine_strategy', return_value=strategy):
            wrapped = DynamicMemoryModule(model)
            assert wrapped.strategy == strategy
    
    def test_gradient_flow(self):
        """Test gradient flow through wrapper."""
        model = nn.Linear(10, 10)
        wrapped = DynamicMemoryModule(model)
        
        # Forward pass
        x = torch.randn(5, 10, requires_grad=True)
        output = wrapped(x)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for param in wrapped.parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
    
    def test_state_dict_compatibility(self):
        """Test state dict save/load compatibility."""
        model = nn.Linear(10, 10)
        wrapped = DynamicMemoryModule(model)
        
        # Get state dict
        state_dict = wrapped.state_dict()
        
        # Create new instance and load
        new_model = nn.Linear(10, 10)
        new_wrapped = DynamicMemoryModule(new_model)
        new_wrapped.load_state_dict(state_dict)
        
        # Check parameters match
        for p1, p2 in zip(wrapped.parameters(), new_wrapped.parameters()):
            assert torch.allclose(p1, p2)


class TestMemoryConfig:
    """Test configuration options."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryConfig()
        
        assert config.enable_profiling == True
        assert 0 < config.checkpoint_threshold < 1
        assert 0 < config.offload_threshold < 1
        assert config.checkpoint_threshold < config.offload_threshold
        assert config.prefetch_size > 0
        assert config.min_gpu_memory_mb > 0
        assert config.profile_interval > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = MemoryConfig(
            enable_profiling=False,
            checkpoint_threshold=0.5,
            offload_threshold=0.7,
            prefetch_size=8,
            min_gpu_memory_mb=2048,
            profile_interval=20
        )
        
        assert config.enable_profiling == False
        assert config.checkpoint_threshold == 0.5
        assert config.offload_threshold == 0.7
        assert config.prefetch_size == 8
        assert config.min_gpu_memory_mb == 2048
        assert config.profile_interval == 20


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_transformer_model(self):
        """Test with a transformer model."""
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4),
            num_layers=2
        )
        wrapped = DynamicMemoryModule(model)
        
        # Test inference
        x = torch.randn(10, 32, 128)  # seq_len, batch, d_model
        output = wrapped(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_training_loop(self):
        """Test complete training loop."""
        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        wrapped = DynamicMemoryModule(model)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training step
        for _ in range(5):
            x = torch.randn(16, 20)
            y = torch.randn(16, 10)
            
            output = wrapped(x)
            loss = criterion(output, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Check training occurred
        stats = wrapped.get_memory_stats()
        assert stats["peak_memory_mb"] >= 0
    
    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create a model that uses significant memory
        model = nn.Sequential(*[nn.Linear(1000, 1000) for _ in range(10)])
        
        config = MemoryConfig(
            checkpoint_threshold=0.5,
            offload_threshold=0.7
        )
        wrapped = DynamicMemoryModule(model, config)
        
        # Should select appropriate strategy
        assert wrapped.strategy != ExecutionStrategy.STANDARD
        
        # Run forward pass
        x = torch.randn(32, 1000)
        output = wrapped(x)
        
        assert output.shape == (32, 1000)


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after each test."""
    yield
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Force garbage collection
    gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])