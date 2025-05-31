# tests/test_dynamic_memory_module.py
"""
Unit tests for DynamicMemoryModule class.
"""

import pytest
import torch
import torch.nn as nn
import gc
from unittest.mock import patch, MagicMock
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


class TestDynamicMemoryModule:
    """Test the main wrapper module."""
    
    def setup_method(self):
        """Setup for each test."""
        # Clear GPU cache before each test
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Force garbage collection and clear GPU cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def test_initialization(self):
        """Test wrapper initialization."""
        model = nn.Linear(100, 100)
        wrapped = DynamicMemoryModule(model)
        
        assert wrapped.wrapped_module is model
        assert wrapped.config is not None
        assert wrapped.strategy in ExecutionStrategy
        assert wrapped.memory_profiler is not None
        assert wrapped.device_manager is not None
    
    def test_strategy_selection_standard(self):
        """Test standard strategy selection for small models."""
        # Very small model should always use standard strategy
        small_model = nn.Linear(10, 10)
        wrapped_small = DynamicMemoryModule(small_model)
        assert wrapped_small.strategy == ExecutionStrategy.STANDARD
    
    def test_strategy_selection_large_model(self):
        """Test strategy selection for large models with mocked memory."""
        # Create a large model
        large_model = nn.Sequential(*[nn.Linear(5000, 5000) for _ in range(20)])
        
        # Mock the device manager to simulate limited GPU memory
        with patch('overflow.module.DeviceManager') as MockDeviceManager:
            mock_manager = MagicMock()
            mock_manager.get_total_gpu_memory.return_value = 1 * 1024**3  # 1GB GPU
            mock_manager.get_available_gpu_memory.return_value = 0.8 * 1024**3
            mock_manager.primary_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mock_manager.devices = [
                MagicMock(device_type='cuda', device_id=0, total_memory=1*1024**3, available_memory=0.8*1024**3, device_name='Mock GPU')
            ] if torch.cuda.is_available() else [
                MagicMock(device_type='cpu', device_id=0, total_memory=8*1024**3, available_memory=4*1024**3, device_name='CPU')
            ]
            MockDeviceManager.return_value = mock_manager
            
            wrapped_large = DynamicMemoryModule(large_model)
            
            # Model is ~1.86GB, GPU is 1GB, so it should NOT use standard strategy
            assert wrapped_large.strategy != ExecutionStrategy.STANDARD
            assert wrapped_large.strategy in [ExecutionStrategy.GRADIENT_CHECKPOINT, ExecutionStrategy.CPU_OFFLOAD]
    
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
        # Move input to same device as model
        if hasattr(wrapped.wrapped_module, 'weight'):
            x = x.to(wrapped.wrapped_module.weight.device)
        elif hasattr(wrapped.wrapped_module[0], 'weight'):
            x = x.to(wrapped.wrapped_module[0].weight.device)
        
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
        # Move to correct device
        device = next(wrapped.parameters()).device
        x = x.to(device)
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
        # Move to correct device
        device = next(wrapped.parameters()).device
        x = x.to(device)
        
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
    
    def test_training_mode_preservation(self):
        """Test that training mode is preserved from wrapped model."""
        # Test with model in training mode
        model_train = nn.Linear(10, 10)
        model_train.train()
        wrapped_train = DynamicMemoryModule(model_train)
        assert wrapped_train.training == True
        
        # Test with model in eval mode
        model_eval = nn.Linear(10, 10)
        model_eval.eval()
        wrapped_eval = DynamicMemoryModule(model_eval)
        assert wrapped_eval.training == False
    
    def test_device_placement(self):
        """Test that model is placed on correct device based on strategy."""
        model = nn.Linear(100, 100)
        
        # Test standard strategy
        with patch.object(DynamicMemoryModule, '_determine_strategy', return_value=ExecutionStrategy.STANDARD):
            wrapped = DynamicMemoryModule(model)
            if torch.cuda.is_available():
                assert next(wrapped.parameters()).is_cuda
            else:
                assert not next(wrapped.parameters()).is_cuda
        
        # Test CPU offload strategy
        with patch.object(DynamicMemoryModule, '_determine_strategy', return_value=ExecutionStrategy.CPU_OFFLOAD):
            wrapped = DynamicMemoryModule(model)
            assert not next(wrapped.parameters()).is_cuda
    
    def test_memory_profiling_disabled(self):
        """Test behavior when memory profiling is disabled."""
        model = nn.Linear(100, 100)
        config = MemoryConfig(enable_profiling=False)
        wrapped = DynamicMemoryModule(model, config)
        
        # Run forward pass
        x = torch.randn(10, 100)
        device = next(wrapped.parameters()).device
        x = x.to(device)
        _ = wrapped(x)
        
        # Stats should still be available but minimal
        stats = wrapped.get_memory_stats()
        assert stats is not None
        assert "strategy" in stats
    
    def test_parameter_access(self):
        """Test that parameters() and named_parameters() work correctly."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10)
        )
        wrapped = DynamicMemoryModule(model)
        
        # Count parameters
        param_count = sum(1 for _ in wrapped.parameters())
        assert param_count == 4  # 2 weights + 2 biases
        
        # Check named parameters
        param_names = [name for name, _ in wrapped.named_parameters()]
        assert len(param_names) == 4
        assert all('weight' in name or 'bias' in name for name in param_names)
    
    def test_module_list_support(self):
        """Test support for ModuleList."""
        model = nn.ModuleList([
            nn.Linear(10, 20),
            nn.Linear(20, 30),
            nn.Linear(30, 10)
        ])
        
        # Wrap in Sequential to make it callable
        sequential_model = nn.Sequential(*model)
        wrapped = DynamicMemoryModule(sequential_model)
        
        # Should initialize without errors
        assert wrapped.strategy in ExecutionStrategy
        
        # Test forward pass
        x = torch.randn(5, 10)
        device = next(wrapped.parameters()).device
        x = x.to(device)
        output = wrapped(x)
        assert output.shape == (5, 10)