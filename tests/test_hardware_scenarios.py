# tests/test_hardware_scenarios.py
"""
Hardware scenario tests for the Overflow framework.
These tests verify correct behavior across different hardware configurations.
"""

import pytest
import torch
import torch.nn as nn
import psutil
import gc
from unittest.mock import patch, MagicMock
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


class TestHardwareScenarios:
    """Test suite for different hardware scenarios."""
    
    def setup_method(self):
        """Setup for each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def teardown_method(self):
        """Cleanup after each test."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
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
        
        # Create a transformer model with appropriate size
        # Approximate parameter count for transformer layer:
        # - Self attention: d_model^2 * 4 (Q,K,V,O projections)
        # - FFN: d_model * ff_dim * 2 + ff_dim * d_model
        # - Layer norms: d_model * 4
        # Total per layer ≈ d_model^2 * 4 + d_model * ff_dim * 3
        
        d_model = 1024
        nhead = 16
        ff_dim = 4096
        
        # More accurate parameter count per layer
        params_per_layer = (
            d_model * d_model * 4 +  # Self-attention projections
            d_model * ff_dim * 2 +   # FFN layers
            d_model * 4              # Layer norms and biases
        )
        
        num_layers = max(1, int(target_params / params_per_layer))
        
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ff_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_small_model(self):
        """Test small model that fits comfortably on single GPU."""
        # Mock single GPU scenario even if we have multiple
        with patch('torch.cuda.device_count', return_value=1):
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Create a truly small model (10% of GPU memory)
            model = self.create_model_of_relative_size(0.1, gpu_memory_gb)
            
            wrapped = DynamicMemoryModule(model)
            
            # Small model should use standard strategy
            assert wrapped.strategy in [ExecutionStrategy.STANDARD, ExecutionStrategy.GRADIENT_CHECKPOINT]
            
            # Test forward pass
            x = torch.randn(4, 128, 1024)
            device = next(wrapped.parameters()).device
            x = x.to(device)
            
            output = wrapped(x)
            assert output.shape == (4, 128, 1024)
            
            # Cleanup
            del model, wrapped, output, x
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_medium_model(self):
        """Test model that might require gradient checkpointing on single GPU."""
        # Mock single GPU scenario
        with patch('torch.cuda.device_count', return_value=1):
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Create medium model (60% of GPU memory)
            model = self.create_model_of_relative_size(0.6, gpu_memory_gb)
            
            wrapped = DynamicMemoryModule(model)
            
            # Should likely use gradient checkpointing
            assert wrapped.strategy in [ExecutionStrategy.GRADIENT_CHECKPOINT, ExecutionStrategy.STANDARD, ExecutionStrategy.CPU_OFFLOAD]
            
            # Test forward pass with smaller batch
            x = torch.randn(2, 64, 1024)
            device = next(wrapped.parameters()).device
            x = x.to(device)
            
            output = wrapped(x)
            assert output.shape == (2, 64, 1024)
            
            # Cleanup
            del model, wrapped, output, x
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_gpu_large_model(self):
        """Test model that requires CPU offloading on single GPU."""
        # First cleanup any existing GPU memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available GPU memory before test
        free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3
        if free_memory < 5:  # Less than 5GB free
            pytest.skip(f"Insufficient GPU memory for test (only {free_memory:.1f}GB free)")
        
        # Mock single GPU scenario
        with patch('torch.cuda.device_count', return_value=1):
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            # Skip if system doesn't have enough RAM
            cpu_memory_gb = psutil.virtual_memory().total / 1024**3
            required_memory = gpu_memory_gb * 1.5
            if cpu_memory_gb < required_memory * 1.2:
                pytest.skip(f"Insufficient RAM for CPU offloading test")
            
            # Create model larger than GPU memory
            model = self.create_model_of_relative_size(1.2, gpu_memory_gb)
            
            wrapped = DynamicMemoryModule(model)
            
            # Should use CPU offloading
            assert wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD
            
            # Test forward pass with small batch
            x = torch.randn(1, 32, 1024)
            output = wrapped(x)
            assert output.shape == (1, 32, 1024)
            
            # Verify CPU offloading happened
            stats = wrapped.get_memory_stats()
            assert stats['swap_stats']['swaps_out'] > 0
            
            # Cleanup
            del model, wrapped, output, x
            torch.cuda.empty_cache()
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    def test_multi_gpu_data_parallel(self):
        """Test data parallel strategy on multiple GPUs."""
        # First cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available GPU memory
        free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3
        if free_memory < 5:  # Less than 5GB free
            pytest.skip(f"Insufficient GPU memory for test (only {free_memory:.1f}GB free)")
        
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Very small model that fits easily on one GPU
        model = self.create_model_of_relative_size(0.1, gpu_memory_gb)
        
        # Force data parallel
        config = MemoryConfig(prefer_data_parallel=True, data_parallel_threshold=0.5)
        wrapped = DynamicMemoryModule(model, config=config)
        
        # Should use data parallel for small model on multi-GPU
        assert wrapped.strategy == ExecutionStrategy.DATA_PARALLEL
        
        # Test with larger batch that benefits from data parallel
        x = torch.randn(32, 128, 1024)
        output = wrapped(x)
        assert output.shape == (32, 128, 1024)
        
        # Cleanup
        del model, wrapped, output, x
        torch.cuda.empty_cache()
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires multiple GPUs")
    def test_multi_gpu_model_parallel(self):
        """Test model parallel strategy on multiple GPUs."""
        # First cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Check available GPU memory
        free_memory = torch.cuda.mem_get_info(0)[0] / 1024**3
        if free_memory < 5:  # Less than 5GB free
            pytest.skip(f"Insufficient GPU memory for test (only {free_memory:.1f}GB free)")
        
        single_gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        total_gpu_memory_gb = single_gpu_memory_gb * torch.cuda.device_count()
        
        # Model larger than single GPU but fits across all GPUs
        # Use 1.5x single GPU memory but less than total
        target_size = min(single_gpu_memory_gb * 1.5, total_gpu_memory_gb * 0.8)
        model = self.create_model_of_relative_size(target_size / single_gpu_memory_gb, single_gpu_memory_gb)
        
        wrapped = DynamicMemoryModule(model)
        
        # Should use model parallel or CPU offload depending on exact sizes
        assert wrapped.strategy in [ExecutionStrategy.MODEL_PARALLEL, ExecutionStrategy.CPU_OFFLOAD]
        
        # Test forward pass
        x = torch.randn(2, 64, 1024)
        output = wrapped(x)
        assert output.shape == (2, 64, 1024)
        
        # Cleanup
        del model, wrapped, output, x
        torch.cuda.empty_cache()
    
    def test_cpu_only_execution(self):
        """Test execution on CPU-only systems."""
        # Create a mock that simulates no CUDA
        with patch('torch.cuda.is_available', return_value=False):
            with patch('overflow.device_manager.torch.cuda.is_available', return_value=False):
                with patch('overflow.module.torch.cuda.is_available', return_value=False):
                    model = nn.Linear(1000, 1000)
                    wrapped = DynamicMemoryModule(model)
                    
                    assert wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD
                    
                    # Test forward pass
                    x = torch.randn(10, 1000)
                    output = wrapped(x)
                    assert output.shape == (10, 1000)
                    assert output.device.type == 'cpu'
    
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
        with patch.object(DynamicMemoryModule, '_determine_strategy', return_value=strategy):
            wrapped = DynamicMemoryModule(model)
            assert wrapped.strategy == strategy
            
            # Test forward pass works with forced strategy
            x = torch.randn(8, 512)
            if strategy != ExecutionStrategy.CPU_OFFLOAD and torch.cuda.is_available():
                device = next(wrapped.parameters()).device
                x = x.to(device)
            
            output = wrapped(x)
            assert output.shape == (8, 512)
    
    def test_low_memory_system(self):
        """Test behavior on system with very limited memory."""
        model = nn.Sequential(*[nn.Linear(100, 100) for _ in range(5)])
        
        # Mock very limited memory - properly mock a CPU-only system
        with patch('overflow.device_manager.torch.cuda.is_available', return_value=False):
            with patch('overflow.device_manager.torch.cuda.device_count', return_value=0):
                with patch('overflow.module.torch.cuda.is_available', return_value=False):
                    # Create a fresh DeviceManager instance for this test
                    from overflow.device_manager import DeviceManager
                    
                    # Mock the DeviceManager to return CPU-only configuration
                    original_detect = DeviceManager._detect_devices
                    
                    def mock_detect_devices(self):
                        """Mock device detection to return only CPU."""
                        import psutil
                        from overflow.config import DeviceInfo
                        
                        vm = psutil.virtual_memory()
                        return [DeviceInfo(
                            device_id=0,
                            device_type='cpu',
                            total_memory=vm.total,
                            available_memory=vm.available,
                            device_name='CPU'
                        )]
                    
                    # Patch the method
                    DeviceManager._detect_devices = mock_detect_devices
                    
                    try:
                        wrapped = DynamicMemoryModule(model)
                        
                        # Should use CPU offload
                        assert wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD
                        
                        # Test with small batch
                        x = torch.randn(2, 100)
                        output = wrapped(x)
                        assert output.shape == (2, 100)
                    finally:
                        # Restore original method
                        DeviceManager._detect_devices = original_detect
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_memory_pressure_adaptation(self):
        """Test adaptation to memory pressure during execution."""
        # Create model that uses significant memory
        model = nn.Sequential(*[nn.Linear(2000, 2000) for _ in range(5)])
        
        config = MemoryConfig(
            checkpoint_threshold=0.7,
            offload_threshold=0.85,
            enable_profiling=True
        )
        
        wrapped = DynamicMemoryModule(model, config=config)
        
        # Run multiple forward passes to trigger memory profiling
        device = next(wrapped.parameters()).device
        
        for i in range(3):
            x = torch.randn(16, 2000).to(device)
            output = wrapped(x)
            
            # Check memory stats
            stats = wrapped.get_memory_stats()
            if stats['peak_memory_mb'] > 0:
                print(f"Pass {i+1}: Peak memory = {stats['peak_memory_mb']:.1f} MB")
        
        # Should complete without OOM
        assert output.shape == (16, 2000)
        
        # Cleanup
        del model, wrapped, output, x
        torch.cuda.empty_cache()