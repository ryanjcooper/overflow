# tests/test_device_manager.py
"""
Unit tests for DeviceManager class.
"""

import pytest
import torch
from overflow import DeviceManager


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
    
    def test_device_info_structure(self):
        """Test DeviceInfo data structure."""
        manager = DeviceManager()
        
        for device in manager.devices:
            # Check required attributes
            assert hasattr(device, 'device_id')
            assert hasattr(device, 'device_type')
            assert hasattr(device, 'total_memory')
            assert hasattr(device, 'available_memory')
            assert hasattr(device, 'device_name')
            
            # Check value constraints
            assert device.device_id >= 0
            assert device.device_type in ['cuda', 'cpu']
            assert device.total_memory > 0
            assert device.available_memory >= 0
            assert device.available_memory <= device.total_memory
            assert isinstance(device.device_name, str)
    
    def test_multi_gpu_detection(self):
        """Test detection of multiple GPUs."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires multiple GPUs")
        
        manager = DeviceManager()
        cuda_devices = [d for d in manager.devices if d.device_type == "cuda"]
        
        # Check that all GPUs are detected
        assert len(cuda_devices) == torch.cuda.device_count()
        
        # Check that each GPU has unique device_id
        device_ids = [d.device_id for d in cuda_devices]
        assert len(device_ids) == len(set(device_ids))
        
        # Check that device IDs are sequential starting from 0
        assert sorted(device_ids) == list(range(len(device_ids)))
    
    def test_primary_device_has_most_memory(self):
        """Test that primary device selection prefers GPU with most available memory."""
        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA")
        
        manager = DeviceManager()
        cuda_devices = [d for d in manager.devices if d.device_type == "cuda"]
        
        if len(cuda_devices) > 1:
            # Find GPU with most available memory
            best_device = max(cuda_devices, key=lambda d: d.available_memory)
            
            # Primary device should be this GPU
            assert manager.primary_device.index == str(best_device.device_id)
    
    def test_cpu_always_detected(self):
        """Test that CPU is always detected as a device."""
        manager = DeviceManager()
        
        cpu_devices = [d for d in manager.devices if d.device_type == "cpu"]
        assert len(cpu_devices) == 1
        
        cpu_device = cpu_devices[0]
        assert cpu_device.device_id == 0
        assert cpu_device.device_name == "CPU"
        assert cpu_device.total_memory > 0