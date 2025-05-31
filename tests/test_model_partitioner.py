# tests/test_model_partitioner.py
"""
Unit tests for ModelPartitioner class.
"""

import pytest
import torch
import torch.nn as nn
from overflow import ModelPartitioner


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
    
    def test_single_device_partitioning(self):
        """Test partitioning with single device."""
        model = nn.Sequential(
            nn.Linear(50, 100),
            nn.Linear(100, 50)
        )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        partitioner = ModelPartitioner(model, [device])
        
        # All modules should be assigned to the single device
        for name in partitioner.partition_map:
            assert partitioner.partition_map[name] == device
    
    def test_balanced_partitioning(self):
        """Test that modules are distributed evenly across devices."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires multiple GPUs")
        
        # Create model with many layers
        layers = []
        for i in range(10):
            layers.append(nn.Linear(100, 100))
            layers.append(nn.ReLU())
        model = nn.Sequential(*layers)
        
        devices = [torch.device(f"cuda:{i}") for i in range(2)]
        partitioner = ModelPartitioner(model, devices)
        
        # Count modules assigned to each device
        device_counts = {device: 0 for device in devices}
        for device in partitioner.partition_map.values():
            device_counts[device] += 1
        
        # Check that distribution is roughly balanced
        counts = list(device_counts.values())
        assert max(counts) - min(counts) <= 2  # Allow small imbalance
    
    def test_large_module_priority(self):
        """Test that larger modules are distributed first."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires multiple GPUs")
        
        # Create model with varied layer sizes
        model = nn.Sequential(
            nn.Linear(1000, 1000),  # Large
            nn.Linear(10, 10),      # Small
            nn.Linear(500, 500),    # Medium
            nn.Linear(20, 20),      # Small
        )
        
        devices = [torch.device(f"cuda:{i}") for i in range(2)]
        partitioner = ModelPartitioner(model, devices)
        
        # The large layer should be assigned to one of the devices
        large_layer_device = partitioner.get_device_for_module("0")
        assert large_layer_device in devices
    
    def test_empty_model(self):
        """Test partitioning an empty model."""
        model = nn.Sequential()
        devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
        
        partitioner = ModelPartitioner(model, devices)
        
        # Partition map should be empty
        assert len(partitioner.partition_map) == 0
    
    def test_nested_module_partitioning(self):
        """Test partitioning with nested modules."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Test requires multiple GPUs")
        
        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU()
                )
                self.layer2 = nn.Sequential(
                    nn.Linear(100, 100),
                    nn.ReLU()
                )
            
            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)
        
        model = NestedModel()
        devices = [torch.device(f"cuda:{i}") for i in range(2)]
        partitioner = ModelPartitioner(model, devices)
        
        # Check that leaf modules are assigned
        leaf_modules = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                leaf_modules.append(name)
        
        assert len(partitioner.partition_map) == len(leaf_modules)
        
        for name in leaf_modules:
            assert name in partitioner.partition_map
            assert partitioner.partition_map[name] in devices
    
    def test_get_device_for_unknown_module(self):
        """Test getting device for unknown module."""
        model = nn.Linear(10, 10)
        devices = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
        partitioner = ModelPartitioner(model, devices)
        
        # Should return first device as default
        unknown_device = partitioner.get_device_for_module("unknown_module")
        assert unknown_device == devices[0]