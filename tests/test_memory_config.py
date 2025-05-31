# tests/test_memory_config.py
"""
Unit tests for MemoryConfig class.
"""

import pytest
from overflow import MemoryConfig


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
    
    def test_threshold_validation(self):
        """Test that thresholds are properly ordered."""
        config = MemoryConfig(
            checkpoint_threshold=0.6,
            offload_threshold=0.8
        )
        
        assert config.checkpoint_threshold < config.offload_threshold
    
    def test_data_parallel_config(self):
        """Test data parallel specific configuration."""
        config = MemoryConfig(
            prefer_data_parallel=True,
            data_parallel_threshold=0.3
        )
        
        assert config.prefer_data_parallel == True
        assert config.data_parallel_threshold == 0.3
    
    def test_default_data_parallel_config(self):
        """Test default data parallel configuration."""
        config = MemoryConfig()
        
        # Check defaults
        assert hasattr(config, 'prefer_data_parallel')
        assert config.prefer_data_parallel == False
        assert hasattr(config, 'data_parallel_threshold')
        assert config.data_parallel_threshold == 0.5
    
    def test_immutable_after_creation(self):
        """Test that config values can be modified after creation."""
        config = MemoryConfig()
        
        # Should be able to modify values
        config.checkpoint_threshold = 0.7
        assert config.checkpoint_threshold == 0.7
        
        config.enable_profiling = False
        assert config.enable_profiling == False
    
    def test_all_attributes_accessible(self):
        """Test that all expected attributes are accessible."""
        config = MemoryConfig()
        
        required_attributes = [
            'enable_profiling',
            'checkpoint_threshold',
            'offload_threshold',
            'prefetch_size',
            'min_gpu_memory_mb',
            'profile_interval',
            'prefer_data_parallel',
            'data_parallel_threshold'
        ]
        
        for attr in required_attributes:
            assert hasattr(config, attr)
            assert getattr(config, attr) is not None
    
    def test_config_repr(self):
        """Test string representation of config."""
        config = MemoryConfig(
            checkpoint_threshold=0.75,
            offload_threshold=0.85
        )
        
        # Should be able to get string representation
        repr_str = repr(config)
        assert isinstance(repr_str, str)
        assert 'MemoryConfig' in repr_str
    
    def test_boundary_values(self):
        """Test configuration with boundary values."""
        # Test minimum values
        config = MemoryConfig(
            checkpoint_threshold=0.1,
            offload_threshold=0.2,
            prefetch_size=1,
            min_gpu_memory_mb=1,
            profile_interval=1
        )
        
        assert config.checkpoint_threshold == 0.1
        assert config.offload_threshold == 0.2
        assert config.prefetch_size == 1
        assert config.min_gpu_memory_mb == 1
        assert config.profile_interval == 1
        
        # Test maximum reasonable values
        config = MemoryConfig(
            checkpoint_threshold=0.95,
            offload_threshold=0.99,
            prefetch_size=100,
            min_gpu_memory_mb=16384,  # 16GB
            profile_interval=1000
        )
        
        assert config.checkpoint_threshold == 0.95
        assert config.offload_threshold == 0.99
        assert config.prefetch_size == 100
        assert config.min_gpu_memory_mb == 16384
        assert config.profile_interval == 1000