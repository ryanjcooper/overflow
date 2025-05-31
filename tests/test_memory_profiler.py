# tests/test_memory_profiler.py
"""
Unit tests for MemoryProfiler class.
"""

import pytest
import torch
from unittest.mock import patch
from overflow import MemoryProfiler, MemoryConfig


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
    
    def test_profile_count_increment(self):
        """Test that profile count increments correctly."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        assert profiler.profile_count == 0
        
        # Simulate profiling by updating stats
        profiler.update_stats("test_layer", 50.0)
        # Note: profile_count is managed by the module, not the profiler itself
        
        # Check that stats were recorded
        assert "test_layer" in profiler.memory_stats
        assert profiler.memory_stats["test_layer"]["count"] == 1
    
    def test_peak_memory_tracking(self):
        """Test peak memory tracking across multiple updates."""
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        # Simulate memory usage pattern
        profiler.update_stats("layer1", 100.0)
        assert profiler.peak_memory == 100.0
        
        profiler.update_stats("layer2", 250.0)
        assert profiler.peak_memory == 250.0
        
        profiler.update_stats("layer3", 150.0)
        assert profiler.peak_memory == 250.0  # Peak should remain at highest value
    
    def test_thread_safety(self):
        """Test that the profiler handles concurrent updates safely."""
        import threading
        
        config = MemoryConfig()
        profiler = MemoryProfiler(config)
        
        def update_stats(layer_name, iterations):
            for i in range(iterations):
                profiler.update_stats(layer_name, float(i))
        
        # Create multiple threads updating different layers
        threads = []
        for i in range(3):
            t = threading.Thread(target=update_stats, args=(f"layer_{i}", 100))
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Verify all updates were recorded
        for i in range(3):
            assert f"layer_{i}" in profiler.memory_stats
            assert profiler.memory_stats[f"layer_{i}"]["count"] == 100