# tests/test_block_swap_manager.py
"""
Unit tests for BlockSwapManager class.
"""

import pytest
import torch
import gc
from overflow import BlockSwapManager, MemoryConfig


class TestBlockSwapManager:
    """Test block swapping functionality."""
    
    def setup_method(self):
        """Setup for each test."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def teardown_method(self):
        """Cleanup after each test."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
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
    
    def test_swap_nonexistent_tensor(self):
        """Test swapping back a tensor that doesn't exist."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        result = manager.swap_from_cpu("nonexistent", device)
        
        assert result is None
    
    def test_pinned_memory(self):
        """Test that swapped tensors use pinned memory for faster transfers."""
        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA")
        
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        # Create GPU tensor
        tensor = torch.randn(1000, 1000).cuda()
        
        # Swap to CPU
        manager.swap_to_cpu("test_tensor", tensor)
        
        # Check that the CPU tensor is pinned
        cpu_tensor = manager.cpu_cache["test_tensor"]
        assert cpu_tensor.is_pinned()
    
    def test_swap_stats_tracking(self):
        """Test accurate tracking of swap statistics."""
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        # Create tensors of known size
        tensor1 = torch.randn(100, 100)  # 100*100*4 bytes = 40,000 bytes
        tensor2 = torch.randn(200, 200)  # 200*200*4 bytes = 160,000 bytes
        
        # Swap to CPU
        manager.swap_to_cpu("tensor1", tensor1)
        manager.swap_to_cpu("tensor2", tensor2)
        
        assert manager.swap_stats["swaps_out"] == 2
        assert manager.swap_stats["total_bytes"] == 200000  # 40,000 + 160,000
        
        # Swap back
        device = torch.device("cpu")
        manager.swap_from_cpu("tensor1", device)
        
        assert manager.swap_stats["swaps_in"] == 1
    
    def test_thread_safety(self):
        """Test thread safety of swap operations."""
        import threading
        
        config = MemoryConfig()
        manager = BlockSwapManager(config)
        
        def swap_tensors(start_idx, count):
            for i in range(count):
                tensor = torch.randn(50, 50)
                manager.swap_to_cpu(f"tensor_{start_idx + i}", tensor)
        
        # Create multiple threads
        threads = []
        for i in range(4):
            t = threading.Thread(target=swap_tensors, args=(i * 25, 25))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check that all swaps were recorded
        assert manager.swap_stats["swaps_out"] == 100
        assert len(manager.cpu_cache) == 100
    
    def test_prefetch_scheduling(self):
        """Test prefetch scheduling functionality."""
        config = MemoryConfig(prefetch_size=3)
        manager = BlockSwapManager(config)
        
        # Add some tensors to cache
        for i in range(5):
            tensor = torch.randn(10, 10)
            manager.swap_to_cpu(f"tensor_{i}", tensor)
        
        # Schedule prefetch
        names_to_prefetch = [f"tensor_{i}" for i in range(5)]
        manager.schedule_prefetch(names_to_prefetch)
        
        # Check that only prefetch_size items are queued
        queued_items = []
        while not manager.swap_queue.empty():
            queued_items.append(manager.swap_queue.get())
        
        assert len(queued_items) == config.prefetch_size