# tests/test_integration.py
"""
Integration tests for complete workflows in the Overflow framework.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import gc
from unittest.mock import patch, MagicMock
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


class TestIntegration:
    """Integration tests for complete workflows."""
    
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
    
    def test_transformer_model(self):
        """Test with a transformer model."""
        model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True),
            num_layers=2
        )
        wrapped = DynamicMemoryModule(model)
        
        # Test inference
        x = torch.randn(32, 10, 128)  # batch, seq_len, d_model
        # Move to correct device
        device = next(wrapped.parameters()).device
        x = x.to(device)
        
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
        
        # Get device
        device = next(wrapped.parameters()).device
        
        # Training step
        for _ in range(5):
            x = torch.randn(16, 20).to(device)
            y = torch.randn(16, 10).to(device)
            
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
        
        # Mock limited GPU memory
        with patch('overflow.module.DeviceManager') as MockDeviceManager:
            mock_manager = MagicMock()
            mock_manager.get_total_gpu_memory.return_value = 50 * 1024**2  # 50MB GPU
            mock_manager.get_available_gpu_memory.return_value = 40 * 1024**2
            mock_manager.primary_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            mock_manager.devices = [
                MagicMock(device_type='cuda', device_id=0, total_memory=50*1024**2, available_memory=40*1024**2, device_name='Mock GPU')
            ] if torch.cuda.is_available() else [
                MagicMock(device_type='cpu', device_id=0, total_memory=8*1024**3, available_memory=4*1024**3, device_name='CPU')
            ]
            MockDeviceManager.return_value = mock_manager
            
            wrapped = DynamicMemoryModule(model, config)
            
            # Should select appropriate strategy (model is ~40MB, GPU is 50MB)
            assert wrapped.strategy != ExecutionStrategy.STANDARD
            
            # Run forward pass
            x = torch.randn(32, 1000)
            # Move to correct device
            if wrapped.strategy == ExecutionStrategy.CPU_OFFLOAD:
                x = x.cpu()  # CPU offload starts with CPU tensors
            else:
                device = mock_manager.primary_device
                x = x.to(device)
            
            output = wrapped(x)
            
            assert output.shape == (32, 1000)
    
    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        if not torch.cuda.is_available():
            pytest.skip("Mixed precision requires CUDA")
        
        from torch.amp import autocast, GradScaler
        
        model = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        wrapped = DynamicMemoryModule(model)
        optimizer = optim.Adam(wrapped.parameters())
        scaler = GradScaler('cuda')
        
        # Get the device where the model is
        device = next(wrapped.parameters()).device
        
        # Mixed precision forward/backward
        x = torch.randn(16, 128).to(device)
        target = torch.randn(16, 128).to(device)
        
        with autocast('cuda'):
            output = wrapped(x)
            # Ensure output is on the same device as target
            if output.device != target.device:
                output = output.to(target.device)
            loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Should complete without errors
        assert not torch.isnan(loss).any()
    
    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        # Create and train model briefly
        model = nn.Linear(64, 32)
        wrapped = DynamicMemoryModule(model)
        optimizer = optim.SGD(wrapped.parameters(), lr=0.1)
        
        # Do one training step
        x = torch.randn(8, 64)
        device = next(wrapped.parameters()).device
        x = x.to(device)
        
        output = wrapped(x)
        loss = output.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Save checkpoint
        checkpoint = {
            'model_state': wrapped.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'strategy': wrapped.strategy.value
        }
        
        # Create new model and load checkpoint
        new_model = nn.Linear(64, 32)
        new_wrapped = DynamicMemoryModule(new_model)
        new_optimizer = optim.SGD(new_wrapped.parameters(), lr=0.1)
        
        new_wrapped.load_state_dict(checkpoint['model_state'])
        new_optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Verify parameters match
        for p1, p2 in zip(wrapped.parameters(), new_wrapped.parameters()):
            # Move to same device (CPU) for comparison
            assert torch.allclose(p1.cpu(), p2.cpu())
    
    def test_data_loader_integration(self):
        """Test integration with PyTorch DataLoader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dataset
        X = torch.randn(100, 32)
        y = torch.randn(100, 16)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
        
        # Create model
        model = nn.Linear(32, 16)
        wrapped = DynamicMemoryModule(model)
        device = next(wrapped.parameters()).device
        
        # Iterate through dataloader
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = wrapped(batch_x)
            assert output.shape == batch_y.shape
            
            # Just test first batch
            break
    
    def test_multi_input_model(self):
        """Test model with multiple inputs."""
        class MultiInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(20, 50)
                self.fc2 = nn.Linear(30, 50)
                self.fc3 = nn.Linear(100, 10)
            
            def forward(self, x1, x2):
                out1 = self.fc1(x1)
                out2 = self.fc2(x2)
                combined = torch.cat([out1, out2], dim=1)
                return self.fc3(combined)
        
        model = MultiInputModel()
        wrapped = DynamicMemoryModule(model)
        
        # Test with multiple inputs
        x1 = torch.randn(5, 20)
        x2 = torch.randn(5, 30)
        
        device = next(wrapped.parameters()).device
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        output = wrapped(x1, x2)
        assert output.shape == (5, 10)
    
    def test_custom_forward_with_kwargs(self):
        """Test model with custom forward that uses kwargs."""
        class CustomModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
            
            def forward(self, x, mask=None, training=True):
                out = self.fc(x)
                if mask is not None:
                    out = out * mask
                if training:
                    out = torch.dropout(out, 0.5, self.training)
                return out
        
        model = CustomModel()
        wrapped = DynamicMemoryModule(model)
        
        x = torch.randn(5, 10)
        mask = torch.ones(5, 10)
        
        device = next(wrapped.parameters()).device
        x = x.to(device)
        mask = mask.to(device)
        
        # Test with kwargs
        output = wrapped(x, mask=mask, training=False)
        assert output.shape == (5, 10)
    
    def test_recursive_model_structure(self):
        """Test deeply nested model structures."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Sequential(
                    nn.Linear(20, 30),
                    nn.ReLU()
                )
            ),
            nn.Linear(30, 10)
        )
        
        wrapped = DynamicMemoryModule(model)
        
        x = torch.randn(5, 10)
        device = next(wrapped.parameters()).device
        x = x.to(device)
        
        output = wrapped(x)
        assert output.shape == (5, 10)