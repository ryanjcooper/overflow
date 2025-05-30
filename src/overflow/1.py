"""
Practical example demonstrating the PyTorch Dynamic Scaling Framework
This example shows how to use the framework to run models larger than GPU memory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from typing import Optional

# Import our framework (in practice, this would be: from pytorch_dynamic_scaling import DynamicMemoryModule)
# For this example, assume the framework is available


class SimpleUsageExample:
    """Demonstrates the simplicity of using the framework."""
    
    @staticmethod
    def standard_pytorch():
        """Standard PyTorch code that might OOM on large models."""
        # Create a large model
        model = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
        )
        
        # This might fail with OOM on smaller GPUs
        model = model.cuda()
        x = torch.randn(32, 10000).cuda()
        output = model(x)
        return output
    
    @staticmethod
    def with_dynamic_scaling():
        """Same code but with automatic memory management."""
        # Create the same large model
        model = nn.Sequential(
            nn.Linear(10000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 5000),
            nn.ReLU(),
            nn.Linear(5000, 1000),
        )
        
        # Wrap with DynamicMemoryModule - that's it!
        model = DynamicMemoryModule(model)
        
        # Use exactly like normal PyTorch
        x = torch.randn(32, 10000)
        output = model(x)
        return output


class TransformerExample:
    """Example with a large transformer model."""
    
    def __init__(self, model_size: str = "large"):
        """
        Initialize transformer example.
        
        Args:
            model_size: "small", "large", or "xlarge"
        """
        self.model_size = model_size
        self.model = self._create_model()
        
    def _create_model(self) -> nn.Module:
        """Create transformer model based on size."""
        configs = {
            "small": {"layers": 12, "d_model": 768, "heads": 12, "vocab": 30000},
            "large": {"layers": 24, "d_model": 1024, "heads": 16, "vocab": 50000},
            "xlarge": {"layers": 48, "d_model": 1280, "heads": 20, "vocab": 50000},
        }
        
        config = configs[self.model_size]
        
        class TransformerLM(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=d_model * 4, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.ln = nn.LayerNorm(d_model)
                self.output = nn.Linear(d_model, vocab_size)
                
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x)
                x = x + self.pos_encoding[:, :seq_len, :]
                x = self.transformer(x)
                x = self.ln(x)
                x = self.output(x)
                return x
        
        return TransformerLM(**config)
    
    def run_inference(self, use_wrapper: bool = True):
        """Run inference with or without the wrapper."""
        print(f"\n{'='*50}")
        print(f"Running {self.model_size} transformer model")
        print(f"Wrapper enabled: {use_wrapper}")
        print(f"{'='*50}\n")
        
        # Optionally wrap the model
        if use_wrapper:
            model = DynamicMemoryModule(self.model)
            print(f"Execution strategy: {model.strategy.value}")
            print(f"Estimated model size: {model._estimate_model_size() / 1024**3:.2f} GB")
        else:
            model = self.model
            # Calculate size manually
            total_params = sum(p.numel() for p in model.parameters())
            model_size_gb = (total_params * 4) / 1024**3  # Assuming fp32
            print(f"Model parameters: {total_params:,}")
            print(f"Estimated model size: {model_size_gb:.2f} GB")
        
        # Create sample input
        batch_size = 4
        seq_length = 128
        vocab_size = 50000 if self.model_size != "small" else 30000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        
        # Move to device (wrapper handles this automatically)
        if use_wrapper:
            device = model.device_manager.primary_device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
        
        input_ids = input_ids.to(device)
        
        # Measure inference time
        print("\nRunning inference...")
        start_time = time.time()
        
        try:
            with torch.no_grad():
                output = model(input_ids)
            
            inference_time = time.time() - start_time
            print(f"✓ Inference successful!")
            print(f"  Output shape: {output.shape}")
            print(f"  Inference time: {inference_time:.2f}s")
            
            # Get memory statistics if using wrapper
            if use_wrapper:
                stats = model.get_memory_stats()
                print(f"\nMemory Statistics:")
                print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB")
                if stats['swap_stats']:
                    print(f"  Swaps to CPU: {stats['swap_stats']['swaps_out']}")
                    print(f"  Swaps from CPU: {stats['swap_stats']['swaps_in']}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"✗ Out of memory error!")
                print(f"  This model is too large for available GPU memory without the wrapper.")
            else:
                raise e
    
    def run_training_loop(self, num_steps: int = 10):
        """Demonstrate training with the wrapper."""
        print(f"\n{'='*50}")
        print(f"Training {self.model_size} transformer model")
        print(f"{'='*50}\n")
        
        # Wrap model
        model = DynamicMemoryModule(self.model)
        device = model.device_manager.primary_device
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy dataset
        vocab_size = 50000 if self.model_size != "small" else 30000
        dataset = TensorDataset(
            torch.randint(0, vocab_size, (100, 128)),  # inputs
            torch.randint(0, vocab_size, (100, 128))   # targets
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        print(f"Starting training with strategy: {model.strategy.value}")
        
        # Training loop
        model.train()
        total_loss = 0
        
        for step, (inputs, targets) in enumerate(dataloader):
            if step >= num_steps:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            start_time = time.time()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step_time = time.time() - start_time
            total_loss += loss.item()
            
            if step % 2 == 0:
                print(f"Step {step}: loss = {loss.item():.4f}, time = {step_time:.2f}s")
        
        # Final statistics
        print(f"\nTraining Summary:")
        print(f"  Average loss: {total_loss / num_steps:.4f}")
        
        stats = model.get_memory_stats()
        print(f"  Peak memory: {stats['peak_memory_mb']:.2f} MB")
        print(f"  Strategy used: {stats['strategy']}")


class ComparisonDemo:
    """Demonstrates the framework with different model sizes."""
    
    @staticmethod
    def run_all_demos():
        """Run demonstrations with different model sizes."""
        print("PyTorch Dynamic Scaling Framework Demo")
        print("=====================================\n")
        
        # 1. Simple usage example
        print("1. Simple Usage Example")
        print("-----------------------")
        print("Standard PyTorch:")
        print("```python")
        print("model = create_large_model()")
        print("model = model.cuda()  # Might OOM!")
        print("output = model(input)")
        print("```")
        
        print("\nWith DynamicMemoryModule:")
        print("```python")
        print("model = create_large_model()")
        print("model = DynamicMemoryModule(model)  # Automatic memory management")
        print("output = model(input)  # Just works!")
        print("```")
        
        # 2. Run inference examples
        print("\n\n2. Inference Examples")
        print("---------------------")
        
        # Small model - should work without wrapper
        small_example = TransformerExample("small")
        small_example.run_inference(use_wrapper=False)
        small_example.run_inference(use_wrapper=True)
        
        # Large model - might need wrapper depending on GPU
        large_example = TransformerExample("large")
        large_example.run_inference(use_wrapper=True)
        
        # 3. Training example
        print("\n\n3. Training Example")
        print("-------------------")
        training_example = TransformerExample("large")
        training_example.run_training_loop(num_steps=5)
        
        print("\n\nDemo complete! The framework automatically:")
        print("✓ Detects available hardware")
        print("✓ Profiles memory usage") 
        print("✓ Selects optimal execution strategy")
        print("✓ Handles models larger than GPU memory")
        print("✓ Maintains standard PyTorch API")


# Advanced features demonstration
class AdvancedFeatures:
    """Demonstrates advanced features of the framework."""
    
    @staticmethod
    def custom_config_example():
        """Show how to use custom configuration."""
        from pytorch_dynamic_scaling import MemoryConfig
        
        # Custom configuration
        config = MemoryConfig(
            checkpoint_threshold=0.7,  # Start checkpointing earlier
            offload_threshold=0.85,    # Start offloading earlier
            prefetch_size=4,          # Prefetch more blocks
            profile_interval=5        # Profile more frequently
        )
        
        # Create model with custom config
        model = create_large_model()
        model = DynamicMemoryModule(model, config=config)
        
        return model
    
    @staticmethod
    def mixed_precision_example():
        """Demonstrate compatibility with mixed precision training."""
        from torch.cuda.amp import autocast, GradScaler
        
        model = create_large_model()
        model = DynamicMemoryModule(model)
        
        optimizer = optim.Adam(model.parameters())
        scaler = GradScaler()
        
        # Training step with mixed precision
        inputs = torch.randn(4, 128)
        
        with autocast():
            outputs = model(inputs)
            loss = outputs.mean()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("Mixed precision training works seamlessly!")
    
    @staticmethod
    def multi_gpu_example():
        """Demonstrate multi-GPU model parallelism."""
        if torch.cuda.device_count() < 2:
            print("Multi-GPU example requires 2+ GPUs")
            return
        
        # Create a very large model
        model = nn.Sequential(
            nn.Linear(10000, 20000),
            nn.ReLU(),
            nn.Linear(20000, 20000),
            nn.ReLU(),
            nn.Linear(20000, 10000),
        )
        
        # Wrap with framework - automatically uses model parallelism
        model = DynamicMemoryModule(model)
        
        print(f"Model distributed across {torch.cuda.device_count()} GPUs")
        print(f"Using strategy: {model.strategy.value}")


def create_large_model() -> nn.Module:
    """Helper function to create a large model for examples."""
    return nn.Sequential(
        nn.Linear(10000, 5000),
        nn.ReLU(),
        nn.Linear(5000, 5000),
        nn.ReLU(),
        nn.Linear(5000, 1000),
    )


if __name__ == "__main__":
    # Run the demonstration
    ComparisonDemo.run_all_demos()