# examples/basic_example.py
"""
Basic example of using Overflow to run large models on limited GPU memory
"""

import torch
import torch.nn as nn
from overflow import DynamicMemoryModule, MemoryConfig


def create_large_transformer():
    """Create a large transformer model that might not fit in GPU memory."""
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=24  # This creates a ~1.5GB model
    )


def standard_pytorch_example():
    """Standard PyTorch approach - might fail with OOM."""
    print("=== Standard PyTorch Example ===")
    print("Creating large transformer model...")
    
    model = create_large_transformer()
    
    # This might fail on GPUs with limited memory
    try:
        model = model.cuda()
        x = torch.randn(16, 512, 1024).cuda()  # batch_size=16, seq_len=512, d_model=1024
        
        print("Running forward pass...")
        output = model(x)
        print(f"✓ Success! Output shape: {output.shape}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("✗ Failed with out of memory error!")
            print("  Your GPU doesn't have enough memory for this model.")
        else:
            raise


def overflow_example():
    """Using Overflow - automatically handles memory constraints."""
    print("\n=== Overflow Example ===")
    print("Creating the same large transformer model...")
    
    model = create_large_transformer()
    
    # Wrap with Overflow - that's it!
    print("Wrapping with Overflow...")
    model = DynamicMemoryModule(model)
    
    print(f"✓ Overflow automatically selected strategy: {model.strategy.value}")
    print(f"  Model size: {model._estimate_model_size() / 1024**3:.2f} GB")
    
    # Create input
    x = torch.randn(16, 512, 1024)  # Same input size
    
    # Run forward pass - Overflow handles device placement
    print("Running forward pass...")
    output = model(x)
    print(f"✓ Success! Output shape: {output.shape}")
    
    # Get memory statistics
    stats = model.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"  Peak memory used: {stats['peak_memory_mb']:.1f} MB")
    print(f"  Execution strategy: {stats['strategy']}")
    
    if stats['swap_stats'].get('swaps_out', 0) > 0:
        print(f"  CPU offloading used: {stats['swap_stats']['swaps_out']} swaps")


def custom_config_example():
    """Example with custom memory configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Configure memory management behavior
    config = MemoryConfig(
        checkpoint_threshold=0.7,    # Start checkpointing at 70% memory
        offload_threshold=0.85,      # Start CPU offloading at 85% memory
        prefetch_size=4,            # Prefetch 4 blocks ahead
        min_gpu_memory_mb=2048      # Keep 2GB GPU memory free
    )
    
    model = create_large_transformer()
    model = DynamicMemoryModule(model, config=config)
    
    print(f"Using custom configuration:")
    print(f"  Checkpoint threshold: {config.checkpoint_threshold:.0%}")
    print(f"  Offload threshold: {config.offload_threshold:.0%}")
    print(f"  Strategy selected: {model.strategy.value}")


def training_example():
    """Example of training with Overflow."""
    print("\n=== Training Example ===")
    
    # Create a smaller model for training demo
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024)
    )
    
    # Wrap with Overflow
    model = DynamicMemoryModule(model)
    
    # Standard PyTorch training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"Training with strategy: {model.strategy.value}")
    
    # Training loop
    for step in range(3):
        # Create dummy data
        x = torch.randn(32, 1024)
        target = torch.randn(32, 1024)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Step {step}: loss = {loss.item():.4f}")
    
    print("✓ Training completed successfully!")


def main():
    """Run all examples."""
    print("Overflow: When your model overflows the GPU")
    print("=" * 50)
    
    # Check available hardware
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {device_name}")
        print(f"Total memory: {total_memory:.1f} GB")
    else:
        print("No GPU detected - using CPU")
    
    print("\n")
    
    # Run examples
    try:
        # This might fail without Overflow
        standard_pytorch_example()
    except:
        pass
    
    # This should always work with Overflow
    overflow_example()
    
    # Show custom configuration
    custom_config_example()
    
    # Show training
    training_example()
    
    print("\n" + "=" * 50)
    print("Key Benefits of Overflow:")
    print("✓ Drop-in replacement - just wrap your model")
    print("✓ Automatic strategy selection")
    print("✓ Handles models larger than GPU memory")
    print("✓ Full PyTorch compatibility")
    print("✓ Built-in memory profiling")


if __name__ == "__main__":
    main()