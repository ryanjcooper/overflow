# examples/basic_usage.py
"""
Basic usage example for the Overflow framework.
Shows how to get started with automatic memory management for large models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from overflow import DynamicMemoryModule, MemoryConfig


def simple_usage_example():
    """
    The simplest way to use Overflow - just wrap your model!
    """
    print("="*60)
    print("Simple Usage Example")
    print("="*60)
    
    # Create any PyTorch model
    model = nn.Sequential(
        nn.Linear(784, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    print("Original model created")
    
    # Wrap it with Overflow - that's it!
    model = DynamicMemoryModule(model)
    
    print(f"✓ Model wrapped with Overflow")
    print(f"  Selected strategy: {model.strategy.value}")
    
    # Use it exactly like a normal PyTorch model
    input_data = torch.randn(32, 784)
    output = model(input_data)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {output.shape}")


def large_model_example():
    """
    Example showing how Overflow handles models that might not fit in GPU memory.
    This adapts to your available hardware automatically.
    """
    print("\n" + "="*60)
    print("Large Model Example (Hardware Adaptive)")
    print("="*60)
    
    # Check available memory to create an appropriately sized model
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU detected: {torch.cuda.get_device_properties(0).name}")
        print(f"GPU memory: {gpu_memory_gb:.1f} GB")
        
        # Create a model that's challenging for your GPU
        if gpu_memory_gb > 16:
            # High-end GPU: Create a very large model
            num_layers = 48
            d_model = 2048
            print(f"Creating large model for high-end GPU...")
        elif gpu_memory_gb > 8:
            # Mid-range GPU: Create a large model
            num_layers = 24
            d_model = 1536
            print(f"Creating large model for mid-range GPU...")
        else:
            # Entry-level GPU: Create a medium model
            num_layers = 12
            d_model = 1024
            print(f"Creating medium model for entry-level GPU...")
    else:
        # CPU only: Create a smaller model
        num_layers = 6
        d_model = 768
        print("No GPU detected, creating CPU-appropriate model...")
    
    # Create transformer model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=16,
            dim_feedforward=d_model * 4,
            batch_first=True
        ),
        num_layers=num_layers
    )
    
    # Estimate model size
    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = (total_params * 4) / 1024**3
    print(f"Model parameters: {total_params:,}")
    print(f"Model size: {model_size_gb:.2f} GB")
    
    # Wrap with Overflow
    print("\nWrapping with Overflow...")
    model = DynamicMemoryModule(model)
    
    print(f"✓ Overflow automatically selected: {model.strategy.value}")
    
    # Explanation of what happened
    if model.strategy.value == "standard":
        print("  → Your GPU has plenty of memory for this model")
    elif model.strategy.value == "gradient_checkpoint":
        print("  → Using gradient checkpointing to fit in available memory")
    elif model.strategy.value == "cpu_offload":
        print("  → Model too large for GPU, using CPU offloading")
    elif model.strategy.value == "model_parallel":
        print("  → Distributing model across multiple GPUs")
    
    # Run inference
    print("\nRunning inference...")
    model.eval()
    
    # Use smaller batch size for larger models
    batch_size = 4 if model_size_gb > 10 else 8
    seq_length = 128
    
    with torch.no_grad():
        input_data = torch.randn(batch_size, seq_length, d_model)
        output = model(input_data)
    
    print(f"✓ Inference successful!")
    print(f"  Input shape: {input_data.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Show memory statistics
    stats = model.get_memory_stats()
    if stats['peak_memory_mb'] > 0:
        print(f"\nMemory statistics:")
        print(f"  Peak GPU memory used: {stats['peak_memory_mb']:.1f} MB")
        if stats['swap_stats'] and stats['swap_stats'].get('swaps_out', 0) > 0:
            print(f"  CPU offload swaps: {stats['swap_stats']['swaps_out']}")


def training_example():
    """
    Example showing how to train a model with Overflow.
    Works exactly like normal PyTorch training.
    """
    print("\n" + "="*60)
    print("Training Example")
    print("="*60)
    
    # Create a simple model for demonstration
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 10)
    )
    
    # Wrap with Overflow
    model = DynamicMemoryModule(model)
    print(f"Model wrapped with strategy: {model.strategy.value}")
    
    # Standard PyTorch training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Create some dummy training data
    batch_size = 32
    num_batches = 5
    
    print(f"\nTraining for {num_batches} batches...")
    model.train()
    
    for batch_idx in range(num_batches):
        # Dummy data
        inputs = torch.randn(batch_size, 128)
        labels = torch.randint(0, 10, (batch_size,))
        
        # Standard training loop
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {batch_idx + 1}/{num_batches}, Loss: {loss.item():.4f}")
    
    print("\n✓ Training completed successfully!")
    print("  Overflow handled all memory management automatically")


def configuration_example():
    """
    Example showing basic configuration options.
    """
    print("\n" + "="*60)
    print("Configuration Example")
    print("="*60)
    
    model = nn.Linear(1000, 1000)
    
    # Example 1: Conservative memory usage
    print("1. Conservative configuration (saves memory):")
    conservative_config = MemoryConfig(
        checkpoint_threshold=0.6,    # Checkpoint early
        offload_threshold=0.8,       # Offload early
        min_gpu_memory_mb=2048      # Keep 2GB free
    )
    
    wrapped_conservative = DynamicMemoryModule(model, config=conservative_config)
    print(f"   Strategy: {wrapped_conservative.strategy.value}")
    
    # Example 2: Aggressive performance
    print("\n2. Performance configuration (uses more memory):")
    performance_config = MemoryConfig(
        checkpoint_threshold=0.95,   # Checkpoint only when necessary
        offload_threshold=0.98,      # Almost never offload
        enable_profiling=False       # Disable profiling overhead
    )
    
    wrapped_performance = DynamicMemoryModule(model, config=performance_config)
    print(f"   Strategy: {wrapped_performance.strategy.value}")
    
    print("\n✓ Choose configuration based on your needs:")
    print("  - Conservative: When running multiple models")
    print("  - Performance: When speed is critical")
    print("  - Default: Balanced for most use cases")


def common_patterns():
    """
    Show common usage patterns and tips.
    """
    print("\n" + "="*60)
    print("Common Usage Patterns")
    print("="*60)
    
    print("1. Inference on large models:")
    print("""
    model = load_pretrained_model()  # Your large model
    model = DynamicMemoryModule(model)
    model.eval()
    
    with torch.no_grad():
        output = model(input)
    """)
    
    print("\n2. Fine-tuning with limited memory:")
    print("""
    model = load_pretrained_model()
    model = DynamicMemoryModule(model)
    
    # Freeze base layers to save memory
    for param in model.wrapped_module.base_layers.parameters():
        param.requires_grad = False
    
    # Train only the head
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters())
    )
    """)
    
    print("\n3. Multi-GPU training:")
    print("""
    model = create_large_model()
    model = DynamicMemoryModule(model)  # Handles memory per GPU
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Standard PyTorch DDP
    """)
    
    print("\n4. Saving and loading:")
    print("""
    # Save
    torch.save(model.state_dict(), 'checkpoint.pt')
    
    # Load
    model = create_model_architecture()
    model = DynamicMemoryModule(model)
    model.load_state_dict(torch.load('checkpoint.pt'))
    """)


def main():
    """Run all basic examples."""
    print("Overflow: Basic Usage Examples")
    print("=" * 60)
    print()
    
    # Check system info
    if torch.cuda.is_available():
        print(f"Running on: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on: CPU")
    print()
    
    # Run examples
    simple_usage_example()
    large_model_example()
    training_example()
    configuration_example()
    common_patterns()
    
    print("\n" + "="*60)
    print("Getting Started Summary")
    print("="*60)
    print("✓ Just wrap your model: DynamicMemoryModule(model)")
    print("✓ Automatic strategy selection based on your hardware")
    print("✓ Full compatibility with PyTorch training")
    print("✓ Configure behavior with MemoryConfig if needed")
    print("\nFor advanced usage, see advanced_usage.py")


if __name__ == "__main__":
    main()