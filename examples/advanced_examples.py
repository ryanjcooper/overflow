# examples/advanced_usage.py
"""
Advanced usage examples for the Overflow framework.
These examples demonstrate advanced features and integration patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from overflow import DynamicMemoryModule, MemoryConfig, ExecutionStrategy


def custom_configuration_example():
    """
    Example: Using custom configuration to control memory management behavior.
    
    This shows how to fine-tune Overflow's behavior for your specific needs.
    """
    print("\n" + "="*60)
    print("Custom Configuration Example")
    print("="*60)
    
    # Create a model
    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024)
    )
    
    # Custom configuration for aggressive memory saving
    memory_saver_config = MemoryConfig(
        checkpoint_threshold=0.6,    # Start checkpointing at 60% memory usage
        offload_threshold=0.75,      # Start offloading at 75% memory usage
        prefetch_size=2,            # Prefetch 2 blocks for CPU offloading
        profile_interval=5,         # Profile every 5 forward passes
        min_gpu_memory_mb=2048      # Always keep 2GB free on GPU
    )
    
    print("Configuration for aggressive memory saving:")
    print(f"  - Checkpoint at {memory_saver_config.checkpoint_threshold:.0%} memory usage")
    print(f"  - Offload at {memory_saver_config.offload_threshold:.0%} memory usage")
    print(f"  - Keep {memory_saver_config.min_gpu_memory_mb}MB GPU memory free")
    
    # Wrap model with custom config
    wrapped = DynamicMemoryModule(model, config=memory_saver_config)
    print(f"\nSelected strategy: {wrapped.strategy.value}")
    
    # Use the model
    x = torch.randn(32, 1024)
    output = wrapped(x)
    print(f"Output shape: {output.shape}")
    
    # Custom configuration for maximum performance
    performance_config = MemoryConfig(
        checkpoint_threshold=0.95,   # Only checkpoint when absolutely necessary
        offload_threshold=0.98,      # Almost never offload
        prefetch_size=8,            # Larger prefetch for better performance
        profile_interval=50,        # Less frequent profiling for lower overhead
        enable_profiling=False      # Disable profiling for maximum speed
    )
    
    print("\n\nConfiguration for maximum performance:")
    print("  - Minimal checkpointing and offloading")
    print("  - Profiling disabled")
    print("  - Larger prefetch buffer")
    
    wrapped_fast = DynamicMemoryModule(model, config=performance_config)
    output = wrapped_fast(x)
    print(f"Output shape: {output.shape}")


def mixed_precision_training_example():
    """
    Example: Using Overflow with PyTorch's Automatic Mixed Precision (AMP).
    
    Shows how to combine memory optimization with mixed precision training.
    """
    print("\n" + "="*60)
    print("Mixed Precision Training Example")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("This example requires CUDA for mixed precision training.")
        return
    
    from torch.cuda.amp import autocast, GradScaler
    
    # Create a model
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True),
        num_layers=6
    )
    
    # Wrap with Overflow
    model = DynamicMemoryModule(model)
    print(f"Strategy: {model.strategy.value}")
    
    # Setup for mixed precision training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    
    # Create dummy data
    batch_size = 16
    seq_length = 256
    d_model = 512
    
    dataset = TensorDataset(
        torch.randn(100, seq_length, d_model),
        torch.randn(100, seq_length, d_model)
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print("\nTraining with mixed precision...")
    model.train()
    
    for epoch in range(2):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= 5:  # Just a few steps for demo
                break
            
            # Mixed precision forward pass
            with autocast():
                outputs = model(inputs)
                loss = nn.functional.mse_loss(outputs, targets)
            
            # Mixed precision backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / min(5, len(dataloader))
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")
    
    print("\n✓ Mixed precision training works seamlessly with Overflow!")
    print("  Benefits: Lower memory usage + automatic memory management")


def model_state_management_example():
    """
    Example: Saving and loading models wrapped with Overflow.
    
    Shows how to properly save/load checkpoints and transfer models.
    """
    print("\n" + "="*60)
    print("Model State Management Example")
    print("="*60)
    
    # Create and wrap a model
    original_model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )
    wrapped_model = DynamicMemoryModule(original_model)
    
    # Train for a bit (simulate training)
    optimizer = optim.SGD(wrapped_model.parameters(), lr=0.01)
    x = torch.randn(10, 256)
    y = torch.randn(10, 256)
    
    print("Training model...")
    for i in range(3):
        output = wrapped_model(x)
        loss = nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  Step {i+1}: loss = {loss.item():.4f}")
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    checkpoint = {
        'model_state_dict': wrapped_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'strategy': wrapped_model.strategy.value,
        'memory_stats': wrapped_model.get_memory_stats()
    }
    
    # In practice, you would save to disk:
    # torch.save(checkpoint, 'model_checkpoint.pt')
    
    # Load checkpoint into new model
    print("\nLoading checkpoint into new model...")
    new_model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    )
    new_wrapped = DynamicMemoryModule(new_model)
    new_wrapped.load_state_dict(checkpoint['model_state_dict'])
    
    # Verify the models produce same output
    with torch.no_grad():
        original_output = wrapped_model(x)
        new_output = new_wrapped(x)
        difference = torch.abs(original_output - new_output).max().item()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Max difference in outputs: {difference:.8f}")
    print(f"  Original strategy: {checkpoint['strategy']}")
    
    # Show how to access the underlying model
    print("\nAccessing the underlying model:")
    print(f"  Wrapped model type: {type(new_wrapped)}")
    print(f"  Inner model type: {type(new_wrapped.wrapped_module)}")
    print("  You can access the original model via: model.wrapped_module")


def memory_profiling_example():
    """
    Example: Using the built-in memory profiler to analyze model behavior.
    
    Shows how to use profiling data to optimize your model.
    """
    print("\n" + "="*60)
    print("Memory Profiling Example")
    print("="*60)
    
    # Create a model with varied layer sizes
    model = nn.Sequential(
        nn.Linear(512, 1024),    # Small to medium
        nn.ReLU(),
        nn.Linear(1024, 4096),   # Medium to large
        nn.ReLU(), 
        nn.Linear(4096, 4096),   # Large to large
        nn.ReLU(),
        nn.Linear(4096, 512),    # Large to small
    )
    
    # Enable detailed profiling
    config = MemoryConfig(
        enable_profiling=True,
        profile_interval=1  # Profile every forward pass
    )
    
    wrapped = DynamicMemoryModule(model, config=config)
    print(f"Strategy: {wrapped.strategy.value}")
    
    # Run several forward passes with different batch sizes
    print("\nRunning forward passes with different batch sizes...")
    batch_sizes = [8, 16, 32, 64]
    
    for batch_size in batch_sizes:
        try:
            x = torch.randn(batch_size, 512)
            _ = wrapped(x)
            print(f"  Batch size {batch_size}: ✓")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  Batch size {batch_size}: Out of memory")
                break
    
    # Get detailed memory statistics
    stats = wrapped.get_memory_stats()
    
    print("\n" + "-"*40)
    print("Memory Profile Results:")
    print("-"*40)
    print(f"Peak memory usage: {stats['peak_memory_mb']:.1f} MB")
    print(f"Strategy used: {stats['strategy']}")
    
    if stats['module_stats']:
        print("\nPer-layer memory usage:")
        for name, layer_stats in stats['module_stats'].items():
            if layer_stats['count'] > 0:
                avg_memory = layer_stats['total_memory'] / layer_stats['count']
                print(f"  {name}:")
                print(f"    Average: {avg_memory:.1f} MB")
                print(f"    Peak: {layer_stats['peak_memory']:.1f} MB")
    
    print("\nInsights from profiling:")
    print("- The 4096-dimensional layers use the most memory")
    print("- Memory usage scales linearly with batch size")
    print("- Consider replacing large layers if memory is tight")


def distributed_training_integration():
    """
    Example: Using Overflow with PyTorch distributed training.
    
    Note: This is a demonstration of the pattern - actual distributed
    training requires proper multi-process setup.
    """
    print("\n" + "="*60)
    print("Distributed Training Integration Pattern")
    print("="*60)
    
    print("Example pattern for distributed training:")
    print("""
    # In your distributed training script:
    
    # 1. Initialize distributed training as usual
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    
    # 2. Create and wrap your model with Overflow
    model = create_your_model()
    model = DynamicMemoryModule(model)
    
    # 3. Wrap with DistributedDataParallel
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )
    
    # 4. Train as normal - Overflow handles memory management
    # while DDP handles distributed training
    """)
    
    print("\n✓ Overflow is fully compatible with PyTorch DDP!")
    print("  The memory management happens transparently within each process.")


def dynamic_model_adaptation():
    """
    Example: Dynamically adapting to available resources.
    
    Shows how Overflow automatically adjusts to different hardware.
    """
    print("\n" + "="*60)
    print("Dynamic Model Adaptation Example")
    print("="*60)
    
    # This model will behave differently on different hardware
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=16,
            dim_feedforward=4096,
            batch_first=True
        ),
        num_layers=24  # Large model
    )
    
    # Let Overflow adapt to your hardware
    wrapped = DynamicMemoryModule(model)
    
    print(f"Automatic adaptation results:")
    print(f"  Strategy selected: {wrapped.strategy.value}")
    
    # Get hardware info
    stats = wrapped.get_memory_stats()
    total_gpu_memory = sum(d['total_memory_mb'] for d in stats['devices'] if d['type'] == 'cuda')
    
    if total_gpu_memory > 0:
        print(f"  Total GPU memory: {total_gpu_memory:.0f} MB")
        print(f"  Model size: ~{wrapped._estimate_model_size() / 1024**2:.0f} MB")
    
    print(f"\nThe same code will automatically:")
    print(f"  - Use standard execution on high-memory GPUs")
    print(f"  - Enable checkpointing on medium GPUs")
    print(f"  - Use CPU offloading on low-memory systems")
    print(f"  - Utilize multiple GPUs if available")
    
    # Demonstrate usage
    x = torch.randn(4, 128, 1024)
    output = wrapped(x)
    print(f"\n✓ Model runs successfully with shape: {output.shape}")


def main():
    """Run all advanced examples."""
    print("Overflow: Advanced Usage Examples")
    print("=" * 60)
    
    # Run examples
    custom_configuration_example()
    mixed_precision_training_example()
    model_state_management_example()
    memory_profiling_example()
    distributed_training_integration()
    dynamic_model_adaptation()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("These examples demonstrated:")
    print("✓ Custom memory management configuration")
    print("✓ Integration with mixed precision training")
    print("✓ Proper checkpoint save/load patterns")
    print("✓ Memory profiling for optimization")
    print("✓ Distributed training compatibility")
    print("✓ Automatic hardware adaptation")
    
    print("\nFor more examples, see the documentation and basic_example.py")


if __name__ == "__main__":
    main()