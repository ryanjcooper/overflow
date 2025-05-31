# README.md
# Overflow: When your model overflows the GPU

A PyTorch memory management framework that enables running models larger than available GPU memory through intelligent memory management, automatic model partitioning, and dynamic execution strategies.

## Features

- **Automatic Memory Management**: Intelligently handles models that exceed GPU memory
- **Multiple Execution Strategies**: Automatically selects between standard execution, gradient checkpointing, model parallelism, and CPU offloading
- **Simple API**: Drop-in replacement for `nn.Module` - just wrap your model
- **Hardware Detection**: Automatically profiles available GPUs and system memory
- **Memory Profiling**: Built-in profiling to track memory usage per module
- **Block Swapping**: Efficient tensor swapping between GPU and CPU memory
- **Multi-GPU Support**: Automatic model parallelism across available GPUs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/overflow.git
cd overflow

# Install dependencies
pip install torch psutil

# Install the package
pip install -e .
```

## Quick Start

```python
from overflow import DynamicMemoryModule

# Your existing PyTorch model
model = create_your_large_model()

# Wrap it with Overflow
model = DynamicMemoryModule(model)

# Use it exactly like a normal PyTorch model
output = model(input_tensor)
```

That's it! Overflow automatically:
- Detects available hardware
- Estimates model memory requirements
- Selects the optimal execution strategy
- Manages memory during forward and backward passes

## Example: Running a 40GB Model on 16GB GPU

```python
import torch
import torch.nn as nn
from overflow import DynamicMemoryModule

# Create a large transformer model (40GB+)
model = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(d_model=2048, nhead=32, dim_feedforward=8192),
    num_layers=48
)

# Wrap with Overflow
model = DynamicMemoryModule(model)

# The framework automatically enables CPU offloading
print(f"Execution strategy: {model.strategy}")  # Output: "cpu_offload"

# Run inference - works even though model > GPU memory!
input_tensor = torch.randn(16, 512, 2048)
output = model(input_tensor)
```


## Execution Strategies

Overflow automatically selects the best strategy based on your hardware:

| Strategy | When Used | Description |
|----------|-----------|-------------|
| **Standard** | Model fits comfortably in GPU memory | Normal PyTorch execution |
| **Gradient Checkpoint** | Model fits but activations don't | Trades compute for memory by recomputing activations |
| **Data Parallel** | Model fits on one GPU + multiple GPUs available | Splits batch across GPUs for faster inference |
| **Model Parallel** | Model too large for one GPU but fits across all | Distributes model layers across GPUs |
| **CPU Offload** | Model exceeds total GPU memory | Dynamically swaps layers between CPU and GPU(s) |
| **Hybrid** | Complex scenarios | Combines multiple strategies |

### Strategy Selection Logic

**Single GPU Systems:**
- Model < 80% GPU memory → **Standard**
- Model fits but activations don't → **Gradient Checkpoint**
- Model > GPU memory → **CPU Offload**

**Multi-GPU Systems:**
- Model < 50% single GPU (inference only) → **Data Parallel** 
- Model < 80% single GPU → **Standard** or **Gradient Checkpoint**
- Model > single GPU but < total → **Model Parallel**
- Model > total GPU memory → **CPU Offload** (uses all GPUs)

### When is Data Parallel Beneficial?

Data parallelism isn't always faster! It works best with:
- **Large batch sizes** (32+ samples)
- **Small models** (< 50% of GPU memory)
- **Inference workloads** (not training)

For small batches or training, the overhead of splitting data and synchronizing across GPUs often outweighs the benefits.

```python
# Force data parallel for specific use cases
config = MemoryConfig(
    prefer_data_parallel=True,  # Force data parallel for small models
    data_parallel_threshold=0.5  # Model must be < 50% of GPU memory
)
model = DynamicMemoryModule(your_model, config=config)
```

## Custom Configuration

```python
from overflow import DynamicMemoryModule, MemoryConfig

# Customize memory management behavior
config = MemoryConfig(
    checkpoint_threshold=0.7,    # Enable checkpointing at 70% memory usage
    offload_threshold=0.85,      # Start CPU offloading at 85% memory usage
    prefetch_size=4,            # Number of blocks to prefetch
    profile_interval=5,         # Profile every 5 forward passes
    min_gpu_memory_mb=2048      # Keep 2GB GPU memory free
)

model = DynamicMemoryModule(your_model, config=config)
```

## Memory Profiling

```python
# Get detailed memory statistics
stats = model.get_memory_stats()

print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")
print(f"Execution strategy: {stats['strategy']}")
print(f"Number of CPU swaps: {stats['swap_stats']['swaps_out']}")

# Per-module memory breakdown
for module_name, module_stats in stats['module_stats'].items():
    print(f"{module_name}: {module_stats['peak_memory']:.2f} MB")
```

## Training Example

```python
import torch.optim as optim

# Wrap your model
model = DynamicMemoryModule(large_transformer)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop works normally
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = model(batch['input_ids'])
        loss = compute_loss(outputs, batch['labels'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Check memory stats
    print(f"Epoch {epoch} - Peak memory: {model.get_memory_stats()['peak_memory_mb']:.2f} MB")
```

## Advanced Features

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

model = DynamicMemoryModule(your_model)
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Manual Strategy Selection

```python
from overflow import ExecutionStrategy

# Force a specific strategy
model = DynamicMemoryModule(
    your_model,
    force_strategy=ExecutionStrategy.CPU_OFFLOAD
)
```

### Integration with Existing Code

Overflow maintains full compatibility with PyTorch:

```python
# Works with all PyTorch features
model = DynamicMemoryModule(your_model)

# State dict operations
torch.save(model.state_dict(), 'checkpoint.pt')
model.load_state_dict(torch.load('checkpoint.pt'))

# Distributed training
model = torch.nn.parallel.DistributedDataParallel(model)

# TorchScript (access wrapped module)
scripted = torch.jit.script(model.wrapped_module)
```

## Architecture

```
DynamicMemoryModule
├── MemoryProfiler      # Tracks memory usage
├── DeviceManager       # Detects and manages devices
├── BlockSwapManager    # Handles CPU-GPU swapping
├── ModelPartitioner    # Automatic model partitioning
└── Strategy Engine     # Selects execution strategy
```

## Performance Considerations

- **CPU Offloading**: ~2-5x slower than pure GPU execution but enables running much larger models
- **Gradient Checkpointing**: ~1.3x slower but reduces memory by ~50%
- **Model Parallelism**: Near-linear scaling with number of GPUs
- **Memory Profiling**: < 1% overhead when enabled

## Troubleshooting

### Out of Memory Errors

If you still get OOM errors:

1. Reduce batch size
2. Enable more aggressive offloading:
   ```python
   config = MemoryConfig(checkpoint_threshold=0.5, offload_threshold=0.7)
   ```
3. Check available system RAM for CPU offloading

### Slow Performance

1. Check if CPU offloading is active (expected to be slower)
2. Ensure CUDA and PyTorch are properly installed
3. Use pinned memory for faster transfers:
   ```python
   torch.cuda.set_per_process_memory_fraction(0.8)
   ```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Roadmap

- [ ] Pipeline parallelism support
- [ ] Automatic batch size tuning
- [ ] Integration with DeepSpeed/FairScale
- [ ] Quantization support
- [ ] Disk offloading for extremely large models
- [ ] ONNX export compatibility

## Citation

If you use Overflow in your research, please cite:

```bibtex
@software{overflow,
  title = {Overflow: When your model overflows the GPU},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/overflow}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.