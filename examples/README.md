# Overflow Examples

This directory contains example scripts demonstrating how to use the Overflow framework. These are **examples**, not tests - they show usage patterns and best practices.

## Examples vs Tests

- **Examples** (in `examples/`): Show how to use the framework
  - Demonstrate features and usage patterns
  - Educational and self-documenting
  - Adapt to available hardware
  - Can be run standalone by users

- **Tests** (in `tests/`): Verify the framework works correctly
  - Assert specific behaviors
  - Check edge cases and error handling
  - May fail if expectations aren't met
  - Use pytest or similar testing frameworks

## Available Examples

### 1. `basic_usage.py` - Getting Started
- Simple model wrapping
- Hardware-adaptive large model example
- Training loop integration
- Basic configuration options
- Common usage patterns

**Run it:** `python examples/basic_usage.py`

### 2. `advanced_usage.py` - Advanced Features
- Custom memory configuration
- Mixed precision training integration
- Model checkpoint save/load
- Memory profiling and analysis
- Distributed training patterns
- Dynamic hardware adaptation

**Run it:** `python examples/advanced_usage.py`

### 3. `performance_comparison.py` - Performance Characteristics
- Compare execution strategies
- Measure impact of batch sizes
- Performance tips and insights
- Strategy selection impact

**Run it:** `python examples/performance_comparison.py`

## Key Features of These Examples

### Hardware Adaptability
All examples automatically adapt to your available hardware:
- High-end GPUs: Larger models, standard execution
- Mid-range GPUs: Gradient checkpointing when needed
- Low-end GPUs: CPU offloading for large models
- CPU-only: Appropriate model sizes and strategies

### Educational Structure
Each example:
1. Explains what it demonstrates
2. Shows the code with comments
3. Provides output and insights
4. Suggests next steps

### No Hard Requirements
Examples work on any hardware configuration:
- Single GPU, multi-GPU, or CPU-only
- Various memory sizes
- Different PyTorch versions (1.9+)

## Running the Examples

1. **Basic usage** - Start here:
   ```bash
   python examples/basic_usage.py
   ```

2. **Advanced features** - Explore more:
   ```bash
   python examples/advanced_usage.py
   ```

3. **Performance insights**:
   ```bash
   python examples/performance_comparison.py
   ```

## Creating Your Own Examples

When adding new examples:

✅ **DO:**
- Show realistic usage patterns
- Adapt to available hardware
- Include explanatory comments
- Handle errors gracefully
- Provide educational value

❌ **DON'T:**
- Assert specific values (that's for tests)
- Assume specific hardware
- Use hard-coded paths
- Require external data files
- Test edge cases (that's for tests)

## Notes

- Examples print informative output, they don't assert conditions
- They demonstrate features, not verify correctness
- All examples should be runnable on any system with PyTorch
- For testing the framework, see `tests/` directory