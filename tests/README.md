# tests/README.md

# Overflow Tests

This directory contains the test suite for the Overflow framework. These are **tests**, not examples - they verify correctness and handle edge cases.

## Test Structure

### Unit Tests
- `test_overflow.py` - Core functionality tests
  - Memory profiler tests
  - Device manager tests
  - Block swap manager tests
  - Model partitioner tests
  - Main module tests

### Integration Tests
- `test_all_scenarios.py` - Hardware scenario tests
  - Single GPU scenarios
  - Multi-GPU scenarios
  - CPU-only execution
  - Strategy selection validation

### Performance Benchmarks
- `benchmarks/test_performance.py` - Performance measurements
  - Strategy comparison benchmarks
  - Throughput measurements
  - Memory efficiency tests
  - Real-world scenario benchmarks

## Running Tests

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test files:
```bash
# Unit tests
python -m pytest tests/test_overflow.py -v

# Integration tests
python -m pytest tests/test_all_scenarios.py -v

# Performance benchmarks
python -m pytest tests/benchmarks/test_performance.py -v --benchmark-only
```

### Run tests by marker:
```bash
# Only benchmarks
python -m pytest tests/ -v -m benchmark

# Skip benchmarks
python -m pytest tests/ -v -m "not benchmark"
```

### Run with coverage:
```bash
python -m pytest tests/ --cov=overflow --cov-report=html
```

## Test Categories

### 1. Unit Tests (`test_overflow.py`)
Test individual components in isolation:
- Memory profiling accuracy
- Device detection
- Swap manager functionality
- Configuration validation
- Hook registration

### 2. Integration Tests (`test_all_scenarios.py`)
Test realistic usage scenarios:
- Different model sizes relative to GPU memory
- Multi-GPU configurations
- Fallback behaviors
- Strategy transitions

### 3. Performance Benchmarks (`benchmarks/test_performance.py`)
Measure performance characteristics:
- Execution time by strategy
- Memory usage patterns
- Throughput scaling
- Overhead measurements

## Hardware Requirements

Tests adapt to available hardware but some require specific configurations:

- **GPU Tests**: Skipped if no CUDA available (`@pytest.mark.skipif`)
- **Multi-GPU Tests**: Skipped if < 2 GPUs available
- **Large Memory Tests**: Skipped if insufficient RAM

## Writing New Tests

### Test Guidelines

✅ **DO:**
- Use pytest fixtures for setup/teardown
- Skip tests that can't run on current hardware
- Clean up GPU memory after tests
- Use relative sizes (% of available memory)
- Test both success and failure cases

❌ **DON'T:**
- Assume specific GPU models or memory sizes
- Leave GPU memory allocated
- Use absolute memory sizes
- Depend on external files
- Print unless debugging

### Example Test Pattern

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gpu_functionality():
    """Test something that requires GPU."""
    # Get relative memory size
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    model_size = gpu_memory * 0.5  # 50% of GPU
    
    # Test the functionality
    model = create_model_of_size(model_size)
    wrapped = DynamicMemoryModule(model)
    
    # Assertions
    assert wrapped.strategy in [ExecutionStrategy.STANDARD, 
                               ExecutionStrategy.GRADIENT_CHECKPOINT]
    
    # Cleanup
    del model, wrapped
    torch.cuda.empty_cache()
```

## Continuous Integration

Tests are designed to work in CI environments:
- Adapt to available hardware
- Skip tests that can't run
- Clean up resources properly
- Provide clear failure messages

## Debugging Failed Tests

1. Run with verbose output:
   ```bash
   python -m pytest tests/test_name.py -vv -s
   ```

2. Run specific test:
   ```bash
   python -m pytest tests/test_name.py::TestClass::test_method -v
   ```

3. Enable debugging:
   ```bash
   python -m pytest tests/ --pdb
   ```