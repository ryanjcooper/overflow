# Overflow Test Suite - Quick Reference

## 🎯 Which Test File Should I Use?

### Testing a specific component?
- **MemoryProfiler** → `test_memory_profiler.py`
- **DeviceManager** → `test_device_manager.py`
- **BlockSwapManager** → `test_block_swap_manager.py`
- **ModelPartitioner** → `test_model_partitioner.py`
- **MemoryConfig** → `test_memory_config.py`
- **DynamicMemoryModule** → `test_dynamic_memory_module.py`

### Testing workflows?
- **Complete workflows** → `test_integration.py`
- **Hardware scenarios** → `test_hardware_scenarios.py`
- **Performance** → `test_performance_benchmarks.py`

## 🚀 Common Commands

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_memory_profiler.py -v

# Run only unit tests (fast)
pytest tests/test_*.py -v -k "not integration and not hardware and not performance"

# Run only integration tests
pytest tests/test_integration.py tests/test_hardware_scenarios.py -v

# Run performance benchmarks
pytest tests/test_performance_benchmarks.py -v --benchmark-only

# Run with coverage
pytest tests/ --cov=overflow --cov-report=html

# Run tests in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run tests matching a pattern
pytest tests/ -k "test_forward_pass"

# Run tests with specific marker
pytest tests/ -m "not benchmark"
```

## 📁 File Structure

```
tests/
├── test_memory_profiler.py      # MemoryProfiler unit tests
├── test_device_manager.py       # DeviceManager unit tests
├── test_block_swap_manager.py   # BlockSwapManager unit tests
├── test_model_partitioner.py    # ModelPartitioner unit tests
├── test_memory_config.py        # MemoryConfig unit tests
├── test_dynamic_memory_module.py # DynamicMemoryModule unit tests
├── test_integration.py          # Integration tests
├── test_hardware_scenarios.py   # Hardware scenario tests
├── test_performance_benchmarks.py # Performance benchmarks
├── TEST_STRUCTURE.md           # Detailed documentation
└── TEST_QUICK_REFERENCE.md     # This file
```

## 🔧 Adding New Tests

### For a new feature in existing component:
```python
# Add to the appropriate test_<component>.py file
def test_my_new_feature(self):
    """Test description."""
    # Your test here
```

### For a new component:
```python
# Create tests/test_my_component.py
import pytest
from overflow import MyComponent

class TestMyComponent:
    def test_initialization(self):
        # Your tests here
```

### For integration testing:
```python
# Add to test_integration.py
def test_my_workflow(self):
    """Test complete workflow."""
    # Your integration test here
```

## 🏷️ Test Markers

```bash
# Skip GPU tests on CPU-only systems
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")

# Mark as benchmark
@pytest.mark.benchmark

# Mark as slow
@pytest.mark.slow

# Parametrize tests
@pytest.mark.parametrize("strategy", [ExecutionStrategy.STANDARD, ...])
```

## 🧹 Test Best Practices

1. **Always cleanup GPU memory:**
   ```python
   def teardown_method(self):
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

2. **Move tensors to correct device:**
   ```python
   device = next(model.parameters()).device
   x = x.to(device)
   ```

3. **Use descriptive test names:**
   ```python
   def test_forward_pass_with_gradient_checkpointing(self):
   ```

4. **Mock external dependencies:**
   ```python
   with patch('overflow.module.DeviceManager') as mock:
       # Your test
   ```

## 📊 Coverage Goals

- Each component should have >90% test coverage
- Integration tests should cover main use cases
- Performance benchmarks should cover different strategies
- Hardware scenarios should test edge cases