# src/overflow/cli.py
"""
Command-line interface for Overflow package
"""

import argparse
import torch
import psutil
from .device_manager import DeviceManager
from . import __version__


def format_bytes(bytes_value):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def print_system_info():
    """Print system information relevant to Overflow."""
    print(f"Overflow v{__version__} - System Information")
    print("=" * 50)
    
    # CPU Information
    print("\nCPU Information:")
    print(f"  Physical cores: {psutil.cpu_count(logical=False)}")
    print(f"  Logical cores: {psutil.cpu_count(logical=True)}")
    
    # Memory Information
    vm = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {format_bytes(vm.total)}")
    print(f"  Available: {format_bytes(vm.available)} ({vm.percent:.1f}% used)")
    
    # PyTorch Information
    print(f"\nPyTorch Information:")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # GPU Information
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory
            free_mem, _ = torch.cuda.mem_get_info(i)
            used_mem = total_mem - free_mem
            
            print(f"\n  GPU {i}: {props.name}")
            print(f"    Compute capability: {props.major}.{props.minor}")
            print(f"    Total memory: {format_bytes(total_mem)}")
            print(f"    Free memory: {format_bytes(free_mem)}")
            print(f"    Used memory: {format_bytes(used_mem)} ({used_mem/total_mem*100:.1f}%)")
            print(f"    Multi-processor count: {props.multi_processor_count}")
    else:
        print("\n  No CUDA GPUs detected")
    
    # Overflow recommendations
    print("\n" + "=" * 50)
    print("Overflow Recommendations:")
    
    device_manager = DeviceManager()
    total_gpu_mem = device_manager.get_total_gpu_memory()
    
    if total_gpu_mem > 0:
        total_gpu_gb = total_gpu_mem / 1024**3
        print(f"\n  Total GPU memory: {total_gpu_gb:.1f} GB")
        print(f"  Estimated model sizes that can run:")
        print(f"    - Standard mode: up to {total_gpu_gb * 0.7:.1f} GB")
        print(f"    - With gradient checkpointing: up to {total_gpu_gb * 0.9:.1f} GB")
        print(f"    - With CPU offloading: up to {vm.total / 1024**3 * 0.8:.1f} GB")
    else:
        print("\n  No GPU detected - CPU execution only")
        print(f"  Maximum model size limited by system RAM: {vm.total / 1024**3 * 0.8:.1f} GB")


def test_model(model_size_gb):
    """Test if a model of given size can run on this system."""
    print(f"\nTesting model size: {model_size_gb} GB")
    print("-" * 30)
    
    device_manager = DeviceManager()
    total_gpu_mem = device_manager.get_total_gpu_memory()
    model_size_bytes = model_size_gb * 1024**3
    
    vm = psutil.virtual_memory()
    
    if total_gpu_mem > 0:
        if model_size_bytes < total_gpu_mem * 0.7:
            print("✓ Can run in STANDARD mode (fastest)")
        elif model_size_bytes < total_gpu_mem * 0.9:
            print("✓ Can run with GRADIENT CHECKPOINTING")
        elif model_size_bytes < vm.total * 0.8:
            print("✓ Can run with CPU OFFLOADING (slower)")
        else:
            print("✗ Model too large for this system")
            print(f"  Maximum supported: {vm.total / 1024**3 * 0.8:.1f} GB")
    else:
        if model_size_bytes < vm.total * 0.8:
            print("✓ Can run on CPU")
        else:
            print("✗ Model too large for system memory")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Overflow: When your model overflows the GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  overflow-info                    # Show system information
  overflow-info --test-size 7.5    # Test if a 7.5GB model can run
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Overflow v{__version__}'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        metavar='GB',
        help='Test if a model of given size (in GB) can run on this system'
    )
    
    args = parser.parse_args()
    
    # Always show system info
    print_system_info()
    
    # Test specific model size if requested
    if args.test_size:
        test_model(args.test_size)


if __name__ == "__main__":
    main()