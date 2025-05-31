# src/overflow/device_manager.py
"""
Device detection and management for the Overflow framework.
"""

import torch
import psutil
from typing import List

from .config import DeviceInfo


class DeviceManager:
    """Manages available compute devices."""
    
    def __init__(self):
        self.devices = self._detect_devices()
        self.primary_device = self._select_primary_device()
    
    def _detect_devices(self) -> List[DeviceInfo]:
        """Detect all available devices."""
        devices = []
        
        # Check CUDA devices
        if torch.cuda.is_available():
            try:
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    device = torch.device(f'cuda:{i}')
                    props = torch.cuda.get_device_properties(i)
                    total_mem = props.total_memory
                    free_mem, _ = torch.cuda.mem_get_info(i)
                    
                    devices.append(DeviceInfo(
                        device_id=i,
                        device_type='cuda',
                        total_memory=total_mem,
                        available_memory=free_mem,
                        device_name=props.name
                    ))
            except Exception:
                # If CUDA operations fail (e.g., in mocked environments), skip CUDA devices
                pass
        
        # Add CPU
        vm = psutil.virtual_memory()
        devices.append(DeviceInfo(
            device_id=0,
            device_type='cpu',
            total_memory=vm.total,
            available_memory=vm.available,
            device_name='CPU'
        ))
        
        return devices
    
    def _select_primary_device(self) -> torch.device:
        """Select the primary compute device."""
        cuda_devices = [d for d in self.devices if d.device_type == 'cuda']
        if cuda_devices:
            # Select GPU with most available memory
            best_device = max(cuda_devices, key=lambda d: d.available_memory)
            return torch.device(f'cuda:{best_device.device_id}')
        return torch.device('cpu')
    
    def get_total_gpu_memory(self) -> int:
        """Get total GPU memory across all devices."""
        return sum(d.total_memory for d in self.devices if d.device_type == 'cuda')
    
    def get_available_gpu_memory(self) -> int:
        """Get available GPU memory across all devices."""
        return sum(d.available_memory for d in self.devices if d.device_type == 'cuda')