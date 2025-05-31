# src/overflow/enums.py
"""
Enumeration types for the Overflow framework.
"""

from enum import Enum


class ExecutionStrategy(Enum):
    """Execution strategies based on hardware capabilities."""
    STANDARD = "standard"
    GRADIENT_CHECKPOINT = "gradient_checkpoint"
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    CPU_OFFLOAD = "cpu_offload"
    HYBRID = "hybrid"