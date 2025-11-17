"""
SPROUT: Self-organizing Progressive Routing with Organic Unified Trees

Utility package for SPROUT training and analysis.
Core model implementation is in src/models/sprout_neuron_based.py
"""

# Data utilities
from .data_utils import (
    SpanMasker,
    TokenDeletion,
    TextValidator,
    DatasetStats,
    CacheLoader,
)

# Training utilities
from .training_utils import (
    CheckpointManager,
    TrainingMonitor,
    format_time,
    count_parameters,
)

__version__ = "0.4.0"

__all__ = [
    # Data utilities
    "SpanMasker",
    "TokenDeletion",
    "TextValidator",
    "DatasetStats",
    "CacheLoader",
    # Training utilities
    "CheckpointManager",
    "TrainingMonitor",
    "format_time",
    "count_parameters",
]
