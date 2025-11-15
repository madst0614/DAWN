"""
DAWN Configuration System

Unified configuration system with single entry point.

Usage:
    from config import get_config, get_phase1_config, get_phase2_config
    
    # Simple usage
    cfg = get_phase1_config()
    
    # Customized
    cfg = get_config(
        phase=2,
        model_preset="large",
        expert_names=["lexical", "semantic"],
        training_preset="aggressive"
    )
    
    # Print summary
    cfg.print_summary()
    
    # Access configuration
    print(cfg.model.hidden_size)
    print(cfg.tasks["mlm"].max_samples)
    print(cfg.training.optimizer.base_lr)
"""

# Main API (legacy - use DAWNConfig instead)
from .registry import (
    get_config,
    get_phase1_config,
    get_phase2_config,
    get_default_config,  # New simplified API
    RuntimeConfig,
    ConfigRegistry,
    ConfigError,
)

# Simple Config API (recommended)
from .simple_config import (
    DAWNConfig,
    get_small_config,
    get_base_config,
    get_large_config,
)

# DAWN adapter (for new refactored package)
from .models_adapter import (
    to_models_config,
    get_models_args,
)

# Configuration classes (for type hints and custom configs)
from .model import (
    ModelConfig,
    ExpertConfig,
    RefinerConfig,
    IntegratorConfig,
    PeerPredictionConfig,
    UnifiedGateConfig,
)

from .data import (
    TaskConfig,
    DatasetConfig,
    TextFilterConfig,
    CurriculumStage,
)

# Training configs removed - now in simple_config.py (DAWNConfig)

from .hardware import (
    HardwareConfig,
    detect_hardware,
    get_hardware_config,
)

# Preset dictionaries (for reference)
from .model import (
    MODEL_PRESETS,
    EXPERT_PRESETS,
    REFINER_PRESETS,
    INTEGRATOR_PRESETS,
    PEER_PREDICTION_PRESETS,
    UNIFIED_GATE_PRESETS,
)

from .data import (
    TASKS,
    DATASETS,
    PHASE1_CURRICULUM,
    PHASE2_TASKS,
    DEFAULT_EPOCHS_PER_TASK,
)


__all__ = [
    # Simple Config API (recommended)
    "DAWNConfig",
    "get_small_config",
    "get_base_config",
    "get_large_config",

    # Main API (legacy)
    "get_config",
    "get_phase1_config",
    "get_phase2_config",
    "get_default_config",  # New simplified API
    "RuntimeConfig",
    "ConfigRegistry",
    "ConfigError",

    # DAWN adapter
    "to_models_config",
    "get_models_args",
    
    # Configuration classes
    "ModelConfig",
    "ExpertConfig",
    "RefinerConfig",
    "IntegratorConfig",
    "PeerPredictionConfig",
    "UnifiedGateConfig",
    "TaskConfig",
    "DatasetConfig",
    "TextFilterConfig",
    "CurriculumStage",
    "HardwareConfig",
    
    # Utilities
    "detect_hardware",
    "get_hardware_config",
    
    # Presets
    "MODEL_PRESETS",
    "EXPERT_PRESETS",
    "REFINER_PRESETS",
    "INTEGRATOR_PRESETS",
    "PEER_PREDICTION_PRESETS",
    "UNIFIED_GATE_PRESETS",
    "TASKS",
    "DATASETS",
    "PHASE1_CURRICULUM",
    "PHASE2_TASKS",
    "DEFAULT_EPOCHS_PER_TASK",
]


# Version
__version__ = "4.0.0"
