"""
Configuration Registry

Central registry for managing and validating all configurations.
This is the single entry point for accessing configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import copy

# Import torch conditionally for hardware detection
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .model import (
    ModelConfig, ExpertConfig, RefinerConfig, IntegratorConfig,
    PeerPredictionConfig, UnifiedGateConfig,
    MODEL_PRESETS, EXPERT_PRESETS, REFINER_PRESETS,
    INTEGRATOR_PRESETS, PEER_PREDICTION_PRESETS, UNIFIED_GATE_PRESETS,
)
from .data import (
    TaskConfig, DatasetConfig, TextFilterConfig,
    TASKS, DATASETS, TEXT_FILTERS,
    PHASE1_CURRICULUM, PHASE2_TASKS,
    get_tasks_for_expert, validate_task_expert_mapping,
)
# Training configs moved inline (training.py deleted)
from dataclasses import dataclass, field
from .data import DEFAULT_EPOCHS_PER_TASK

@dataclass
class Phase1TrainingConfig:
    """Phase 1: Independent expert training"""
    base_lr: float = 2e-4
    embedding_lr: float = 1e-4
    warmup_ratio: float = 0.08
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scheduler: str = "linear"
    mixed_precision: bool = True
    default_epochs: int = 6
    epochs_per_task: Dict[str, int] = field(default_factory=lambda: DEFAULT_EPOCHS_PER_TASK.copy())
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "./checkpoints/phase1"
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1

    def get_epochs_for_task(self, task_name: str) -> int:
        return self.epochs_per_task.get(task_name, self.default_epochs)

@dataclass
class Phase2TrainingConfig:
    """Phase 2: Collaborative training"""
    base_lr: float = 2e-5
    embedding_lr: float = 5e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    scheduler: str = "linear"
    mixed_precision: bool = True
    epochs: int = 5
    freeze_experts: bool = False
    train_integrator: bool = True
    train_task_heads: bool = True
    task_loss_weight: float = 1.0
    peer_prediction_loss_weight: float = 0.5
    task_sampling: str = "uniform"
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "./checkpoints/phase2"
    log_every_n_steps: int = 100
    eval_every_n_epochs: int = 1

@dataclass
class DataLoaderConfig:
    """DataLoader settings"""
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
from .hardware import HardwareConfig, DEFAULT_HARDWARE


class ConfigError(Exception):
    """Configuration validation error"""
    pass


@dataclass
class RuntimeConfig:
    """
    Complete runtime configuration

    This is the unified configuration object used throughout the system.
    All components reference values from this single source.
    """
    # Core architecture
    model: ModelConfig
    refiner: RefinerConfig
    integrator: IntegratorConfig

    # Experts (filtered based on selection)
    expert_names: List[str]
    experts: Dict[str, ExpertConfig]

    # Tasks (filtered based on expert selection)
    tasks: Dict[str, TaskConfig]

    # Datasets
    datasets: Dict[str, DatasetConfig]

    # Text filtering
    text_filters: TextFilterConfig

    # Training
    phase: int  # 1 or 2
    training: Phase1TrainingConfig | Phase2TrainingConfig
    dataloader: DataLoaderConfig
    hardware: HardwareConfig

    # Preset names (for checkpoint naming)
    model_preset: str = "base"
    training_preset: str = "standard"

    # Phase 2 specific (None for Phase 1)
    peer_prediction: Optional[PeerPredictionConfig] = None
    unified_gate: Optional[UnifiedGateConfig] = None
    phase2_tasks: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate configuration after creation"""
        self._validate()
    
    def _validate(self):
        """Run all validation checks"""
        errors = []
        
        # 1. Expert names consistency
        if set(self.expert_names) != set(self.experts.keys()):
            errors.append(
                f"Expert name mismatch: expert_names={self.expert_names}, "
                f"experts.keys()={list(self.experts.keys())}"
            )
        
        # 2. All tasks have valid experts
        for task_name, task_cfg in self.tasks.items():
            if task_cfg.expert_name not in self.expert_names:
                errors.append(
                    f"Task '{task_name}' requires expert '{task_cfg.expert_name}' "
                    f"which is not in selected experts"
                )
        
        # 3. All tasks reference valid datasets
        for task_name, task_cfg in self.tasks.items():
            if task_cfg.dataset_source not in self.datasets:
                errors.append(
                    f"Task '{task_name}' references unknown dataset '{task_cfg.dataset_source}'"
                )
        
        # 4. Phase-specific validation
        if self.phase == 1:
            if self.peer_prediction is not None:
                errors.append("Phase 1 should not have peer_prediction config")
        elif self.phase == 2:
            if self.peer_prediction is None:
                errors.append("Phase 2 requires peer_prediction config")
            if self.phase2_tasks is None:
                errors.append("Phase 2 requires phase2_tasks list")
            # Validate phase2_tasks
            for task_name in self.phase2_tasks or []:
                if task_name not in self.tasks:
                    errors.append(f"Phase 2 task '{task_name}' not in available tasks")
        else:
            errors.append(f"Invalid phase: {self.phase} (must be 1 or 2)")
        
        # 5. Model architecture consistency
        if self.model.num_steps != len(self.experts[self.expert_names[0]].step_weights):
            errors.append(
                f"Model num_steps ({self.model.num_steps}) must match "
                f"expert step_weights length ({len(self.experts[self.expert_names[0]].step_weights)})"
            )
        
        # 6. Hardware compatibility
        if self.hardware.mixed_precision and self.hardware.device == "cpu":
            errors.append("Mixed precision training requires CUDA device")
        
        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(f"  ❌ {e}" for e in errors))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (for backward compatibility with old code)"""
        config_dict = {
            # Model
            "hidden_size": self.model.hidden_size,
            "vocab_size": self.model.vocab_size,
            "max_length": self.model.max_length,
            "num_heads": self.model.num_heads,
            "intermediate_size": self.model.intermediate_size,
            "num_steps": self.model.num_steps,
            "dropout": self.model.dropout,
            
            # Experts
            "expert_names": self.expert_names,
            "expert": {name: expert.__dict__ for name, expert in self.experts.items()},
            
            # Components
            "refiner": self.refiner.__dict__,
            "integrator": self.integrator.__dict__,
        }
        
        # Phase 2 specific
        if self.phase == 2:
            # Convert nested dataclasses to dict
            peer_pred_dict = {
                "collaboration_mode": self.peer_prediction.collaboration_mode,
                "aspect_dim": self.peer_prediction.aspect_dim,
                "peer_graph": self.peer_prediction.peer_graph,
                "feature_extraction": self.peer_prediction.feature_extraction.__dict__,
                "complement_gate": self.peer_prediction.complement_gate.__dict__,
                "num_iterations": self.peer_prediction.num_iterations,
                "prediction_loss_weight": self.peer_prediction.prediction_loss_weight,
                "predictor_type": self.peer_prediction.predictor_type,
                "predictor_hidden_size": self.peer_prediction.predictor_hidden_size,
                "detach_predictions": self.peer_prediction.detach_predictions,
                "symmetric_loss": self.peer_prediction.symmetric_loss,
            }
            config_dict["peer_prediction"] = peer_pred_dict
            config_dict["unified_gate"] = self.unified_gate.__dict__

        return config_dict
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print(f"DAWN Configuration (Phase {self.phase})")
        print("="*70)
        
        print("\n📐 Model Architecture:")
        print(f"  Size: {self.model.hidden_size}d x {self.model.num_heads} heads")
        print(f"  Refinement steps: {self.model.num_steps}")
        print(f"  Max length: {self.model.max_length}")
        print(f"  Dropout: {self.model.dropout}")
        
        print("\n👥 Experts:")
        for name in self.expert_names:
            expert = self.experts[name]
            tasks = get_tasks_for_expert(name)
            tasks_in_config = [t for t in tasks if t in self.tasks]
            print(f"  {name:12s}: {len(tasks_in_config)} tasks")
            print(f"  {'':12s}  Step weights: {expert.step_weights}")
        
        print("\n🎯 Tasks:")
        for task_name, task_cfg in sorted(self.tasks.items()):
            print(f"  {task_name:20s} → {task_cfg.expert_name:12s} ({task_cfg.max_samples:,} samples)")
        
        print("\n⚙️  Hardware:")
        print(f"  {self.hardware}")
        
        print("\n📚 Training:")
        if self.phase == 1:
            print(f"  LR: {self.training.base_lr} (base), {self.training.embedding_lr} (emb)")
            print(f"  Warmup: {self.training.warmup_ratio}")
            print(f"  Gradient clip: {self.training.gradient_clip}")
            print(f"  Default epochs: {self.training.default_epochs}")
        else:
            print(f"  LR: {self.training.base_lr} (base), {self.training.embedding_lr} (emb)")
            print(f"  Epochs: {self.training.epochs}")
            print(f"  Freeze experts: {self.training.freeze_experts}")
            print(f"  Peer loss weight: {self.training.peer_prediction_loss_weight}")
        
        if self.phase == 2 and self.phase2_tasks:
            print(f"\n🤝 Phase 2 Tasks:")
            print(f"  {', '.join(self.phase2_tasks)}")
        
        print("\n" + "="*70 + "\n")


def get_default_config(
    phase: int = 1,
    # Model parameters (directly customizable) - defaults match PNN hierarchical structure
    hidden_size: int = 768,
    vocab_size: int = 30522,
    max_length: int = 512,  # PNN uses 128, but 512 allows longer sequences
    num_heads: int = 12,
    intermediate_size = [1024, 1536, 2048, 1536, 1024],  # Mountain-shaped (PNN hierarchical)
    num_steps: int = 4,
    dropout: float = 0.1,
    # Expert selection
    expert_names: Optional[List[str]] = None,
    # Training parameters (directly customizable) - Conservative for DAWN stability
    base_lr: float = 1e-5,  # Very conservative for stability (was 2e-5)
    embedding_lr: float = 3e-6,  # Very conservative for stability (was 5e-6)
    warmup_ratio: float = 0.25,  # Longer warmup for stability (was 0.15)
    weight_decay: float = 0.01,
    gradient_clip: float = 1.0,  # Reduced to prevent gradient explosion (was 3.0)
    default_epochs: int = 5,
    epochs_per_task: Optional[Dict[str, int]] = None,
    # Hardware
    hardware_config: Optional[HardwareConfig] = None,
    # Checkpoint directory
    checkpoint_dir: Optional[str] = None,
) -> RuntimeConfig:
    """
    Create configuration with direct parameters (no presets required)

    This is the new simplified way to configure DAWN without preset names.
    All parameters have sensible defaults and can be overridden individually.

    Args:
        phase: Training phase (1 or 2)
        hidden_size: Model hidden dimension
        vocab_size: Vocabulary size
        max_length: Maximum sequence length
        num_heads: Number of attention heads
        intermediate_size: FFN intermediate size
        num_steps: Number of refinement steps
        dropout: Dropout probability
        expert_names: List of expert names (None = all three: lexical, syntactic, semantic)
        base_lr: Base learning rate
        embedding_lr: Embedding learning rate (usually lower)
        warmup_ratio: Warmup ratio for scheduler
        weight_decay: Weight decay for AdamW
        gradient_clip: Gradient clipping threshold
        default_epochs: Default epochs for tasks
        epochs_per_task: Task-specific epoch overrides
        hardware_config: Hardware config (None = auto-detect)
        checkpoint_dir: Checkpoint directory override

    Returns:
        RuntimeConfig instance ready to use
    """

    # Default expert names
    if expert_names is None:
        expert_names = ["lexical", "syntactic", "semantic"]

    # Create model config
    model = ModelConfig(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        max_length=max_length,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_steps=num_steps,
        dropout=dropout,
    )

    # Create expert configs with standard settings
    experts = {}
    for name in expert_names:
        if num_steps == 4:
            step_weights = [0.1, 0.2, 0.3, 0.4]
        elif num_steps == 5:
            step_weights = [0.05, 0.05, 0.1, 0.2, 0.6]
        else:
            # Equal weights
            step_weights = [1.0 / num_steps] * num_steps

        experts[name] = ExpertConfig(
            name=name,
            step_weights=step_weights,
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        )

    # Create refiner config
    refiner = RefinerConfig(
        num_blocks=5,
        gating_type="query_key",
        temperature_scale="sqrt",
        attention_dropout=0.1,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.1,
        init_std=0.02,
        zero_init_final_layer=True,
        zero_init_gate=False,
    )

    # Create integrator config
    integrator = IntegratorConfig(
        integration_method="attention",
        use_query_from_mean=True,
        attention_temperature=1.0,
        use_residual=True,
        use_layer_norm=True,
        output_projection=True,
        init_std=0.02,
    )

    # Create training config with flattened structure (no nested OptimizerConfig)
    if epochs_per_task is None:
        epochs_per_task = DEFAULT_EPOCHS_PER_TASK.copy()

    if phase == 1:
        training = Phase1TrainingConfig(
            base_lr=base_lr,
            embedding_lr=embedding_lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            scheduler="linear",
            mixed_precision=True,
            default_epochs=default_epochs,
            epochs_per_task=epochs_per_task,
            save_every_n_epochs=1,
            checkpoint_dir=checkpoint_dir or "./checkpoints/phase1",
            log_every_n_steps=100,
            eval_every_n_epochs=1,
        )
        peer_prediction = None
        unified_gate = None
        phase2_tasks = None
    else:
        training = Phase2TrainingConfig(
            base_lr=base_lr,
            embedding_lr=embedding_lr,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            scheduler="linear",
            mixed_precision=True,
            epochs=default_epochs,
            freeze_experts=False,
            train_integrator=True,
            train_task_heads=True,
            task_loss_weight=1.0,
            peer_prediction_loss_weight=0.5,
            task_sampling="uniform",
            save_every_n_epochs=1,
            checkpoint_dir=checkpoint_dir or "./checkpoints/phase2",
            log_every_n_steps=100,
            eval_every_n_epochs=1,
        )
        peer_prediction = PEER_PREDICTION_PRESETS["standard"]
        unified_gate = UNIFIED_GATE_PRESETS["standard"]
        phase2_tasks = PHASE2_TASKS

    # Get hardware config
    if hardware_config is None:
        hardware_config = DEFAULT_HARDWARE

    # Get dataloader config
    dataloader = DataLoaderConfig(
        num_workers=hardware_config.num_workers,
        pin_memory=hardware_config.device == "cuda",
        prefetch_factor=2,
        persistent_workers=hardware_config.num_workers > 0,
    )

    # Filter tasks based on selected experts
    filtered_tasks = {
        name: task
        for name, task in TASKS.items()
        if task.expert_name in expert_names
    }

    # Create runtime config
    config = RuntimeConfig(
        model=model,
        refiner=refiner,
        integrator=integrator,
        expert_names=expert_names,
        experts=experts,
        tasks=filtered_tasks,
        datasets=DATASETS,
        text_filters=TEXT_FILTERS,
        phase=phase,
        training=training,
        dataloader=dataloader,
        hardware=hardware_config,
        model_preset="custom",  # Mark as custom config
        training_preset="custom",
        peer_prediction=peer_prediction,
        unified_gate=unified_gate,
        phase2_tasks=phase2_tasks,
    )

    return config


class ConfigRegistry:
    """
    Central configuration registry
    
    Manages all configuration presets and builds validated runtime configs.
    """
    
    def __init__(self):
        self.model_presets = MODEL_PRESETS
        self.expert_presets = EXPERT_PRESETS
        self.refiner_presets = REFINER_PRESETS
        self.integrator_presets = INTEGRATOR_PRESETS
        self.peer_prediction_presets = PEER_PREDICTION_PRESETS
        self.unified_gate_presets = UNIFIED_GATE_PRESETS
        # Training presets removed - use get_default_config() instead
        self.phase1_presets = {}
        self.phase2_presets = {}
        self.dataloader_presets = {"standard": DataLoaderConfig()}

        self.tasks = TASKS
        self.datasets = DATASETS
        self.text_filters = TEXT_FILTERS
    
    def build(
        self,
        # Phase
        phase: int = 1,
        
        # Model architecture
        model_preset: str = "base",
        
        # Experts
        expert_names: Optional[List[str]] = None,
        expert_preset: str = "standard",
        
        # Components
        refiner_preset: str = "standard",
        integrator_preset: str = "standard",
        
        # Phase 2 specific
        peer_prediction_preset: Optional[str] = None,
        unified_gate_preset: Optional[str] = None,
        phase2_tasks: Optional[List[str]] = None,
        
        # Training
        training_preset: Optional[str] = None,
        dataloader_preset: str = "standard",
        
        # Hardware
        hardware_config: Optional[HardwareConfig] = None,
        auto_detect_hardware: bool = True,
        
    ) -> RuntimeConfig:
        """
        Build complete runtime configuration
        
        Args:
            phase: Training phase (1 or 2)
            model_preset: Model size preset name
            expert_names: List of expert names to use (None = all)
            expert_preset: Expert configuration preset
            refiner_preset: Refiner configuration preset
            integrator_preset: Integrator configuration preset
            peer_prediction_preset: Peer prediction preset (Phase 2 only)
            unified_gate_preset: Unified gate preset (Phase 2 only)
            phase2_tasks: Tasks for Phase 2 training
            training_preset: Training configuration preset
            dataloader_preset: DataLoader configuration preset
            hardware_config: Hardware configuration (None = auto-detect)
            auto_detect_hardware: Whether to auto-detect hardware
            
        Returns:
            RuntimeConfig instance
        """
        
        # 1. Get model configuration
        if model_preset not in self.model_presets:
            raise ConfigError(f"Unknown model preset: {model_preset}")
        model = copy.deepcopy(self.model_presets[model_preset])
        
        # 2. Get expert configurations
        if expert_names is None:
            expert_names = ["lexical", "syntactic", "semantic"]
        
        if expert_preset not in self.expert_presets:
            raise ConfigError(f"Unknown expert preset: {expert_preset}")
        
        experts = {}
        for name in expert_names:
            if name not in self.expert_presets[expert_preset]:
                raise ConfigError(f"Expert '{name}' not found in preset '{expert_preset}'")
            experts[name] = copy.deepcopy(self.expert_presets[expert_preset][name])
        
        # 3. Filter tasks based on selected experts
        filtered_tasks = {
            task_name: copy.deepcopy(task_cfg)
            for task_name, task_cfg in self.tasks.items()
            if task_cfg.expert_name in expert_names
        }
        
        # 4. Get component configurations
        if refiner_preset not in self.refiner_presets:
            raise ConfigError(f"Unknown refiner preset: {refiner_preset}")
        refiner = copy.deepcopy(self.refiner_presets[refiner_preset])
        
        if integrator_preset not in self.integrator_presets:
            raise ConfigError(f"Unknown integrator preset: {integrator_preset}")
        integrator = copy.deepcopy(self.integrator_presets[integrator_preset])
        
        # 5. Phase-specific configuration
        peer_prediction = None
        unified_gate = None
        
        if phase == 2:
            # Peer prediction
            if peer_prediction_preset is None:
                peer_prediction_preset = "standard"
            if peer_prediction_preset not in self.peer_prediction_presets:
                raise ConfigError(f"Unknown peer prediction preset: {peer_prediction_preset}")
            peer_prediction = copy.deepcopy(self.peer_prediction_presets[peer_prediction_preset])
            
            # Unified gate
            if unified_gate_preset is None:
                unified_gate_preset = "standard"
            if unified_gate_preset not in self.unified_gate_presets:
                raise ConfigError(f"Unknown unified gate preset: {unified_gate_preset}")
            unified_gate = copy.deepcopy(self.unified_gate_presets[unified_gate_preset])
            
            # Phase 2 tasks
            if phase2_tasks is None:
                phase2_tasks = PHASE2_TASKS
        
        # 6. Get training configuration
        if training_preset is None:
            training_preset = "standard"
        
        if phase == 1:
            if training_preset not in self.phase1_presets:
                raise ConfigError(f"Unknown Phase 1 training preset: {training_preset}")
            training = copy.deepcopy(self.phase1_presets[training_preset])
        else:
            if training_preset not in self.phase2_presets:
                raise ConfigError(f"Unknown Phase 2 training preset: {training_preset}")
            training = copy.deepcopy(self.phase2_presets[training_preset])
        
        # 7. Get dataloader configuration
        if dataloader_preset not in self.dataloader_presets:
            raise ConfigError(f"Unknown dataloader preset: {dataloader_preset}")
        dataloader = copy.deepcopy(self.dataloader_presets[dataloader_preset])
        
        # 8. Get hardware configuration
        if hardware_config is None and auto_detect_hardware:
            hardware_config = DEFAULT_HARDWARE
        elif hardware_config is None:
            # Use default settings without auto-detection
            device = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
            hardware_config = HardwareConfig(
                device=device,
                batch_size=32,
                num_workers=4,
            )
        
        # 9. Build runtime configuration
        runtime_config = RuntimeConfig(
            model=model,
            refiner=refiner,
            integrator=integrator,
            expert_names=expert_names,
            experts=experts,
            tasks=filtered_tasks,
            datasets=copy.deepcopy(self.datasets),
            text_filters=copy.deepcopy(self.text_filters),
            phase=phase,
            model_preset=model_preset,
            training_preset=training_preset,
            training=training,
            dataloader=dataloader,
            hardware=hardware_config,
            peer_prediction=peer_prediction,
            unified_gate=unified_gate,
            phase2_tasks=phase2_tasks,
        )
        
        return runtime_config


# ============================================================================
# Global Registry Instance
# ============================================================================

_registry = ConfigRegistry()


# ============================================================================
# Public API
# ============================================================================

def get_config(**kwargs) -> RuntimeConfig:
    """
    Get configuration (single entry point)
    
    See ConfigRegistry.build() for all available arguments.
    """
    return _registry.build(**kwargs)


def get_phase1_config(
    model_preset: str = "base",
    expert_preset: str = "standard",
    training_preset: str = "standard",
    **kwargs
) -> RuntimeConfig:
    """Get Phase 1 configuration"""
    return get_config(
        phase=1,
        model_preset=model_preset,
        expert_preset=expert_preset,
        training_preset=training_preset,
        **kwargs
    )


def get_phase2_config(
    model_preset: str = "base",
    peer_prediction_preset: str = "standard",
    training_preset: str = "standard",
    **kwargs
) -> RuntimeConfig:
    """Get Phase 2 configuration"""
    return get_config(
        phase=2,
        model_preset=model_preset,
        peer_prediction_preset=peer_prediction_preset,
        training_preset=training_preset,
        **kwargs
    )
