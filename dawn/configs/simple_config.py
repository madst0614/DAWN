"""
Simple DAWN Configuration

Single config class with sensible defaults.
No preset names required - just customize what you need!
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from types import SimpleNamespace

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class DAWNConfig:
    """
    Simple DAWN Configuration

    All parameters have sensible defaults based on PNN paper.
    Just override what you want to change!

    Example:
        # Minimal - use all defaults
        config = DAWNConfig()

        # Custom size
        config = DAWNConfig(hidden_size=512, num_heads=8)

        # Phase 2
        config = DAWNConfig(phase=2, expert_names=["lexical", "semantic"])
    """

    # ============================================================
    # Core Architecture (PNN defaults)
    # ============================================================
    hidden_size: int = 768
    vocab_size: int = 30522  # BERT tokenizer
    max_length: int = 128  # Match PNN (512 causes 16x memory usage due to O(L²) attention)
    num_heads: int = 12
    num_steps: int = 4  # Refinement steps
    dropout: float = 0.1

    # Mountain-shaped FFN (PNN hierarchical structure)
    intermediate_size: List[int] = field(
        default_factory=lambda: [1024, 1536, 2048, 1536, 1024]
    )

    # ============================================================
    # Experts
    # ============================================================
    expert_names: List[str] = field(
        default_factory=lambda: ["lexical", "syntactic", "semantic"]
    )

    # ============================================================
    # Training Phase
    # ============================================================
    phase: int = 1  # 1 or 2

    # ============================================================
    # Training Hyperparameters (Match PNN hierarchical success)
    # ============================================================
    learning_rate: float = 3e-4  # Match PNN hierarchical
    embedding_lr: float = 1e-4  # Match PNN hierarchical
    warmup_ratio: float = 0.1  # Match PNN hierarchical
    weight_decay: float = 0.01
    gradient_clip: float = 1.0  # Match PNN hierarchical

    batch_size: int = 32
    epochs: int = 6  # Match PNN default
    gradient_accumulation_steps: int = 1

    # Task-specific epochs (optional)
    epochs_per_task: Optional[Dict[str, int]] = None

    # ============================================================
    # Component Settings (rarely need to change)
    # ============================================================

    # Debug mode (enables detailed logging and NaN checks)
    debug_mode: bool = False

    # Delta Refiner
    num_refiner_blocks: int = 5
    refiner_dropout: float = 0.1
    temperature_scale: str = "sqrt"  # or float
    init_std: float = 0.02

    # Peer Context (Phase 2)
    peer_projection_rank: int = 256

    # Expert Integrator (Phase 2)
    integrator_blocks: int = 3
    base_expert_name: Optional[str] = None  # Auto: first in expert_names

    # ============================================================
    # Hardware
    # ============================================================
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = True
    num_workers: int = 4

    # ============================================================
    # Paths
    # ============================================================
    checkpoint_dir: Optional[str] = None

    def __post_init__(self):
        """Validate and set defaults"""

        # Validate architecture
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

        # Validate phase
        assert self.phase in [1, 2], f"phase must be 1 or 2, got {self.phase}"

        # Auto-detect hardware (batch size, device, etc.)
        from .hardware import detect_hardware
        hw = detect_hardware()

        # Apply hardware-detected settings
        self.batch_size = hw.batch_size
        self.gradient_accumulation_steps = hw.gradient_accumulation_steps
        self.num_workers = hw.num_workers
        self.mixed_precision = hw.mixed_precision

        # Auto-detect device
        if self.device == "auto":
            self.device = hw.device

        # Set checkpoint dir
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"./checkpoints/phase{self.phase}"

        # Set base expert for Phase 2
        if self.phase == 2 and self.base_expert_name is None:
            self.base_expert_name = self.expert_names[0]

        # Import task epochs from data.py (single source of truth)
        from .data import DEFAULT_EPOCHS_PER_TASK
        if self.epochs_per_task is None:
            self.epochs_per_task = DEFAULT_EPOCHS_PER_TASK.copy()

        # Legacy compatibility: Create nested namespaces like RuntimeConfig
        self._create_legacy_compat()

    def _create_legacy_compat(self):
        """Create nested attributes for backward compatibility with legacy scripts"""
        # Model namespace
        self.model = SimpleNamespace(
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            max_length=self.max_length,
            num_heads=self.num_heads,
            intermediate_size=self.intermediate_size,
            num_steps=self.num_steps,
            dropout=self.dropout,
        )

        # Training namespace (flattened optimizer params - no nesting)
        self.training = SimpleNamespace(
            base_lr=self.learning_rate,
            embedding_lr=self.embedding_lr,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip,
            scheduler="linear",
            mixed_precision=self.mixed_precision,
            default_epochs=self.epochs,
            epochs_per_task=self.epochs_per_task,
            checkpoint_dir=self.checkpoint_dir,
            get_epochs_for_task=self.get_epochs_for_task,
        )

        # Hardware namespace
        self.hardware = SimpleNamespace(
            batch_size=self.batch_size,
            device=self.device,
            mixed_precision=self.mixed_precision,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        # Dataloader namespace
        self.dataloader = SimpleNamespace(
            num_workers=self.num_workers,
            pin_memory=self.device == "cuda",
            prefetch_factor=2,
            persistent_workers=self.num_workers > 0,
        )

        # Load tasks from data.py
        try:
            from .data import TASKS, PHASE2_TASKS
            # Filter tasks by expert names
            self.tasks = {
                name: task
                for name, task in TASKS.items()
                if task.expert_name in self.expert_names
            }
            # Phase 2 tasks
            if self.phase == 2:
                self.phase2_tasks = PHASE2_TASKS
            else:
                self.phase2_tasks = None
        except ImportError:
            self.tasks = {}
            self.phase2_tasks = None

        # Preset names (for checkpoint naming)
        self.model_preset = "custom"
        self.training_preset = "custom"

    # ============================================================
    # Conversion to models format
    # ============================================================

    def to_model_config(self) -> Dict[str, Any]:
        """
        Convert to models.DAWN config format

        Returns:
            Dict ready for DAWN(**config)
        """
        config = {
            # Model basics
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "max_length": self.max_length,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "num_steps": self.num_steps,
            "dropout": self.dropout,
            "expert_names": self.expert_names,
            "init_std": self.init_std,

            # Delta Module
            "delta_module": {
                "num_blocks": self.num_refiner_blocks,
                "refiner": {
                    "type": getattr(self, 'refiner_type', 'gated'),  # 'gated' or 'simple'
                    "attention_dropout": self.refiner_dropout,
                    "ffn_dropout": self.refiner_dropout,
                    "zero_init_final_layer": True,
                    "init_std": self.init_std,
                    "use_gradient_checkpointing": True,  # Memory optimization
                    "debug_mode": self.debug_mode,  # Enable debug logging
                    "gate": {
                        "temperature_scale": self.temperature_scale,
                        "init_std": self.init_std,
                        "debug_mode": self.debug_mode,
                    },
                },
            },

            # Integration
            "integration": {
                "use_context_modulation": True,
                "context_modulation": {
                    "hidden_dim": self.hidden_size,
                },
                "dropout": self.dropout,
                "init_std": self.init_std,
                "debug_mode": self.debug_mode,
            },
        }

        # Phase 2 specific
        if self.phase == 2:
            config["peer_context"] = {
                "num_heads": self.num_heads,
                "projection_rank": self.peer_projection_rank,
                "dropout": self.dropout,
                "init_std": self.init_std,
            }

            config["expert_integrator"] = {
                "base_expert": self.base_expert_name,
                "expert_context": {
                    "num_heads": self.num_heads,
                    "projection_rank": self.hidden_size // 2,
                    "dropout": self.dropout,
                    "init_std": self.init_std,
                },
                "delta_module": {
                    "num_blocks": self.integrator_blocks,
                    "num_heads": self.num_heads,
                    "intermediate_size": self.hidden_size * 4,
                    "dropout": self.dropout,
                    "refiner": {
                        "type": getattr(self, 'refiner_type', 'gated'),  # 'gated' or 'simple'
                        "attention_dropout": self.dropout,
                        "ffn_dropout": self.dropout,
                        "zero_init_final_layer": True,
                        "init_std": self.init_std,
                        "use_gradient_checkpointing": True,
                        "gate": {
                            "temperature_scale": self.temperature_scale,
                            "init_std": self.init_std,
                        },
                    },
                },
                "integration": {
                    "use_context_modulation": True,
                    "dropout": self.dropout,
                    "init_std": self.init_std,
                },
                "gate": {
                    "temperature_scale": self.temperature_scale,
                    "init_std": self.init_std,
                },
            }

        return config

    def get_model_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for models.DAWN constructor

        Returns:
            Dict with config, enable_peer_prediction, active_experts
        """
        return {
            "config": self.to_model_config(),
            "enable_peer_prediction": self.phase == 2,
            "active_experts": self.expert_names,
        }

    # ============================================================
    # Utility
    # ============================================================

    def get_epochs_for_task(self, task_name: str) -> int:
        """Get epochs for specific task"""
        return self.epochs_per_task.get(task_name, self.epochs)

    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*70)
        print(f"DAWN Configuration (Phase {self.phase})")
        print("="*70)

        print("\n📐 Architecture:")
        print(f"  Hidden: {self.hidden_size}d x {self.num_heads} heads")
        print(f"  Steps: {self.num_steps}")
        print(f"  FFN: {self.intermediate_size}")
        print(f"  Dropout: {self.dropout}")

        print("\n👥 Experts:")
        for name in self.expert_names:
            print(f"  - {name}")

        if self.phase == 2:
            print(f"\n🤝 Phase 2 Settings:")
            print(f"  Base expert: {self.base_expert_name}")
            print(f"  Peer projection: {self.peer_projection_rank}d")
            print(f"  Integrator blocks: {self.integrator_blocks}")

        print("\n⚙️  Training:")
        print(f"  LR: {self.learning_rate} (base), {self.embedding_lr} (emb)")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Warmup: {self.warmup_ratio}")
        print(f"  Gradient clip: {self.gradient_clip}")

        print("\n💻 Hardware:")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.mixed_precision}")
        print(f"  Workers: {self.num_workers}")

        print("\n📁 Checkpoints:")
        print(f"  Directory: {self.checkpoint_dir}")

        print("\n" + "="*70 + "\n")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DAWNConfig':
        """Create config from dictionary"""
        # Filter out any keys that aren't in the dataclass
        import inspect
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


# ============================================================
# Quick Presets (optional convenience)
# ============================================================

def get_small_config(**kwargs) -> DAWNConfig:
    """Small model (for testing)"""
    defaults = {
        "hidden_size": 256,
        "num_heads": 4,
        "intermediate_size": [512, 768, 1024, 768, 512],
        "batch_size": 64,
    }
    return DAWNConfig(**{**defaults, **kwargs})


def get_base_config(**kwargs) -> DAWNConfig:
    """Base model (default)"""
    return DAWNConfig(**kwargs)


def get_large_config(**kwargs) -> DAWNConfig:
    """Large model"""
    defaults = {
        "hidden_size": 1024,
        "num_heads": 16,
        "intermediate_size": [2048, 3072, 4096, 3072, 2048],
        "batch_size": 16,
    }
    return DAWNConfig(**{**defaults, **kwargs})
