"""
Model Architecture Configuration

Single source of truth for all model architecture parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Core model architecture parameters"""
    hidden_size: int
    vocab_size: int
    max_length: int
    num_heads: int
    intermediate_size: int | List[int]  # Single value or mountain-shaped list
    num_steps: int  # Refinement steps for PNN
    dropout: float
    use_parallel_experts: bool = True  # Use CUDA streams for parallel expert execution (Phase 2)

    def __post_init__(self):
        # Validation
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

        # Validate intermediate_size
        if isinstance(self.intermediate_size, list):
            assert all(size > 0 for size in self.intermediate_size), \
                "All intermediate sizes must be positive"
        else:
            assert self.intermediate_size > self.hidden_size, \
                "intermediate_size should be larger than hidden_size"


@dataclass
class RefinerConfig:
    """Hierarchical PNN Refiner configuration"""
    refiner_type: str = "gated"  # "gated" (with GeGLU gates) or "simple" (no gates)
    num_blocks: int = 5  # Number of hierarchical blocks
    gating_type: str = "query_key"  # "query_key" or "unified" (only for gated type)
    temperature_scale: str = "sqrt"  # "sqrt" or float (only for gated type)
    attention_dropout: float = 0.1
    attention_type: str = "multihead"  # "multihead" or "linear"
    ffn_activation: str = "gelu"
    ffn_dropout: float = 0.1
    init_std: float = 0.02
    zero_init_final_layer: bool = True
    zero_init_gate: bool = False  # Only for gated type


@dataclass
class ExpertConfig:
    """Expert-specific configuration"""
    name: str
    step_weights: List[float]  # Weight for each refinement step
    share_embeddings: bool = False
    use_positional_encoding: bool = True
    use_recurrent_loss: bool = True
    final_step_only: bool = False
    
    def __post_init__(self):
        # Normalize step weights
        total = sum(self.step_weights)
        if total > 0:
            self.step_weights = [w / total for w in self.step_weights]


@dataclass
class IntegratorConfig:
    """Multi-expert integration configuration"""
    integration_method: str = "attention"  # "attention", "learned_weights", "average"
    use_query_from_mean: bool = True
    attention_temperature: float = 1.0
    use_residual: bool = True
    use_layer_norm: bool = True
    output_projection: bool = True
    init_std: float = 0.02


@dataclass
class FeatureExtractionConfig:
    """Feature extraction settings for peer collaboration"""
    num_extraction_heads: int = 4
    extraction_dropout: float = 0.1
    init_std: float = 0.02


@dataclass
class ComplementGateConfig:
    """Complement-based gating network settings"""
    hidden_multiplier: int = 2
    num_layers: int = 2
    dropout: float = 0.1
    use_layer_norm: bool = True
    init_std: float = 0.02


@dataclass
class PeerPredictionConfig:
    """Phase 2 peer collaboration configuration"""
    # Collaboration mode
    collaboration_mode: str = "feature_extraction"  # "feature_extraction" or "prediction"

    # Feature extraction settings (NEW)
    aspect_dim: Optional[int] = None  # None = hidden_size // 2

    # Peer graph: defines which peers each expert learns from
    # None = fully connected (each expert learns from all others)
    # Example: {"lexical": ["syntactic", "semantic"], ...}
    peer_graph: Optional[Dict[str, List[str]]] = None

    # Feature extraction sub-configs
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    complement_gate: ComplementGateConfig = field(default_factory=ComplementGateConfig)

    # Legacy prediction mode settings
    num_iterations: int = 2
    prediction_loss_weight: float = 0.1
    predictor_type: str = "linear"  # "linear" or "mlp"
    predictor_hidden_size: Optional[int] = None
    detach_predictions: bool = True
    symmetric_loss: bool = True


@dataclass
class UnifiedGateConfig:
    """Phase 2 unified gate configuration"""
    hidden_multiplier: int = 2
    num_layers: int = 2
    activation: str = "gelu"
    dropout: float = 0.1
    use_layer_norm: bool = False
    detach_peer_errors: bool = True
    normalize_errors: bool = False


# ============================================================================
# Model Size Presets
# ============================================================================

MODEL_PRESETS: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        hidden_size=256,
        vocab_size=30522,
        max_length=512,
        num_heads=4,
        intermediate_size=1024,
        num_steps=4,  # Match expert step_weights length
        dropout=0.1,
    ),
    "small": ModelConfig(
        hidden_size=512,
        vocab_size=30522,
        max_length=512,
        num_heads=8,
        intermediate_size=2048,
        num_steps=4,
        dropout=0.1,
    ),
    "base": ModelConfig(
        hidden_size=768,
        vocab_size=30522,
        max_length=128,  # Match PNN for memory efficiency
        num_heads=12,
        intermediate_size=[1024, 1536, 2048, 1536, 1024],  # Mountain-shaped (PNN hierarchical)
        num_steps=4,
        dropout=0.1,
    ),
    "large": ModelConfig(
        hidden_size=1024,
        vocab_size=30522,
        max_length=512,
        num_heads=16,
        intermediate_size=4096,
        num_steps=5,
        dropout=0.1,
    ),
}


# ============================================================================
# Expert Presets
# ============================================================================

EXPERT_PRESETS: Dict[str, Dict[str, ExpertConfig]] = {
    "standard": {
        "lexical": ExpertConfig(
            name="lexical",
            step_weights=[0.1, 0.2, 0.3, 0.4],
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
        "syntactic": ExpertConfig(
            name="syntactic",
            step_weights=[0.1, 0.2, 0.3, 0.4],
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
        "semantic": ExpertConfig(
            name="semantic",
            step_weights=[0.1, 0.2, 0.3, 0.4],
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
    },
    "efficient": {
        "lexical": ExpertConfig(
            name="lexical",
            step_weights=[0.0, 0.0, 0.0, 1.0],
            share_embeddings=True,
            use_positional_encoding=True,
            use_recurrent_loss=False,
            final_step_only=True,
        ),
        "syntactic": ExpertConfig(
            name="syntactic",
            step_weights=[0.0, 0.0, 0.0, 1.0],
            share_embeddings=True,
            use_positional_encoding=True,
            use_recurrent_loss=False,
            final_step_only=True,
        ),
        "semantic": ExpertConfig(
            name="semantic",
            step_weights=[0.0, 0.0, 0.0, 1.0],
            share_embeddings=True,
            use_positional_encoding=True,
            use_recurrent_loss=False,
            final_step_only=True,
        ),
    },
    "deep_refinement": {
        "lexical": ExpertConfig(
            name="lexical",
            step_weights=[0.05, 0.05, 0.1, 0.2, 0.6],  # 5 steps
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
        "syntactic": ExpertConfig(
            name="syntactic",
            step_weights=[0.05, 0.05, 0.1, 0.2, 0.6],  # 5 steps
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
        "semantic": ExpertConfig(
            name="semantic",
            step_weights=[0.05, 0.05, 0.1, 0.2, 0.6],  # 5 steps
            share_embeddings=False,
            use_positional_encoding=True,
            use_recurrent_loss=True,
            final_step_only=False,
        ),
    },
}


# ============================================================================
# Component Presets
# ============================================================================

REFINER_PRESETS: Dict[str, RefinerConfig] = {
    "standard": RefinerConfig(
        refiner_type="gated",
        num_blocks=5,  # Match PNN hierarchical
        gating_type="query_key",
        temperature_scale="sqrt",
        attention_dropout=0.1,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.1,
        init_std=0.02,
        zero_init_final_layer=True,
        zero_init_gate=False,
    ),
    "simple": RefinerConfig(
        refiner_type="simple",  # No gates, direct delta generation
        num_blocks=5,
        gating_type="query_key",  # Ignored for simple type
        temperature_scale="sqrt",  # Ignored for simple type
        attention_dropout=0.1,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.1,
        init_std=0.02,
        zero_init_final_layer=True,
        zero_init_gate=False,
    ),
    "deep": RefinerConfig(
        refiner_type="gated",
        num_blocks=8,  # Deeper hierarchy
        gating_type="query_key",
        temperature_scale="sqrt",
        attention_dropout=0.1,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.1,
        init_std=0.02,
        zero_init_final_layer=True,
        zero_init_gate=False,
    ),
    "aggressive_gating": RefinerConfig(
        refiner_type="gated",
        num_blocks=5,
        gating_type="query_key",
        temperature_scale="0.5",
        attention_dropout=0.1,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.1,
        init_std=0.02,
        zero_init_final_layer=True,
        zero_init_gate=True,
    ),
    "conservative": RefinerConfig(
        refiner_type="gated",
        num_blocks=5,
        gating_type="query_key",
        temperature_scale="2.0",
        attention_dropout=0.15,
        attention_type="multihead",
        ffn_activation="gelu",
        ffn_dropout=0.15,
        init_std=0.01,
        zero_init_final_layer=True,
        zero_init_gate=False,
    ),
}


INTEGRATOR_PRESETS: Dict[str, IntegratorConfig] = {
    "standard": IntegratorConfig(
        integration_method="attention",
        use_query_from_mean=True,
        attention_temperature=1.0,
        use_residual=True,
        use_layer_norm=True,
        output_projection=True,
        init_std=0.02,
    ),
    "simple_average": IntegratorConfig(
        integration_method="average",
        use_query_from_mean=False,
        attention_temperature=1.0,
        use_residual=False,
        use_layer_norm=True,
        output_projection=False,
        init_std=0.02,
    ),
    "learned_weights": IntegratorConfig(
        integration_method="learned_weights",
        use_query_from_mean=False,
        attention_temperature=1.0,
        use_residual=True,
        use_layer_norm=True,
        output_projection=True,
        init_std=0.02,
    ),
}


PEER_PREDICTION_PRESETS: Dict[str, PeerPredictionConfig] = {
    "standard": PeerPredictionConfig(
        collaboration_mode="feature_extraction",
        aspect_dim=None,  # hidden_size // 2
        peer_graph={
            "lexical": ["syntactic", "semantic"],
            "syntactic": ["lexical", "semantic"],
            "semantic": ["lexical", "syntactic"],
        },
        feature_extraction=FeatureExtractionConfig(
            num_extraction_heads=4,
            extraction_dropout=0.1,
            init_std=0.02,
        ),
        complement_gate=ComplementGateConfig(
            hidden_multiplier=2,
            num_layers=2,
            dropout=0.1,
            use_layer_norm=True,
            init_std=0.02,
        ),
        num_iterations=2,
        prediction_loss_weight=0.1,
        predictor_type="linear",
        predictor_hidden_size=None,
        detach_predictions=True,
        symmetric_loss=True,
    ),
    "aggressive": PeerPredictionConfig(
        collaboration_mode="feature_extraction",
        aspect_dim=None,
        peer_graph={
            "lexical": ["syntactic", "semantic"],
            "syntactic": ["lexical", "semantic"],
            "semantic": ["lexical", "syntactic"],
        },
        feature_extraction=FeatureExtractionConfig(
            num_extraction_heads=8,  # More heads
            extraction_dropout=0.15,
            init_std=0.02,
        ),
        complement_gate=ComplementGateConfig(
            hidden_multiplier=4,  # Larger gate
            num_layers=3,
            dropout=0.15,
            use_layer_norm=True,
            init_std=0.02,
        ),
        num_iterations=3,
        prediction_loss_weight=0.3,
        predictor_type="mlp",
        predictor_hidden_size=None,
        detach_predictions=False,
        symmetric_loss=True,
    ),
    "conservative": PeerPredictionConfig(
        collaboration_mode="feature_extraction",
        aspect_dim=None,
        peer_graph={
            "lexical": ["syntactic", "semantic"],
            "syntactic": ["lexical", "semantic"],
            "semantic": ["lexical", "syntactic"],
        },
        feature_extraction=FeatureExtractionConfig(
            num_extraction_heads=2,  # Fewer heads
            extraction_dropout=0.05,
            init_std=0.01,
        ),
        complement_gate=ComplementGateConfig(
            hidden_multiplier=1,  # Smaller gate
            num_layers=1,
            dropout=0.05,
            use_layer_norm=False,
            init_std=0.01,
        ),
        num_iterations=1,
        prediction_loss_weight=0.05,
        predictor_type="linear",
        predictor_hidden_size=None,
        detach_predictions=True,
        symmetric_loss=False,
    ),
}


UNIFIED_GATE_PRESETS: Dict[str, UnifiedGateConfig] = {
    "standard": UnifiedGateConfig(
        hidden_multiplier=2,
        num_layers=2,
        activation="gelu",
        dropout=0.1,
        use_layer_norm=False,
        detach_peer_errors=True,
        normalize_errors=False,
    ),
    "deep_gate": UnifiedGateConfig(
        hidden_multiplier=4,
        num_layers=3,
        activation="gelu",
        dropout=0.15,
        use_layer_norm=True,
        detach_peer_errors=True,
        normalize_errors=True,
    ),
    "lightweight": UnifiedGateConfig(
        hidden_multiplier=1,
        num_layers=1,
        activation="gelu",
        dropout=0.1,
        use_layer_norm=False,
        detach_peer_errors=True,
        normalize_errors=False,
    ),
}
