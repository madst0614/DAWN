"""
Utils Module

Unified interface for data loading, task management, and model creation.

Usage:
    # Data loading
    from utils import get_dataloader, DATASET_MAP
    loader = get_dataloader("mlm", split="train", batch_size=32, max_samples=100000)

    # Model creation
    from utils import create_model_from_config, create_phase1_model, ModelFactory
    model, config = create_phase1_model("base")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Utilities
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from .data_utils import (
    # Core utilities
    get_tokenizer,
    get_spacy_nlp,
    get_cached_pos_tags,
    # Data loading
    load_wikipedia_streaming,
    load_huggingface_dataset,
    # Base classes
    BaseDataset,
    TextValidationMixin,
    MaskingStrategy,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from .task_registry import (
    # Pre-training tasks
    MLMDataset,
    SpanMaskingDataset,
    TokenDeletionDataset,
    TextInfillingDataset,
    SentencePermutationDataset,
    # Understanding tasks
    NLIDataset,
    WiCDataset,
    NSPDataset,
    # Reasoning tasks
    COPAStyleDataset,
    COPADataset,
    BoolQDataset,
    # Generation tasks
    StoryCompletionDataset,
    StoryClozeDataset,
    SentenceOrderingDataset,
    # Paraphrase tasks
    ParaphraseDataset,
    STSDataset,
    # Registry and factory
    DATASET_MAP,
    get_dataloader,
    get_task_max_samples,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Model Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from .model_factory import (
    # Factory class
    ModelFactory,
    # Convenience functions
    create_model_from_config,
    create_phase1_model,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Exports
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

__all__ = [
    # === Data Utilities ===
    "get_tokenizer",
    "get_spacy_nlp",
    "get_cached_pos_tags",
    "load_wikipedia_streaming",
    "load_huggingface_dataset",
    "BaseDataset",
    "TextValidationMixin",
    "MaskingStrategy",
    # === Pre-training Tasks ===
    "MLMDataset",
    "SpanMaskingDataset",
    "TokenDeletionDataset",
    "TextInfillingDataset",
    "SentencePermutationDataset",
    # === Understanding Tasks ===
    "NLIDataset",
    "WiCDataset",
    "NSPDataset",
    # === Reasoning Tasks ===
    "COPAStyleDataset",
    "COPADataset",
    "BoolQDataset",
    # === Generation Tasks ===
    "StoryCompletionDataset",
    "StoryClozeDataset",
    "SentenceOrderingDataset",
    # === Paraphrase Tasks ===
    "ParaphraseDataset",
    "STSDataset",
    # === Data Factory ===
    "DATASET_MAP",
    "get_dataloader",
    "get_task_max_samples",
    # === Model Factory ===
    "ModelFactory",
    "create_model_from_config",
    "create_phase1_model",
]
