"""
Data and Task Configuration

Defines all tasks, datasets, and their properties.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class TaskConfig:
    """Configuration for a single task"""
    name: str
    type: Literal["token_level", "sequence_level"]
    expert_name: str  # Which expert handles this task
    max_samples: int
    dataset_source: str
    
    # Task-specific parameters
    task_params: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        assert self.type in ["token_level", "sequence_level"]
        assert self.max_samples > 0


@dataclass
class DatasetConfig:
    """Dataset source configuration"""
    name: str
    path: str
    dataset_name: Optional[str] = None
    streaming: bool = True
    split: str = "train"


@dataclass
class TextFilterConfig:
    """Text filtering parameters"""
    min_words: int = 10  # 5 → 10 (filter out very short texts)
    max_words: int = 300  # 150 → 300 (WikiText-103 has longer documents)
    min_chars: int = 30  # 20 → 30 (stricter filtering)
    min_alpha_ratio: float = 0.5
    exclude_starts_with: List[str] = field(default_factory=lambda: ["=", "#", "@"])
    exclude_contains: List[str] = field(default_factory=lambda: ["http://", "https://"])
    sep_exclusion_margin: int = 2


# ============================================================================
# Task Definitions
# ============================================================================

TASKS: Dict[str, TaskConfig] = {
    # Lexical Tasks
    "mlm": TaskConfig(
        name="mlm",
        type="token_level",
        expert_name="lexical",
        max_samples=200_000,
        dataset_source="wikitext",  # wikipedia → wikitext (WikiText-103)
        task_params={
            "mask_prob": 0.15,
            "mask_token_ratio": 0.8,
            "random_token_ratio": 0.5,
        },
    ),
    "wic": TaskConfig(
        name="wic",
        type="sequence_level",
        expert_name="lexical",
        max_samples=100_000,
        dataset_source="wic",
        task_params={},
    ),
    
    # Syntactic Tasks
    "span_masking": TaskConfig(
        name="span_masking",
        type="token_level",
        expert_name="syntactic",
        max_samples=200_000,
        dataset_source="wikipedia",
        task_params={
            "mask_prob": 0.15,
            "mean_span": 3.8,
            "max_span_length": 8,
            "span_difficulty": "medium",
        },
    ),
    "token_deletion": TaskConfig(
        name="token_deletion",
        type="token_level",
        expert_name="lexical",  # BART: Token-level corruption (Lexical)
        max_samples=200_000,
        dataset_source="wikipedia",
        task_params={
            "delete_prob": 0.15,
        },
    ),
    "text_infilling": TaskConfig(
        name="text_infilling",
        type="token_level",
        expert_name="syntactic",
        max_samples=200_000,
        dataset_source="wikipedia",
        task_params={
            "mask_prob": 0.30,
            "mean_span": 3.0,
            "max_span_length": 10,
            "single_mask_per_span": True,
        },
    ),
    "sentence_permutation": TaskConfig(
        name="sentence_permutation",
        type="sequence_level",
        expert_name="semantic",  # BART: Discourse/Document-level (Semantic)
        max_samples=200_000,
        dataset_source="wikipedia",
        task_params={
            "shuffle_prob": 0.5,  # 50% shuffled, 50% original order
            "min_sentences": 3,
            "max_sentences": 6,
        },
    ),
    
    # Semantic Tasks
    "nli": TaskConfig(
        name="nli",
        type="sequence_level",
        expert_name="semantic",
        max_samples=100_000,
        dataset_source="snli",
        task_params={},
    ),
    "paraphrase": TaskConfig(
        name="paraphrase",
        type="sequence_level",
        expert_name="semantic",
        max_samples=50_000,
        dataset_source="paws",
        task_params={},
    ),
    "sts": TaskConfig(
        name="sts",
        type="sequence_level",
        expert_name="semantic",
        max_samples=40_000,
        dataset_source="stsb",
        task_params={},
    ),
}


# ============================================================================
# Dataset Sources
# ============================================================================

DATASETS: Dict[str, DatasetConfig] = {
    "wikipedia": DatasetConfig(
        name="wikipedia",
        path="wikimedia/wikipedia",
        dataset_name="20231101.en",
        streaming=True,
        split="train",
    ),
    "wikitext": DatasetConfig(
        name="wikitext",
        path="Salesforce/wikitext",
        dataset_name="wikitext-103-raw-v1",
        streaming=True,
        split="train",
    ),
    "snli": DatasetConfig(
        name="snli",
        path="snli",
        dataset_name=None,
        streaming=False,
        split="train",
    ),
    "paws": DatasetConfig(
        name="paws",
        path="paws",
        dataset_name="labeled_final",
        streaming=False,
        split="train",
    ),
    "stsb": DatasetConfig(
        name="stsb",
        path="stsb_multi_mt",
        dataset_name="en",
        streaming=False,
        split="train",
    ),
    "wic": DatasetConfig(
        name="wic",
        path="super_glue",
        dataset_name="wic",
        streaming=False,
        split="train",
    ),
}


# ============================================================================
# Text Filters
# ============================================================================

TEXT_FILTERS = TextFilterConfig()


# ============================================================================
# Curriculum Definitions
# ============================================================================

@dataclass
class CurriculumStage:
    """
    Single stage in curriculum

    NOTE: Epochs are NOT defined here - they come from training preset.
    Curriculum only defines the order of expert-task training.
    """
    name: str
    expert_name: str
    task_name: str


# Phase 1 Curriculum: Expert specialization
# NOTE: Epochs are defined in simple_config.py (DAWNConfig.epochs_per_task)
PHASE1_CURRICULUM: List[CurriculumStage] = [
    # Lexical Expert (BART: token-level corruption)
    CurriculumStage("lexical_mlm", "lexical", "mlm"),
    # WiC removed - too small dataset (5428 samples)
    CurriculumStage("lexical_deletion", "lexical", "token_deletion"),  # BART: Lexical

    # Syntactic Expert (structural corruption)
    CurriculumStage("syntactic_span", "syntactic", "span_masking"),
    CurriculumStage("syntactic_infilling", "syntactic", "text_infilling"),

    # Semantic Expert (meaning & discourse)
    CurriculumStage("semantic_nli", "semantic", "nli"),
    CurriculumStage("semantic_paraphrase", "semantic", "paraphrase"),
    CurriculumStage("semantic_sts", "semantic", "sts"),
    CurriculumStage("semantic_permutation", "semantic", "sentence_permutation"),  # BART: Discourse
]


# Phase 2: Multi-task collaborative training
PHASE2_TASKS: List[str] = ["mlm", "nli", "paraphrase"]


# ============================================================================
# Training Epochs per Task (centralized)
# ============================================================================

# Single source of truth for task epochs
# Used by DAWNConfig.get_epochs_for_task()
DEFAULT_EPOCHS_PER_TASK: Dict[str, int] = {
    "mlm": 15,  # Stage 1 (lexical_mlm)
    # "wic": 5,  # Removed - dataset too small (5428 samples)
    "span_masking": 8,
    "token_deletion": 4,
    "text_infilling": 6,
    "sentence_permutation": 4,
    "nli": 8,
    "paraphrase": 5,
    "sts": 5,
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_tasks_for_expert(expert_name: str) -> List[str]:
    """Get all task names for an expert"""
    return [
        task_name 
        for task_name, task_config in TASKS.items() 
        if task_config.expert_name == expert_name
    ]


def get_expert_for_task(task_name: str) -> str:
    """Get expert name for a task"""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASKS[task_name].expert_name


def get_primary_task_for_expert(expert_name: str) -> str:
    """Get primary (first) task for an expert"""
    tasks = get_tasks_for_expert(expert_name)
    if not tasks:
        raise ValueError(f"No tasks found for expert: {expert_name}")
    return tasks[0]


def validate_task_expert_mapping(expert_names: List[str], task_names: List[str]):
    """Validate that all tasks have corresponding experts"""
    errors = []
    
    for task_name in task_names:
        if task_name not in TASKS:
            errors.append(f"Unknown task: {task_name}")
            continue
            
        expert = TASKS[task_name].expert_name
        if expert not in expert_names:
            errors.append(
                f"Task '{task_name}' requires expert '{expert}' which is not in expert_names"
            )
    
    if errors:
        raise ValueError("\n".join(errors))


# ============================================================================
# Legacy Constants for Backwards Compatibility
# ============================================================================

# Export task-specific configs for data_utils.py compatibility
MLM_CONFIG = TASKS["mlm"].task_params
SPAN_MASKING_CONFIG = {
    "mask_prob": TASKS["span_masking"].task_params.get("mask_prob", 0.15),
    "mean_span": TASKS["span_masking"].task_params.get("mean_span", 3.8),
    "max_span_length": TASKS["span_masking"].task_params.get("max_span_length", 8),
}
TOKEN_DELETION_CONFIG = TASKS["token_deletion"].task_params
TEXT_INFILLING_CONFIG = TASKS["text_infilling"].task_params
SENTENCE_PERMUTATION_CONFIG = TASKS["sentence_permutation"].task_params

# Dataset sizes
DATASET_SIZES = {task_name: task_config.max_samples for task_name, task_config in TASKS.items()}

# Tokenizer config
TOKENIZER_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,  # Match PNN (512 causes 16x memory usage)
    "use_fast": True,
}

# Dataloader config (basic defaults)
DATALOADER_CONFIG = {
    "batch_size": 64,
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
    "persistent_workers": True,
}

# Dataset sources mapping (legacy format for data_utils.py)
DATASET_SOURCES = {
    "wikipedia": {
        "primary": {
            "path": "wikimedia/wikipedia",
            "name": "20231101.en",
        },
        "fallback": {
            "path": "Salesforce/wikitext",
            "name": "wikitext-103-raw-v1",
        },
    },
}
