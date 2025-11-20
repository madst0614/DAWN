"""
Data utilities for SPROUT training.

Self-contained implementations inspired by dawn but standalone.
Includes advanced masking strategies and data loading.
"""

import torch
import numpy as np
import random
import pickle
import os
from typing import List, Set, Tuple, Optional


class SpanMasker:
    """
    Span-based masking for MLM.

    Masks contiguous spans of tokens instead of individual tokens.
    More challenging than standard MLM.
    """

    @staticmethod
    def create_span_mask(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        mask_prob: float = 0.15,
        mean_span: float = 3.0,
        max_span_length: int = 5
    ) -> Set[int]:
        """
        Create span-based masking indices.

        Args:
            input_ids: Token IDs [seq_len]
            attention_mask: Attention mask [seq_len]
            tokenizer: HuggingFace tokenizer
            mask_prob: Probability of masking
            mean_span: Mean span length (Poisson λ)
            max_span_length: Maximum span length

        Returns:
            Set of indices to mask
        """
        # Get valid indices (exclude padding and special tokens)
        valid_indices = torch.where(attention_mask == 1)[0].tolist()
        valid_indices = [
            i for i in valid_indices
            if input_ids[i] not in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id
            ]
        ]

        if len(valid_indices) == 0:
            return set()

        # Calculate number of tokens to mask
        num_to_mask = max(1, int(len(valid_indices) * mask_prob))
        masked_indices = set()

        # Safety parameters
        max_iterations = num_to_mask * 3
        iteration = 0

        while len(masked_indices) < num_to_mask and iteration < max_iterations:
            iteration += 1

            # Select start index
            available_set = set(valid_indices) - masked_indices
            if not available_set:
                break

            available = list(available_set)
            start_idx = random.choice(available)

            # Sample span length from Poisson distribution
            span_len = min(
                np.random.poisson(mean_span - 1) + 1,
                max_span_length,
                num_to_mask - len(masked_indices)
            )

            # Create contiguous span
            span = []
            for offset in range(span_len):
                next_idx = start_idx + offset
                if next_idx in valid_indices and next_idx not in masked_indices:
                    span.append(next_idx)
                else:
                    break

            if span:
                masked_indices.update(span)

            # Avoid over-masking
            if len(masked_indices) >= num_to_mask * 1.2:
                break

        return masked_indices


class TokenDeletion:
    """
    Token deletion for denoising.

    Randomly delete tokens - model must predict what was deleted.
    """

    @staticmethod
    def apply_deletion(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        delete_prob: float = 0.15
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply token deletion.

        Args:
            input_ids: Token IDs [seq_len]
            attention_mask: Attention mask [seq_len]
            tokenizer: HuggingFace tokenizer
            delete_prob: Probability of deleting each token

        Returns:
            Deleted input_ids and labels
        """
        # Get valid indices
        valid_indices = torch.where(attention_mask == 1)[0].tolist()
        valid_indices = [
            i for i in valid_indices
            if input_ids[i] not in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id
            ]
        ]

        # Sample deletion positions
        delete_mask = torch.bernoulli(
            torch.full((len(input_ids),), delete_prob)
        ).bool()

        # Don't delete special tokens
        special_mask = torch.tensor([
            tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0]
            for val in input_ids.tolist()
        ], dtype=torch.bool)
        delete_mask = delete_mask & ~special_mask

        # Create new sequence without deleted tokens
        keep_indices = ~delete_mask
        new_input_ids = input_ids[keep_indices]
        new_attention_mask = attention_mask[keep_indices]

        # Pad back to original length
        padding_length = len(input_ids) - len(new_input_ids)
        if padding_length > 0:
            new_input_ids = torch.cat([
                new_input_ids,
                torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            new_attention_mask = torch.cat([
                new_attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])

        labels = input_ids.clone()
        labels[~delete_mask] = -100  # Only predict deleted tokens

        return new_input_ids, labels


class TextValidator:
    """
    Validate text quality for training.

    Standalone text validator compatible with dawn's validation logic.
    """

    # Default validation parameters (matching dawn/configs/data.py)
    DEFAULT_MIN_WORDS = 10
    DEFAULT_MAX_WORDS = 300
    DEFAULT_MIN_CHARS = 30
    DEFAULT_MIN_ALPHA_RATIO = 0.5
    DEFAULT_EXCLUDE_STARTS = ["=", "#", "@", "*", "-", "Category:", "File:"]
    DEFAULT_EXCLUDE_CONTAINS = ["http://", "https://"]

    @staticmethod
    def is_valid_sentence(
        text: str,
        min_words: int = None,
        max_words: int = None,
        min_chars: int = None,
        min_alpha_ratio: float = None,
        exclude_starts_with: List[str] = None,
        exclude_contains: List[str] = None
    ) -> bool:
        """
        Check if sentence is valid for training.

        Args:
            text: Text to validate
            min_words: Minimum word count (default: 10)
            max_words: Maximum word count (default: 300)
            min_chars: Minimum character count (default: 30)
            min_alpha_ratio: Minimum alphabetic character ratio (default: 0.5)
            exclude_starts_with: Patterns to exclude if text starts with them
            exclude_contains: Patterns to exclude if text contains them

        Returns:
            True if valid
        """
        if not text:
            return False

        # Apply defaults
        min_words = min_words if min_words is not None else TextValidator.DEFAULT_MIN_WORDS
        max_words = max_words if max_words is not None else TextValidator.DEFAULT_MAX_WORDS
        min_chars = min_chars if min_chars is not None else TextValidator.DEFAULT_MIN_CHARS
        min_alpha_ratio = min_alpha_ratio if min_alpha_ratio is not None else TextValidator.DEFAULT_MIN_ALPHA_RATIO
        exclude_starts_with = exclude_starts_with if exclude_starts_with is not None else TextValidator.DEFAULT_EXCLUDE_STARTS
        exclude_contains = exclude_contains if exclude_contains is not None else TextValidator.DEFAULT_EXCLUDE_CONTAINS

        word_count = len(text.split())
        char_count = len(text)

        # Length checks
        if not (min_words <= word_count <= max_words):
            return False

        if char_count < min_chars:
            return False

        # Exclude special formats
        for pattern in exclude_starts_with:
            if text.startswith(pattern):
                return False

        for pattern in exclude_contains:
            if pattern in text:
                return False

        # Minimum alphabetic ratio
        alpha_count = sum(c.isalpha() for c in text)
        if char_count > 0 and alpha_count / char_count < min_alpha_ratio:
            return False

        return True

    @staticmethod
    def filter_sentences(sentences: List[str], **kwargs) -> List[str]:
        """
        Filter list of sentences.

        Args:
            sentences: List of sentences
            **kwargs: Arguments for is_valid_sentence

        Returns:
            Filtered list
        """
        return [
            s for s in sentences
            if TextValidator.is_valid_sentence(s, **kwargs)
        ]


class DatasetStats:
    """
    Track dataset statistics during training.
    """

    def __init__(self):
        self.total_samples = 0
        self.total_tokens = 0
        self.masked_tokens = 0
        self.sequence_lengths = []

    def update(self, batch_size: int, seq_len: int, num_masked: int):
        """Update statistics."""
        self.total_samples += batch_size
        self.total_tokens += batch_size * seq_len
        self.masked_tokens += num_masked
        self.sequence_lengths.append(seq_len)

    def get_summary(self) -> dict:
        """Get summary statistics."""
        return {
            'total_samples': self.total_samples,
            'total_tokens': self.total_tokens,
            'masked_tokens': self.masked_tokens,
            'mask_ratio': self.masked_tokens / self.total_tokens if self.total_tokens > 0 else 0,
            'avg_seq_len': np.mean(self.sequence_lengths) if self.sequence_lengths else 0,
            'max_seq_len': max(self.sequence_lengths) if self.sequence_lengths else 0
        }


class CacheLoader:
    """
    Load cached datasets from pickle files.

    Uses the cache system at /content/drive/MyDrive/dawn_v4/cache/
    Uses wikitext_5to1_texts.pkl which has train/validation already split 5:1
    """

    CACHE_BASE_DIR = "/content/drive/MyDrive/dawn_v4/cache"

    @staticmethod
    def load_texts(
        split: str = "validation",
        dataset: str = "wikitext",
        use_cache: bool = True
    ) -> Optional[List[str]]:
        """
        Load cached text data from cache/{split}/wikitext_5to1_texts.pkl.

        Args:
            split: Dataset split ('train' or 'validation')
            dataset: Dataset name (e.g., 'wikitext')
            use_cache: Whether to use cache (if False, returns None)

        Returns:
            List of text strings, or None if cache not available
        """
        if not use_cache:
            return None

        # Cache structure: cache/{split}/{dataset}_5to1_texts.pkl
        cache_path = os.path.join(
            CacheLoader.CACHE_BASE_DIR,
            split,
            f"{dataset}_5to1_texts.pkl"
        )

        if not os.path.exists(cache_path):
            print(f"⚠️  Cache not found: {cache_path}")
            return None

        try:
            with open(cache_path, 'rb') as f:
                texts = pickle.load(f)

            # Should be a list of text strings
            if isinstance(texts, list):
                print(f"✅ Loaded {len(texts)} {split} texts from cache: {cache_path}")
                return texts
            else:
                print(f"❌ Invalid cache format. Expected list, got {type(texts)}")
                return None
        except Exception as e:
            print(f"❌ Error loading cache: {e}")
            return None

    @staticmethod
    def load_validation_texts(dataset: str = "wikitext") -> Optional[List[str]]:
        """
        Load validation texts from cache.

        Convenience method for loading validation split.

        Args:
            dataset: Dataset name (default: 'wikitext')

        Returns:
            List of validation texts, or None if not available
        """
        return CacheLoader.load_texts(split="validation", dataset=dataset)

    @staticmethod
    def load_train_texts(dataset: str = "wikitext") -> Optional[List[str]]:
        """
        Load training texts from cache.

        Convenience method for loading train split.

        Args:
            dataset: Dataset name (default: 'wikitext')

        Returns:
            List of training texts, or None if not available
        """
        return CacheLoader.load_texts(split="train", dataset=dataset)


# ============================================================
# MLM Configuration
# ============================================================

MLM_CONFIG = {
    "mask_prob": 0.15,
    "mask_token_ratio": 0.8,
    "random_token_ratio": 0.5
}


# ============================================================
# MLM Masking Function
# ============================================================

def apply_mlm_masking(input_ids, tokenizer, config=None):
    """
    Apply MLM-style masking (80% [MASK], 10% random, 10% keep).

    Args:
        input_ids: [B, S] Token IDs to mask
        tokenizer: Tokenizer instance
        config: Optional config dict with mask_prob, mask_token_ratio, random_token_ratio

    Returns:
        Tuple of (masked_input_ids, labels)
    """
    if config is None:
        config = MLM_CONFIG

    labels = input_ids.clone()
    mask_prob = config.get("mask_prob", 0.15)
    device = input_ids.device

    probability_matrix = torch.full(labels.shape, mask_prob, device=device)

    # Exclude special tokens (CLS, SEP, PAD, etc.)
    special_tokens_mask = []
    for seq in labels.tolist():
        seq_mask = [
            tokenizer.get_special_tokens_mask([val], already_has_special_tokens=True)[0]
            for val in seq
        ]
        special_tokens_mask.append(seq_mask)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=device)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # Exclude padding tokens
    padding_mask = input_ids == tokenizer.pad_token_id
    probability_matrix.masked_fill_(padding_mask, value=0.0)

    # Sample masked positions
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Apply masking strategy
    mask_ratio = config.get("mask_token_ratio", 0.8)
    random_ratio = config.get("random_token_ratio", 0.5)

    # 80% [MASK]
    indices_replaced = masked_indices & (torch.rand(labels.shape, device=device) < mask_ratio)
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% random (of remaining)
    indices_random = (
        masked_indices
        & ~indices_replaced
        & (torch.rand(labels.shape, device=device) < random_ratio)
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% keep original (implicit)
    return input_ids, labels


# ============================================================
# Dataset Classes
# ============================================================

from torch.utils.data import DataLoader, Dataset

class TextDataset(Dataset):
    """Dataset for tokenized texts"""
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize (NO padding here - will be done dynamically in collate_fn)
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def collate_fn_dynamic_padding(batch, tokenizer):
    """
    Collate function with DYNAMIC padding (배치별 최대 길이만큼만 padding)
    """
    # Find max length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        seq_len = input_ids.size(0)

        # Pad to batch max length
        if seq_len < max_len:
            padding_len = max_len - seq_len
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_len, dtype=torch.long)
            ])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def load_data(data_config, max_length=128, batch_size=128, tokenizer_path=None):
    """Load data from config paths with DYNAMIC padding"""
    from transformers import AutoTokenizer
    from functools import partial

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load data from config paths
    base_dir = data_config['base_dir']
    train_path = os.path.join(base_dir, data_config['train_file'])
    val_path = os.path.join(base_dir, data_config['val_file'])

    print(f"Loading data from: {base_dir}")

    # Load train texts
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found: {train_path}")
    with open(train_path, 'rb') as f:
        train_texts = pickle.load(f)

    # Load validation texts
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")
    with open(val_path, 'rb') as f:
        val_texts = pickle.load(f)

    print(f"Loaded {len(train_texts)} train texts, {len(val_texts)} val texts")

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length)

    # Create collate function with tokenizer
    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer)

    # Create dataloaders with DYNAMIC padding
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=2,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, tokenizer
