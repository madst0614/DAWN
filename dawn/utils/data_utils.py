"""
Data Utilities Module

Core utilities for data loading, tokenization, and text processing.
Contains base classes and common functionality used across all tasks.
"""

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from datasets import load_dataset
import random
import numpy as np
import re
import spacy
import os
from functools import lru_cache

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration Import
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

try:
    from configs.data import (
        TEXT_FILTERS,
        MLM_CONFIG,
        SPAN_MASKING_CONFIG,
        TOKEN_DELETION_CONFIG,
        TEXT_INFILLING_CONFIG,
        SENTENCE_PERMUTATION_CONFIG,
        DATASET_SIZES,
        TOKENIZER_CONFIG,
        DATALOADER_CONFIG,
        DATASET_SOURCES,
        DATASETS,
        TASKS,
    )
except ImportError:
    print("⚠️ Warning: Could not import data config. Using fallback values.")
    TEXT_FILTERS = {
        "min_words": 5,
        "max_words": 50,
        "min_chars": 20,
        "min_alpha_ratio": 0.5,
    }
    MLM_CONFIG = {"mask_prob": 0.15, "mask_token_ratio": 0.8, "random_token_ratio": 0.5}
    SPAN_MASKING_CONFIG = {"mean_span": 3.8, "max_span_length": 8}
    TOKEN_DELETION_CONFIG = {"delete_prob": 0.15}
    TEXT_INFILLING_CONFIG = {
        "mask_prob": 0.30,
        "mean_span": 3.0,
        "single_mask_per_span": True,
    }
    SENTENCE_PERMUTATION_CONFIG = {"shuffle_prob": 1.0, "min_sentences": 3}
    DATASET_SIZES = {"mlm": 1000000, "sampling_multiplier": 10}
    TOKENIZER_CONFIG = {"model_name": "bert-base-uncased", "max_length": 128}  # Match PNN (512 causes 16x memory)
    DATALOADER_CONFIG = {"num_workers": 2}
    DATASET_SOURCES = {}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Global Singletons (Lazy Loading)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_tokenizer = None
_nlp = None


def get_tokenizer():
    """Get or initialize the shared tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(TOKENIZER_CONFIG["model_name"])
    return _tokenizer


def get_spacy_nlp():
    """Get or initialize SpaCy model (lazy loading)."""
    global _nlp
    if _nlp is None:
        pos_config = SPAN_MASKING_CONFIG.get("pos_tagging", {})
        if pos_config.get("enabled", False):
            try:
                model_name = pos_config.get("spacy_model", "en_core_web_sm")
                _nlp = spacy.load(model_name)
                print(f"✅ Loaded SpaCy model: {model_name}")
            except OSError:
                print(
                    f"⚠️ SpaCy model not found. Run: python -m spacy download en_core_web_sm"
                )
                _nlp = None
    return _nlp


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SpaCy Caching (Performance Optimization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@lru_cache(maxsize=10000)
def _cached_spacy_parse(text: str, nlp_id: int):
    """
    Cached SpaCy parsing for 10x speed improvement.

    Args:
        text: Text to parse
        nlp_id: SpaCy model ID (for cache invalidation)

    Returns:
        Tuple of POS tags
    """
    nlp = get_spacy_nlp()
    if nlp is None:
        return None
    doc = nlp(text)
    return tuple(token.pos_ for token in doc)


def get_cached_pos_tags(text: str):
    """Get POS tags with caching."""
    nlp = get_spacy_nlp()
    if nlp is None:
        return None

    nlp_id = id(nlp)
    try:
        pos_tags = _cached_spacy_parse(text, nlp_id)
        return list(pos_tags) if pos_tags else None
    except Exception:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Common Data Loading Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def load_wikipedia_streaming(
    split="train",
    max_samples=None,
    min_words=5,
    max_words=50,
    streaming=True,
    dataset_source="wikipedia",  # NEW: dataset source name
):
    """
    Load dataset with automatic fallback (supports wikipedia, wikitext, etc.).

    Tries to load from DATASETS config, falls back if needed.
    Applies basic text filtering during loading.

    Args:
        split: Dataset split
        max_samples: Maximum samples to load
        min_words: Minimum word count filter
        max_words: Maximum word count filter
        streaming: Use streaming mode
        dataset_source: Dataset source name from DATASETS config (e.g., "wikipedia", "wikitext")

    Returns:
        List of filtered sentences
    """
    # Get dataset config
    dataset_config = DATASETS.get(dataset_source)

    if dataset_config:
        # Use DATASETS config
        try:
            print(f"🔄 Loading {dataset_source} dataset (streaming={streaming})...")
            dataset_name = dataset_config.dataset_name if dataset_config.dataset_name else None

            if dataset_name:
                dataset = load_dataset(
                    dataset_config.path,
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=False,
                )
            else:
                dataset = load_dataset(
                    dataset_config.path,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=False,
                )
            print(f"✅ Successfully loaded {dataset_config.path}")
        except Exception as e:
            print(f"⚠️ Failed to load {dataset_source}: {e}")
            print(f"   Trying fallback: Salesforce/wikitext")
            fallback_config = DATASET_SOURCES.get("wikipedia", {}).get("fallback", {})
            dataset = load_dataset(
                fallback_config.get("path", "Salesforce/wikitext"),
                fallback_config.get("name", "wikitext-103-raw-v1"),
                split=split,
                streaming=streaming,
                trust_remote_code=False,
            )
            print(f"✅ Successfully loaded fallback")
    else:
        # Fallback to old behavior (DATASET_SOURCES)
        try:
            print(f"🔄 Loading Wikipedia dataset (streaming={streaming})...")
            wiki_config = DATASET_SOURCES.get("wikipedia", {}).get("primary", {})
            dataset = load_dataset(
                wiki_config.get("path", "wikimedia/wikipedia"),
                wiki_config.get("name", "20231101.en"),
                split=split,
                streaming=streaming,
                trust_remote_code=False,
            )
            print(f"✅ Successfully loaded {wiki_config.get('path')}")
        except Exception as e:
            print(f"⚠️ Failed to load primary source: {e}")
            print(f"   Trying fallback: Salesforce/wikitext")
            fallback_config = DATASET_SOURCES.get("wikipedia", {}).get("fallback", {})
            dataset = load_dataset(
                fallback_config.get("path", "Salesforce/wikitext"),
                fallback_config.get("name", "wikitext-103-raw-v1"),
                split=split,
                streaming=streaming,
                trust_remote_code=False,
            )
            print(f"✅ Successfully loaded fallback")

    # Filter and collect sentences
    valid_sentences = []
    multiplier = DATASET_SIZES.get("sampling_multiplier", 10)
    max_items = (max_samples * multiplier) if max_samples else 100000

    for idx, item in enumerate(dataset):
        if idx >= max_items:
            break

        text = item.get("text", "").strip()
        if not text or text.startswith("="):  # Skip titles
            continue

        # Split into sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", text)]

        for sent in sentences:
            word_count = len(sent.split())
            char_count = len(sent)

            if min_words <= word_count <= max_words and char_count >= 20:
                valid_sentences.append(sent)

            if max_samples and len(valid_sentences) >= max_samples:
                break

        if max_samples and len(valid_sentences) >= max_samples:
            break

    return valid_sentences


def load_huggingface_dataset(
    dataset_name, config_name=None, split="train", fallback=None
):
    """
    Load HuggingFace dataset with optional fallback.

    Args:
        dataset_name: Primary dataset name
        config_name: Dataset configuration
        split: Split to load
        fallback: Tuple of (fallback_name, fallback_config, fallback_split)

    Returns:
        Loaded dataset or None
    """
    try:
        return load_dataset(dataset_name, config_name, split=split)
    except Exception as e:
        if fallback:
            print(f"⚠️ Failed to load {dataset_name}: {e}")
            fallback_name, fallback_config, fallback_split = fallback
            print(f"   Trying fallback: {fallback_name}")
            try:
                return load_dataset(
                    fallback_name, fallback_config, split=fallback_split or split
                )
            except Exception as e2:
                print(f"⚠️ Fallback also failed: {e2}")
                return None
        raise


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text Validation Mixin
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TextValidationMixin:
    """Mixin for text validation functionality."""

    def _is_valid_text(
        self,
        text,
        min_words=None,
        max_words=None,
        min_chars=None,
        min_alpha_ratio=None,
    ):
        """
        Validate text based on configurable criteria.

        Args:
            text: Text to validate
            min_words: Minimum word count (None uses config default)
            max_words: Maximum word count (None uses config default)
            min_chars: Minimum character count (None uses config default)
            min_alpha_ratio: Minimum alphabetic ratio (None uses config default)

        Returns:
            bool: True if valid
        """
        # Apply defaults from config
        min_words = min_words if min_words is not None else TEXT_FILTERS.min_words
        max_words = max_words if max_words is not None else TEXT_FILTERS.max_words
        min_chars = min_chars if min_chars is not None else TEXT_FILTERS.min_chars
        min_alpha_ratio = (
            min_alpha_ratio
            if min_alpha_ratio is not None
            else TEXT_FILTERS.min_alpha_ratio
        )

        word_count = len(text.split())
        char_count = len(text)

        # Length checks
        if not (min_words <= word_count <= max_words and char_count >= min_chars):
            return False

        # Exclude special formats
        for pattern in TEXT_FILTERS.exclude_starts_with:
            if text.startswith(pattern):
                return False

        for pattern in TEXT_FILTERS.exclude_contains:
            if pattern in text:
                return False

        # Minimum alphabetic ratio
        alpha_count = sum(c.isalpha() for c in text)
        if char_count > 0 and alpha_count / char_count < min_alpha_ratio:
            return False

        return True

    def _is_valid_sentence(self, sent):
        """Backward compatibility wrapper for _is_valid_text."""
        return self._is_valid_text(sent)

    def _is_valid_pair(self, text1, text2, **kwargs):
        """Validate a pair of texts."""
        return self._is_valid_text(text1, **kwargs) and self._is_valid_text(
            text2, **kwargs
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Base Dataset Class
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class BaseDataset(Dataset, TextValidationMixin):
    """
    Base dataset class with common functionality.

    Provides:
    - Tokenization
    - Text validation
    - Common configuration handling
    """

    def __init__(self, split="train", max_length=None, max_samples=None):
        self.tokenizer = get_tokenizer()
        self.max_length = max_length or TOKENIZER_CONFIG["max_length"]
        self.split = split
        self.max_samples = max_samples
        self.data = []

    def __len__(self):
        return len(self.data)

    def encode_text(self, text):
        """Encode text with tokenizer."""
        return self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Masking Strategy Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MaskingStrategy:
    """Helper class for various masking strategies."""

    @staticmethod
    def apply_mlm_masking(input_ids, labels, mask_prob, tokenizer, config=None):
        """
        Apply MLM-style masking (80% [MASK], 10% random, 10% keep).

        Args:
            input_ids: Token IDs to mask
            labels: Labels tensor
            mask_prob: Probability of masking
            tokenizer: Tokenizer instance
            config: Optional config dict with mask_token_ratio and random_token_ratio

        Returns:
            Tuple of (masked_input_ids, labels)
        """
        if config is None:
            config = MLM_CONFIG

        probability_matrix = torch.full(labels.shape, mask_prob)

        # ✅ Exclude special tokens (CLS, SEP, PAD, etc.)
        special_tokens_mask = torch.tensor([
            tokenizer.get_special_tokens_mask(
                [val], already_has_special_tokens=True
            )[0]
            for val in labels.tolist()
        ], dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # ✅ Exclude padding tokens (belt and suspenders)
        padding_mask = input_ids == tokenizer.pad_token_id
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Sample masked positions
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # Apply masking strategy
        mask_ratio = config.get("mask_token_ratio", 0.8)
        random_ratio = config.get("random_token_ratio", 0.5)

        # 80% [MASK]
        indices_replaced = masked_indices & (torch.rand(labels.shape) < mask_ratio)
        input_ids[indices_replaced] = tokenizer.mask_token_id

        # 10% random (of remaining)
        indices_random = (
            masked_indices
            & ~indices_replaced
            & (torch.rand(labels.shape) < random_ratio)
        )
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # 10% keep original (implicit)
        return input_ids, labels

    @staticmethod
    def create_span_mask(
        input_ids, attention_mask, mask_prob, mean_span, max_span_length, tokenizer
    ):
        """
        Create span-based masking.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            mask_prob: Probability of masking
            mean_span: Mean span length (Poisson λ)
            max_span_length: Maximum span length
            tokenizer: Tokenizer instance

        Returns:
            Set of masked indices
        """
        # Get valid indices (exclude padding and special tokens)
        valid_indices = torch.where(attention_mask == 1)[0].tolist()
        valid_indices = [
            i
            for i in valid_indices
            if input_ids[i]
            not in [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            ]
        ]

        # Exclude tokens near SEP
        sep_exclusion_margin = TEXT_FILTERS.sep_exclusion_margin
        sep_positions = [
            i for i, tok in enumerate(input_ids) if tok == tokenizer.sep_token_id
        ]
        for sep_pos in sep_positions:
            valid_indices = [
                i
                for i in valid_indices
                if not (sep_pos - sep_exclusion_margin <= i < sep_pos)
            ]

        if len(valid_indices) == 0:
            return set()

        # Calculate number of tokens to mask
        num_to_mask = max(1, int(len(valid_indices) * mask_prob))
        masked_indices = set()

        # Safety parameters
        max_iterations = num_to_mask * 3
        max_mask_ratio = 1.2
        iteration = 0

        while len(masked_indices) < num_to_mask and iteration < max_iterations:
            iteration += 1

            # Select start index
            available_set = set(valid_indices) - masked_indices
            if not available_set:
                break

            available = list(available_set)
            start_idx = random.choice(available)

            # Sample span length
            span_len = min(
                np.random.poisson(mean_span - 1) + 1,
                max_span_length,
                num_to_mask - len(masked_indices),
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
            if len(masked_indices) >= num_to_mask * max_mask_ratio:
                break

        return masked_indices
