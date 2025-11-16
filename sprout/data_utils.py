"""
Data utilities for SPROUT training.

Self-contained implementations inspired by dawn but standalone.
Includes advanced masking strategies and data loading.
"""

import torch
import numpy as np
import random
from typing import List, Set, Tuple


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
    """Validate text quality for training."""

    @staticmethod
    def is_valid_sentence(
        text: str,
        min_words: int = 5,
        max_words: int = 50,
        min_chars: int = 20,
        min_alpha_ratio: float = 0.5
    ) -> bool:
        """
        Check if sentence is valid for training.

        Args:
            text: Text to validate
            min_words: Minimum word count
            max_words: Maximum word count
            min_chars: Minimum character count
            min_alpha_ratio: Minimum alphabetic character ratio

        Returns:
            True if valid
        """
        if not text:
            return False

        word_count = len(text.split())
        char_count = len(text)

        # Length checks
        if not (min_words <= word_count <= max_words):
            return False

        if char_count < min_chars:
            return False

        # Exclude special formats
        if text.startswith(("=", "#", "*", "-", "Category:", "File:")):
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
