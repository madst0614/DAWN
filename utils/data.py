"""
Data utilities for DAWN training.

Includes data loading and MLM masking utilities.
"""

import torch
import numpy as np
import pickle
import os
from typing import List, Optional


class CacheLoader:
    """
    Load cached datasets from pickle files.

    Uses wikitext_5to1_texts.pkl which has train/validation already split 5:1
    """

    CACHE_BASE_DIR = os.environ.get('DAWN_DATA_DIR', './data')

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

    # Ensure int64 dtype (pt files may have int32)
    input_ids = input_ids.long()

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


class TokenDataset(Dataset):
    """Dataset for pre-tokenized data (from .pt files)"""
    def __init__(self, tokens, max_length=128):
        """
        Args:
            tokens: Tensor of shape [N, seq_len] containing token IDs
            max_length: Maximum sequence length (truncate if longer)
        """
        self.tokens = tokens
        self.max_length = max_length

        # Debug: print shape info
        if hasattr(tokens, 'shape'):
            print(f"  TokenDataset: tokens shape = {tokens.shape}")

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        input_ids = self.tokens[idx]

        # Handle different tensor shapes
        if input_ids.dim() == 0:
            # 0-d tensor (scalar) - shouldn't happen, but wrap in 1D
            input_ids = input_ids.unsqueeze(0)

        seq_len = input_ids.shape[0]

        # Truncate if needed
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            seq_len = self.max_length

        # Create attention mask (1 for real tokens, will handle padding in collate)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }


def collate_fn_dynamic_padding(batch, tokenizer, max_seq_len=None):
    """
    Collate function with padding.

    Args:
        batch: List of samples
        tokenizer: Tokenizer with pad_token_id
        max_seq_len: If provided, pad to this fixed length. Otherwise, pad to batch max (dynamic).
    """
    # Determine target length
    if max_seq_len is not None:
        # Fixed padding to max_seq_len
        max_len = max_seq_len
    else:
        # Dynamic padding to batch max
        max_len = max(item['input_ids'].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        seq_len = input_ids.size(0)

        # Truncate if longer than max_len
        if seq_len > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            seq_len = max_len

        # Pad to max_len
        if seq_len < max_len:
            padding_len = max_len - seq_len
            input_ids = torch.cat([
                input_ids,
                torch.full((padding_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_len, dtype=torch.long)  # 0 for padding positions
            ])

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        'input_ids': torch.stack(input_ids_list),
        'attention_mask': torch.stack(attention_mask_list)
    }


def load_single_file(filepath, max_length=128):
    """
    Load data from a single file (.pkl or .pt format).

    Args:
        filepath: Path to the data file
        max_length: Sequence length for reshaping flat token tensors

    Returns:
        Tuple of (data, is_pretokenized)
        - For .pkl: (list of texts, False)
        - For .pt: (tensor of tokens [N, seq_len], True)
    """
    if filepath.endswith('.pt'):
        # Pre-tokenized data
        data = torch.load(filepath)
        if isinstance(data, dict) and 'tokens' in data:
            tokens = data['tokens']
        elif isinstance(data, torch.Tensor):
            tokens = data
        else:
            raise ValueError(f"Unknown .pt format. Expected dict with 'tokens' or tensor, got {type(data)}")

        # Ensure int64 dtype for embedding indexing
        if tokens.dtype != torch.int64:
            print(f"  Converting tokens from {tokens.dtype} to int64")
            tokens = tokens.long()

        # Reshape flat 1D tokens to [N, seq_len]
        if tokens.dim() == 1:
            total_tokens = tokens.shape[0]
            num_sequences = total_tokens // max_length
            discarded = total_tokens - (num_sequences * max_length)
            if discarded > 0:
                print(f"  Warning: Discarding {discarded:,} tokens ({discarded/total_tokens*100:.2f}%) that don't fit into sequences")
            tokens = tokens[:num_sequences * max_length]
            tokens = tokens.view(num_sequences, max_length)
            print(f"  Reshaped flat tokens to {tokens.shape[0]:,} sequences of length {max_length}")

        return tokens, True
    else:
        # Text data (.pkl)
        with open(filepath, 'rb') as f:
            texts = pickle.load(f)
        return texts, False


def load_data(data_config, max_length=128, batch_size=128, tokenizer_path=None):
    """Load data from config paths with FIXED padding to max_length

    Supports both single file and multiple files:
    - Single file: data_config['train_file'], data_config['val_file']
    - Multiple files: data_config['train_files'], data_config['val_files'] (list)

    Supports both formats:
    - .pkl: List of text strings (will be tokenized)
    - .pt: Pre-tokenized tensor with 'tokens' key

    Optional config options:
    - max_train_tokens: Limit training data (e.g., 360000000 for 360M)
    - max_val_tokens: Limit validation data (default: 10M)
    """
    from transformers import AutoTokenizer
    from functools import partial

    # Load tokenizer
    if tokenizer_path is None:
        tokenizer_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Get optional token limits from config
    max_train_tokens = data_config.get('max_train_tokens', None)
    max_val_tokens = data_config.get('max_val_tokens', 10_000_000)  # Default 10M

    # Load data from config paths
    base_dir = data_config['base_dir']

    # Support both single file and multiple files
    train_files = data_config.get('train_files', [data_config.get('train_file')])
    val_files = data_config.get('val_files', [data_config.get('val_file')])

    # Ensure lists
    if isinstance(train_files, str):
        train_files = [train_files]
    if isinstance(val_files, str):
        val_files = [val_files]

    print(f"Loading data from: {base_dir}")

    # Load train data from all files
    train_data = []
    train_is_pretokenized = None
    for train_file in train_files:
        train_path = os.path.join(base_dir, train_file)
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train data not found: {train_path}")

        data, is_pretokenized = load_single_file(train_path, max_length)

        # Check consistency
        if train_is_pretokenized is None:
            train_is_pretokenized = is_pretokenized
        elif train_is_pretokenized != is_pretokenized:
            raise ValueError("Cannot mix .pkl and .pt files in the same split")

        if is_pretokenized:
            train_data.append(data)
            print(f"  Loaded {len(data):,} sequences from {train_file} (pre-tokenized)")
        else:
            train_data.extend(data)
            print(f"  Loaded {len(data):,} texts from {train_file}")

    # Load validation data from all files
    val_data = []
    val_is_pretokenized = None
    for val_file in val_files:
        val_path = os.path.join(base_dir, val_file)
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")

        data, is_pretokenized = load_single_file(val_path, max_length)

        # Check consistency
        if val_is_pretokenized is None:
            val_is_pretokenized = is_pretokenized
        elif val_is_pretokenized != is_pretokenized:
            raise ValueError("Cannot mix .pkl and .pt files in the same split")

        if is_pretokenized:
            val_data.append(data)
            print(f"  Loaded {len(data):,} sequences from {val_file} (pre-tokenized)")
        else:
            val_data.extend(data)
            print(f"  Loaded {len(data):,} texts from {val_file}")

    # Create datasets based on data type
    if train_is_pretokenized:
        # Concatenate all token tensors
        train_tokens = torch.cat(train_data, dim=0) if len(train_data) > 1 else train_data[0]
        # Limit training data if specified
        if max_train_tokens is not None:
            max_train_seqs = max_train_tokens // max_length
            if len(train_tokens) > max_train_seqs:
                train_tokens = train_tokens[:max_train_seqs]
                print(f"  Limited training to {max_train_seqs:,} sequences ({max_train_tokens/1e6:.0f}M tokens)")
        train_dataset = TokenDataset(train_tokens, max_length)
        print(f"Total: {len(train_tokens):,} train sequences (pre-tokenized)")
    else:
        train_dataset = TextDataset(train_data, tokenizer, max_length)
        print(f"Total: {len(train_data):,} train texts")

    if val_is_pretokenized:
        val_tokens = torch.cat(val_data, dim=0) if len(val_data) > 1 else val_data[0]
        # Limit validation data
        max_val_seqs = max_val_tokens // max_length
        if len(val_tokens) > max_val_seqs:
            val_tokens = val_tokens[:max_val_seqs]
            print(f"  Limited validation to {max_val_seqs:,} sequences ({max_val_tokens/1e6:.0f}M tokens)")
        val_dataset = TokenDataset(val_tokens, max_length)
        print(f"Total: {len(val_tokens):,} val sequences (pre-tokenized)")
    else:
        val_dataset = TextDataset(val_data, tokenizer, max_length)
        print(f"Total: {len(val_data):,} val texts")

    # Create collate function with tokenizer and fixed max_length
    collate_fn = partial(collate_fn_dynamic_padding, tokenizer=tokenizer, max_seq_len=max_length)

    # Create dataloaders with FIXED padding to max_length
    # shuffle=False: reproducibility, efficient resume, pre-shuffled data
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False
    )

    return train_loader, val_loader, tokenizer


# ============================================================
# MLM Accuracy Calculation
# ============================================================

def compute_mlm_accuracy(logits, labels):
    """
    Compute accuracy for MLM task (only on masked tokens).

    Args:
        logits: Model output logits [B, S, V]
        labels: Labels with -100 for non-masked tokens [B, S]

    Returns:
        Tuple of (num_correct, num_valid_tokens)
    """
    predictions = logits.argmax(dim=-1)  # [B, S]

    # Only count tokens that are not masked (-100)
    valid_mask = (labels != -100)  # [B, S]
    correct_predictions = (predictions == labels) & valid_mask

    num_correct = correct_predictions.sum().item()
    num_valid = valid_mask.sum().item()

    return num_correct, num_valid


def evaluate_mlm_batch(model, input_ids, tokenizer, device, config=None):
    """
    Evaluate a single batch with MLM masking.

    Args:
        model: The model to evaluate
        input_ids: Input token IDs [B, S]
        tokenizer: Tokenizer for masking
        device: Device to use
        config: MLM config (optional)

    Returns:
        Dict with loss, correct, valid_tokens
    """
    # Apply MLM masking
    masked_input_ids, labels = apply_mlm_masking(input_ids.clone(), tokenizer, config)

    with torch.no_grad():
        outputs = model(
            input_ids=masked_input_ids,
            labels=labels
        )
        # Handle both DAWN (tuple) and HuggingFace (dict) output formats
        if isinstance(outputs, tuple):
            loss, logits = outputs[0], outputs[1]
        else:
            loss = outputs['loss']
            logits = outputs['logits']

        num_correct, num_valid = compute_mlm_accuracy(logits, labels)

    return {
        'loss': loss.item(),
        'correct': num_correct,
        'valid_tokens': num_valid,
        'logits': logits,
        'labels': labels
    }
