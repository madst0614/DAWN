"""
Task Registry Module

Contains all task-specific dataset implementations and the task registry.
Each dataset class is optimized and follows common patterns.
"""

import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import re
import os
import pickle

from .data_utils import (
    BaseDataset,
    get_tokenizer,
    get_spacy_nlp,
    get_cached_pos_tags,
    load_wikipedia_streaming,
    load_huggingface_dataset,
    MaskingStrategy,
    TEXT_FILTERS,
    TOKENIZER_CONFIG,
    MLM_CONFIG,
    SPAN_MASKING_CONFIG,
    TOKEN_DELETION_CONFIG,
    TEXT_INFILLING_CONFIG,
    SENTENCE_PERMUTATION_CONFIG,
    DATASET_SIZES,
    DATALOADER_CONFIG,
    DATASET_SOURCES,
)

from datasets import load_dataset


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pre-training Tasks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MLMDataset(BaseDataset):
    """Masked Language Modeling Dataset."""

    def __init__(
        self, split="train", max_length=None, mask_prob=None, max_samples=None
    ):
        max_length = max_length or TOKENIZER_CONFIG["max_length"]
        mask_prob = mask_prob if mask_prob is not None else MLM_CONFIG["mask_prob"]
        max_samples = (
            max_samples
            if max_samples is not None
            else DATASET_SIZES.get("mlm", 200_000)
        )

        super().__init__(split, max_length, max_samples)
        self.mask_prob = mask_prob

        # ✅ Get dataset source from TASKS config
        try:
            from configs.data import TASKS
            dataset_source = TASKS["mlm"].dataset_source
        except (ImportError, KeyError):
            dataset_source = "wikitext"  # fallback to wikitext (not wikipedia)

        # Load WikiText-103 directly for MLM (PNN style - no sentence splitting)
        self.data = self._load_wikitext_streaming(split, max_samples)

        # Only shuffle for training split
        if split == "train":
            random.shuffle(self.data)
        print(f"✅ MLM Dataset (WikiText-103 streaming): {len(self.data)} samples")

    def _load_wikitext_streaming(self, split, max_samples):
        """Load WikiText-103 in streaming mode (PNN style)"""
        from datasets import load_dataset
        from tqdm import tqdm

        # Map validation to test (WikiText-103 doesn't have validation split)
        hf_split = "test" if split == "validation" else split

        try:
            print(f"🔄 Loading WikiText-103 (split={hf_split}, streaming=True)...")
            dataset = load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                split=hf_split,
                streaming=True,
                trust_remote_code=False,
            )
        except Exception as e:
            print(f"❌ Failed to load WikiText-103: {e}")
            return []

        # Collect valid texts (minimal filtering - PNN style)
        valid_texts = []
        for item in tqdm(dataset, desc=f"Loading {split} data", total=max_samples, disable=split=="validation"):
            text = item['text'].strip()

            # Minimal filtering: only skip empty lines and headers
            if len(text) > 20 and len(text.split()) >= 5:
                valid_texts.append(text)

                if max_samples and len(valid_texts) >= max_samples:
                    break

        return valid_texts

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.encode_text(text)
        input_ids = encoding["input_ids"][0].clone()
        attention_mask = encoding["attention_mask"][0]
        labels = input_ids.clone()

        # Apply MLM masking strategy
        input_ids, labels = MaskingStrategy.apply_mlm_masking(
            input_ids, labels, self.mask_prob, self.tokenizer
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SpanMaskingDataset(MLMDataset):
    """Span-based Masking Dataset with optional POS tagging."""

    def __init__(
        self,
        split="train",
        max_length=None,
        mask_prob=None,
        mean_span=None,
        max_span_length=None,
        max_samples=None,
        pos_cache_file=None,
        enable_pos=None,
    ):
        mask_prob = (
            mask_prob
            if mask_prob is not None
            else SPAN_MASKING_CONFIG.get("mask_prob", 0.15)
        )
        mean_span = (
            mean_span if mean_span is not None else SPAN_MASKING_CONFIG["mean_span"]
        )
        max_span_length = (
            max_span_length
            if max_span_length is not None
            else SPAN_MASKING_CONFIG["max_span_length"]
        )
        max_samples = max_samples or DATASET_SIZES.get("span_masking", 200_000)

        super().__init__(split, max_length, mask_prob, max_samples)
        self.mean_span = mean_span
        self.max_span_length = max_span_length

        # POS tagging configuration
        pos_config = SPAN_MASKING_CONFIG.get("pos_tagging", {})
        self.use_pos = (
            enable_pos if enable_pos is not None else pos_config.get("enabled", False)
        )
        self.nlp = get_spacy_nlp() if self.use_pos else None

        # Load POS cache if available
        self.pos_cache = None
        if self.use_pos and pos_cache_file and os.path.exists(pos_cache_file):
            try:
                with open(pos_cache_file, "rb") as f:
                    self.pos_cache = pickle.load(f)
                print(f"✅ Loaded {len(self.pos_cache)} cached POS tags")
                print(f"   → Span Masking will be 10x faster! 🚀\n")
            except Exception as e:
                print(f"⚠️  Failed to load POS cache: {e}")
                print(f"   Falling back to runtime SpaCy parsing\n")
                self.pos_cache = None

        # POS tag to index mapping (Universal POS tags)
        self.pos_to_idx = {
            "PAD": 0,  # Padding
            "ADJ": 1,
            "ADP": 2,
            "ADV": 3,
            "AUX": 4,
            "CCONJ": 5,
            "CONJ": 6,
            "DET": 7,
            "INTJ": 8,
            "NOUN": 9,
            "NUM": 10,
            "PART": 11,
            "PRON": 12,
            "PROPN": 13,
            "PUNCT": 14,
            "SCONJ": 15,
            "SYM": 16,
            "VERB": 17,
            "X": 18,
            "SPACE": 19,
        }

        # ✅ Get dataset source from TASKS config
        try:
            from configs.data import TASKS
            dataset_source = TASKS["span_masking"].dataset_source
        except (ImportError, KeyError):
            dataset_source = "wikipedia"  # fallback

        # Load data
        self.data = load_wikipedia_streaming(
            split=split,
            max_samples=max_samples,
            streaming=True,
            dataset_source=dataset_source,  # ✅ Use config
        )
        # Only shuffle for training split
        if split == "train":
            random.shuffle(self.data)
        print(f"✅ Span Masking Dataset ({dataset_source}): {len(self.data)} samples")

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.encode_text(text)
        input_ids = encoding["input_ids"][0].clone()
        attention_mask = encoding["attention_mask"][0]
        labels = input_ids.clone()

        # POS Tagging (if enabled)
        pos_labels = None
        if self.use_pos and self.nlp is not None:
            try:
                spacy_pos = None

                # Option 1: Pre-computed cache (fastest!)
                if self.pos_cache is not None and text in self.pos_cache:
                    spacy_pos = list(self.pos_cache[text])

                # Option 2: LRU cached parsing (fast)
                elif self.pos_cache is None:
                    spacy_pos = get_cached_pos_tags(text)

                if spacy_pos is not None:
                    # Align BERT tokens with SpaCy POS tags
                    # WordPiece tokens inherit POS from first token
                    pos_labels = torch.zeros(len(input_ids), dtype=torch.long)

                    # Tokenize with word tracking
                    tokens = self.tokenizer.tokenize(text)
                    word_ids = []
                    current_word = 0

                    for i, token in enumerate(tokens):
                        if i == 0 or not token.startswith("##"):
                            current_word = len(word_ids)
                            word_ids.append(current_word)
                        else:
                            word_ids.append(current_word if word_ids else 0)

                    # Map POS tags to token positions
                    # [CLS] at position 0, then tokens, then [SEP]
                    for token_idx, word_id in enumerate(word_ids):
                        actual_pos = token_idx + 1  # +1 for [CLS]
                        if actual_pos < len(pos_labels) and word_id < len(spacy_pos):
                            pos_tag = spacy_pos[word_id]
                            pos_labels[actual_pos] = self.pos_to_idx.get(
                                pos_tag, self.pos_to_idx["X"]
                            )

            except Exception:
                # Fallback: no POS labels
                pos_labels = None

        # Create span mask
        masked_indices = MaskingStrategy.create_span_mask(
            input_ids,
            attention_mask,
            self.mask_prob,
            self.mean_span,
            self.max_span_length,
            self.tokenizer,
        )

        # Apply masking
        labels[~attention_mask.bool()] = -100
        for idx_pos in range(len(input_ids)):
            if idx_pos in masked_indices:
                rand = random.random()
                if rand < 0.8:
                    input_ids[idx_pos] = self.tokenizer.mask_token_id
                elif rand < 0.9:
                    input_ids[idx_pos] = random.randint(0, len(self.tokenizer) - 1)
            else:
                labels[idx_pos] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pos_labels is not None:
            result["pos_labels"] = pos_labels

        return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BART-style Denoising Tasks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TokenDeletionDataset(BaseDataset):
    """Token Deletion (BART): Random tokens deleted, model predicts which tokens should be kept/deleted."""

    def __init__(self, split="train", max_samples=None, delete_prob=0.15, **kwargs):
        max_samples = max_samples or DATASET_SIZES.get("token_deletion", 200_000)
        super().__init__(split, max_samples=max_samples)

        self.delete_prob = delete_prob or TOKEN_DELETION_CONFIG.get("delete_prob", 0.15)

        # ✅ Get dataset source from TASKS config
        try:
            from configs.data import TASKS
            dataset_source = TASKS["token_deletion"].dataset_source
        except (ImportError, KeyError):
            dataset_source = "wikipedia"  # fallback

        self.data = load_wikipedia_streaming(
            split=split, max_samples=max_samples, dataset_source=dataset_source
        )
        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ Token Deletion Dataset ({dataset_source}): {len(self.data)} samples")

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.encode_text(text)
        input_ids = encoding["input_ids"][0].clone()
        attention_mask = encoding["attention_mask"][0]

        # Create binary labels (0=keep, 1=delete)
        seq_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        labels = torch.zeros(len(input_ids), dtype=torch.long)

        # Mark tokens for deletion
        for i in range(1, seq_length - 1):  # Skip [CLS] and [SEP]
            if random.random() < self.delete_prob:
                labels[i] = 1  # Mark as "delete"

        # Padding positions get ignore_index
        labels[~attention_mask.bool()] = -100
        # [CLS] and [SEP] also get ignore_index
        labels[0] = -100
        if seq_length > 0:
            labels[seq_length - 1] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class TextInfillingDataset(BaseDataset):
    """Text Infilling (BART): Spans replaced with single [MASK]."""

    def __init__(
        self,
        split="train",
        max_samples=None,
        mask_prob=0.30,
        mean_span=3.0,
        max_span_length=10,
        single_mask_per_span=True,
    ):
        max_samples = max_samples or DATASET_SIZES.get("text_infilling", 200_000)
        super().__init__(split, max_samples=max_samples)

        config = TEXT_INFILLING_CONFIG
        self.mask_prob = mask_prob or config.get("mask_prob", 0.30)
        self.mean_span = mean_span or config.get("mean_span", 3.0)
        self.max_span_length = max_span_length or config.get("max_span_length", 10)
        self.single_mask_per_span = (
            single_mask_per_span
            if single_mask_per_span is not None
            else config.get("single_mask_per_span", True)
        )

        # ✅ Get dataset source from TASKS config
        try:
            from configs.data import TASKS
            dataset_source = TASKS["text_infilling"].dataset_source
        except (ImportError, KeyError):
            dataset_source = "wikipedia"  # fallback

        self.data = load_wikipedia_streaming(
            split=split, max_samples=max_samples, dataset_source=dataset_source
        )
        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ Text Infilling Dataset ({dataset_source}): {len(self.data)} samples")
        print(
            f"   Mode: {'BART-style (single [MASK])' if self.single_mask_per_span else 'Multiple [MASK]'}"
        )

    def __getitem__(self, idx):
        text = self.data[idx]
        encoding = self.encode_text(text)
        input_ids = encoding["input_ids"][0].clone()

        # Create span mask (simplified version)
        num_tokens = (input_ids != self.tokenizer.pad_token_id).sum().item() - 2
        num_to_mask = max(1, int(num_tokens * self.mask_prob))

        labels = input_ids.clone()
        labels[:] = -100

        masked_count = 0
        while masked_count < num_to_mask:
            span_length = min(
                np.random.poisson(self.mean_span) + 1, self.max_span_length
            )
            start_idx = random.randint(1, max(1, num_tokens - span_length + 1))
            end_idx = min(start_idx + span_length, num_tokens + 1)

            if (input_ids[start_idx:end_idx] == self.tokenizer.mask_token_id).any():
                continue

            # Store original for labels
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx].clone()

            if self.single_mask_per_span:
                input_ids[start_idx] = self.tokenizer.mask_token_id
                # Shift left
                remaining = end_idx - start_idx - 1
                if remaining > 0:
                    # Clone to avoid in-place operation overlap
                    input_ids[start_idx + 1 : num_tokens + 1 - remaining] = input_ids[
                        end_idx : num_tokens + 1
                    ].clone()
                    input_ids[num_tokens + 1 - remaining : num_tokens + 1] = (
                        self.tokenizer.pad_token_id
                    )
                    num_tokens -= remaining
            else:
                input_ids[start_idx:end_idx] = self.tokenizer.mask_token_id

            masked_count += span_length

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SentencePermutationDataset(BaseDataset):
    """Sentence Permutation (BART): Detect shuffled sentence order."""

    def __init__(
        self,
        split="train",
        max_samples=None,
        shuffle_prob=None,
        min_sentences=None,
        max_sentences=None,
    ):
        max_samples = max_samples or DATASET_SIZES.get("sentence_permutation", 200_000)
        super().__init__(split, max_samples=max_samples)

        config = SENTENCE_PERMUTATION_CONFIG
        self.shuffle_prob = shuffle_prob if shuffle_prob is not None else config.get("shuffle_prob", 0.5)
        self.min_sentences = min_sentences if min_sentences is not None else config.get("min_sentences", 3)
        self.max_sentences = max_sentences if max_sentences is not None else config.get("max_sentences", 6)

        # Load paragraphs with multiple sentences
        self.data = self._load_paragraphs()
        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ Sentence Permutation Dataset: {len(self.data)} samples")

    def _load_paragraphs(self):
        """Load Wikipedia paragraphs with multiple sentences."""
        dataset_source = DATASET_SOURCES.get("wikipedia", {}).get("primary", {})

        try:
            hf_dataset = load_dataset(
                dataset_source.get("path", "wikimedia/wikipedia"),
                dataset_source.get("name", "20231101.en"),
                split=self.split,
                streaming=True,
            )
        except Exception:
            fallback = DATASET_SOURCES.get("wikipedia", {}).get("fallback", {})
            hf_dataset = load_dataset(
                fallback.get("path", "Salesforce/wikitext"),
                fallback.get("name", "wikitext-103-raw-v1"),
                split=self.split,
                streaming=True,
            )

        valid_samples = []
        for idx, item in enumerate(hf_dataset):
            if idx >= (self.max_samples or 100000) * 10:
                break

            text = item.get("text", "").strip()
            if not text:
                continue

            sentences = [s.strip() for s in text.split(".") if s.strip()]

            if len(sentences) < self.min_sentences:
                continue

            valid_sents = [s for s in sentences if 3 <= len(s.split()) <= 30]

            if len(valid_sents) >= self.min_sentences:
                valid_samples.append(valid_sents[: self.max_sentences])

            if len(valid_samples) >= (self.max_samples or 100000):
                break

        return valid_samples

    def __getitem__(self, idx):
        sentences = self.data[idx]

        is_shuffled = random.random() < self.shuffle_prob
        if is_shuffled:
            shuffled = sentences.copy()
            random.shuffle(shuffled)
            text = ". ".join(shuffled) + "."
            label = 1
        else:
            text = ". ".join(sentences) + "."
            label = 0

        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Natural Language Understanding Tasks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class NLIDataset(BaseDataset):
    """Natural Language Inference (MultiNLI)."""

    def __init__(self, split="train", max_length=256, max_samples=None):
        max_samples = max_samples or DATASET_SIZES.get("nli", 100_000)
        super().__init__(split, max_length, max_samples)

        actual_split = "validation_matched" if split == "validation" else split
        dataset = load_dataset("multi_nli", split=actual_split)

        for item in dataset:
            if item["label"] >= 0:
                self.data.append(
                    {
                        "premise": item["premise"],
                        "hypothesis": item["hypothesis"],
                        "label": item["label"],
                    }
                )
                if self.max_samples and len(self.data) >= self.max_samples:
                    break

        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ NLI Dataset (MultiNLI): {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["premise"] + " [SEP] " + item["hypothesis"]
        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"]),
        }


class WiCDataset(BaseDataset):
    """Word-in-Context (SuperGLUE)."""

    def __init__(
        self, split="train", max_length=256, max_samples=None, difficulty="hard"
    ):
        max_samples = max_samples or DATASET_SIZES.get("wic", 100_000)
        super().__init__(split, max_length, max_samples)

        dataset = load_dataset("super_glue", "wic", split=split)
        limit = max_samples if max_samples else len(dataset)

        for i, item in enumerate(dataset):
            if i >= limit:
                break
            self.data.append(
                {
                    "sentence1": item["sentence1"],
                    "sentence2": item["sentence2"],
                    "word": item["word"],
                    "label": item["label"],
                }
            )

        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        self.difficulty = difficulty
        print(f"✅ WiC Dataset ({difficulty} mode): {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.difficulty == "easy":
            text = (
                f"Word: {item['word']}. {item['sentence1']} [SEP] {item['sentence2']}"
            )
        elif self.difficulty == "medium":
            text = (
                f"{item['sentence1']} [SEP] {item['sentence2']}. Word: {item['word']}"
            )
        else:  # hard
            text = f"{item['sentence1']} [SEP] {item['sentence2']}"

        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


class NSPDataset(BaseDataset):
    """Next Sentence Prediction."""

    def __init__(self, split="train", max_length=256, max_samples=None):
        max_samples = max_samples or DATASET_SIZES.get("nsp", 500_000)
        super().__init__(split, max_length, max_samples)

        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        )

        # Collect filtered sentences
        all_sentences = []
        for item in dataset:
            text = item["text"].strip()
            if text and not text.startswith("="):  # Skip titles
                sentences = [s.strip() for s in re.split(r"[.!?]+", text)]

                for sent in sentences:
                    if self._is_valid_sentence(sent):
                        all_sentences.append(sent)

        # Create positive/negative pairs
        for i in range(0, len(all_sentences) - 1, 2):
            # Positive pair (consecutive)
            self.data.append(
                {"sent1": all_sentences[i], "sent2": all_sentences[i + 1], "label": 1}
            )

            # Negative pair (random)
            j = random.randint(0, len(all_sentences) - 1)
            if j != i + 1:
                self.data.append(
                    {"sent1": all_sentences[i], "sent2": all_sentences[j], "label": 0}
                )

            if self.max_samples and len(self.data) >= self.max_samples:
                break

        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ NSP Dataset: {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["sent1"] + " [SEP] " + item["sent2"]
        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"]),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Reasoning Tasks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class COPAStyleDataset(BaseDataset):
    """COPA-style dataset with data augmentation."""

    def __init__(
        self,
        split="train",
        max_length=128,
        max_samples=None,
        augment=True,
        aug_factor=4,
    ):
        max_samples = max_samples or DATASET_SIZES.get("copa_style", 100_000)
        super().__init__(split, max_length, max_samples)

        copa_split = "train" if split == "train" else "validation"
        dataset = load_dataset("super_glue", "copa", split=copa_split)

        # Basic synonyms for augmentation
        self.synonyms = {
            "man": ["guy", "person"],
            "woman": ["lady", "person"],
            "because": ["since", "as"],
            "so": ["therefore", "thus"],
            "big": ["large", "huge"],
            "small": ["tiny", "little"],
        }

        original_count = 0
        for item in dataset:
            self.data.append(
                {
                    "premise": item["premise"],
                    "choice1": item["choice1"],
                    "choice2": item["choice2"],
                    "label": item["label"],
                }
            )
            original_count += 1

            # Augmentation
            if augment and split == "train":
                for aug_type in range(1, aug_factor):
                    aug_item = self._augment_item(item, aug_type)
                    self.data.append(aug_item)

        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ COPA Dataset: {len(self.data)} samples (original: {original_count})")

    def _augment_item(self, item, aug_type):
        """Simple text augmentation."""
        premise = item["premise"]

        if aug_type == 1:  # Synonym replacement
            words = premise.split()
            for i in range(len(words)):
                word = words[i].lower().rstrip(".,!?")
                if word in self.synonyms and random.random() < 0.4:
                    replacement = random.choice(self.synonyms[word])
                    if words[i][0].isupper():
                        replacement = replacement.capitalize()
                    words[i] = words[i].replace(word, replacement)
            premise = " ".join(words)

        return {
            "premise": premise,
            "choice1": item["choice1"],
            "choice2": item["choice2"],
            "label": item["label"],
        }

    def __getitem__(self, idx):
        item = self.data[idx]

        text1 = item["premise"] + " [SEP] " + item["choice1"]
        text2 = item["premise"] + " [SEP] " + item["choice2"]

        enc1 = self.encode_text(text1)
        enc2 = self.encode_text(text2)

        input_ids = torch.stack([enc1["input_ids"][0], enc2["input_ids"][0]])
        attention_mask = torch.stack(
            [enc1["attention_mask"][0], enc2["attention_mask"][0]]
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


COPADataset = COPAStyleDataset


class BoolQDataset(BaseDataset):
    """Boolean Question Answering."""

    def __init__(self, split="train", max_length=256, max_samples=None):
        max_samples = max_samples or DATASET_SIZES.get("boolq", 1_000)
        super().__init__(split, max_length, max_samples)

        dataset = load_dataset("google/boolq", split=split)

        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            self.data.append(
                {
                    "question": item["question"],
                    "passage": item["passage"],
                    "label": 1 if item["answer"] else 0,
                }
            )

        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ BoolQ Dataset: {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["question"] + " [SEP] " + item["passage"]
        encoding = self.encode_text(text)

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"]),
        }


class StoryCompletionDataset(BaseDataset):
    """Story Completion (context + 4 choices)."""

    def __init__(self, split="train", max_length=256, max_samples=None):
        max_samples = max_samples or DATASET_SIZES.get("story_completion", 100_000)
        super().__init__(split, max_length, max_samples)

        dataset = load_dataset(
            "Salesforce/wikitext", "wikitext-103-raw-v1", split=split
        )

        for item in dataset:
            text = item["text"].strip()

            if len(text) > 200 and not text.startswith("="):
                sentences = [s.strip() for s in re.split(r"[.!?]+", text)]
                valid_sentences = [s for s in sentences if self._is_valid_sentence(s)]

                if len(valid_sentences) >= 7:
                    context = " ".join(valid_sentences[:3])
                    correct = valid_sentences[3]
                    available = valid_sentences[4:]

                    if len(available) >= 3:
                        distractors = random.sample(available, 3)
                        choices = [correct] + distractors
                        random.shuffle(choices)
                        label = choices.index(correct)

                        self.data.append(
                            {"context": context, "choices": choices, "label": label}
                        )

                        if self.max_samples and len(self.data) >= self.max_samples:
                            break

        # Only shuffle for training split
        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ Story Completion Dataset: {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]

        input_ids_list = []
        attention_mask_list = []

        for choice in item["choices"]:
            text = item["context"] + " [SEP] " + choice
            enc = self.encode_text(text)
            input_ids_list.append(enc["input_ids"][0])
            attention_mask_list.append(enc["attention_mask"][0])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.tensor(item["label"]),
        }


StoryClozeDataset = StoryCompletionDataset


class SentenceOrderingDataset(BaseDataset):
    """Sentence Ordering Task."""

    def __init__(self, split="train", max_length=256, max_samples=None):
        max_samples = max_samples or DATASET_SIZES.get("sentence_ordering", 100_000)
        super().__init__(split, max_length, max_samples)

        sentences = load_wikipedia_streaming(split=split, max_samples=max_samples * 10)

        # Create sentence ordering pairs
        for i in range(0, len(sentences) - 3, 3):
            if len(self.data) >= max_samples:
                break

            sents = sentences[i : i + 3]
            if all(self._is_valid_sentence(s) for s in sents):
                # Correct order
                self.data.append({"text": " ".join(sents), "label": 1})

                # Shuffled order
                shuffled = sents.copy()
                random.shuffle(shuffled)
                self.data.append({"text": " ".join(shuffled), "label": 0})

        if self.split == "train":
            random.shuffle(self.data)
        print(f"✅ Sentence Ordering Dataset: {len(self.data)} samples")

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.encode_text(item["text"])

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"]),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Semantic Understanding Tasks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ParaphraseDataset(BaseDataset):
    """Paraphrase Detection (QQP/PAWS)."""

    def __init__(self, split="train", max_length=None, max_samples=None, source="qqp"):
        max_samples = max_samples or DATASET_SIZES.get("paraphrase", 50_000)
        super().__init__(split, max_length, max_samples)
        self.source = source

        # Try primary source, fallback if needed
        if source == "qqp":
            if not self._load_qqp():
                print("⚠️ QQP failed, trying PAWS...")
                self.source = "paws"
                self._load_paws()
        else:
            self._load_paws()

        if self.split == "train":
            random.shuffle(self.data)
        print(
            f"✅ Paraphrase Dataset ({self.source.upper()}): {len(self.data)} samples"
        )

    def _load_qqp(self):
        """Load Quora Question Pairs."""
        try:
            dataset = load_dataset("quora", split=self.split)
        except Exception:
            return False

        for i, item in enumerate(dataset):
            if i >= self.max_samples:
                break

            q1 = item["questions"]["text"][0].strip()
            q2 = item["questions"]["text"][1].strip()
            label = 1 if item["is_duplicate"] else 0

            if self._is_valid_pair(q1, q2, min_words=3, max_words=100):
                self.data.append({"text1": q1, "text2": q2, "label": label})

        return len(self.data) > 0

    def _load_paws(self):
        """Load PAWS."""
        try:
            dataset = load_dataset("paws", "labeled_final", split=self.split)
        except Exception:
            return False

        for i, item in enumerate(dataset):
            if i >= self.max_samples:
                break

            s1 = item["sentence1"].strip()
            s2 = item["sentence2"].strip()

            if self._is_valid_pair(s1, s2, min_words=3, max_words=100):
                self.data.append({"text1": s1, "text2": s2, "label": item["label"]})

        return len(self.data) > 0

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer.encode_plus(
            item["text1"],
            item["text2"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(item["label"]),
        }


class STSDataset(BaseDataset):
    """Semantic Textual Similarity (regression)."""

    def __init__(self, split="train", max_length=None, max_samples=None, augment=True):
        max_samples = max_samples or DATASET_SIZES.get("sts", 40_000)
        super().__init__(split, max_length, max_samples)
        self.augment = augment

        # Load STS-B
        try:
            dataset = load_dataset("stsb_multi_mt", "en", split=self.split)
        except Exception:
            if self.split == "validation":
                dataset = load_dataset("stsb_multi_mt", "en", split="dev")
            else:
                self.data = []
                print(f"✅ STS Dataset: 0 samples (failed to load)")
                return

        base_samples = []
        for item in dataset:
            s1 = item["sentence1"].strip()
            s2 = item["sentence2"].strip()
            score = item["similarity_score"]

            if self._is_valid_pair(s1, s2, min_words=3, max_words=100):
                base_samples.append({"text1": s1, "text2": s2, "score": score})

        self.data = base_samples.copy()

        # Augmentation: swap sentences (symmetric)
        if self.augment and split == "train":
            for sample in base_samples:
                self.data.append(
                    {
                        "text1": sample["text2"],
                        "text2": sample["text1"],
                        "score": sample["score"],
                    }
                )

        if self.max_samples and len(self.data) > self.max_samples:
            random.shuffle(self.data)
            self.data = self.data[: self.max_samples]

        print(f"✅ STS Dataset: {len(self.data)} samples (augmented: {self.augment})")

    def __getitem__(self, idx):
        item = self.data[idx]

        encoding = self.tokenizer.encode_plus(
            item["text1"],
            item["text2"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        normalized_score = item["score"] / 5.0

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(normalized_score, dtype=torch.float),
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Registry
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET_MAP = {
    # Pre-training
    "mlm": MLMDataset,
    "span_masking": SpanMaskingDataset,
    # Denoising (BART-style)
    "token_deletion": TokenDeletionDataset,
    "text_infilling": TextInfillingDataset,
    "sentence_permutation": SentencePermutationDataset,
    # Understanding
    "nli": NLIDataset,
    "wic": WiCDataset,
    "nsp": NSPDataset,
    # Reasoning
    "copa_style": COPAStyleDataset,
    "copa": COPADataset,
    "boolq": BoolQDataset,
    "story_completion": StoryCompletionDataset,
    "story_cloze": StoryClozeDataset,
    "sentence_ordering": SentenceOrderingDataset,
    # Semantic
    "paraphrase": ParaphraseDataset,
    "sts": STSDataset,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DataLoader Factory
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_task_max_samples(task, max_samples=None, global_limit=None):
    """
    Resolve max_samples with proper priority:
    1. Explicit max_samples parameter
    2. Global limit (backward compatibility)
    3. Task-specific config
    4. None
    """
    if max_samples is not None:
        return max_samples
    if global_limit is not None:
        return global_limit
    if task in DATASET_SIZES:
        return DATASET_SIZES[task]
    return None


def get_dataloader(
    task,
    split="train",
    batch_size=32,
    num_workers=None,
    task_config=None,
    max_samples=None,
    global_limit=None,
    verbose=True,
    **kwargs,
):
    """
    Create DataLoader for a task.

    Args:
        task: Task name
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of workers
        task_config: Task-specific configuration
        max_samples: Explicit max samples (recommended)
        global_limit: Global limit (deprecated)
        verbose: Print info
        **kwargs: Additional args for Dataset/DataLoader

    Returns:
        DataLoader instance
    """
    # Apply defaults
    num_workers = (
        num_workers
        if num_workers is not None
        else DATALOADER_CONFIG.get("num_workers", 2)
    )

    if task not in DATASET_MAP:
        raise ValueError(f"Unknown task: {task}. Available: {list(DATASET_MAP.keys())}")

    dataset_class = DATASET_MAP[task]

    # Resolve max_samples
    final_max_samples = get_task_max_samples(task, max_samples, global_limit)

    if verbose and final_max_samples:
        source = "explicit" if max_samples else "global" if global_limit else "config"
        print(f"📊 {task.upper()}: max_samples={final_max_samples:,} (from {source})")

    # Separate DataLoader vs Dataset args
    dataloader_arg_names = {
        "pin_memory",
        "prefetch_factor",
        "persistent_workers",
        "timeout",
        "worker_init_fn",
        "multiprocessing_context",
        "generator",
        "drop_last",
    }

    dataloader_args = {}
    dataset_kwargs = {}

    for key, value in kwargs.items():
        if key in dataloader_arg_names:
            dataloader_args[key] = value
        else:
            dataset_kwargs[key] = value

    dataset_kwargs["max_samples"] = final_max_samples

    # Task-specific configuration
    if task == "mlm" and task_config and "mask_prob" in task_config:
        dataset_kwargs["mask_prob"] = task_config["mask_prob"]

    if task == "span_masking":
        # Span difficulty curriculum
        if task_config and "span_difficulty" in task_config:
            difficulty = task_config["span_difficulty"]
            curriculum = SPAN_MASKING_CONFIG.get("curriculum", {})
            if difficulty in curriculum:
                span_params = curriculum[difficulty]
                dataset_kwargs["mean_span"] = span_params["mean_span"]
                dataset_kwargs["max_span_length"] = span_params["max_span_length"]
                if verbose:
                    print(f"✅ Span difficulty: {difficulty.upper()}")

        # Auto-detect POS cache
        if "pos_cache_file" not in dataset_kwargs:
            cache_dir = os.environ.get("DAWN_CACHE_DIR", "./cache")
            cache_file = os.path.join(cache_dir, f"pos_tags_{split}_100000.pkl")
            if os.path.exists(cache_file):
                dataset_kwargs["pos_cache_file"] = cache_file

        # POS auxiliary task control
        if task_config and "enable_pos" in task_config:
            dataset_kwargs["enable_pos"] = task_config["enable_pos"]

    if task == "wic" and "difficulty" not in dataset_kwargs:
        dataset_kwargs["difficulty"] = "medium"

    # Create dataset
    dataset = dataset_class(split=split, **dataset_kwargs)

    # Collate function for multi-choice tasks
    needs_collate = task in ["copa_style", "copa", "story_completion", "story_cloze"]

    if needs_collate:

        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    else:
        collate_fn = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        collate_fn=collate_fn,
        **dataloader_args,
    )
