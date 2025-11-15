"""MLM Dataset wrapper"""

import torch
from torch.utils.data import Dataset


class MLMDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=128, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        labels = input_ids.clone()

        # Mask tokens
        probability_matrix = torch.full(labels.shape, self.mask_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                [val], already_has_special_tokens=True
            )[0]
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        input_ids[masked_indices] = self.tokenizer.mask_token_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
