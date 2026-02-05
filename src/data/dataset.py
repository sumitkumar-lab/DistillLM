import json
import torch
from torch.utils.data import Dataset


class IntentDataset(Dataset):
    def __init__(self, path, tokenizer, label2id, use_soft=False):
        self.samples = json.load(open(path))
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.use_soft = use_soft

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=32,
            return_tensors="pt"
        )

        if self.use_soft:
            soft = torch.tensor(
                [item["soft_labels"][k] for k in self.label2id],
                dtype=torch.float
            )
            return enc["input_ids"].squeeze(0), soft

        return (
            enc["input_ids"].squeeze(0),
            torch.tensor(self.label2id[item["label"]])
        )
