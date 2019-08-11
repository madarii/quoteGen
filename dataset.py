import torch
from torch.utils.data import Dataset
import numpy as np


class QuoteDataset(Dataset):
    def __init__(self, quote_file):
        self.quote_dataset = open(quote_file).read().strip()
        self.chars = sorted(set(self.quote_dataset))
        self.quote_dataset = self.quote_dataset.split("\n")
        self.n_chars = len(self.chars) + 2  # 2 extra for SOS and EOS
        self.char_to_idx = {c: i + 1
                            for i, c in enumerate(self.chars)}  # 0 = SOS
        self.idx_to_char = {i + 1: c
                            for i, c in enumerate(self.chars)
                            }  # n_chars-1 = EOS

    def __len__(self):
        return len(self.quote_dataset)

    def __getitem__(self, idx):
        quote = self.quote_dataset[idx].split("\t")[1]
        input_quote = quote
        target_quote = quote
        sample = {"input": input_quote, "target": target_quote}
        sample = self.encode(sample, self.char_to_idx, self.n_chars)
        sample = self.one_hot(sample, self.n_chars)
        sample = self.to_tensor(sample)

        return sample

    def encode(self, sample, char_to_idx, n_char):
        input_quote = sample["input"]
        target_quote = sample["target"]

        input_quote = [0] + [self.char_to_idx[c] for c in input_quote]
        target_quote = [self.char_to_idx[c]
                        for c in target_quote] + [self.n_chars - 1]

        return {"input": input_quote, "target": target_quote}

    def one_hot(self, sample, n_chars):
        encoding = np.zeros((len(sample["input"]), n_chars))
        input_quote = sample["input"]
        for i, e in enumerate(input_quote):
            encoding[i][e] = 1
        sample["input"] = encoding
        return sample

    def to_tensor(self, sample):
        return {"input" : torch.Tensor(sample["input"]),\
                "target" : torch.LongTensor(sample["target"])}
