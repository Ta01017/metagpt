# dataset_stage1.py
from typing import List

import torch
from torch.utils.data import Dataset


class Stage1Dataset(Dataset):
    """
    Build (input_ids, labels) for Stage1 LM training.
    sequences: list of token-id lists without BOS/EOS.
    """

    def __init__(self, sequences: List[List[int]], max_len: int, bos_id: int, eos_id: int, pad_id: int):
        self.sequences = sequences
        self.max_len = int(max_len)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.pad_id = int(pad_id)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        # Reserve 1 position for BOS/EOS so input/label lengths stay == max_len.
        seq = seq[: max(self.max_len - 1, 0)]

        inp = [self.bos_id] + seq
        tgt = seq + [self.eos_id]

        pad = self.max_len - len(inp)
        if pad > 0:
            inp = inp + [self.pad_id] * pad
            tgt = tgt + [-100] * pad  # ignore padding in loss

        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
