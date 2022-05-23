import torch
import io, tokenize, re
import ast, astunparse

from typing import Tuple, Optional, List, Union

def right_pad_sequences(sequences: List[torch.Tensor], batch_first: bool = True, padding_value: Union[int, bool] = 0, 
                       max_len: int = -1, device: torch.device = None) -> torch.Tensor:
    assert all([len(seq.shape) == 1 for seq in sequences])
    max_len = max_len if max_len > 0 else max(len(s) for s in sequences)
    device = device if device is not None else sequences[0].device

    padded_seqs = []
    for seq in sequences:
        padded_seqs.append(torch.cat(seq, (torch.full((max_len - seq.shape[0],), padding_value, dtype=torch.long).to(device))))
    return torch.stack(padded_seqs)