from typing import List, Tuple, Any
from modules.tokenizers.base_tokenizer import CodeTokenizer
from torch.utils.data import Dataset
import random
import torch

class LineDataset(Dataset):
    """
    One sample = (prefix_tokens, full_line_tokens).
    The model learns to predict the rest of the current line given a prefix.
    """
    def __init__(self, texts: List[str], tokenizer: CodeTokenizer,
                 max_prefix: int = 96, max_line: int = 64) -> None:
        self.samples: List[Tuple[List[int], List[int]]] = []
        for text in texts:
            for line in text.splitlines():
                commentary_pos = line.find('#') 
                if commentary_pos != -1 and not line[commentary_pos - 1] in ['\'', '"']:
                    # print(line)
                    line = line[:line.find('#')]
                line = line.rstrip()
                if len(line.strip()) < 10:
                    continue
                full = tokenizer.encode(line)
                if len(full) < 4:
                    continue
                split = random.randint(2, max(2, len(full) - 2))
                prefix = full[:split][-max_prefix:]
                target = full[split:][:max_line]
                target.append(tokenizer.eos_id)
                self.samples.append((prefix, target))
        print(f"[LineDataset] {len(self.samples)} samples")

    def __len__(self) -> int: return len(self.samples)

    def __getitem__(self, i) -> Any:
        return self.samples[i]



def collate_line(batch, pad_id: int):
    prefixes, targets = zip(*batch)
    max_p = max(len(p) for p in prefixes)
    max_t = max(len(t) for t in targets)
    P = torch.full((len(batch), max_p), pad_id, dtype=torch.long)
    T = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    for i, (p, t) in enumerate(zip(prefixes, targets)):
        P[i, :len(p)] = torch.tensor(p)
        T[i, :len(t)] = torch.tensor(t)
    return P, T