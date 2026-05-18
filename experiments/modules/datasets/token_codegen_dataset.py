from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from typing import List

class HFTokenDataset(Dataset):
    """
    Sliding-window dataset for next-token prediction over a single token-ID stream.

    All files are tokenised once with the HF tokenizer, concatenated into one
    long stream, then chopped into fixed-length contexts. Because every sample
    is exactly `ctx` tokens, no padding is needed → loss is computed everywhere.
    """
    def __init__(self, texts: List[str], hf_tokenizer, ctx: int = 256):
        self.ctx = ctx
        all_ids: List[int] = []
        eos_id = hf_tokenizer.eos_token_id
        # encode each file separately and join with EOS so the model sees document boundaries
        for t in tqdm(texts, desc="[Tokenising]", unit="file"):
            ids = hf_tokenizer.encode(t, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(eos_id)
        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"[HFTokenDataset] {len(self.data):,} tokens, "
              f"{max(0, len(self.data) - ctx - 1):,} samples")

    def __len__(self):
        return max(0, len(self.data) - self.ctx - 1)

    def __getitem__(self, i):
        # for causal LM we just return the chunk — the loop builds inputs/labels
        return self.data[i: i + self.ctx + 1]   # length ctx+1 → shift for next-token target