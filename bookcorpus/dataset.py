import os
import torch
import numpy as np
from torch.utils.data import Dataset
from jaxtyping import Int
from torch import Tensor

class BinaryDataset(Dataset):
    def __init__(self, data_dir: str, context_len: int):
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".bin")])
        self.context_len = context_len
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")
        
        # Load all shards (Zero-copy memmap)
        self.shards = [np.memmap(f, dtype=np.uint16, mode='r') for f in self.files]
        self.shard_lengths = [len(s) for s in self.shards]
        
    def get_batch(self, batch_size: int, device: str) -> tuple[Int[Tensor, "B T"], Int[Tensor, "B T"]]:
        # Fast path for batch_size=1 (Your specific case)
        if batch_size == 1:
            ix_shard = np.random.randint(0, len(self.shards))
            shard = self.shards[ix_shard]
            idx = np.random.randint(0, len(shard) - self.context_len - 1)
            
            # Slice directly without list overhead
            chunk = shard[idx : idx + self.context_len + 1].astype(np.int64)
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
            return x, y

        # General path
        x_batch = torch.empty((batch_size, self.context_len), dtype=torch.long, device=device)
        y_batch = torch.empty((batch_size, self.context_len), dtype=torch.long, device=device)
        
        for i in range(batch_size):
            ix_shard = np.random.randint(0, len(self.shards))
            shard = self.shards[ix_shard]
            idx = np.random.randint(0, len(shard) - self.context_len - 1)
            
            # Convert directly from uint16 to tensor
            chunk = torch.from_numpy(shard[idx : idx + self.context_len + 1].astype(np.int64)).to(device)
            x_batch[i] = chunk[:-1]
            y_batch[i] = chunk[1:]
            
        return x_batch, y_batch