import torch
from torch.utils.data import Dataset

class GetData(Dataset):
    def __init__(self, data: list, seq_len: int, device: str) -> None:
        self.data = data
        self.seq_len = seq_len
        self.device = device

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx: int) -> tuple:
        x = torch.tensor(self.data[idx:idx + self.seq_len], device = self.device).long()
        y = torch.tensor(self.data[idx + 1:idx + 1 + self.seq_len], device = self.device).long()

        return x, y