import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout

        self.linear1 = nn.Linear(self.emb_size, 4 * self.emb_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(4 * self.emb_size, self.emb_size)
        self.dropout1 = nn.Dropout(self.dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        linear1_output = self.linear1(X)
        relu1_output = self.relu1(linear1_output)
        linear2_output = self.linear2(relu1_output)
        output = self.dropout1(linear2_output)

        return output