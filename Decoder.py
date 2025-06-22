import torch
import torch.nn as nn

from Attention import MultiHeadAttention
from FeedForward import FeedForward

class Decoder(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.multi_head_attention = MultiHeadAttention(self.num_heads, self.emb_size, self.head_size, self.max_seq_len, self.dropout)
        self.norm1 = nn.LayerNorm(self.emb_size)
        self.feed_forward = FeedForward(self.emb_size, self.dropout)
        self.norm2 = nn.LayerNorm(self.emb_size)

    def forward(self, X: torch.tensor) -> torch.tensor:
        multi_head_X = self.multi_head_attention(X)
        residual1_X = X + multi_head_X
        norm1_X = self.norm1(residual1_X)
        feed_forward_X = self.feed_forward(norm1_X)
        residual2_X = feed_forward_X + norm1_X
        output = self.norm2(residual2_X)

        return output