import torch
import torch.nn as nn
import torch.nn.functional as F

class HeadAttention(nn.Module):
    def __init__(self, emb_size: int, head_size: int, max_seq_len: int) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.key_w = nn.Linear(self.emb_size, self.head_size)
        self.query_w = nn.Linear(self.emb_size, self.head_size)
        self.value_w = nn.Linear(self.emb_size, self.head_size)

        self.mask = torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).byte()

    def forward(self, X: torch.tensor) -> torch.tensor:
        key = self.key_w(X)
        query = self.query_w(X)
        value = self.value_w(X)

        seq_len = X.shape[1]
        mask = (self.mask[:seq_len, :seq_len].bool()).to(X.device)

        attention = query @ key.transpose(-2, -1)
        norm_attention = attention / (self.head_size ** 0.5)
        masked_attention = norm_attention.masked_fill(~mask, float('-inf'))

        probabilities = F.softmax(masked_attention, dim = -1)
        output = probabilities @ value

        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, emb_size: int, head_size: int, max_seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        self.heads = nn.ModuleList([HeadAttention(self.emb_size, self.head_size, self.max_seq_len) for _ in range(self.num_heads)])
        self.multi_head_weight = nn.Linear(self.head_size * self.num_heads, self.emb_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, X: torch.tensor) -> torch.tensor:
        heads = [head(X) for head in self.heads]
        heads_concat = torch.cat(heads, dim = -1)
        heads_output = self.multi_head_weight(heads_concat)
        output = self.dropout_layer(heads_output)

        return output