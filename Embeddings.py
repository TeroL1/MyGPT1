import torch
import torch.nn as nn

class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.token_embedding = nn.Embedding(self.vocab_size, self.emb_size)

    def forward(self, X: torch.tensor):
        return self.token_embedding(X)

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.emb_size)

    def forward(self, seq_len: int, device: str = 'cpu'):
        pos_embedding_tensor = torch.tensor([i for i in range(seq_len)], device=device)
        return self.pos_embedding(pos_embedding_tensor)