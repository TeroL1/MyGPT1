import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from Embeddings import TokenEmbeddings
from Embeddings import PositionalEmbeddings
from Decoder import Decoder

from torch.optim import Adam

class GPT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, emb_size: int, num_heads: int, head_size: int, num_layers: int, dropout: float = 0.1, device: str = 'cpu') -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.token_embedding = TokenEmbeddings(self.vocab_size, self.emb_size)
        self.positional_embedding = PositionalEmbeddings(self.max_seq_len, self.emb_size)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.decoders = nn.Sequential(*[Decoder(self.num_heads, self.emb_size, self.head_size, self.max_seq_len, self.dropout) for _ in range(self.num_layers)])
        self.linear = nn.Linear(self.emb_size, self.vocab_size)

        self.train_loss = []
        self.valid_loss = []

    def forward(self, X: torch.tensor) -> torch.tensor:
        token_emb_output = self.token_embedding(X)
        pos_emb_output = self.positional_embedding(X.shape[1], device=X.device)
        X_emb = token_emb_output + pos_emb_output
        X_dropout = self.dropout_layer(X_emb)
        X_decoder = self.decoders(X_dropout)
        output = self.linear(X_decoder)

        return output

    def generate(self, X: torch.tensor, max_new_tokens: int, do_sample: bool, temperature: float = 1.0, top_k: int = None, top_p: int = None) -> torch.tensor:
        for _ in range(max_new_tokens):
            input_tensor = X[:, -self.max_seq_len:]
            output_tensor = self.forward(input_tensor)
            last_vector = output_tensor[:, -1, :] / temperature
            last_vector = __class__._recomputeTopPK(last_vector, do_sample, top_k, top_p)
            probabilities = F.softmax(last_vector, dim = -1)
            next_token = __class__._findNewToken(probabilities, do_sample)
            X = torch.cat([X, next_token], dim = -1)

        return X

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader, num_epoch: int, learning_rate: float, device: str = 'cpu', saver = False, verbose: int = None) -> None:
        self.to(device)
        optimizer = Adam(self.parameters(), lr = learning_rate)

        for epoch in tqdm(range(num_epoch)):
            epoch_train_loss = []
            epoch_valid_loss = []

            self.train()
            for inputs, targets in train_loader:
                ouputs = self.forward(inputs)
                ouputs = torch.reshape(ouputs, (-1, self.vocab_size))
                targets = torch.flatten(targets)
                loss = F.cross_entropy(ouputs, targets)
                epoch_train_loss.append(float(loss))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_loss = torch.Tensor(epoch_train_loss).mean()
            self.train_loss.append(average_loss)

            if verbose and epoch % verbose == 0: print(f"Train loss [epoch: {epoch}]: {average_loss}")

            self.eval()
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    ouputs = self.forward(inputs)
                    ouputs = torch.reshape(ouputs, (-1, self.vocab_size))
                    targets = torch.flatten(targets)
                    loss = F.cross_entropy(ouputs, targets)
                    epoch_valid_loss.append(float(loss))

                average_val_loss = torch.Tensor(epoch_valid_loss).mean()
                self.valid_loss.append(average_val_loss)

                if verbose and epoch % verbose == 0: print(f"Valid loss [epoch: {epoch}]: {average_val_loss}")

            if saver:
                self.save(f"Models\MyGPT1_vol{epoch}")

    def save(self, path) -> None:
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)

    @classmethod
    def load(cls, path, device) -> 'GPT':
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model

    @staticmethod
    def _findNewToken(probabilities: torch.Tensor, do_sample: bool) -> torch.Tensor:
        if do_sample:
            return torch.multinomial(probabilities, num_samples = 1)

        return torch.argmax(probabilities, dim = -1, keepdim = True)

    @staticmethod
    def _recomputeTopPK(output_tensor: torch.Tensor, do_sample: bool, top_k: int, top_p: float) -> torch.Tensor:
        new_tensor = output_tensor
        if do_sample and (top_k is not None or top_p is not None):
            tensor, sorted_indices = torch.sort(new_tensor, descending = True)

            if top_k is not None:
                tensor[:, top_k:] = -float('Inf')

            if top_p is not None:
                probabilities = F.softmax(tensor, dim = -1)
                cumprobs = torch.cumsum(probabilities, dim = -1)
                indexes = (cumprobs >= top_p)
                indexes[:, 0] = 0
                tensor[indexes] = -float('Inf')

            new_tensor = torch.full_like(tensor, -float('Inf'))
            new_tensor.scatter_(dim = -1, index = sorted_indices, src = tensor)

        return new_tensor