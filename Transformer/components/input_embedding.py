import torch
import torch.nn as nn

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=d_model)
        self.register_buffer("scale",torch.sqrt(torch.tensor(d_model, dtype=torch.float)))

    def forward(self, x):
        return self.embedding(x) * self.scale