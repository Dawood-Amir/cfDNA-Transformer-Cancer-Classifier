import torch
import torch.nn as nn


class CFDNATokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim =128 ,padding_idx=0 ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,embedding_dim=embed_dim ,padding_idx=padding_idx
        )


    def forward(self,x):
        """
        x:
        token ids

        shape:
        (batch, sequence_length)

        output:

        (batch, sequence_length, embed_dim)
        """
        return self.embedding(x)