import torch

from tokenizer import CFDNATokenizer
from token_embedding import CFDNATokenEmbedding


tokenizer = CFDNATokenizer()


# Example regions
tokens=[
    "<cfdna>",
    "G",
    "G",
    "A",
    "<m>",
    "C",
    "<um>",
    "</cfdna>"
]


ids = tokenizer.encode(tokens)


print("Token IDs:")
print(ids)


# add batch dimension 1

ids = ids.unsqueeze(0)


#it expects x of shape (batch, seq_len). thats why we added batch dim for test

model = CFDNATokenEmbedding(
    vocab_size=len(tokenizer.vocab),
    embed_dim=128
) #Output shape: (batch, seq_len, embed_dim).


output=model(ids)


print("\nEmbedding output shape:")
print(output.shape)