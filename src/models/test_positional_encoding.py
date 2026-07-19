import torch

from tokenizer import CFDNATokenizer
from token_embedding import CFDNATokenEmbedding
from positional_encoding import PositionalEncoding


tokenizer = CFDNATokenizer()

tokens = [
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

ids = ids.unsqueeze(0)

embedding = CFDNATokenEmbedding(
    vocab_size=len(tokenizer.vocab),
    embed_dim=128
)

position = PositionalEncoding(
    embed_dim=128
)


embedded = embedding(ids)

output = position(embedded)


print("Input shape:")
print(embedded.shape)

print()

print("Output shape:")
print(output.shape)

print()

print("First token before PE:")
print(embedded[0, 0, :8])

print()

print("First token after PE:")
print(output[0, 0, :8])