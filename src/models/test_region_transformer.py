import torch

from tokenizer import CFDNATokenizer
from region_transformer import RegionTransformerEncoder


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


ids = tokenizer.encode(tokens=tokens)
# add batch dimension

ids = ids.unsqueeze(0)

print("Input IDs:")
print(ids)

print()

model = RegionTransformerEncoder(
    vocab_size=len(tokenizer.vocab),
    embed_dim=128,
    num_heads=4,
    num_layers=4
)

output = model(ids)


print("Region embedding shape:")
print(output.shape)