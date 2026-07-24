import torch
from torch.utils.data import DataLoader

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate
from models.tokenizer import CFDNATokenizer
from models.region_transformer import RegionTransformerEncoder


##########################################
# Dataset
##########################################

dataset = PatientDataset(
    "src/data/processed/patient_tensors"
)

tokenizer = CFDNATokenizer()

loader = DataLoader(

    dataset,

    batch_size=2,

    shuffle=False,

    collate_fn=PatientCollate(
        pad_token_id=tokenizer.vocab["<pad>"]
    )

)

##########################################
# First batch
##########################################

batch = next(iter(loader))

input_ids = batch["input_ids"]

token_mask = batch["token_padding_mask"]

region_mask = batch["region_padding_mask"]

labels = batch["labels"]

print()

print("Input IDs")

print(input_ids.shape)

print()

print("Token Mask")

print(token_mask.shape)

print()

print("Region Mask")

print(region_mask.shape)

print()

print("Labels")

print(labels)

##########################################
# Flatten Regions
##########################################

B,R,T = input_ids.shape

flat_input = input_ids.view(

    B*R,

    T

)

flat_mask = token_mask.view(

    B*R,

    T

)

print()

print("Flattened")

print(flat_input.shape)

##########################################
# Region Transformer
##########################################

model = RegionTransformerEncoder(

    vocab_size=len(tokenizer.vocab),

    embed_dim=128,

    num_heads=4,

    num_layers=4

)

# region_embeddings = model(

#     flat_input,

#     padding_mask=flat_mask

# )

# Only test the first 8 regions
flat_input = flat_input[:8]
flat_mask = flat_mask[:8]

region_embeddings = model(
    flat_input,
    padding_mask=flat_mask
)

print()

print("Region Embeddings")

print(region_embeddings.shape)

##########################################
# Restore Patient Layout
##########################################

# region_embeddings = region_embeddings.view(

#     B,

#     R,

#     -1

# )

# print()

# print("Patient Representation")

# print(region_embeddings.shape)