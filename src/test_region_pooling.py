import torch

from models.region_pooling import RegionAttentionPooling



####################################
# Fake fragment embeddings
####################################

# Suppose we have 10 fragments

fragment_embeddings = torch.randn(
    10,
    128
)



####################################
# Fragment -> Region mapping
####################################


# Region 0:
# fragments 0,1,2

# Region 1:
# fragments 3,4,5,6

# Region 2:
# fragments 7,8,9


fragment_to_region = torch.tensor(
    [
        0,0,0,
        1,1,1,1,
        2,2,2
    ]
)



print("Input fragment embeddings:")
print(fragment_embeddings.shape)



print(
    "Fragment mapping:"
)

print(
    fragment_to_region
)



####################################
# Model
####################################

model = RegionAttentionPooling(
    embed_dim=128
)



output = model(
    fragment_embeddings,
    fragment_to_region
)



print()
print(
    "Region embeddings:"
)

print(
    output.shape
)