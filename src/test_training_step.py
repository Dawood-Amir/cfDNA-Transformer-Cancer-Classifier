import sys
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


sys.path.insert(
    0,
    str(Path(__file__).parent)
)


from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer

from models.cfdna_transformer import CFDNATransformer



# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "src/data/processed/patient_tensors"

BATCH_SIZE = 2

CHUNK_SIZE = 64

EMBED_DIM = 128



# ============================================================
# START
# ============================================================

print("="*70)
print("TRAINING STEP TEST")
print("="*70)



tokenizer = CFDNATokenizer()



dataset = PatientDataset(
    DATA_DIR
)



collate_fn = PatientCollate(
    pad_token_id=tokenizer.vocab["<pad>"]
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)


device=torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


print()

print("Patients:",len(dataset))
print("Device:", device)


# ============================================================
# MODEL
# ============================================================


model = CFDNATransformer(
    vocab_size=len(tokenizer.vocab),
    embed_dim=EMBED_DIM,
    num_heads=4,
    fragment_layers=2,
    patient_layers=2,
    num_classes=4
).to(device)

model.train()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# ============================================================
# ONE TRAINING STEP
# ============================================================


batch = next(iter(loader))

input_ids = batch["input_ids"]

token_mask = batch["token_padding_mask"]

fragment_mask = batch["fragment_padding_mask"]

labels = batch["labels"]



B,R,F,T = input_ids.shape



print("\nINPUT")
print("-"*50)

print("Input:", input_ids.shape)

print("Labels:",labels)


# ------------------------------------------------------------
# Flatten fragments
# ------------------------------------------------------------

flat_ids = input_ids.view(
    -1,
    T
)

flat_masks = token_mask.view(
    -1,
    T
)

flat_fragment_mask = fragment_mask.view(
    -1
)

real_ids = flat_ids[
    ~flat_fragment_mask
]

real_masks = flat_masks[
    ~flat_fragment_mask
]


print()

print("Real fragments:", real_ids.shape[0])


# ------------------------------------------------------------
# Create mappings
# ------------------------------------------------------------

fragment_patient_ids=[]

fragment_region_ids=[]

region_counter=0


for b in range(B):

    for r in range(R):

        n_fragments = ((~fragment_mask[b,r]).sum().item())

        if n_fragments>0:
            fragment_patient_ids.extend(
                [b]*n_fragments
            )

            fragment_region_ids.extend(
                [r]*n_fragments
            )



fragment_patient_ids=torch.tensor(
    fragment_patient_ids
)


fragment_region_ids=torch.tensor(
    fragment_region_ids
)



# ------------------------------------------------------------
# Region padding mask
# ------------------------------------------------------------


region_padding_mask=torch.ones(

    B,

    R,

    dtype=torch.bool

)



for b in range(B):

    for r in range(R):

        if (~fragment_mask[b,r]).sum()>0:

            region_padding_mask[b,r]=False



# ------------------------------------------------------------
# Move
# ------------------------------------------------------------


real_ids=real_ids.to(device)

real_masks=real_masks.to(device)

fragment_patient_ids=fragment_patient_ids.to(device)

fragment_region_ids=fragment_region_ids.to(device)

region_padding_mask=region_padding_mask.to(device)

labels=labels.to(device)



# ============================================================
# Fragment encoder
# ============================================================


fragment_embeddings=[]


print("\nFragment Encoder")

with torch.no_grad():

    for i in tqdm(range(0,real_ids.size(0),CHUNK_SIZE)):

        emb=model.fragment_encoder(

            real_ids[i:i+CHUNK_SIZE],

            real_masks[i:i+CHUNK_SIZE]

        )

        fragment_embeddings.append(
            emb
        )



fragment_embeddings=torch.cat(
    fragment_embeddings
)



print(
    "Fragment embeddings:",
    fragment_embeddings.shape
)



# ============================================================
# Full model remaining part
# ============================================================


region_embeddings, mapping = model.region_pool(

    fragment_embeddings,

    fragment_patient_ids,

    fragment_region_ids

)



print(
    "Region embeddings:",
    region_embeddings.shape
)



# Build region batch


region_batch=torch.zeros(

    B,

    R,

    EMBED_DIM,

    device=device

)



for i in range(
    region_embeddings.size(0)
):

    p=mapping[i,0]

    r=mapping[i,1]


    region_batch[p,r]=region_embeddings[i]



patient_embeddings=model.patient_encoder(

    region_batch,

    region_padding_mask

)



logits=model.classifier(
    patient_embeddings
)



print()

print("LOGITS")
print("-"*50)

print(
    logits.shape
)

print(
    logits
)



loss=criterion(
    logits,
    labels
)



print()

print(
    "Loss:",
    loss.item()
)



# ============================================================
# BACKPROP
# ============================================================


optimizer.zero_grad()

loss.backward()

optimizer.step()



print()

print("="*70)
print("SUCCESS ✅")
print("Forward + Loss + Backprop works")
print("="*70)