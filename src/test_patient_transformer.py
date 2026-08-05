import sys
import time
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer
from models.fragment_transformer import FragmentTransformer
from models.region_pooling import RegionAttentionPooling
from models.region_pooling import RegionPooling

from models.region_batch_builder import RegionBatchBuilder
from models.patient_transformer import PatientTransformer


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

DATA_DIR = "src/data/processed/patient_tensors"

BATCH_SIZE = 2
CHUNK_SIZE = 64

EMBED_DIM = 128


# ----------------------------------------------------------
# DATA
# ----------------------------------------------------------

print("="*70)
print("FULL PIPELINE TEST")
print("="*70)

tokenizer = CFDNATokenizer()

dataset = PatientDataset(DATA_DIR)

collate = PatientCollate(
    pad_token_id=tokenizer.vocab["<pad>"]
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)
print("Patients:", len(dataset))


# ----------------------------------------------------------
# MODELS
# ----------------------------------------------------------

fragment_model = FragmentTransformer(
    vocab_size=len(tokenizer.vocab),
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=2
).to(device)

# region_pool = RegionAttentionPooling(
#     embed_dim=EMBED_DIM
# ).to(device)

region_pool = RegionPooling().to(device)

builder = RegionBatchBuilder(
    embed_dim=EMBED_DIM
)

patient_model = PatientTransformer(
    embed_dim=EMBED_DIM,
    num_heads=4,
    num_layers=2
).to(device)

fragment_model.eval()
patient_model.eval()


# ----------------------------------------------------------
# LOAD ONE BATCH
# ----------------------------------------------------------

batch = next(iter(loader))

input_ids = batch["input_ids"]
token_mask = batch["token_padding_mask"]
fragment_mask = batch["fragment_padding_mask"]

B,R,F,T = input_ids.shape

print()
print("="*70)
print("INPUT")
print("="*70)

print(input_ids.shape)

# ----------------------------------------------------------
# FLATTEN FRAGMENTS
# ----------------------------------------------------------

flat_ids = input_ids.view(-1,T)

flat_token_mask = token_mask.view(-1,T)

flat_fragment_mask = fragment_mask.view(-1)

real_ids = flat_ids[~flat_fragment_mask]

real_masks = flat_token_mask[~flat_fragment_mask]

print()
print("Real fragments:", real_ids.shape[0])

# ----------------------------------------------------------
# BUILD MAPPINGS
# ----------------------------------------------------------

fragment_region_ids = []

fragment_patient_ids = []

current_region = 0

for patient in range(B):

    for region in range(R):

        n = (~fragment_mask[patient,region]).sum().item()

        if n == 0:
            continue

        #fragment_region_ids.extend([current_region]*n) #used when RegionAttentionPolling is used
        # region index INSIDE this patient
        fragment_region_ids.extend([region] * n)
        
        fragment_patient_ids.extend([patient]*n)

        current_region += 1


fragment_region_ids = torch.tensor(fragment_region_ids)
fragment_patient_ids = torch.tensor(fragment_patient_ids)

print("Regions:", current_region)

# ----------------------------------------------------------
# FRAGMENT TRANSFORMER
# ----------------------------------------------------------

print()
print("="*70)
print("Fragment Transformer")
print("="*70)

fragment_embeddings = []

start = time.time()

with torch.no_grad():

    pbar = tqdm(
        range(0,real_ids.shape[0],CHUNK_SIZE),
        desc="Fragments"
    )

    for i in pbar:

        emb = fragment_model(

            real_ids[i:i+CHUNK_SIZE].to(device),

            real_masks[i:i+CHUNK_SIZE].to(device)

        )

        fragment_embeddings.append(
            emb.cpu()
        )

fragment_embeddings = torch.cat(fragment_embeddings)

print()

print(
    "Fragment embeddings:",
    fragment_embeddings.shape
)

print(
    "Time:",
    round(time.time()-start,2),
    "sec"
)

# ----------------------------------------------------------
# REGION POOLING
# ----------------------------------------------------------

print()
print("="*70)
print("Region Pooling")
print("="*70)

with torch.no_grad():
    #For RegionAttentionPooling
    # region_embeddings, region_mapping = region_pool(

    #     fragment_embeddings.to(device),

    #     fragment_region_ids.to(device),

    #     fragment_patient_ids.to(device)

    # )
     #For RegionPooling
    region_embeddings, region_mapping = region_pool(
    fragment_embeddings.to(device),
    fragment_patient_ids.to(device),
    fragment_region_ids.to(device)
)

print(region_embeddings.shape)
print(region_mapping.shape)

# ----------------------------------------------------------
# BUILD REGION BATCH
# ----------------------------------------------------------

print()
print("="*70)
print("Region Batch")
print("="*70)

patient_ids = region_mapping[:,0]

region_ids = region_mapping[:,1]

region_batch, region_mask = builder(

    region_embeddings.cpu(),

    patient_ids.cpu(),

    region_ids.cpu(),

    B

)

print(region_batch.shape)
print(region_mask.shape)

# ----------------------------------------------------------
# PATIENT TRANSFORMER
# ----------------------------------------------------------

print()
print("="*70)
print("Patient Transformer")
print("="*70)

with torch.no_grad():

    patient_embeddings = patient_model(

        region_batch.to(device),

        region_mask.to(device)

    )

print()

print("Patient embeddings:")
print(patient_embeddings.shape)

print()
print("="*70)
print("SUCCESS")
print("="*70)