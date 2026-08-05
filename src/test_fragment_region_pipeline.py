import sys
import torch
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader


# ----------------------------------------------------
# Imports
# ----------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent)
)


from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer
from models.fragment_transformer import FragmentTransformer
from models.region_pooling import  RegionPooling




# ----------------------------------------------------
# Config
# ----------------------------------------------------

DATA_DIR = "src/data/processed/patient_tensors"

BATCH_SIZE = 2

CHUNK_SIZE = 64



# ----------------------------------------------------
# Load data
# ----------------------------------------------------

print("="*70)
print("FRAGMENT -> REGION PIPELINE TEST")
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



print()

print(
    "Patients:",
    len(dataset)
)



# ----------------------------------------------------
# Models
# ----------------------------------------------------


fragment_model = FragmentTransformer(

    vocab_size=len(tokenizer.vocab),

    embed_dim=128,

    num_heads=4,

    num_layers=2

)




region_pool = RegionPooling()


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



fragment_model.to(device)

region_pool.to(device)



fragment_model.eval()

region_pool.eval()



print(
    "Device:",
    device
)



# ----------------------------------------------------
# Get one batch
# ----------------------------------------------------


batch = next(iter(loader))



input_ids = batch["input_ids"]

token_mask = batch["token_padding_mask"]

fragment_mask = batch["fragment_padding_mask"]



B,R,F,T = input_ids.shape



print()
print("="*70)
print("INPUT")
print("="*70)


print(
    "Input:",
    input_ids.shape
)



print(
    "Fragment mask:",
    fragment_mask.shape
)



# ----------------------------------------------------
# Flatten fragments
# ----------------------------------------------------


flat_ids = input_ids.view(
    -1,
    T
)


flat_token_mask = token_mask.view(
    -1,
    T
)


flat_fragment_mask = fragment_mask.view(
    -1
)



# remove padded fragments


real_ids = flat_ids[
    ~flat_fragment_mask
]


real_masks = flat_token_mask[
    ~flat_fragment_mask
]



print()

print(
    "Real fragments:",
    real_ids.shape[0]
)


# ----------------------------------------------------
# Create fragment -> patient + region mapping
# ----------------------------------------------------

fragment_patient_ids = []

fragment_region_ids = []


for patient_id in range(B):

    region_counter = 0


    for r in range(R):


        number_fragments = (
            (~fragment_mask[patient_id,r])
            .sum()
            .item()
        )


        if number_fragments > 0:


            fragment_patient_ids.extend(

                [patient_id] * number_fragments

            )


            fragment_region_ids.extend(

                [region_counter] * number_fragments

            )


            region_counter += 1



fragment_patient_ids = torch.tensor(
    fragment_patient_ids,
    dtype=torch.long
)


fragment_region_ids = torch.tensor(
    fragment_region_ids,
    dtype=torch.long
)



print()

print("Patient mapping:")
print(fragment_patient_ids.shape)


print("Region mapping:")
print(fragment_region_ids.shape)



# ----------------------------------------------------
# Fragment Transformer with chunks
# ----------------------------------------------------


fragment_embeddings = []



start=time.time()



with torch.no_grad():


    for i in tqdm(

        range(
            0,
            real_ids.shape[0],
            CHUNK_SIZE
        ),

        desc="Fragment Transformer"

    ):


        ids_chunk = real_ids[
            i:i+CHUNK_SIZE
        ].to(device)



        mask_chunk = real_masks[
            i:i+CHUNK_SIZE
        ].to(device)



        emb = fragment_model(

            ids_chunk,

            mask_chunk

        )



        fragment_embeddings.append(

            emb.cpu()

        )



fragment_embeddings = torch.cat(
    fragment_embeddings,
    dim=0
)



elapsed=time.time()-start



print()

print(
    "Fragment embeddings:"
)

print(
    fragment_embeddings.shape
)



print(
    "Time:",
    round(elapsed,2),
    "seconds"
)



# ----------------------------------------------------
# Region pooling
# ----------------------------------------------------


with torch.no_grad():


    region_embeddings, region_mapping = region_pool(

    fragment_embeddings.to(device),

    fragment_patient_ids.to(device),

    fragment_region_ids.to(device)

)



print()

print("="*70)

print(
    "REGION OUTPUT"
)


print("="*70)



print(
    "Region embeddings:"
)

print(
    region_embeddings.shape
)


print()

print(
    "Region mapping:"
)

print(
    region_mapping.shape
)


print(region_mapping[:10])