import sys
import torch
import time

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader


# ----------------------------------------------------
# Import path
# ----------------------------------------------------

sys.path.insert(
    0,
    str(Path(__file__).parent)
)


# ----------------------------------------------------
# Imports
# ----------------------------------------------------

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer

from models.fragment_transformer import FragmentTransformer
from models.region_pooling import RegionPooling
from models.patient_transformer import PatientTransformer
from models.classification_head import ClassificationHead



# ----------------------------------------------------
# Config
# ----------------------------------------------------

DATA_DIR = "src/data/processed/patient_tensors"

BATCH_SIZE = 2

CHUNK_SIZE = 64

EMBED_DIM = 128



# ----------------------------------------------------
# Header
# ----------------------------------------------------

print("="*70)
print("FULL cfDNA TRANSFORMER PIPELINE TEST")
print("="*70)



# ----------------------------------------------------
# Dataset
# ----------------------------------------------------

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



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



print()

print(
    "Patients:",
    len(dataset)
)

print(
    "Device:",
    device
)



# ----------------------------------------------------
# Models
# ----------------------------------------------------


fragment_model = FragmentTransformer(

    vocab_size=len(tokenizer.vocab),

    embed_dim=EMBED_DIM,

    num_heads=4,

    num_layers=2

).to(device)



region_pool = RegionPooling()



patient_model = PatientTransformer(

    embed_dim=EMBED_DIM,

    num_heads=4,

    num_layers=2

).to(device)



classifier = ClassificationHead(

    embed_dim=EMBED_DIM,

    num_classes=4

).to(device)



fragment_model.eval()
patient_model.eval()
classifier.eval()



# ----------------------------------------------------
# Get batch
# ----------------------------------------------------

batch = next(iter(loader))


input_ids = batch["input_ids"]

token_mask = batch["token_padding_mask"]

fragment_mask = batch["fragment_padding_mask"]

region_mask = batch["region_padding_mask"]

labels = batch["labels"]



B,R,F,T = input_ids.shape



print("\n")
print("="*70)
print("INPUT")
print("="*70)


print(
    "Input IDs:",
    input_ids.shape
)

print(
    "Labels:",
    labels
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
# Create fragment mappings
# ----------------------------------------------------

fragment_patient_ids = []

fragment_region_ids = []


region_counter = 0



for b in range(B):

    for r in range(R):


        number_fragments = (

            (~fragment_mask[b,r])

            .sum()

            .item()

        )


        if number_fragments > 0:


            fragment_patient_ids.extend(

                [b] * number_fragments

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

print(
    "Regions:",
    region_counter
)



# ----------------------------------------------------
# Fragment Transformer
# ----------------------------------------------------


print("\n")
print("="*70)
print("FRAGMENT TRANSFORMER")
print("="*70)



fragment_outputs = []

start = time.time()



with torch.no_grad():


    for i in tqdm(

        range(
            0,
            real_ids.shape[0],
            CHUNK_SIZE
        ),

        desc="Fragments"

    ):


        ids = real_ids[
            i:i+CHUNK_SIZE
        ].to(device)



        masks = real_masks[
            i:i+CHUNK_SIZE
        ].to(device)



        emb = fragment_model(

            ids,

            masks

        )


        fragment_outputs.append(
            emb.cpu()
        )



fragment_embeddings = torch.cat(
    fragment_outputs,
    dim=0
)



print()

print(
    "Fragment embeddings:",
    fragment_embeddings.shape
)


print(
    "Time:",
    round(time.time()-start,2),
    "seconds"
)



# ----------------------------------------------------
# Region Pooling
# ----------------------------------------------------


print("\n")
print("="*70)
print("REGION POOLING")
print("="*70)



with torch.no_grad():

    region_embeddings, region_mapping = region_pool(

        fragment_embeddings.to(device),

        fragment_patient_ids.to(device),

        fragment_region_ids.to(device)

    )



print()

print(
    "Region embeddings:",
    region_embeddings.shape
)


print(
    "Region mapping:",
    region_mapping.shape
)



# ----------------------------------------------------
# Build patient region batch
# ----------------------------------------------------


print("\n")
print("="*70)
print("PATIENT TRANSFORMER")
print("="*70)


max_regions = region_mask.shape[1]


region_batch = torch.zeros(

    B,

    max_regions,

    EMBED_DIM,

    device=device

)



# keep separate counter for every patient

patient_region_counter = torch.zeros(
    B,
    dtype=torch.long
)



for i in range(
    region_embeddings.size(0)
):

    patient = region_mapping[i,0]


    region_position = patient_region_counter[patient]


    region_batch[
        patient,
        region_position
    ] = region_embeddings[i]


    patient_region_counter[patient] += 1


with torch.no_grad():

    patient_embeddings = patient_model(

        region_batch,

        region_mask.to(device)

    )



print()

print(
    "Patient embeddings:",
    patient_embeddings.shape
)



# ----------------------------------------------------
# Classification
# ----------------------------------------------------


print("\n")
print("="*70)
print("CLASSIFICATION")
print("="*70)



with torch.no_grad():

    logits = classifier(
        patient_embeddings
    )



print()

print(
    "Logits:",
    logits.shape
)


print(
    logits
)



print("\n")
print("="*70)
print("SUCCESS ✅")
print("Complete cfDNA Transformer forward pass works")
print("="*70)