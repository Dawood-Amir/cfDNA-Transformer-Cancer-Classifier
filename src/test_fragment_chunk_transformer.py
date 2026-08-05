import sys
import os
import torch
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate       # your updated collate
from models.tokenizer import CFDNATokenizer
from models.fragment_transformer import FragmentTransformer

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "src/data/processed/patient_tensors"   # your new data folder
BATCH_SIZE = 2        
CHUNK_SIZE = 64       # number of fragments per chunk

# ------------------------------------------------------------
# LOAD DATA & MODEL
# ------------------------------------------------------------
print("=" * 60)
print("🧪 TESTING FRAGMENT TRANSFORMER (Chunked)")
print("=" * 60)

tokenizer = CFDNATokenizer()
dataset = PatientDataset(DATA_DIR)
collate_fn = PatientCollate(pad_token_id=tokenizer.vocab["<pad>"])

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

print(f" Loaded {len(dataset)} patients.")
print(f" Batch size: {BATCH_SIZE}")

model = FragmentTransformer(
    vocab_size=len(tokenizer.vocab),
    embed_dim=128,
    num_heads=4,
    num_layers=2
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f" Model on: {device}")

# ------------------------------------------------------------
# PROCESS ONE BATCH IN CHUNKS
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("📦 Processing first batch...")

for batch_idx, batch in enumerate(loader):
    input_ids = batch["input_ids"]                 # [B, R, F, T]
    token_mask = batch["token_padding_mask"]       # [B, R, F, T]
    frag_mask = batch["fragment_padding_mask"]     # [B, R, F]

    B, R, F, T = input_ids.shape
    print(f"\n--- Batch {batch_idx + 1} ---")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Fragment mask shape: {frag_mask.shape}")

    # Flatten all fragments
    flat_ids = input_ids.view(-1, T)               # [B*R*F, T]
    flat_token_mask = token_mask.view(-1, T)
    flat_frag_mask = frag_mask.view(-1)            # True = padding

    # Keep only real fragments
    real_flat_ids = flat_ids[~flat_frag_mask]
    real_flat_mask = flat_token_mask[~flat_frag_mask]

    print(f"   Real fragments: {real_flat_ids.shape[0]}")

    if real_flat_ids.shape[0] == 0:
        print("   ⚠️  No real fragments in this batch. Skipping.")
        continue

    # Move to device
    real_flat_ids = real_flat_ids.to(device)
    real_flat_mask = real_flat_mask.to(device)

    # ------------------------------------------------------------
    # PROCESS IN CHUNKS WITH PROGRESS BAR
    # ------------------------------------------------------------
    all_embeddings = []
    start_time = time.time()

    with torch.no_grad():
        # Create a progress bar over chunks
        pbar = tqdm(
            range(0, real_flat_ids.shape[0], CHUNK_SIZE),
            desc="Processing fragments",
            unit="chunk",
            leave=False
        )
        for i in pbar:
            chunk_ids = real_flat_ids[i:i+CHUNK_SIZE]
            chunk_mask = real_flat_mask[i:i+CHUNK_SIZE]
            emb = model(chunk_ids, chunk_mask)          # [chunk_size, embed_dim]
            all_embeddings.append(emb.cpu())

    elapsed = time.time() - start_time

    # Concatenate all chunk embeddings
    fragment_embeddings = torch.cat(all_embeddings, dim=0)

    print(f"\n    Processed {fragment_embeddings.shape[0]} fragments in {elapsed:.2f} seconds")
    print(f"   Output shape: {fragment_embeddings.shape}")
    print(f"   Mean: {fragment_embeddings.mean().item():.4f}")
    print(f"   Std:  {fragment_embeddings.std().item():.4f}")

    # (Optional) Restore to [B, R, F, embed_dim] for later
    # We need to scatter the embeddings back into the original positions.
    # This requires keeping track of which fragments were real.
    # For now, just show the flattened output.

    # Only one batch for test
    break

print("\n" + "=" * 60)
print(" Test complete.")
print("=" * 60)