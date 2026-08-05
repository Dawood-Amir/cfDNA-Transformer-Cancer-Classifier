import sys
import os
import torch
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate
from models.tokenizer import CFDNATokenizer
from models.fragment_transformer import FragmentTransformer
from models.region_pooling import RegionAttentionPooling   # your pooling module

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
DATA_DIR = "src/data/processed/patient_tensors"   # your new data folder
BATCH_SIZE = 2
CHUNK_SIZE = 64

# ------------------------------------------------------------
# LOAD DATA & MODELS
# ------------------------------------------------------------
print("=" * 60)
print("🧪 TESTING REGION ATTENTION POOLING")
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

print(f"✅ Loaded {len(dataset)} patients.")
print(f"✅ Batch size: {BATCH_SIZE}")

# Fragment Transformer
frag_model = FragmentTransformer(
    vocab_size=len(tokenizer.vocab),
    embed_dim=128,
    num_heads=4,
    num_layers=2
)
frag_model.eval()

# Region Attention Pooling
pool_model = RegionAttentionPooling(embed_dim=128)
pool_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frag_model.to(device)
pool_model.to(device)
print(f"✅ Models on: {device}")

# ------------------------------------------------------------
# PROCESS ONE BATCH
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
    # STEP 1: Fragment Transformer (chunked)
    # ------------------------------------------------------------
    all_embeddings = []
    start_time = time.time()

    with torch.no_grad():
        pbar = tqdm(
            range(0, real_flat_ids.shape[0], CHUNK_SIZE),
            desc="Processing fragments",
            unit="chunk",
            leave=False
        )
        for i in pbar:
            chunk_ids = real_flat_ids[i:i+CHUNK_SIZE]
            chunk_mask = real_flat_mask[i:i+CHUNK_SIZE]
            emb = frag_model(chunk_ids, chunk_mask)
            all_embeddings.append(emb.cpu())

    elapsed = time.time() - start_time
    fragment_embeddings = torch.cat(all_embeddings, dim=0)   # [N_real_frags, D]

    print(f"\n   ✅ Fragment embeddings: {fragment_embeddings.shape} in {elapsed:.2f} s")

    # ------------------------------------------------------------
    # STEP 2: Create fragment_to_region mapping
    # ------------------------------------------------------------
    # We need a tensor of shape [N_real_frags] with region IDs.
    # The region IDs should be unique per region, across all patients.
    # We'll create a flat tensor of region indices for all fragments,
    # then filter using the same mask.

    # Create a flat tensor of region IDs of shape [B*R*F]
    # We can generate indices: each region gets a unique ID within the batch.
    # For simplicity, we'll assign a unique ID per (patient, region) pair.
    # We'll create a tensor of shape [B, R] with region IDs, then expand to [B, R, F].

    # Start with region indices per (patient, region): 0 .. (B*R - 1)
    region_id_tensor = torch.arange(B * R).view(B, R)  # [B, R]
    # Expand to fragments: [B, R, F]
    region_id_expanded = region_id_tensor.unsqueeze(-1).expand(-1, -1, F)  # [B, R, F]
    # Flatten
    flat_region_ids = region_id_expanded.reshape(-1)  # [B*R*F]

    # Apply the same mask to keep only real fragments
    real_region_ids = flat_region_ids[~flat_frag_mask]  # [N_real_frags]

    # Now fragment_embeddings and real_region_ids are aligned.
    # Move to CPU if needed (they are already on CPU after chunking).
    real_region_ids = real_region_ids.to(fragment_embeddings.device)

    print(f"   Unique regions in batch: {torch.unique(real_region_ids).numel()}")

    # ------------------------------------------------------------
    # STEP 3: Region Attention Pooling
    # ------------------------------------------------------------
    with torch.no_grad():
        region_embeddings = pool_model(fragment_embeddings, real_region_ids)

    print(f"\n   ✅ Region embeddings: {region_embeddings.shape}")
    print(f"      Mean: {region_embeddings.mean().item():.4f}")
    print(f"      Std:  {region_embeddings.std().item():.4f}")

    # ------------------------------------------------------------
    # STEP 4: Verify shape – each region gets one embedding
    # ------------------------------------------------------------
    expected_num_regions = (~batch["region_padding_mask"]).sum().item()
    print(f"   Expected number of real regions (from mask): {expected_num_regions}")
    print(f"   Actual region embeddings count: {region_embeddings.shape[0]}")

    # Only one batch for test
    break

print("\n" + "=" * 60)
print("✅ Test complete.")
print("=" * 60)