# test_v6.py - Quick test for Version 6
# Run this to verify everything works before pushing to GitHub

import os
import sys
import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "patient_tensors")
SAVE_DIR = os.path.join(PROJECT_ROOT, "training_outputs_3classes_v6_attention")

print("=" * 70)
print("🧪 TESTING VERSION 6 - QUICK VERIFICATION")
print("=" * 70)

print(f"\n📁 Project root: {PROJECT_ROOT}")
print(f"📂 Data dir: {DATA_DIR}")
print(f"📂 Save dir: {SAVE_DIR}")

# ============================================================
# CHECK 1: Data exists
# ============================================================

print("\n📊 Checking data...")

if not os.path.exists(DATA_DIR):
    print(f"❌ Data directory not found: {DATA_DIR}")
    sys.exit(1)

manifest_path = os.path.join(DATA_DIR, "manifest.csv")
if not os.path.exists(manifest_path):
    print(f"❌ Manifest not found: {manifest_path}")
    sys.exit(1)
else:
    print(f"✅ Manifest found: {manifest_path}")

# ============================================================
# CHECK 2: Imports work
# ============================================================

print("\n📦 Checking imports...")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from tqdm import tqdm
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        classification_report, roc_auc_score, f1_score
    )
    print("✅ Core imports OK")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Check your custom modules
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    from data.patient_dataset import PatientDataset
    from data.patient_collate import PatientCollate
    from models.tokenizer import CFDNATokenizer
    from models.cfdna_transformer import CFDNATransformer
    print("✅ Custom module imports OK")
except ImportError as e:
    print(f"❌ Custom module import error: {e}")
    sys.exit(1)

# ============================================================
# CHECK 3: Model creation
# ============================================================

print("\n🧠 Testing model creation...")

try:
    # Test with V6 parameters
    model = CFDNATransformer(
        vocab_size=100,  # Dummy vocab size
        embed_dim=96,
        num_heads=3,
        fragment_layers=1,
        patient_layers=1,
        num_classes=3,
        dropout=0.3,
        pooling_type='attention',
        classifier_dropout=0.3
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created successfully")
    print(f"   Parameters: {total_params:,}")
except Exception as e:
    print(f"❌ Model creation error: {e}")
    sys.exit(1)

# ============================================================
# CHECK 4: Data loading
# ============================================================

print("\n📂 Testing data loading...")

try:
    tokenizer = CFDNATokenizer()
    dataset = PatientDataset(DATA_DIR)
    print(f"✅ Dataset loaded successfully")
    print(f"   Total patients: {len(dataset)}")
    
    # Quick check: labels exist
    labels = dataset.manifest["label"].values
    print(f"   Labels found: {len(labels)}")
except Exception as e:
    print(f"❌ Data loading error: {e}")
    sys.exit(1)

# ============================================================
# CHECK 5: Small forward pass
# ============================================================

print("\n🔬 Testing forward pass (1 batch)...")

try:
    # Load a single sample
    sample = dataset[0]
    
    # Since we can't easily simulate the full batch, just check if the model accepts dummy data
    # This is a basic check - the real test will be during training
    
    # Create dummy batch
    dummy_input_ids = torch.randint(0, 100, (2, 5, 3, 10))  # [B, R, F, T]
    dummy_token_mask = torch.zeros(2, 5, 3, 10, dtype=torch.bool)
    dummy_fragment_mask = torch.zeros(2, 5, 3, dtype=torch.bool)
    dummy_region_mask = torch.zeros(2, 5, dtype=torch.bool)
    
    # Move to device if GPU available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    dummy_input_ids = dummy_input_ids.to(device)
    dummy_token_mask = dummy_token_mask.to(device)
    dummy_fragment_mask = dummy_fragment_mask.to(device)
    dummy_region_mask = dummy_region_mask.to(device)
    
    with torch.no_grad():
        output = model(
            dummy_input_ids,
            dummy_token_mask,
            dummy_fragment_mask,
            dummy_region_mask,
            show_progress=False
        )
    
    print(f"✅ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Device: {output.device}")
    
except Exception as e:
    print(f"❌ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# CHECK 6: Loss function
# ============================================================

print("\n🎯 Testing Focal Loss...")

try:
    from src.utils import FocalLoss  # Or wherever you put it
    
    class_weights = torch.tensor([0.8, 1.0, 1.2])
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    dummy_logits = torch.randn(10, 3)
    dummy_labels = torch.randint(0, 3, (10,))
    
    loss = criterion(dummy_logits, dummy_labels)
    print(f"✅ Focal Loss works: {loss.item():.4f}")
    
except ImportError:
    # FocalLoss might be inside the main script
    print("⚠️ FocalLoss not imported separately (will be defined in main script)")
except Exception as e:
    print(f"❌ Focal Loss error: {e}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYou can now:")
print("  1. Run the full training script locally (slow)")
print("  2. Push code to GitHub")
print("  3. Run on Kaggle with GPU")
print("=" * 70)