# ============================================================
# cfDNA HIERARCHICAL TRANSFORMER - VERSION 4 (Final KAggle RUN)
# ============================================================
#
# VERSION 4 CHANGES
# -----------------
# 1. HEALTHY_SAMPLING_BOOST: 1.23 → 1.27
#
# 2. GBM sampling weight reduced by 5% (class index 1)
#    This slightly reduces GBM's dominance
#
# 3. LABEL_SMOOTHING: 0.05 → 0.07

# ============================================================

import os
import gc
import json
import time
import shutil
import random
from datetime import datetime

import numpy as np
import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler
)

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)

# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("🧬 cfDNA HIERARCHICAL TRANSFORMER - VERSION 4 (FINAL)")
print("🚀 BALANCED TRAINING PIPELINE")
print("=" * 70)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DEVICE_TYPE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\n Device: {DEVICE}")
print(f" CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():

    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# ============================================================
# PATHS
# ============================================================

DATA_DIR = (
    "/kaggle/input/datasets/"
    "dawoodaamar/cfdna-patient-tensors/"
    "patient_tensors_3classes"
)

SAVE_DIR = "/kaggle/working/training_outputs_3classes_v4"

CHECKPOINT_DIR = f"{SAVE_DIR}/checkpoints"
METRICS_DIR = f"{SAVE_DIR}/metrics"
LOGS_DIR = f"{SAVE_DIR}/logs"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print("\n Output directory:")
print(SAVE_DIR)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1
GRAD_ACCUMULATION = 4
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 7

NUM_CLASSES = 3

CLASS_NAMES = ["Healthy", "GBM", "LGG"]

# Full dataset class counts.
CLASS_COUNTS = [656, 450, 534]

# ------------------------------------------------------------
# MODEL ARCHITECTURE
# ------------------------------------------------------------

EMBED_DIM = 128
NUM_HEADS = 4
FRAGMENT_LAYERS = 2
PATIENT_LAYERS = 2

# ------------------------------------------------------------
# VERSION 4: UPDATED HYPERPARAMETERS
# ------------------------------------------------------------

# 1. Healthy boost: 1.23 → 1.27 (small, safe increase)
HEALTHY_SAMPLING_BOOST = 1.27

# 2. Label smoothing: 0.05 → 0.07 (slight increase)
LABEL_SMOOTHING = 0.07

# 3. GBM weight reduction: 5% (class index 1)
GBM_WEIGHT_REDUCTION = 0.95

# ============================================================
# MODEL SELECTION
# ============================================================

BEST_METRIC = "macro_f1"

# ============================================================
# PRINT CONFIG
# ============================================================

print("\n" + "=" * 70)
print("⚙️ VERSION 4 CONFIGURATION")
print("=" * 70)

print(f"DATA_DIR              : {DATA_DIR}")
print(f"SAVE_DIR              : {SAVE_DIR}")
print(f"BATCH_SIZE            : {BATCH_SIZE}")
print(f"GRAD_ACCUMULATION     : {GRAD_ACCUMULATION}")
print(f"EFFECTIVE BATCH       : {BATCH_SIZE * GRAD_ACCUMULATION}")
print(f"EPOCHS                : {EPOCHS}")
print(f"LEARNING RATE         : {LR}")
print(f"WEIGHT DECAY          : {WEIGHT_DECAY}")
print(f"PATIENCE              : {PATIENCE}")
print(f"NUM CLASSES           : {NUM_CLASSES}")
print(f"BEST METRIC           : {BEST_METRIC}")
print(f"EMBED DIM             : {EMBED_DIM}")
print(f"NUM HEADS             : {NUM_HEADS}")
print(f"FRAGMENT LAYERS       : {FRAGMENT_LAYERS}")
print(f"PATIENT LAYERS        : {PATIENT_LAYERS}")

print(f"\n🔧 VERSION 4 SPECIFIC:")
print(f"HEALTHY SAMPLE BOOST  : {HEALTHY_SAMPLING_BOOST} (was 1.23)")
print(f"LABEL SMOOTHING       : {LABEL_SMOOTHING} (was 0.05)")
print(f"GBM WEIGHT REDUCTION  : {GBM_WEIGHT_REDUCTION} (5% reduction)")

print("=" * 70)

# ============================================================
# MODEL IMPORTS
# ============================================================

import sys

PROJECT_SRC = "/kaggle/working/cfDNA-Transformer-Cancer-Classifier/src"

if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate

from models.tokenizer import CFDNATokenizer
from models.cfdna_transformer import CFDNATransformer

# ============================================================
# LOAD TOKENIZER + DATASET
# ============================================================

print("\n Loading dataset...")

tokenizer = CFDNATokenizer()
dataset = PatientDataset(DATA_DIR)

print(f"Total patients: {len(dataset)}")

# ============================================================
# FULL DATASET DISTRIBUTION
# ============================================================

labels = dataset.manifest["label"].values.astype(int)

dataset_counts = np.bincount(labels, minlength=NUM_CLASSES)

print("\n Full dataset distribution:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} - {name:<10}: {dataset_counts[i]}")

# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

rng = np.random.default_rng(SEED)

train_indices = []
val_indices = []

TRAIN_RATIO = 0.80

for class_id in range(NUM_CLASSES):

    class_indices = np.where(labels == class_id)[0]
    rng.shuffle(class_indices)

    class_train_size = int(len(class_indices) * TRAIN_RATIO)

    train_indices.extend(class_indices[:class_train_size].tolist())
    val_indices.extend(class_indices[class_train_size:].tolist())

rng.shuffle(train_indices)
rng.shuffle(val_indices)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

print("\n Stratified dataset split:")
print(f"  Train      : {len(train_dataset)}")
print(f"  Validation : {len(val_dataset)}")

# ============================================================
# TRAINING LABELS
# ============================================================

train_labels = labels[np.asarray(train_indices)]
val_labels = labels[np.asarray(val_indices)]

train_class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
val_class_counts = np.bincount(val_labels, minlength=NUM_CLASSES)

print("\n Training distribution:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} - {name:<10}: {train_class_counts[i]}")

print("\n Validation distribution:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} - {name:<10}: {val_class_counts[i]}")

# ============================================================
# COLLATE FUNCTION
# ============================================================

collate_fn = PatientCollate(tokenizer.vocab["<pad>"])

# ============================================================
# CLASS-IMBALANCE CORRECTION
# ============================================================

class_sampling_weights = 1.0 / torch.tensor(train_class_counts, dtype=torch.float32)

# Boost Healthy
class_sampling_weights[0] *= HEALTHY_SAMPLING_BOOST

# Reduce GBM (class index 1)
class_sampling_weights[1] *= GBM_WEIGHT_REDUCTION

print("\n Sampling weights:")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} - {name:<10}: {class_sampling_weights[i].item():.6f}")

# Create one weight per training sample.
sample_weights = torch.tensor(
    [class_sampling_weights[label].item() for label in train_labels],
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print("\n WeightedRandomSampler enabled")
print(" No class weights will be used inside CrossEntropyLoss.")

# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=torch.cuda.is_available()
)

print("\n📦 DataLoaders created")
print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")

# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 70)
print(" CREATING VERSION 4 MODEL")
print("=" * 70)

model = CFDNATransformer(
    vocab_size=len(tokenizer.vocab),
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    fragment_layers=FRAGMENT_LAYERS,
    patient_layers=PATIENT_LAYERS,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

# ============================================================
# COUNT PARAMETERS
# ============================================================

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters    : {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)

print("\n Loss:")
print(f"  CrossEntropyLoss")
print(f"  Label smoothing: {LABEL_SMOOTHING}")
print("  Class weights: NONE")
print("  Imbalance correction: WeightedRandomSampler only")

# ============================================================
# AMP
# ============================================================

if DEVICE_TYPE == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    print("\n Mixed Precision: ENABLED")
else:
    scaler = None
    print("\n Mixed Precision: DISABLED")

# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(targets, predictions, probabilities):

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    metrics = {}

    # --- Accuracy ---
    metrics["accuracy"] = float(accuracy_score(targets, predictions))

    # --- Balanced Accuracy ---
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(targets, predictions))

    # --- Macro F1 ---
    metrics["macro_f1"] = float(f1_score(targets, predictions, average="macro", zero_division=0))

    # --- Weighted F1 ---
    metrics["weighted_f1"] = float(f1_score(targets, predictions, average="weighted", zero_division=0))

    # --- ROC-AUC (FIXED) ---
    try:
        probs = probabilities / probabilities.sum(axis=1, keepdims=True)

        metrics["roc_auc_macro"] = float(
            roc_auc_score(targets, probs, multi_class="ovr", average="macro")
        )

        metrics["roc_auc_weighted"] = float(
            roc_auc_score(targets, probs, multi_class="ovr", average="weighted")
        )

    except Exception:
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_weighted"] = 0.0

    # --- Confusion Matrix ---
    metrics["confusion_matrix"] = (
        confusion_matrix(targets, predictions, labels=list(range(NUM_CLASSES))).tolist()
    )

    # --- Per-class metrics ---
    metrics["per_class"] = {}

    for i, name in enumerate(CLASS_NAMES):

        tp = np.sum((predictions == i) & (targets == i))
        fp = np.sum((predictions == i) & (targets != i))
        fn = np.sum((predictions != i) & (targets == i))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics["per_class"][name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(targets == i))
        }

    # --- Prediction distribution ---
    metrics["prediction_distribution"] = (
        np.bincount(predictions, minlength=NUM_CLASSES).tolist()
    )

    # --- Actual distribution ---
    metrics["actual_distribution"] = (
        np.bincount(targets, minlength=NUM_CLASSES).tolist()
    )

    return metrics

# ============================================================
# TRAINING HISTORY
# ============================================================

history = {

    "train_loss": [],
    "val_loss": [],
    "accuracy": [],
    "balanced_accuracy": [],
    "macro_f1": [],
    "weighted_f1": [],
    "roc_auc_macro": [],
    "roc_auc_weighted": [],
    "learning_rate": [],
    "epoch_time": [],
    "prediction_distribution": [],
    "actual_distribution": [],

    "best_epoch": None,
    "best_metric": None,
    "best_val_loss": None,
    "best_accuracy": None,
    "best_macro_f1": None,

    "start_time": datetime.now().isoformat(),
    "end_time": None

}

# ============================================================
# BEST MODEL TRACKING
# ============================================================

best_metric_value = -float("inf")
best_epoch = 0
patience_counter = 0

# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print(" STARTING VERSION 4 TRAINING")
print("=" * 70)

training_start = time.time()

for epoch in range(EPOCHS):

    epoch_start = time.time()

    print("\n" + "=" * 70)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 70)

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()
    train_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    progress = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")

    for step, batch in progress:

        input_ids = batch["input_ids"].to(DEVICE, non_blocking=True)
        token_mask = batch["token_padding_mask"].to(DEVICE, non_blocking=True)
        fragment_mask = batch["fragment_padding_mask"].to(DEVICE, non_blocking=True)
        region_mask = batch["region_padding_mask"].to(DEVICE, non_blocking=True)
        labels = batch["labels"].to(DEVICE, non_blocking=True)

        if DEVICE_TYPE == "cuda":

            with torch.amp.autocast("cuda"):
                logits = model(input_ids, token_mask, fragment_mask, region_mask)
                loss = criterion(logits, labels)

        else:

            logits = model(input_ids, token_mask, fragment_mask, region_mask)
            loss = criterion(logits, labels)

        loss_for_backward = loss / GRAD_ACCUMULATION

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((step + 1) % GRAD_ACCUMULATION == 0) or ((step + 1) == len(train_loader))

        if should_step:

            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

        del input_ids, token_mask, fragment_mask, region_mask, labels, logits, loss

    train_loss /= len(train_loader)

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        validation_progress = tqdm(val_loader, desc="Validation")

        for batch in validation_progress:

            input_ids = batch["input_ids"].to(DEVICE)
            token_mask = batch["token_padding_mask"].to(DEVICE)
            fragment_mask = batch["fragment_padding_mask"].to(DEVICE)
            region_mask = batch["region_padding_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            if DEVICE_TYPE == "cuda":

                with torch.amp.autocast("cuda"):
                    logits = model(input_ids, token_mask, fragment_mask, region_mask)
                    loss = criterion(logits, labels)

            else:

                logits = model(input_ids, token_mask, fragment_mask, region_mask)
                loss = criterion(logits, labels)

            val_loss += loss.item()

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

            del input_ids, token_mask, fragment_mask, region_mask, labels, logits, loss, probabilities, predictions

    val_loss /= len(val_loader)

    all_targets = np.asarray(all_targets)
    all_predictions = np.asarray(all_predictions)
    all_probabilities = np.asarray(all_probabilities)

    metrics = calculate_metrics(all_targets, all_predictions, all_probabilities)

    # ========================================================
    # EPOCH INFORMATION
    # ========================================================

    epoch_time = time.time() - epoch_start
    current_lr = optimizer.param_groups[0]["lr"]

    print("\n" + "-" * 70)
    print(f" Train Loss       : {train_loss:.4f}")
    print(f" Val Loss         : {val_loss:.4f}")
    print(f" Accuracy         : {metrics['accuracy']:.4f}")
    print(f" Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f" Macro F1         : {metrics['macro_f1']:.4f}")
    print(f" Weighted F1      : {metrics['weighted_f1']:.4f}")

    print(f" Macro AUC        : {metrics['roc_auc_macro']:.4f}")
    print(f" Weighted AUC     : {metrics['roc_auc_weighted']:.4f}")

    print(f" Epoch time       : {epoch_time:.1f}s")
    print(f" Learning rate    : {current_lr:.8f}")

    print("\nPer-class recall:")
    for name in CLASS_NAMES:
        print(f"  {name:<10}: {metrics['per_class'][name]['recall']:.4f}")

    print("\nPer-class F1:")
    for name in CLASS_NAMES:
        print(f"  {name:<10}: {metrics['per_class'][name]['f1']:.4f}")

    print("\nPrediction distribution:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10}: predicted={metrics['prediction_distribution'][i]:<4}  actual={val_class_counts[i]:<4}")

    print("-" * 70)

    # ========================================================
    # HISTORY
    # ========================================================

    history["train_loss"].append(float(train_loss))
    history["val_loss"].append(float(val_loss))
    history["accuracy"].append(metrics["accuracy"])
    history["balanced_accuracy"].append(metrics["balanced_accuracy"])
    history["macro_f1"].append(metrics["macro_f1"])
    history["weighted_f1"].append(metrics["weighted_f1"])
    history["roc_auc_macro"].append(metrics["roc_auc_macro"])
    history["roc_auc_weighted"].append(metrics["roc_auc_weighted"])
    history["learning_rate"].append(current_lr)
    history["epoch_time"].append(epoch_time)
    history["prediction_distribution"].append(metrics["prediction_distribution"])
    history["actual_distribution"].append(metrics["actual_distribution"])

    # ========================================================
    # CURRENT CHECKPOINT
    # ========================================================

    checkpoint = {

        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "metrics": metrics,
        "best_metric": best_metric_value,
        "patience_counter": patience_counter,

        "config": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "grad_accumulation": GRAD_ACCUMULATION,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "class_counts": CLASS_COUNTS,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "fragment_layers": FRAGMENT_LAYERS,
            "patient_layers": PATIENT_LAYERS,
            "label_smoothing": LABEL_SMOOTHING,
            "healthy_sampling_boost": HEALTHY_SAMPLING_BOOST,
            "gbm_weight_reduction": GBM_WEIGHT_REDUCTION,
            "imbalance_method": "WeightedRandomSampler only",
            "class_weights": None
        }

    }

    # ========================================================
    # SAVE EVERY EPOCH
    # ========================================================

    epoch_checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch + 1:02d}.pth"
    torch.save(checkpoint, epoch_checkpoint_path)
    print(f"💾 Checkpoint saved: {epoch_checkpoint_path}")

    # ========================================================
    # BEST MODEL
    # ========================================================

    current_metric = metrics["macro_f1"]

    if current_metric > best_metric_value:

        best_metric_value = current_metric
        best_epoch = epoch + 1
        patience_counter = 0

        history["best_epoch"] = best_epoch
        history["best_metric"] = best_metric_value
        history["best_val_loss"] = val_loss
        history["best_accuracy"] = metrics["accuracy"]
        history["best_macro_f1"] = metrics["macro_f1"]

        checkpoint["best_metric"] = best_metric_value
        checkpoint["is_best"] = True

        best_model_path = f"{CHECKPOINT_DIR}/best_model.pth"
        torch.save(checkpoint, best_model_path)

        with open(f"{METRICS_DIR}/best_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("\n NEW BEST MODEL!")
        print(f"   Macro F1: {best_metric_value:.4f}")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")

    else:

        patience_counter += 1
        print(f"\n No improvement: {patience_counter}/{PATIENCE}")

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(f"{LOGS_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ========================================================
    # SAVE CONFIG
    # ========================================================

    config_to_save = {

        "version": "v4",
        "seed": SEED,
        "data_dir": DATA_DIR,
        "batch_size": BATCH_SIZE,
        "grad_accumulation": GRAD_ACCUMULATION,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUMULATION,
        "epochs": EPOCHS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "patience": PATIENCE,
        "best_metric": BEST_METRIC,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_counts": CLASS_COUNTS,
        "train_class_counts": train_class_counts.tolist(),
        "validation_class_counts": val_class_counts.tolist(),

        "model": {
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "fragment_layers": FRAGMENT_LAYERS,
            "patient_layers": PATIENT_LAYERS
        },

        "loss": {
            "label_smoothing": LABEL_SMOOTHING,
            "class_weights": None
        },

        "imbalance": {
            "method": "WeightedRandomSampler only",
            "healthy_sampling_boost": HEALTHY_SAMPLING_BOOST,
            "gbm_weight_reduction": GBM_WEIGHT_REDUCTION
        }

    }

    with open(f"{LOGS_DIR}/config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step()

    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if patience_counter >= PATIENCE:

        print("\n🛑 EARLY STOPPING")
        break

# ============================================================
# TRAINING FINISHED
# ============================================================

history["end_time"] = datetime.now().isoformat()
total_training_time = time.time() - training_start
history["total_training_time_seconds"] = total_training_time

with open(f"{LOGS_DIR}/training_history.json", "w") as f:
    json.dump(history, f, indent=2)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n" + "=" * 70)
print(" TRAINING COMPLETE")
print("=" * 70)

print(f" Best epoch: {best_epoch}")
print(f" Best Macro F1: {best_metric_value:.4f}")

best_model_path = f"{CHECKPOINT_DIR}/best_model.pth"
best_checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)

model.load_state_dict(best_checkpoint["model_state_dict"])
model.eval()

# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n🔬 Running final evaluation...")

all_targets = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch in tqdm(val_loader, desc="Final Evaluation"):

        input_ids = batch["input_ids"].to(DEVICE)
        token_mask = batch["token_padding_mask"].to(DEVICE)
        fragment_mask = batch["fragment_padding_mask"].to(DEVICE)
        region_mask = batch["region_padding_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids, token_mask, fragment_mask, region_mask)
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_targets.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# ============================================================
# FINAL METRICS
# ============================================================

final_targets = np.asarray(all_targets)
final_predictions = np.asarray(all_predictions)
final_probabilities = np.asarray(all_probabilities)

final_metrics = calculate_metrics(final_targets, final_predictions, final_probabilities)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print(" FINAL VERSION 4 RESULTS")
print("=" * 70)

print(f"\n Accuracy: {final_metrics['accuracy']:.4f}")
print(f" Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
print(f" Macro F1: {final_metrics['macro_f1']:.4f}")
print(f" Weighted F1: {final_metrics['weighted_f1']:.4f}")
print(f" Macro ROC-AUC: {final_metrics['roc_auc_macro']:.4f}")
print(f" Weighted ROC-AUC: {final_metrics['roc_auc_weighted']:.4f}")

# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n" + "-" * 70)
print(" PER-CLASS RESULTS")
print("-" * 70)

for name in CLASS_NAMES:

    scores = final_metrics["per_class"][name]

    print(f"\n{name}")
    print(f"  Precision: {scores['precision']:.4f}")
    print(f"  Recall:    {scores['recall']:.4f}")
    print(f"  F1:        {scores['f1']:.4f}")
    print(f"  Support:   {scores['support']}")

# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "-" * 70)
print("CONFUSION MATRIX")
print("-" * 70)

cm = np.asarray(final_metrics["confusion_matrix"])

print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))

# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print("\nPrediction distribution:")

for i, name in enumerate(CLASS_NAMES):

    print(f"  {name:<10}: predicted={final_metrics['prediction_distribution'][i]:<4}  actual={val_class_counts[i]:<4}")

# ============================================================
# SAVE FINAL METRICS
# ============================================================

with open(f"{METRICS_DIR}/final_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=2)

# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_df.to_csv(f"{METRICS_DIR}/confusion_matrix.csv")

# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    final_targets,
    final_predictions,
    target_names=CLASS_NAMES,
    zero_division=0
)

with open(f"{METRICS_DIR}/classification_report.txt", "w") as f:
    f.write(report)

print("\n📄 Classification report saved.")

# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = f"{CHECKPOINT_DIR}/final_model.pth"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": len(tokenizer.vocab),
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "fragment_layers": FRAGMENT_LAYERS,
            "patient_layers": PATIENT_LAYERS,
            "num_classes": NUM_CLASSES
        },
        "class_names": CLASS_NAMES,
        "metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_metric": best_metric_value,
        "training_config": config_to_save
    },
    final_model_path
)

print(f"\n💾 Final model saved:\n   {final_model_path}")

# ============================================================
# SAVE TOKENIZER VOCABULARY
# ============================================================

try:

    tokenizer_vocab = tokenizer.vocab

    with open(f"{SAVE_DIR}/tokenizer_vocab.json", "w") as f:
        json.dump(tokenizer_vocab, f, indent=2)

    print(" Tokenizer vocabulary saved.")

except Exception as e:

    print(f" Could not save tokenizer vocabulary: {e}")

# ============================================================
# CREATE ZIP
# ============================================================

zip_base = "/kaggle/working/training_outputs_3classes_v4"
zip_path = shutil.make_archive(zip_base, "zip", SAVE_DIR)

print(f"\n ZIP created:\n   {zip_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(" EVERYTHING SAVED")
print("=" * 70)

print(f"\n Directory:\n{SAVE_DIR}")
print(f"\n Best model:\n{best_model_path}")
print(f"\n Final model:\n{final_model_path}")
print(f"\n ZIP:\n{zip_path}")

print("\n FINAL PERFORMANCE")
print(f"Accuracy          : {final_metrics['accuracy']:.4f}")
print(f"Balanced Accuracy : {final_metrics['balanced_accuracy']:.4f}")
print(f"Macro F1          : {final_metrics['macro_f1']:.4f}")
print(f"Macro ROC-AUC     : {final_metrics['roc_auc_macro']:.4f}")

print("\n FINAL PREDICTION DISTRIBUTION")

for i, name in enumerate(CLASS_NAMES):
    print(f"  {name:<10}: predicted={final_metrics['prediction_distribution'][i]:<4}  actual={val_class_counts[i]:<4}")

print("\n" + "=" * 70)
print(" VERSION 4 RUN COMPLETE")
print("=" * 70)