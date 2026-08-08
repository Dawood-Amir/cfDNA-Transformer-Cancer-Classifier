import os
import gc
import json
import time
from datetime import datetime
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm

from data.patient_dataset import PatientDataset
from data.patient_collate import PatientCollate
from models.tokenizer import CFDNATokenizer
from models.cfdna_transformer import CFDNATransformer

# Import the utils you created
from utils import FocalLoss, compute_all_metrics, save_metrics, print_metrics


# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    # Paths
    DATA_DIR = "/content/drive/MyDrive/cfdna-transformer-data/patient_tensors"
    SAVE_DIR = "/content/drive/MyDrive/cfdna-transformer-data/training_v2"
    CHECKPOINT_DIR = f"{SAVE_DIR}/checkpoints"
    METRICS_DIR = f"{SAVE_DIR}/metrics"
    LOGS_DIR = f"{SAVE_DIR}/logs"
    
    # Create directories
    for d in [SAVE_DIR, CHECKPOINT_DIR, METRICS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Training parameters
    BATCH_SIZE = 1
    GRAD_ACCUMULATION = 4
    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    PATIENCE = 7
    NUM_CLASSES = 4
    
    # Class names
    CLASS_NAMES = ['Healthy', 'GBM', 'LGG', 'DMG_H3K27M']
    
    # Known class counts (from your data)
    CLASS_COUNTS = [511, 374, 427, 160]  # Training set counts
    
    # Loss function: 'focal' or 'ce' (cross-entropy)
    LOSS_TYPE = 'focal'  # 'focal' or 'ce'
    FOCAL_GAMMA = 2.0
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Resume training?
    RESUME_CHECKPOINT = None  # Set to path to resume, or None to start fresh
    
    @classmethod
    def print_config(cls):
        print("=" * 70)
        print("🚀 TRAINING CONFIGURATION")
        print("=" * 70)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"  {key}: {value}")
        print("=" * 70)


# ============================================================
# DATA LOADING
# ============================================================

def create_data_loaders(config):
    """Create train and validation data loaders with optional oversampling."""
    
    print("\n📂 Loading dataset...")
    tokenizer = CFDNATokenizer()
    dataset = PatientDataset(config.DATA_DIR)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"  Total patients: {len(dataset)}")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    
    # Collate function
    collate_fn = PatientCollate(tokenizer.vocab["<pad>"])
    
    # --- Create sampler with oversampling ---
    print("\n📊 Computing class weights for oversampling...")
    
    # Get all labels from training set
    train_labels = []
    for i in train_dataset.indices:
        patient = dataset[i]
        # Patient is a dict with 'label' key
        train_labels.append(patient["label"])
    
    class_counts = np.bincount(train_labels, minlength=config.NUM_CLASSES)
    print(f"  Class counts: {class_counts}")
    print(f"  Class names: {config.CLASS_NAMES}")
    
    # Calculate sampling weights (inverse frequency)
    # Higher weight = more likely to be sampled
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label].item() for label in train_labels]
    
    # Create sampler with replacement
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights) * 2,  # Oversample to balance
        replacement=True
    )
    
    print(f"  Sampling weights: {class_weights.tolist()}")
    print("  ✅ WeightedRandomSampler created (oversampling enabled)")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,  # Use sampler instead of shuffle
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    return train_loader, val_loader, tokenizer


# ============================================================
# MODEL CREATION
# ============================================================

def create_model(config, tokenizer):
    """Create the model, optimizer, scheduler, and loss function."""
    
    model = CFDNATransformer(
        vocab_size=len(tokenizer.vocab),
        embed_dim=128,
        num_heads=4,
        fragment_layers=2,
        patient_layers=2,
        num_classes=config.NUM_CLASSES
    )
    model.to(config.DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS
    )
    
    # Loss function
    # Class weights (balanced)
    total = sum(config.CLASS_COUNTS)
    class_weights = total / (config.NUM_CLASSES * torch.tensor(config.CLASS_COUNTS, dtype=torch.float32))
    class_weights = class_weights.to(config.DEVICE)
    
    if config.LOSS_TYPE == 'focal':
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=config.FOCAL_GAMMA
        )
        print(f"\n  Using Focal Loss (gamma={config.FOCAL_GAMMA})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("\n  Using Cross-Entropy Loss with class weights")
    
    print(f"  Class weights: {class_weights.tolist()}")
    
    return model, optimizer, scheduler, criterion


# ============================================================
# TRAINING LOOP
# ============================================================

def train_model(config, train_loader, val_loader, model, optimizer, scheduler, criterion):
    """Main training loop with full checkpointing and metrics."""
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_balanced_accuracy': [],
        'val_macro_f1': [],
        'val_weighted_f1': [],
        'val_roc_auc_macro': [],
        'best_epoch': 0,
        'best_val_loss': float('inf'),
        'best_val_accuracy': 0.0,
        'start_time': datetime.now().isoformat(),
        'end_time': None,
        'config': {
            'batch_size': config.BATCH_SIZE,
            'grad_accumulation': config.GRAD_ACCUMULATION,
            'epochs': config.EPOCHS,
            'lr': config.LR,
            'weight_decay': config.WEIGHT_DECAY,
            'patience': config.PATIENCE,
            'loss_type': config.LOSS_TYPE,
            'class_names': config.CLASS_NAMES,
            'class_counts': config.CLASS_COUNTS
        }
    }
    
    # Variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # AMP Scaler
    scaler = torch.amp.GradScaler(config.DEVICE_TYPE)
    
    print(f"\n🔍 Training on: {config.DEVICE_TYPE.upper()}")
    print("=" * 70)
    print("🚀 STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        
        print("\n" + "=" * 70)
        print(f"EPOCH {epoch+1}/{config.EPOCHS}")
        print("=" * 70)
        
        # --- Training Phase ---
        model.train()
        train_loss = 0
        optimizer.zero_grad()
        
        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        
        for step, batch in progress:
            # Move to device
            input_ids = batch["input_ids"].to(config.DEVICE)
            token_mask = batch["token_padding_mask"].to(config.DEVICE)
            fragment_mask = batch["fragment_padding_mask"].to(config.DEVICE)
            region_mask = batch["region_padding_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            # Forward pass with AMP
            if config.DEVICE_TYPE == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, token_mask, fragment_mask, region_mask)
                    loss = criterion(logits, labels)
                    loss = loss / config.GRAD_ACCUMULATION
            else:
                logits = model(input_ids, token_mask, fragment_mask, region_mask)
                loss = criterion(logits, labels)
                loss = loss / config.GRAD_ACCUMULATION
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (step + 1) % config.GRAD_ACCUMULATION == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            train_loss += loss.item() * config.GRAD_ACCUMULATION
            progress.set_description(f"loss={loss.item() * config.GRAD_ACCUMULATION:.4f}")
            
            # Free memory
            del input_ids, token_mask, fragment_mask, region_mask, labels, logits, loss
        
        train_loss /= len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        all_preds = []
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(config.DEVICE)
                token_mask = batch["token_padding_mask"].to(config.DEVICE)
                fragment_mask = batch["fragment_padding_mask"].to(config.DEVICE)
                region_mask = batch["region_padding_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)
                
                if config.DEVICE_TYPE == 'cuda':
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, token_mask, fragment_mask, region_mask)
                        loss = criterion(logits, labels)
                else:
                    logits = model(input_ids, token_mask, fragment_mask, region_mask)
                    loss = criterion(logits, labels)
                
                val_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                
                del input_ids, token_mask, fragment_mask, region_mask, labels, logits, loss, probs, preds
        
        val_loss /= len(val_loader)
        
        # --- Compute metrics ---
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        metrics = compute_all_metrics(
            all_targets, 
            all_preds, 
            all_probs,
            class_names=config.CLASS_NAMES
        )
        
        # --- Logging ---
        epoch_time = time.time() - epoch_start_time
        
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Bal Acc:    {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
        print(f"  Macro AUC:  {metrics['roc_auc_macro']:.4f}")
        print(f"  Time:       {epoch_time:.2f}s")
        
        # Per-class recall (most important for rare classes)
        print("\n  Per-class Recall:")
        for class_name, scores in metrics['per_class'].items():
            print(f"    {class_name}: {scores['recall']:.4f} (support: {scores['support']})")
        
        # --- Save history ---
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(metrics['accuracy'])
        history['val_balanced_accuracy'].append(metrics['balanced_accuracy'])
        history['val_macro_f1'].append(metrics['macro_f1'])
        history['val_weighted_f1'].append(metrics['weighted_f1'])
        history['val_roc_auc_macro'].append(metrics['roc_auc_macro'])
        
        # --- Save latest checkpoint ---
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'metrics': metrics,
            'config': history['config']
        }
        
        torch.save(checkpoint, f"{config.CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pth")
        
        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            history['best_epoch'] = epoch
            history['best_val_loss'] = val_loss
            history['best_val_accuracy'] = metrics['accuracy']
            
            # Save best checkpoint
            checkpoint['is_best'] = True
            torch.save(checkpoint, f"{config.CHECKPOINT_DIR}/best_model.pth")
            
            # Save best metrics
            save_metrics(metrics, f"{config.METRICS_DIR}/best_metrics.json")
            
            print(f"\n  ✅ NEW BEST MODEL! (Loss: {val_loss:.4f}, Acc: {metrics['accuracy']:.4f})")
        else:
            patience_counter += 1
            print(f"\n  ⏳ Early stopping counter: {patience_counter}/{config.PATIENCE}")
        
        # --- Early stopping ---
        if patience_counter >= config.PATIENCE:
            print("\n🛑 Early stopping triggered!")
            break
        
        # --- Scheduler step ---
        scheduler.step()
        
        # --- Cleanup ---
        torch.cuda.empty_cache()
        gc.collect()
        
        # --- Save history periodically ---
        history['end_time'] = datetime.now().isoformat()
        with open(f"{config.LOGS_DIR}/training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
    
    # --- Final metrics on best model ---
    print("\n" + "=" * 70)
    print("🏁 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Best Epoch: {history['best_epoch'] + 1}")
    print(f"Best Val Loss: {history['best_val_loss']:.4f}")
    print(f"Best Val Accuracy: {history['best_val_accuracy']:.4f}")
    
    # Load best model and compute final metrics
    best_checkpoint = torch.load(f"{config.CHECKPOINT_DIR}/best_model.pth", map_location=config.DEVICE)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    
    # Run final validation with best model
    all_preds = []
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Final Evaluation"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            token_mask = batch["token_padding_mask"].to(config.DEVICE)
            fragment_mask = batch["fragment_padding_mask"].to(config.DEVICE)
            region_mask = batch["region_padding_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            logits = model(input_ids, token_mask, fragment_mask, region_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    final_metrics = compute_all_metrics(
        np.array(all_targets),
        np.array(all_preds),
        np.array(all_probs),
        class_names=config.CLASS_NAMES
    )
    
    # Print final metrics
    print_metrics(final_metrics)
    
    # Save final metrics
    save_metrics(final_metrics, f"{config.METRICS_DIR}/final_metrics.json")
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(
        final_metrics['confusion_matrix'],
        index=config.CLASS_NAMES,
        columns=config.CLASS_NAMES
    )
    cm_df.to_csv(f"{config.METRICS_DIR}/confusion_matrix.csv")
    
    print(f"\n📁 Results saved to: {config.SAVE_DIR}")
    
    return model, history, final_metrics


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # Load config
    config = Config
    config.print_config()
    
    # Create data loaders
    train_loader, val_loader, tokenizer = create_data_loaders(config)
    
    # Create model
    model, optimizer, scheduler, criterion = create_model(config, tokenizer)
    
    # Train
    model, history, final_metrics = train_model(
        config, train_loader, val_loader, 
        model, optimizer, scheduler, criterion
    )
    
    print("\n🎉 TRAINING PIPELINE COMPLETE!")
    print(f"📁 All outputs saved to: {config.SAVE_DIR}")


if __name__ == "__main__":
    main()