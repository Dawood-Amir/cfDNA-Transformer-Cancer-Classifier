
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from utils.metrics import compute_metrics

# =====================================================================
# 1. DATASET AND MASKING PIPELINE
# =====================================================================

class PretrainFineTuneDataset(Dataset):
    def __init__(self, base_dir="src/data/processed/region_tensors"):
        self.base_dir = base_dir
        manifest_path = os.path.join(base_dir, "manifest.csv")
        if os.path.exists(manifest_path):
            raw_manifest = pd.read_csv(manifest_path)
            
            # Balance classes inside initialization before splitting
            SEED = 42
            min_class_size = raw_manifest['label'].value_counts().min()
            
            balanced_chunks = []
            for lbl in raw_manifest['label'].unique():
                class_subset = raw_manifest[raw_manifest['label'] == lbl]
                sampled_subset = class_subset.sample(n=min_class_size, random_state=SEED)
                balanced_chunks.append(sampled_subset)
                
            self.manifest = pd.concat(balanced_chunks, axis=0).reset_index(drop=True)
            print(f"[Dataset] Explicit balancing complete. Total samples: {len(self.manifest)} ({min_class_size} per class)")
        else:
            print(f"Warning: {manifest_path} not found. Creating a mock manifest for execution testing.")
            self.manifest = pd.DataFrame([{"filename": f"mock_{i}.npy", "label": i % 4} for i in range(800)])
            
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, index):
        row = self.manifest.iloc[index]
        file_path = os.path.join(self.base_dir, row['filename'])
        
        if os.path.exists(file_path):
            matrix = np.load(file_path)
        else:
            mock_g = np.random.randint(100, 200)
            matrix = np.random.randn(mock_g, 16).astype(np.float32)
            
        return torch.tensor(matrix, dtype=torch.float32), torch.tensor(row['label'], dtype=torch.long)


def custom_pretrain_collate(batch, mask_prob=0.15):
    """
    Custom collate function that pads sequences dynamically AND generates 
    the random 15% masks required for BERT-style pre-training.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    targets = padded_sequences.clone()
    
    random_prob_matrix = torch.rand(padded_sequences.shape[:2])
    max_len = padded_sequences.size(1)
    valid_lengths_mask = torch.arange(max_len)[None, :] < lengths[:, None]
    
    mask_indices = (random_prob_matrix < mask_prob) & valid_lengths_mask
    padded_sequences[mask_indices] = 0.0
    
    return padded_sequences, targets, lengths, mask_indices, labels


# =====================================================================
# 2.  MODULAR ARCHITECTURE WITH RESIDUALS & LAYERNORM
# =====================================================================

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.ffn(x))


class CompleteGenomicModel(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, num_classes=4):
        super().__init__()
        
        # --- PRE-TRAINED BACKBONE ---
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2, 
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.residual_block = ResidualBlock(hidden_dim=hidden_dim)
        
        #  BACKBONE NORMALIZATION FIX: Scales the volatile 698.0+ raw values 
        # down into a clean 0-1 distribution before pushing them to the classification heads.
        self.backbone_norm = nn.LayerNorm(hidden_dim)
        
        # --- PHASE 1 HEAD: Reconstruction (Pre-training) ---
        self.reconstruction_head = nn.Linear(hidden_dim, input_dim)
        
        # --- PHASE 2 HEADS: Attention & Classification (Fine-Tuning) ---
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x, lengths, mode="pretrain"):
        x_proj = torch.relu(self.input_projection(x))
        
        packed_x = pack_padded_sequence(x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_x)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        out = self.residual_block(out)
        
        if mode == "pretrain":
            return self.reconstruction_head(out)
            
        elif mode == "finetune":
            # Normalize feature scales right before computing attention scores
            out = self.backbone_norm(out)
            
            attn_scores = self.attention(out)
            mask = torch.arange(out.size(1), device=out.device)[None, :] < lengths[:, None]
            attn_scores[~mask] = float('-inf')
            
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = torch.sum(attn_weights * out, dim=1)
            
            return self.classifier(context)


# =====================================================================
# 3. EXECUTIVE EXECUTION FLOW CONTROL PIPELINES
# =====================================================================

def execute_pretraining_phase(model, train_loader, device, epochs=40):
    print("\n" + "="*60)
    print(" STARTING PHASE 1: MASKED REGION PRE-TRAINING")
    print("="*60)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    mse_criterion = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, total_masked_elements = 0.0, 0
        
        for batch_x, batch_targets, lengths, mask_indices, _ in train_loader:
            batch_x, batch_targets = batch_x.to(device), batch_targets.to(device)
            mask_indices = mask_indices.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x, lengths, mode="pretrain")
            
            max_len = predictions.size(1)
            valid_lengths_mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None].to(device)
            loss_mask = valid_lengths_mask & mask_indices
            
            if loss_mask.sum() == 0: continue
                
            loss = mse_criterion(predictions[loss_mask], batch_targets[loss_mask])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * loss_mask.sum().item()
            total_masked_elements += loss_mask.sum().item()
            
        avg_loss = epoch_loss / max(total_masked_elements, 1)
        print(f"Pretrain Epoch {epoch:<2}/{epochs} | Reconstructive MSE Loss: {avg_loss:.5f}")
        
    print("\n Phase 1 complete! Saved structural backbone representation weights.")
    torch.save(model.state_dict(), "pretrained_genomic_backbone.pth")


def execute_finetuning_phase(model, train_loader, val_loader, device, epochs=40):
    print("\n" + "="*60)
    print(" STARTING PHASE 2: BRAIN CANCER CLASSIFICATION FINE-TUNING")
    print("="*60)
    
    # -----------------------------------------------------------------
    # STAGE 1: FREEZE THE BACKBONE (First 5 Epochs)
    # -----------------------------------------------------------------
    print("\n[Warmup Phase] Freezing pre-trained backbone parameters...")
    for name, param in model.named_parameters():
        if "classifier" not in name and "attention" not in name:
            param.requires_grad = False
            
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=0.01)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        # -----------------------------------------------------------------
        # STAGE 2: UNFREEZE AND FINE-TUNE ENTIRE NETWORK (At Epoch 6)
        # -----------------------------------------------------------------
        if epoch == 6:
            print("\n[Fine-Tuning Phase] Unfreezing backbone. Engaging discriminative learning rates...")
            for param in model.parameters():
                param.requires_grad = True
                
            # Differential Learning Rates setup
            optimizer = optim.AdamW([
                {"params": model.input_projection.parameters(), "lr": 1e-5},
                {"params": model.lstm.parameters(), "lr": 1e-5},
                {"params": model.residual_block.parameters(), "lr": 1e-5},
                {"params": model.backbone_norm.parameters(), "lr": 1e-5},
                {"params": model.attention.parameters(), "lr": 3e-4},
                {"params": model.classifier.parameters(), "lr": 3e-4}
            ], weight_decay=0.05)
            
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(epochs - 5))

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_x, _, lengths, _, labels in train_loader:
            batch_x, labels = batch_x.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, lengths, mode="finetune")
            loss = ce_criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, _, lengths, _, labels in val_loader:
                batch_x, labels = batch_x.to(device), labels.to(device)
                outputs = model(batch_x, lengths, mode="finetune")
                loss = ce_criterion(outputs, labels)
                
                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        if epoch >= 6:
            scheduler.step()
            
        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
        
        phase_prefix = "[Warmup]" if epoch < 6 else "[Tune]"
        print(f"{phase_prefix} Epoch {epoch:<2} | Train Loss: {train_loss/train_total:.4f} ({train_acc:.1f}%) | Val Loss: {val_loss/val_total:.4f} ({val_acc:.1f}%)")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_brain_cancer_classifier.pth")
            
    print("\n[System] Loading best model weights for comprehensive test evaluation...")
    model.load_state_dict(torch.load("best_brain_cancer_classifier.pth"))
    model.eval()
    
    all_preds, all_probs, all_targets = [], [], []
    test_loss, test_total = 0.0, 0
    
    with torch.no_grad():
        for batch_x, _, lengths, _, labels in val_loader:
            batch_x, labels = batch_x.to(device), labels.to(device)
            outputs = model(batch_x, lengths, mode="finetune")
            loss = ce_criterion(outputs, labels)
            
            test_loss += loss.item() * batch_x.size(0)
            test_total += labels.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            
    final_probs = np.concatenate(all_probs, axis=0)
    final_preds = np.concatenate(all_preds, axis=0)
    final_targets = np.concatenate(all_targets, axis=0)
    final_test_loss = test_loss / test_total
    
    metrics = compute_metrics(final_targets, final_preds, final_probs) 
    
    print("\n========== TEST RESULTS ==========")
    print(f"Loss : {final_test_loss:.4f}")
    print(f"AUC  : {metrics['auc']:.4f}")
    print(metrics["classification_report"])
    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
    print("==================================\n")


# =====================================================================
# 4. ORCHESTRATION PIPELINE CONTROL
# =====================================================================

def main():
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialization sequence ready. Target Device: {device}")
    
    DATA_DIR = "src/data/processed/region_tensors"
    full_dataset = PretrainFineTuneDataset(base_dir=DATA_DIR)
    
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_pretrain_collate)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_pretrain_collate)
    
    model = CompleteGenomicModel(input_dim=16, hidden_dim=64, num_classes=4).to(device)
    
    # STEP A: RUN PRE-TRAINING
    execute_pretraining_phase(model, train_loader, device, epochs=100)
    
    # STEP B: SWAP HEADS AND INITIALIZE BACKBONE WEIGHTS
    print("\n[System Alert] Swapping heads: Loading structural parameters into downstream tasks...")
    pretrained_states = torch.load("pretrained_genomic_backbone.pth")
    model.load_state_dict(pretrained_states, strict=False)  # strict=False allows loading despite the new LayerNorm parameter
    
    # Re-initialize only the custom classifier and attention weights before warm-up
    for layer in [model.attention, model.classifier]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    # RUN FINE-TUNING (Classification via Attention Pooling)
    execute_finetuning_phase(model, train_loader, val_loader, device, epochs=40)


if __name__ == "__main__":
    main()

