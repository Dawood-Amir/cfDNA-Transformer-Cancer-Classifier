# src/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)
import json
import os


# ============================================================
# FOCAL LOSS (for handling class imbalance)
# ============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    Focuses training on hard-to-classify examples.
    
    Args:
        alpha: Class weights (tensor of shape [num_classes])
        gamma: Focusing parameter (default: 2.0)
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Standard cross-entropy (without reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Focal loss: (1 - pt)^gamma * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_all_metrics(targets, preds, probs, class_names=None):
    """
    Compute all classification metrics.
    
    Args:
        targets: Ground truth labels
        preds: Predicted labels
        probs: Prediction probabilities (shape [n_samples, n_classes])
        class_names: List of class names (optional)
    
    Returns:
        dict: All computed metrics
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(probs.shape[1])]
    
    n_classes = probs.shape[1]
    metrics = {}
    
    # --- Basic metrics ---
    metrics['accuracy'] = accuracy_score(targets, preds)
    metrics['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
    
    # --- F1 scores ---
    metrics['macro_f1'] = f1_score(targets, preds, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(targets, preds, average='weighted', zero_division=0)
    
    # --- Per-class metrics ---
    metrics['per_class'] = {}
    for i in range(n_classes):
        # For each class, compute precision, recall, f1
        tp = np.sum((preds == i) & (targets == i))
        fp = np.sum((preds == i) & (targets != i))
        fn = np.sum((preds != i) & (targets == i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': np.sum(targets == i)
        }
    
    # --- ROC-AUC ---
    try:
        # Macro ROC-AUC (one-vs-rest)
        metrics['roc_auc_macro'] = roc_auc_score(
            targets, probs, 
            multi_class='ovr', 
            average='macro'
        )
        
        # Weighted ROC-AUC
        metrics['roc_auc_weighted'] = roc_auc_score(
            targets, probs, 
            multi_class='ovr', 
            average='weighted'
        )
        
        # Per-class ROC-AUC
        metrics['roc_auc_per_class'] = {}
        for i in range(n_classes):
            try:
                # Binary ROC-AUC for each class
                metrics['roc_auc_per_class'][class_names[i]] = roc_auc_score(
                    (targets == i).astype(int),
                    probs[:, i]
                )
            except:
                metrics['roc_auc_per_class'][class_names[i]] = 0.0
                
    except Exception as e:
        # If ROC-AUC fails (e.g., only one class predicted), set to 0
        metrics['roc_auc_macro'] = 0.0
        metrics['roc_auc_weighted'] = 0.0
        metrics['roc_auc_per_class'] = {name: 0.0 for name in class_names}
    
    # --- Confusion Matrix ---
    metrics['confusion_matrix'] = confusion_matrix(targets, preds).tolist()
    
    # --- Prediction Distribution ---
    pred_counts = np.bincount(preds, minlength=n_classes)
    metrics['prediction_distribution'] = pred_counts.tolist()
    
    # --- Classification Report ---
    report = classification_report(
        targets, preds, 
        target_names=class_names, 
        zero_division=0,
        output_dict=True
    )
    metrics['classification_report'] = report
    
    return metrics


def save_metrics(metrics, save_path):
    """Save metrics to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def print_metrics(metrics):
    """Pretty print metrics."""
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION METRICS")
    print("=" * 70)
    
    print(f"\n🎯 Accuracy:          {metrics['accuracy']:.4f}")
    print(f"⚖️  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"📊 Macro F1:          {metrics['macro_f1']:.4f}")
    print(f"📊 Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"📈 Macro ROC-AUC:     {metrics['roc_auc_macro']:.4f}")
    print(f"📈 Weighted ROC-AUC:  {metrics['roc_auc_weighted']:.4f}")
    
    print("\n" + "-" * 70)
    print("📋 PER-CLASS PERFORMANCE")
    print("-" * 70)
    
    for class_name, scores in metrics['per_class'].items():
        print(f"\n{class_name}:")
        print(f"  Precision: {scores['precision']:.4f}")
        print(f"  Recall:    {scores['recall']:.4f}")
        print(f"  F1:        {scores['f1']:.4f}")
        print(f"  Support:   {scores['support']}")
    
    print("\n" + "-" * 70)
    print("📊 PREDICTION DISTRIBUTION")
    print("-" * 70)
    for class_name, count in zip(metrics['classification_report'].keys(), metrics['prediction_distribution']):
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"  {class_name}: {count} predictions")
    
    print("\n" + "=" * 70)