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
# JSON SERIALIZATION HELPERS
# ============================================================

def convert_to_serializable(obj):
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


# ============================================================
# FOCAL LOSS
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
        dict: All computed metrics (with Python-native types)
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(probs.shape[1])]
    
    n_classes = probs.shape[1]
    metrics = {}
    
    # --- Basic metrics ---
    metrics['accuracy'] = float(accuracy_score(targets, preds))
    metrics['balanced_accuracy'] = float(balanced_accuracy_score(targets, preds))
    metrics['macro_f1'] = float(f1_score(targets, preds, average='macro', zero_division=0))
    metrics['weighted_f1'] = float(f1_score(targets, preds, average='weighted', zero_division=0))
    
    # --- Per-class metrics ---
    metrics['per_class'] = {}
    for i in range(n_classes):
        tp = np.sum((preds == i) & (targets == i))
        fp = np.sum((preds == i) & (targets != i))
        fn = np.sum((preds != i) & (targets == i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['per_class'][class_names[i]] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'support': int(np.sum(targets == i))
        }
    
    # --- ROC-AUC ---
    try:
        metrics['roc_auc_macro'] = float(roc_auc_score(targets, probs, multi_class='ovr', average='macro'))
        metrics['roc_auc_weighted'] = float(roc_auc_score(targets, probs, multi_class='ovr', average='weighted'))
        
        metrics['roc_auc_per_class'] = {}
        for i in range(n_classes):
            try:
                auc_val = roc_auc_score(
                    (targets == i).astype(int),
                    probs[:, i]
                )
                metrics['roc_auc_per_class'][class_names[i]] = float(auc_val)
            except:
                metrics['roc_auc_per_class'][class_names[i]] = 0.0
    except Exception as e:
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
    metrics['classification_report'] = convert_to_serializable(report)
    
    return metrics


def save_metrics(metrics, save_path):
    """Save metrics to a JSON file with NumPy type conversion."""
    serializable_metrics = convert_to_serializable(metrics)
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


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
    
    # Handle both dict and list cases
    if isinstance(metrics.get('classification_report'), dict):
        # For sklearn classification_report output
        for class_name in metrics['classification_report'].keys():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                # Try to find prediction count for this class
                class_idx = list(metrics['per_class'].keys()).index(class_name) if class_name in metrics['per_class'] else None
                if class_idx is not None:
                    count = metrics['prediction_distribution'][class_idx] if class_idx < len(metrics['prediction_distribution']) else 0
                    print(f"  {class_name}: {count} predictions")
    else:
        # Fallback: just show distribution
        for i, count in enumerate(metrics.get('prediction_distribution', [])):
            class_name = list(metrics['per_class'].keys())[i] if i < len(metrics['per_class']) else f"Class_{i}"
            print(f"  {class_name}: {count} predictions")
    
    print("\n" + "=" * 70)