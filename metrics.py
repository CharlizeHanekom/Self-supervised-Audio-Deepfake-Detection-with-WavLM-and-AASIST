
import torch
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score

def calculate_eer(y_true, y_scores):
    """Calculates the Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def calculate_performance_metrics(labels: torch.Tensor, scores: torch.Tensor):
    """
    Calculates a dictionary of performance metrics.
    Args:
        labels: Ground truth labels (0 or 1)
        scores: Model prediction scores (logits or probabilities)
    """
    y_true = labels.cpu().numpy()
    y_scores = scores.cpu().numpy()
    y_pred_class = (y_scores > 0.5).astype(int)

    try:
        eer = calculate_eer(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        # Happens if only one class is present in a batch
        eer = -1.0
        auc = -1.0
        
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_class, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred_class)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "EER": eer
    }