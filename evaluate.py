import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Local imports
from config import get_args
from model import WavLM_AASIST_Model
from dataset import UnifiedAudioDataset
from engine import evaluate

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Spoof', 'Bonafide'] # 0 is Spoof, 1 is Bonafide

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()


def main(args):
    """
    Main function to evaluate a pre-trained model on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for evaluation.")

    # --- Data Loading for Test Set ---
    _ = UnifiedAudioDataset(data_root=args.data_dir, partition='train')
    test_dataset = UnifiedAudioDataset(data_root=args.data_dir, partition='eval')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- Model Loading ---
    print("Loading pre-trained model for evaluation...")
    model = WavLM_AASIST_Model(
        model_path=args.model_path,
        freeze_wavlm=True
    ).to(device)

    model_save_path = os.path.join(args.output_dir, args.model_checkpoint)
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"FATAL: Model checkpoint not found at {model_save_path}. Please train a model first.")

    model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded from {model_save_path}")

    # --- Evaluation ---
    criterion = nn.BCEWithLogitsLoss()
    # The evaluate function now returns three items
    test_metrics, true_labels, pred_labels = evaluate(model, test_loader, criterion, device, "Testing on Eval Set")

    print("\n--- Test Performance ---")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    # --- Generate and Save Confusion Matrix ---
    print("\n--- Generating Confusion Matrix ---")
    cm_save_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(true_labels.cpu().numpy(), pred_labels.cpu().numpy(), cm_save_path)

    # --- Latency Test ---
    print("\n--- Latency Test ---")
    dummy_input = torch.randn(1, 16000 * 5).to(device)
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        _ = model(dummy_input)
        end_time = time.time()
    latency = (end_time - start_time) * 1000
    print(f"Inference Latency for a single sample: {latency:.2f} ms")


if __name__ == "__main__":
    args = get_args()
    main(args)