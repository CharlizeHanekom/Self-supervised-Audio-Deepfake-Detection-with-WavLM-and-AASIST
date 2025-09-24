import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random
import time
import warnings

# Local imports
from config import get_args
from model import WavLM_AASIST_Model
from dataset import UnifiedAudioDataset
from engine import evaluate

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def main(args):
    """
    Main function to evaluate a pre-trained model on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for evaluation.")

    # --- Data Loading for Test Set ---
    # Note: We initialize the dataset to ensure partitions are created if they don't exist
    # but we only need the 'eval' partition.
    _ = UnifiedAudioDataset(data_root=args.data_dir, partition='train') # This ensures splits are made
    test_dataset = UnifiedAudioDataset(data_root=args.data_dir, partition='eval')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # --- Model Loading ---
    print("Loading pre-trained model for evaluation...")
    model = WavLM_AASIST_Model(
        model_path=args.model_path,
        freeze_wavlm=True  # Always freeze for evaluation
    ).to(device)

    model_save_path = os.path.join(args.output_dir, args.model_checkpoint)
    if not os.path.exists(model_save_path):
        raise FileNotFoundError(f"FATAL: Model checkpoint not found at {model_save_path}. Please train a model first.")

    model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded from {model_save_path}")

    # --- Evaluation ---
    criterion = nn.BCEWithLogitsLoss() # Loss function for evaluation
    test_metrics = evaluate(model, test_loader, criterion, device, "Testing on Eval Set")

    print("\n--- Test Performance ---")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

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