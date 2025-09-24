import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import time
import numpy as np
import random
from tqdm import tqdm
# import torchaudio
# from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, ApplyImpulseResponse

from model import WavLM_AASIST_Model
from dataset import UnifiedAudioDataset
from engine import train_one_epoch, evaluate

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets (this may take a moment)...")
    train_dataset = UnifiedAudioDataset(data_root=args.data_dir, partition='train')
    val_dataset = UnifiedAudioDataset(data_root=args.data_dir, partition='dev')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, pin_memory=True
    )

    # --- Model, Loss, Optimizer ---
    # --- 1. UNFREEZE THE MODEL ---
    print("Initializing model for fine-tuning...")
    model = WavLM_AASIST_Model(
        model_path=args.model_path,
        freeze_wavlm=False  # Set to False to allow fine-tuning
    ).to(device)

    pos_weight = train_dataset.get_class_weights().to(device)
    print(f"Using positive class weight: {pos_weight.item():.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- 2. SET UP DIFFERENTIAL LEARNING RATES ---
    # Separate parameters for the feature extractor (WavLM) and the backend (AASIST)
    wavlm_params = model.feature_extractor.parameters()
    backend_params = model.backend.parameters()

    # Create two parameter groups for the optimizer
    optimizer = torch.optim.AdamW([
        {'params': wavlm_params, 'lr': args.learning_rate / 10},  # Lower LR for WavLM
        {'params': backend_params, 'lr': args.learning_rate}      # Higher LR for the new backend
    ], weight_decay=1e-5)

    # It's often good to use a more advanced scheduler for fine-tuning
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    best_val_eer = float('inf')
    patience_counter = 0
    
    # Main progress bar for epochs
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Total Progress")

    for epoch in epoch_pbar:
        # Dynamic description for the main progress bar
        epoch_pbar.set_description(f"Epoch {epoch}/{args.epochs}")

        # Pass descriptions to the engine functions
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, f"Training Epoch {epoch}")
        val_metrics = evaluate(model, val_loader, criterion, device, f"Validating Epoch {epoch}")
        
        scheduler.step(val_metrics['EER'])

        # Update the main progress bar's postfix with the latest metrics
        epoch_pbar.set_postfix(
            TrainLoss=f"{train_loss:.4f}",
            ValLoss=f"{val_metrics['Loss']:.4f}",
            ValEER=f"{val_metrics['EER']:.4f}"
        )

        # Save best model and implement early stopping
        if val_metrics['EER'] < best_val_eer:
            best_val_eer = val_metrics['EER']
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("\nEarly stopping triggered.")
            break

    # --- Final Evaluation ---
    print("\n--- Final Evaluation on Test Set ---")
    test_dataset = UnifiedAudioDataset(data_root=args.data_dir, partition='eval')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pth")))
    
    # The evaluate function already has a progress bar
    test_metrics = evaluate(model, test_loader, criterion, device, "Testing")

    print("\n--- Test Performance ---")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")
    print(f"Best Validation EER during training: {best_val_eer:.4f}")

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
    parser = argparse.ArgumentParser(description="WavLM-AASIST Audio Deepfake Detection")
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory of datasets')
    parser.add_argument('--model_path', type=str, default='./models/wavlm-base', help='Path to pretrained WavLM model')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save model checkpoints')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    main(args)