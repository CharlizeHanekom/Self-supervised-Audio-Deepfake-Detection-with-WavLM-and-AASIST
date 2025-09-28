import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import warnings

# Local imports
from config import get_args
from model import WavLM_AASIST_Model
from dataset import UnifiedAudioDataset
from engine import train_one_epoch, evaluate
import evaluate as final_evaluator # To run evaluation automatically after training

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

def set_seed(seed=42):
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
    """
    Main function to orchestrate the training and validation process.
    """
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Loading datasets for training...")
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

    # --- Model Initialization ---
    print("Initializing model...")
    model = WavLM_AASIST_Model(
        model_path=args.model_path,
        freeze_wavlm=not args.fine_tune # Unfreeze if --fine_tune is set
    ).to(device)
    if args.fine_tune:
        print("Model is being prepared for fine-tuning (WavLM is unfrozen).")

    # --- Loss, Optimizer, and Scheduler ---
    pos_weight = train_dataset.get_class_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    if args.fine_tune:
        print("Using differential learning rates for fine-tuning.")
        wavlm_params = model.feature_extractor.parameters()
        backend_params = model.backend.parameters()
        optimizer = torch.optim.AdamW([
            {'params': wavlm_params, 'lr': args.learning_rate * args.wavlm_lr_ratio},
            {'params': backend_params, 'lr': args.learning_rate}
        ], weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=False)
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop ---
    best_val_eer = float('inf')
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    model_save_path = os.path.join(args.output_dir, args.model_checkpoint)

    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Total Progress")
    for epoch in epoch_pbar:
        epoch_pbar.set_description(f"Epoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, f"Training Epoch {epoch}")
        val_metrics, _, _ = evaluate(model, val_loader, criterion, device, f"Validating Epoch {epoch}")

        scheduler.step(val_metrics['EER'])
        epoch_pbar.set_postfix(TrainLoss=f"{train_loss:.4f}", ValEER=f"{val_metrics['EER']:.4f}")

        if val_metrics['EER'] < best_val_eer:
            best_val_eer = val_metrics['EER']
            torch.save(model.state_dict(), model_save_path)
            print(f"\nNew best model saved to {model_save_path} with EER: {best_val_eer:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs with no improvement.")
            break

    print(f"\nTraining complete. Best validation EER: {best_val_eer:.4f}")
    print("-" * 30)

    # --- Automatically run final evaluation ---
    print("Proceeding to final evaluation on the test set...")
    final_evaluator.main(args)


if __name__ == "__main__":
    args = get_args()
    main(args)