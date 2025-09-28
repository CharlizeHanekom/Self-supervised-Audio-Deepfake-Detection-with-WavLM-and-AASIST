import torch
from tqdm import tqdm
from typing import Dict, Any, Tuple
from torch.cuda.amp import GradScaler, autocast
from metrics import calculate_performance_metrics

def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, epoch_desc="Training") -> float:
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    
    # The progress bar now includes the epoch description
    progress_bar = tqdm(dataloader, desc=epoch_desc, leave=False)

    for waveforms, labels in progress_bar:
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        optimizer.zero_grad()

        with autocast(): # Automatic Mixed Precision
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        
        # Display the running average loss in the progress bar
        progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, eval_desc="Evaluating") -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
    """Evaluates the model on the given dataset."""
    model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []
    
    # The progress bar now includes the evaluation description
    progress_bar = tqdm(dataloader, desc=eval_desc, leave=False)
    
    for waveforms, labels in progress_bar:
        waveforms, labels = waveforms.to(device), labels.to(device)
        
        with autocast():
            outputs = model(waveforms)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        all_labels.append(labels)
        all_scores.append(torch.sigmoid(outputs)) # Convert logits to scores
        
        # Display the running average loss in the evaluation progress bar
        progress_bar.set_postfix(loss=f"{total_loss / (progress_bar.n + 1):.4f}")

    all_labels = torch.cat(all_labels)
    all_scores = torch.cat(all_scores)
    
    # Generate predictions (0 or 1) based on a 0.5 threshold
    all_preds = (all_scores > 0.5).long()

    metrics = calculate_performance_metrics(all_labels, all_scores)
    metrics["Loss"] = total_loss / len(dataloader)
    
    return metrics, all_labels, all_preds