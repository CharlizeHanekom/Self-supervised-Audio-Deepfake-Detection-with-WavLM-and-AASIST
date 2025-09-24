import multiprocessing
multiprocessing.cpu_count()

# # Environment Setup
import sys
# get_ipython().system('{sys.executable} -m pip install torch torchaudio transformers librosa matplotlib numpy scikit-learn pandas seaborn tqdm')

# # Data Preparation
import os
import torchaudio
import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
ITW_BASE_PATH = "./data/InTheWild"
ASV_BASE_PATH = "./data/ASVspoof"
FOR_BASE_PATH = "./data/FOR"

ITW_DATASET_PATH = os.path.join(ITW_BASE_PATH, "ITW_dataset")
ASV_DATASET_PATH = os.path.join(ASV_BASE_PATH, "ASV_dataset")
FOR_DATASET_PATH = os.path.join(FOR_BASE_PATH, "FOR_dataset")

def load_InTheWild_metadata(base_path, dataset_path):
    meta_file = os.path.join(base_path, "ITW_meta.csv")
    print("Meta file: ", meta_file)
    metadata = pd.read_csv(meta_file)
    metadata['filepath'] = metadata['file'].apply(lambda x: os.path.join(dataset_path, x))
    metadata['label'] = metadata['label'].apply(lambda x: 1 if x == 'fake' else 0)
    return metadata

def load_ASVspoof_metadata(base_path, dataset_path):
    meta_file = os.path.join(base_path, "ASV_meta.csv")
    print("Meta file: ", meta_file)
    metadata = pd.read_csv(meta_file)
    metadata['filepath'] = metadata['file'].apply(lambda x: os.path.join(dataset_path, x))
    metadata['label'] = metadata['label'].apply(lambda x: 1 if x == 'fake' else 0)
    return metadata
    
def load_FOR_metadata_by_split(base_path, dataset_path):
    meta_file = os.path.join(base_path, "FOR_meta.csv")
    print("Meta file:", meta_file)
    df = pd.read_csv(meta_file)

    df['filepath'] = df['file'].apply(lambda x: os.path.join(dataset_path, x))
    df['label'] = df['label'].apply(lambda x: 1 if x == 'fake' else 0)

    # Make sure 'split' column is standardized
    df['split'] = df['split'].str.lower().str.strip()

    train_df = df[df['split'] == 'train'].copy()
    val_df   = df[df['split'] == 'val'].copy()
    test_df  = df[df['split'] == 'test'].copy()

    return train_df, val_df, test_df


# Load datasets
inthewild_meta = load_InTheWild_metadata(ITW_BASE_PATH, ITW_DATASET_PATH)
asv_meta = load_ASVspoof_metadata(ASV_BASE_PATH, ASV_DATASET_PATH)
# inthewild_meta = load_InTheWild_metadata(BASE_PATH, DATASET_PATH)

# Combine and split
combined_meta = pd.concat([inthewild_meta, asv_meta], ignore_index=True)
train_df, test_df = train_test_split(combined_meta, test_size=0.2, random_state=42, stratify=combined_meta['label'])
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

# Load FOR metadata by split
for_train, for_val, for_test = load_FOR_metadata_by_split(FOR_BASE_PATH, FOR_DATASET_PATH)

# Combine with other datasets
train_df = pd.concat([train_df, for_train], ignore_index=True)
val_df   = pd.concat([val_df, for_val], ignore_index=True)
test_df  = pd.concat([test_df, for_test], ignore_index=True)

print(f"✅ Final splits:")
print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")


# ## Data Loaders
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

class AudioDataset(Dataset):
    def __init__(self, metadata, sample_rate=16000, max_length=64600, name="dataset"):
        self.metadata = metadata
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.name = name
        self._analyze_dataset()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        try:
            waveform, sr = torchaudio.load(row['filepath'])

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)

            if waveform.shape[1] < self.max_length:
                pad_length = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            else:
                waveform = waveform[:, :self.max_length]

            return waveform.squeeze(0), torch.tensor(row['label'], dtype=torch.float32)
        except Exception as e:
            print(f"\nError loading {row['filepath']}: {str(e)}")
            return torch.zeros(self.max_length), torch.tensor(-1, dtype=torch.float32)

    def _analyze_dataset(self):
        print(f"\n{'='*50}")
        print(f"Initializing {self.name} dataset")
        print(f"{'='*50}")
        print(f"Total samples: {len(self.metadata)}")
        print(f"Real/Fake ratio: {sum(self.metadata['label']==0)}/{sum(self.metadata['label']==1)}")

# Create datasets and data loaders
train_dataset = AudioDataset(train_df, name="Training")
val_dataset = AudioDataset(val_df, name="Validation")
test_dataset = AudioDataset(test_df, name="Test")

batch_size = 36
num_workers = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory = False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Skip batches if resuming mid-epoch
if 'start_batch' in locals():
    # Create iterator and skip batches
    train_iterator = iter(train_loader)
    for _ in range(start_batch):
        next(train_iterator)
else:
    train_iterator = None
    
print(f"Using {num_workers} DataLoader workers on {os.cpu_count()} CPU cores.")

# # Model Implementation
import zipfile

def unzip_dataset(zip_path, extract_to):
    try:
        if not os.path.exists(zip_path):
            print(f"Zip file not found at {zip_path}")
            return False
            
        print(f"Unzipping {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzip completed successfully!")
        return True
    except Exception as e:
        print(f"Error unzipping file: {e}")
        return False

zip_file_path = "models/models--microsoft--wavlm-base.zip"  # Change if your zip has different name

if os.path.exists(zip_file_path):
    unzip_success = unzip_dataset(zip_file_path, "models")
else:
    print("No zip file found, assuming dataset is already extracted")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, WavLMConfig
from typing import Optional, Tuple

class WavLMFeatureExtractor(nn.Module):
    def __init__(self, model_name: str = "models/wavlm-base", freeze: bool = True):
        """
        WavLM feature extractor with optional fine-tuning
        
        Args:
            model_name: Path to local pretrained WavLM model
            freeze: Whether to freeze WavLM parameters
        """
        super().__init__()
        self.config = WavLMConfig.from_pretrained(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name)

        if freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False
                
        self.sample_rate = 16000  # WavLM's expected sample rate
        self.output_dim = self.config.hidden_size

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: Input audio tensor of shape (batch, seq_len) or (batch, 1, seq_len)
        Returns:
            features: Extracted features of shape (batch, seq_len, hidden_size)
        """
        # Input validation and reshaping
        if waveforms.dim() == 1:
            waveforms = waveforms.unsqueeze(0)
        elif waveforms.dim() == 3:
            waveforms = waveforms.squeeze(1)
            
        # Normalize waveform to [-1, 1] if not already
        if waveforms.abs().max() > 1:
            waveforms = waveforms / (waveforms.abs().max() + 1e-8)
            
        outputs = self.wavlm(waveforms)
        return outputs.last_hidden_state

class AASIST(nn.Module):
    def __init__(self, input_dim: int = 768, num_heads: int = 4, dropout: float = 0.3):
        """
        AASIST model for audio spoofing detection
        
        Args:
            input_dim: Dimension of input features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        # Spectro-temporal processing
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
        Returns:
            predictions: Output scores of shape (batch,)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        
        # Global average pooling
        x = self.pool(x).squeeze(2)
        
        # Self-attention (expects seq_len, batch, channels)
        x = x.unsqueeze(0)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = x.mean(dim=0)    # Average over sequence
        
        return self.classifier(x).squeeze(1)

class WavLM_AASIST_Model(nn.Module):
    def __init__(self, wavlm_model: str = "microsoft/wavlm-base", freeze_wavlm: bool = True):
        """
        Combined WavLM + AASIST model for audio deepfake detection
        
        Args:
            wavlm_model: Name of pretrained WavLM model
            freeze_wavlm: Whether to freeze WavLM parameters
        """
        super().__init__()
        self.feature_extractor = WavLMFeatureExtractor(wavlm_model, freeze_wavlm)
        self.aasist = AASIST(input_dim=self.feature_extractor.output_dim)
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveforms: Input audio tensor of shape (batch, seq_len) or (batch, 1, seq_len)
        Returns:
            predictions: Output scores of shape (batch,)
        """
        features = self.feature_extractor(waveforms)
        return self.aasist(features)
    
    def get_feature_dim(self) -> int:
        """Returns the dimension of the extracted features"""
        return self.feature_extractor.output_dim

# Ignore the tqdm warning (optional)
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="tqdm.auto")

# # Training
# ## Training Setup

# ===== NEW: Checkpoint Setup =====
checkpoint_frequency = 500  # Save checkpoint every 500 batches
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize training variables
start_epoch = 0
best_val_loss = float('inf')
train_losses, val_losses = [], []
train_accs, val_accs = [], []

# Resume if checkpoint exists
last_checkpoint_path = f"{CHECKPOINT_DIR}/last_checkpoint.pth"
if os.path.exists(last_checkpoint_path):
    print("Resuming training from checkpoint...")
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore early stopping state
    es_state = checkpoint['early_stopping_state']
    early_stopping.counter = es_state['counter']
    early_stopping.best_score = es_state['best_score']
    early_stopping.early_stop = es_state['early_stop']
    
    # Restore training state
    start_epoch = checkpoint['epoch'] + 1
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accs = checkpoint['train_accs']
    val_accs = checkpoint['val_accs']
    best_val_loss = checkpoint['best_val_loss']
    print(f"Resumed from epoch {start_epoch}")
# ===== END NEW =====

def save_checkpoint(epoch, batch_idx, model, optimizer, scheduler, early_stopping, train_losses, val_losses, train_accs, val_accs, best_val_loss):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping_state': {
            'counter': early_stopping.counter,
            'best_score': early_stopping.best_score,
            'early_stop': early_stopping.early_stop
        },
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_val_loss': best_val_loss
    }
    
    filename = f"checkpoint_epoch_{epoch}_batch_{batch_idx}.pth"
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, filename))
    torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth"))  # Always save last
    print(f"Saved checkpoint for epoch {epoch+1}, batch {batch_idx+1}")

if os.path.exists(last_checkpoint_path):
    print("Resuming training from checkpoint...")
    checkpoint = torch.load(last_checkpoint_path, map_location=device)
    
    # Restore training position
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint.get('batch_idx', 0) + 1  # Start from next batch
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Restore other states
    es_state = checkpoint['early_stopping_state']
    early_stopping.counter = es_state['counter']
    early_stopping.best_score = es_state['best_score']
    early_stopping.early_stop = es_state['early_stop']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    train_accs = checkpoint['train_accs']
    val_accs = checkpoint['val_accs']
    best_val_loss = checkpoint['best_val_loss']
    
    print(f"Resumed from epoch {start_epoch+1}, batch {start_batch}")

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print(f"Initializeing model")
model = WavLM_AASIST_Model(
    wavlm_model="models/wavlm-base",  # Point to your local model directory
    freeze_wavlm=True
).to(device)
print(f"Criterion")
criterion = nn.BCELoss()
print(f"Optimizer")
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
print(f"Scheduler")
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

early_stopping = EarlyStopping(patience=5)


# ## Training Loop
import sys
import time
import torch
from tqdm import tqdm

# === Redirect output to a log file ===
log_file = open("training_log.txt", "w")
sys.stdout = log_file
sys.stderr = log_file

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use position=0 and leave=True for the main bar
    with tqdm(dataloader, desc="Training", position=0, leave=True) as pbar:
        for batch_idx, (waveforms, labels) in enumerate(pbar):
            waveforms, labels = waveforms.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(waveforms)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Metrics calculation
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Save intra-epoch checkpoint
            if (batch_idx + 1) % checkpoint_frequency == 0:
                save_checkpoint(
                    epoch=epoch,
                    batch_idx=batch_idx,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    early_stopping=early_stopping,
                    train_losses=train_losses,
                    val_losses=val_losses,
                    train_accs=train_accs,
                    val_accs=val_accs,
                    best_val_loss=best_val_loss
                )
                
            # Update the progress bar description with current metrics
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': correct/total
            })
            
            if epoch == start_epoch and batch_idx == 0:
                start_batch = 0

        # End of epoch processing
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
            
    return running_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Use position=0 and leave=True for the main bar
        with tqdm(dataloader, desc="Validating", position=0, leave=True, file=sys.stdout) as pbar:
            for i, (waveforms, labels) in enumerate(pbar):
                waveforms, labels = waveforms.to(device), labels.to(device)
                outputs = model(waveforms)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                # Update the progress bar description with current metrics
                pbar.set_postfix({
                    'loss': running_loss/(i+1),
                    'acc': correct/total
                })
                
    return running_loss / len(dataloader), correct / total

# Training loop
num_epochs = 20
train_losses, val_losses = [], []
train_accs, val_accs = [], []

waveforms, labels = next(iter(train_loader))
waveforms, labels = waveforms.to(device), labels.to(device)

start = time.time()
for _ in range(10):
    optimizer.zero_grad()
    outputs = model(waveforms)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
print(f"Time for 10 training steps: {time.time() - start:.2f} sec")

# Modified training loop
for epoch in range(start_epoch, num_epochs):
    loader = train_iterator if train_iterator else train_loader
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # Append metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # Update best validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
    
    # ===== NEW: Save checkpoint =====
    save_checkpoint(epoch)
    # ===== END NEW =====
    
    scheduler.step(val_loss)
    early_stopping(val_loss)
    
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
    
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break
    
    train_iterator = None
    
log_file.close()
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Save Model
torch.save(model.state_dict(), 'audio_deepfake_model.pth')

# # Evaluation and Visualization
# ## Load Model

# # Load saved model
# print("\nLoading model")
# model = WavLM_AASIST_Model(
#     wavlm_model="models/wavlm-base",  # Point to your local model directory
#     freeze_wavlm=True).to(device)
# model.load_state_dict(torch.load('audio_deepfake_model.pth'))
# print("\nModel successfully loaded")

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for waveforms, labels in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    preds = [1 if x > 0.5 else 0 for x in all_outputs]
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return {
        'accuracy': accuracy_score(all_labels, preds),
        'precision': precision_score(all_labels, preds),
        'recall': recall_score(all_labels, preds),
        'f1': f1_score(all_labels, preds),
        'auc': roc_auc_score(all_labels, all_outputs),
        'eer': eer
    }

# Evaluate
print("\nEvaluating")
test_metrics = evaluate_model(model, test_loader, device)
print("\nTest Metrics:")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# Plot training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train')
plt.plot(val_accs, label='Validation')
plt.title('Accuracy')
plt.legend()
plt.show()


# ## Confusion Matrices
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for waveforms, labels in tqdm(dataloader, desc="Generating predictions"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms)
            preds = (outputs > 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], 
                yticklabels=['Real', 'Fake'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return cm

# Plot confusion matrices for train, validation and test sets
print("\nTraining Set Confusion Matrix:")
train_cm = plot_confusion_matrix(model, train_loader, device)

print("\nValidation Set Confusion Matrix:")
val_cm = plot_confusion_matrix(model, val_loader, device)

print("\nTest Set Confusion Matrix:")
test_cm = plot_confusion_matrix(model, test_loader, device)

# Print detailed metrics from confusion matrix
def print_metrics_from_cm(cm, set_name):
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{set_name} Metrics from Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

print_metrics_from_cm(train_cm, "Training")
print_metrics_from_cm(val_cm, "Validation")
print_metrics_from_cm(test_cm, "Test")


# # Save Model

torch.save(model.state_dict(), 'audio_deepfake_model.pth')

# To load later:
# model = WavLM_AASIST_Model().to(device)
# model.load_state_dict(torch.load('audio_deepfake_model.pth'))


# # For later
# 1. First run all the cell with model class definitions
# (WavLMFeatureExtractor, AASIST, WavLM_AASIST_Model)

# 2. Then run:
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Initialize and load model
model = WavLM_AASIST_Model().to(device)
model.load_state_dict(torch.load('audio_deepfake_model.pth', map_location=device))
model.eval()

# 4. Now you can use it for predictions
def predict_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    # Add any preprocessing you used during training
    waveform = waveform.to(device)
    with torch.no_grad():
        output = model(waveform)
    return "Fake" if output > 0.5 else "Real", float(output)

# Example usage:
# prediction, confidence = predict_audio("test_audio.wav")

