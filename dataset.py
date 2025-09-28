import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, PolarityInversion 

class AudioAugmentation:
    """Applies a set of diverse augmentations to the audio."""
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # p=0.5 means each augmentation has a 50% chance of being applied
        self.augmenter = Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
            # --- QUICK FIX ---
            # Replaced AddReverb with a more basic, backward-compatible augmentation.
            PolarityInversion(p=0.5),
        ])

    def __call__(self, waveform: torch.Tensor):
        # audiomentations expects a numpy array, not a tensor
        np_waveform = waveform.numpy()
        # The augmentation is applied
        augmented_waveform = self.augmenter(samples=np_waveform, sample_rate=self.sample_rate)
        # Convert back to a tensor
        return torch.from_numpy(augmented_waveform)
    
class UnifiedAudioDataset(Dataset):
    """
    Dataset to load audio files, combine them, and then create
    train, validation, and test partitions automatically.
    Labels: 1 for 'bonafide' (real), 0 for 'spoof' (fake).
    """
    # Class-level variable to store the split data so it's only done once
    _partitions = None

    def __init__(self, data_root: str, partition: str, max_len_sec: int = 5, test_size=0.2, val_size=0.1, random_state=42):
        self.data_root = data_root
        self.partition = partition
        self.max_len_sec = max_len_sec
        self.target_sr = 16000
        self.max_len_samples = self.target_sr * self.max_len_sec
        
        # Load and split data only once and store it in the class variable
        if UnifiedAudioDataset._partitions is None:
            print("No pre-existing data partitions found. Creating them now...")
            all_metadata = self._load_all_metadata()
            UnifiedAudioDataset._partitions = self._split_data(all_metadata, test_size, val_size, random_state)
            print("Partitions created successfully.")

        self.metadata = UnifiedAudioDataset._partitions[self.partition]
        print(f"Loaded {len(self.metadata)} samples for partition '{self.partition}'.")

        self.label_map = {'real': 1, 'fake': 0}
        self.augmenter = AudioAugmentation(self.target_sr) if partition == 'train' else None

    def _load_all_metadata(self):
        """Loads and concatenates metadata from all specified datasets."""
        all_meta = []
        EXPECTED_COLUMNS = ['file', 'label']

        for dataset_name in ["ASV", "FOR", "ITW"]:
            meta_path = os.path.join(self.data_root, dataset_name,f"{dataset_name}_meta.csv")
            dataset_path = os.path.join(self.data_root, dataset_name,f"{dataset_name}_dataset")

            if os.path.exists(meta_path):
                print(f"Found metadata file: {meta_path}")
                df = pd.read_csv(meta_path)
                
                missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Metadata file '{meta_path}' is missing: {missing_cols}")

                df['full_path'] = df['file'].apply(lambda x: os.path.join(dataset_path, x))
                all_meta.append(df)
            else:
                print(f"WARNING: Could not find metadata file: {meta_path}. Skipping this dataset.")
        
        if not all_meta:
            raise FileNotFoundError(f"No metadata files found in {self.data_root}. Searched for ASV_meta.csv, FOR_meta.csv, etc.")
        
        print("-" * 30) # Separator for clarity
        return pd.concat(all_meta, ignore_index=True)

    def _split_data(self, df, test_size, val_size, random_state):
        """Performs a stratified split into train, validation (dev), and test (eval)."""
        # First, split into training and a temporary set (test + validation)
        train_df, temp_df = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=random_state
        )

        # Calculate the proportion of the validation set relative to the temp set
        relative_val_size = val_size / (test_size)

        # Split the temporary set into validation and test sets
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1.0 - relative_val_size, # The rest goes to testing
            stratify=temp_df['label'],
            random_state=random_state
        )

        return {
            'train': train_df.reset_index(drop=True),
            'dev': val_df.reset_index(drop=True),
            'eval': test_df.reset_index(drop=True)
        }
        
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.loc[idx, 'full_path']
        label_str = self.metadata.loc[idx, 'label']
        label = torch.tensor(self.label_map[label_str], dtype=torch.float32)

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_len_samples), torch.tensor(0.0, dtype=torch.float32)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if self.augmenter:
            waveform = self.augmenter(waveform)

        current_len = waveform.shape[0]
        if current_len > self.max_len_samples:
            waveform = waveform[:self.max_len_samples]
        else:
            padding = self.max_len_samples - current_len
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform, label

    def get_class_weights(self):
        counts = self.metadata['label'].value_counts()
        weight_spoof = len(self.metadata) / (2 * counts.get('spoof', 1))
        weight_bonafide = len(self.metadata) / (2 * counts.get('bonafide', 1))
        return torch.tensor(weight_spoof / weight_bonafide)