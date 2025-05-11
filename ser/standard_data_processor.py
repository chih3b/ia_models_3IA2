import os
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TunisianSERDataset(Dataset):
    """Dataset for Tunisian Speech Emotion Recognition"""
    
    def __init__(self, audio_paths, labels=None, sample_rate=16000, max_length=250000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length  # ~15 seconds at 16kHz
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        
        # Load audio file
        try:
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a dummy waveform if loading fails
            waveform = np.zeros(self.max_length)
            sr = self.sample_rate
            
        # Pad or truncate to fixed length
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            padding = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
            
        # Convert to tensor
        waveform = torch.from_numpy(waveform).float()
        
        if self.labels is not None:
            label = self.labels[idx]
            return waveform, label
        else:
            return waveform

def prepare_standard_data(data_dir, test_size=0.2, val_size=0.1, batch_size=32, return_paths_and_labels=False):
    """
    Prepare data with standard train/validation/test splits as used in the research paper.
    This uses a random split rather than a speaker-independent approach.
    """
    
    # Get emotion categories from subdirectories
    emotion_categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotion_categories)}
    
    all_paths = []
    all_labels = []
    
    # Collect all audio paths and labels
    for emotion in emotion_categories:
        emotion_dir = os.path.join(data_dir, emotion)
        for audio_file in os.listdir(emotion_dir):
            if audio_file.endswith(('.wav', '.mp3', '.m4a')):
                audio_path = os.path.join(emotion_dir, audio_file)
                all_paths.append(audio_path)
                all_labels.append(emotion_to_idx[emotion])
    
    # First split into train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, 
        stratify=all_labels, random_state=42
    )
    
    # Then split train+val into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val size
        stratify=train_val_labels, random_state=42
    )
    
    print(f"Train set: {len(train_paths)} samples")
    print(f"Validation set: {len(val_paths)} samples")
    print(f"Test set: {len(test_paths)} samples")
    
    if return_paths_and_labels:
        return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, emotion_to_idx
    
    # Create datasets
    train_dataset = TunisianSERDataset(train_paths, train_labels)
    val_dataset = TunisianSERDataset(val_paths, val_labels)
    test_dataset = TunisianSERDataset(test_paths, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, emotion_to_idx
