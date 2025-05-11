import numpy as np
import librosa
import random
from torch.utils.data import Dataset

class AugmentedSERDataset(Dataset):
    """Dataset for Speech Emotion Recognition with data augmentation"""
    
    def __init__(self, audio_paths, labels=None, sample_rate=16000, max_length=250000, augment=True):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.augment = augment
        
    def __len__(self):
        return len(self.audio_paths)
    
    def _apply_augmentation(self, waveform, sr):
        """Apply random augmentation to the audio waveform"""
        # Choose a random augmentation or no augmentation
        aug_type = random.choice(['pitch', 'stretch', 'noise', 'none'])
        
        if aug_type == 'pitch':
            # Pitch shift (up or down by 0-4 semitones)
            n_steps = random.uniform(-4, 4)
            return librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)
        
        elif aug_type == 'stretch':
            # Time stretch (0.8x to 1.2x)
            rate = random.uniform(0.8, 1.2)
            return librosa.effects.time_stretch(waveform, rate=rate)
        
        elif aug_type == 'noise':
            # Add random noise (SNR between 10-20dB)
            noise_level = random.uniform(0.005, 0.02)
            noise = np.random.randn(len(waveform))
            return waveform + noise_level * noise
        
        else:
            # No augmentation
            return waveform
    
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
        
        # Apply augmentation if enabled and during training
        if self.augment and self.labels is not None:
            waveform = self._apply_augmentation(waveform, sr)
            
        # Pad or truncate to fixed length
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        else:
            padding = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), 'constant')
            
        # Convert to tensor
        waveform = np.float32(waveform)
        
        if self.labels is not None:
            label = self.labels[idx]
            return waveform, label
        else:
            return waveform

def prepare_augmented_data(train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, 
                          batch_size=32, sample_rate=16000, max_length=250000):
    """Prepare data loaders with augmentation for training"""
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = AugmentedSERDataset(train_paths, train_labels, sample_rate, max_length, augment=True)
    val_dataset = AugmentedSERDataset(val_paths, val_labels, sample_rate, max_length, augment=False)
    test_dataset = AugmentedSERDataset(test_paths, test_labels, sample_rate, max_length, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader
