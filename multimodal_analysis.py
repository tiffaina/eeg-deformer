"""
Multi-Modal Emotion Recognition: EEG + ECG
DREAMER contains both EEG and ECG - combining them often improves performance
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

sys.path.append('./EEG-Deformer')
from models.EEGDeformer import Deformer


class MultiModalDataset(Dataset):
    """Dataset that includes both EEG and ECG signals"""
    
    def __init__(self, data_path, task='valence', segment_length_eeg=384,
                 segment_length_ecg=768, overlap=0.5, split='train',
                 use_ecg=True, fusion_method='late'):
        
        self.data_path = data_path
        self.task = task
        self.segment_length_eeg = segment_length_eeg  # 3s at 128 Hz
        self.segment_length_ecg = segment_length_ecg  # 3s at 256 Hz
        self.overlap = overlap
        self.use_ecg = use_ecg
        self.fusion_method = fusion_method  # 'early', 'late', 'feature'
        
        # Load data
        self.eeg_data, self.ecg_data, self.labels = self._load_data()
        
        # Split
        if self.use_ecg:
            X_train_eeg, X_test_eeg, X_train_ecg, X_test_ecg, y_train, y_test = \
                train_test_split(self.eeg_data, self.ecg_data, self.labels,
                               test_size=0.2, random_state=42, stratify=self.labels)
            
            if split == 'train':
                self.eeg_data = X_train_eeg
                self.ecg_data = X_train_ecg
                self.labels = y_train
            else:
                self.eeg_data = X_test_eeg
                self.ecg_data = X_test_ecg
                self.labels = y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.eeg_data, self.labels, test_size=0.2,
                random_state=42, stratify=self.labels
            )
            
            if split == 'train':
                self.eeg_data = X_train
                self.labels = y_train
            else:
                self.eeg_data = X_test
                self.labels = y_test
        
        # Normalize
        self._normalize()
    
    def _load_data(self):
        """Load both EEG and ECG data"""
        print(f"Loading multi-modal DREAMER data...")
        
        mat_data = sio.loadmat(self.data_path)
        dreamer = mat_data['DREAMER']
        data_field = dreamer['Data'][0, 0]
        num_subjects = data_field.shape[1]
        
        all_eeg_segments = []
        all_ecg_segments = []
        all_labels = []
        
        for subject_idx in range(num_subjects):
            subject_data = data_field[0, subject_idx]
            
            eeg_struct = subject_data['EEG'][0, 0]
            ecg_struct = subject_data['ECG'][0, 0]
            
            # Get scores
            if self.task == 'valence':
                scores = subject_data['ScoreValence'][0, 0].flatten()
            elif self.task == 'arousal':
                scores = subject_data['ScoreArousal'][0, 0].flatten()
            elif self.task == 'dominance':
                scores = subject_data['ScoreDominance'][0, 0].flatten()
            
            # Get stimuli
            eeg_stimuli = eeg_struct['stimuli'][0, 0]
            ecg_stimuli = ecg_struct['stimuli'][0, 0]
            
            for video_idx in range(eeg_stimuli.shape[0]):
                eeg_recording = eeg_stimuli[video_idx, 0]
                ecg_recording = ecg_stimuli[video_idx, 0]
                
                if eeg_recording.size == 0 or (self.use_ecg and ecg_recording.size == 0):
                    continue
                
                label = 1 if scores[video_idx] > 3 else 0
                
                # Segment EEG
                eeg_segments = self._segment_signal(
                    eeg_recording, self.segment_length_eeg
                )
                
                # Segment ECG if using
                if self.use_ecg:
                    ecg_segments = self._segment_signal(
                        ecg_recording, self.segment_length_ecg
                    )
                    
                    # Ensure same number of segments
                    min_segments = min(len(eeg_segments), len(ecg_segments))
                    eeg_segments = eeg_segments[:min_segments]
                    ecg_segments = ecg_segments[:min_segments]
                    
                    all_ecg_segments.extend(ecg_segments)
                
                all_eeg_segments.extend(eeg_segments)
                all_labels.extend([label] * len(eeg_segments))
        
        print(f"Generated {len(all_eeg_segments)} segments")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        eeg_data = np.array(all_eeg_segments)
        ecg_data = np.array(all_ecg_segments) if self.use_ecg else None
        labels = np.array(all_labels)
        
        return eeg_data, ecg_data, labels
    
    def _segment_signal(self, signal, segment_length):
        """Segment signal"""
        num_samples = signal.shape[0]
        step_size = int(segment_length * (1 - self.overlap))
        
        segments = []
        start = 0
        
        while start + segment_length <= num_samples:
            segment = signal[start:start + segment_length, :]
            segment = segment.T
            segments.append(segment)
            start += step_size
        
        return segments
    
    def _normalize(self):
        """Normalize data"""
        # Normalize EEG
        original_shape = self.eeg_data.shape
        reshaped = self.eeg_data.reshape(-1, self.eeg_data.shape[-1])
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped.T).T
        self.eeg_data = normalized.reshape(original_shape)
        
        # Normalize ECG
        if self.use_ecg:
            original_shape = self.ecg_data.shape
            reshaped = self.ecg_data.reshape(-1, self.ecg_data.shape[-1])
            scaler = StandardScaler()
            normalized = scaler.fit_transform(reshaped.T).T
            self.ecg_data = normalized.reshape(original_shape)
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg = torch.FloatTensor(self.eeg_data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        if self.use_ecg:
            ecg = torch.FloatTensor(self.ecg_data[idx])
            return eeg, ecg, label
        else:
            return eeg, label


class MultiModalModel(nn.Module):
    """
    Multi-modal model combining EEG and ECG
    Supports different fusion strategies
    """
    
    def __init__(self, fusion='late', num_classes=2):
        super().__init__()
        self.fusion = fusion
        
        # EEG branch (14 channels, 384 samples)
        self.eeg_model = Deformer(
            num_chan=14,
            num_time=384,
            temporal_kernel=13,
            num_kernel=64,
            num_classes=num_classes if fusion == 'late' else 128,
            depth=4,
            heads=16,
            mlp_dim=16,
            dim_head=16,
            dropout=0.5
        )
        
        # ECG branch (2 channels, 768 samples)
        self.ecg_model = Deformer(
            num_chan=2,
            num_time=768,
            temporal_kernel=25,  # Odd[0.1 * 256]
            num_kernel=32,
            num_classes=num_classes if fusion == 'late' else 128,
            depth=3,
            heads=8,
            mlp_dim=16,
            dim_head=16,
            dropout=0.5
        )
        
        # Fusion layer
        if fusion == 'late':
            # Late fusion: average predictions
            pass
        elif fusion == 'feature':
            # Feature fusion: concatenate features before classification
            self.fusion_layer = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, eeg, ecg=None):
        if ecg is None:
            # EEG only
            return self.eeg_model(eeg)
        
        if self.fusion == 'late':
            # Get predictions from both branches
            eeg_out = self.eeg_model(eeg)
            ecg_out = self.ecg_model(ecg)
            
            # Average (or you could learn weights)
            return (eeg_out + ecg_out) / 2
        
        elif self.fusion == 'feature':
            # Get features from both branches
            eeg_features = self.eeg_model(eeg)
            ecg_features = self.ecg_model(ecg)
            
            # Concatenate and classify
            combined = torch.cat([eeg_features, ecg_features], dim=1)
            return self.fusion_layer(combined)


def compare_modalities(data_path, task='valence'):
    """
    Compare EEG-only, ECG-only, and multi-modal approaches
    """
    
    print("="*80)
    print("MULTI-MODAL COMPARISON")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    # 1. EEG Only
    print("\n1. Training EEG-Only Model...")
    # Use your existing train_dreamer.py results
    
    # 2. Multi-Modal (Late Fusion)
    print("\n2. Training Multi-Modal Model (Late Fusion)...")
    
    train_dataset = MultiModalDataset(
        data_path, task=task, split='train', use_ecg=True, fusion_method='late'
    )
    test_dataset = MultiModalDataset(
        data_path, task=task, split='test', use_ecg=True, fusion_method='late'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = MultiModalModel(fusion='late', num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    best_acc = 0
    for epoch in range(50):
        # Train
        model.train()
        for eeg, ecg, labels in train_loader:
            eeg, ecg, labels = eeg.to(device), ecg.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(eeg, ecg)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for eeg, ecg, labels in test_loader:
                eeg, ecg, labels = eeg.to(device), ecg.to(device), labels.to(device)
                outputs = model(eeg, ecg)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        acc = 100. * correct / total
        if acc > best_acc:
            best_acc = acc
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Test Acc = {acc:.2f}%")
    
    results['multi_modal'] = best_acc
    print(f"\nBest Multi-Modal Accuracy: {best_acc:.2f}%")
    
    return results


if __name__ == "__main__":
    data_path = 'DREAMER.mat'
    results = compare_modalities(data_path, task='valence')