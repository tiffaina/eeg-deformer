"""
DREAMER Dataset Loader for EEGDeformer
Loads and preprocesses DREAMER dataset for emotion recognition
"""

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DREAMERDataset(Dataset):
    """
    DREAMER Dataset for EEG-based emotion recognition
    
    Parameters:
    -----------
    data_path : str
        Path to DREAMER.mat file
    task : str
        Classification task: 'valence', 'arousal', or 'dominance'
    segment_length : int
        Length of each EEG segment in samples (default: 384 for 3 seconds at 128 Hz)
    overlap : float
        Overlap ratio between segments (default: 0.5)
    split : str
        'train' or 'test'
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
    normalize : bool
        Whether to apply z-score normalization
    binary_classification : bool
        If True, converts ratings to binary (high/low) based on threshold
    threshold : int
        Threshold for binary classification (default: 3 for 5-point scale)
    """
    
    def __init__(self, data_path, task='valence', segment_length=384, overlap=0.5,
                 split='train', test_size=0.2, random_state=42, normalize=True,
                 binary_classification=True, threshold=3):
        
        self.data_path = data_path
        self.task = task
        self.segment_length = segment_length
        self.overlap = overlap
        self.normalize = normalize
        self.binary_classification = binary_classification
        self.threshold = threshold
        
        # Load data
        self.data, self.labels = self._load_and_process_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, 
            random_state=random_state, stratify=self.labels
        )
        
        if split == 'train':
            self.data = X_train
            self.labels = y_train
        else:
            self.data = X_test
            self.labels = y_test
            
        # Normalize if requested
        if self.normalize:
            self._normalize_data()
    
    def _load_and_process_data(self):
        """Load DREAMER.mat and process into segments"""
        print(f"Loading DREAMER dataset from {self.data_path}...")
        
        # Load MATLAB file
        mat_data = sio.loadmat(self.data_path)
        dreamer = mat_data['DREAMER']
        
        all_segments = []
        all_labels = []
        
        # Get data from the structure
        data_field = dreamer['Data'][0, 0]
        num_subjects = data_field.shape[1]
        
        print(f"Processing {num_subjects} subjects...")
        
        for subject_idx in range(num_subjects):
            subject_data = data_field[0, subject_idx]
            
            # Get EEG data and scores
            eeg_struct = subject_data['EEG'][0, 0]
            
            # Get the appropriate score based on task
            if self.task == 'valence':
                scores = subject_data['ScoreValence'][0, 0].flatten()
            elif self.task == 'arousal':
                scores = subject_data['ScoreArousal'][0, 0].flatten()
            elif self.task == 'dominance':
                scores = subject_data['ScoreDominance'][0, 0].flatten()
            else:
                raise ValueError(f"Unknown task: {self.task}")
            
            # Process each video (18 videos)
            stimuli_data = eeg_struct['stimuli'][0, 0]
            num_videos = stimuli_data.shape[0]
            
            for video_idx in range(num_videos):
                # Get EEG recording for this video
                eeg_recording = stimuli_data[video_idx, 0]  # Shape: (samples, 14 channels)
                
                if eeg_recording.size == 0:
                    continue
                
                # Get corresponding label
                label = scores[video_idx]
                
                # Convert to binary if requested
                if self.binary_classification:
                    label = 1 if label > self.threshold else 0
                else:
                    label = int(label) - 1  # Convert to 0-indexed
                
                # Segment the EEG data
                segments = self._segment_signal(eeg_recording)
                
                for segment in segments:
                    all_segments.append(segment)
                    all_labels.append(label)
        
        print(f"Generated {len(all_segments)} segments")
        print(f"Label distribution: {np.bincount(all_labels)}")
        
        return np.array(all_segments), np.array(all_labels)
    
    def _segment_signal(self, signal):
        """
        Segment EEG signal into fixed-length windows
        
        Parameters:
        -----------
        signal : ndarray
            EEG signal of shape (samples, channels)
            
        Returns:
        --------
        segments : list
            List of segmented signals
        """
        num_samples = signal.shape[0]
        step_size = int(self.segment_length * (1 - self.overlap))
        
        segments = []
        start = 0
        
        while start + self.segment_length <= num_samples:
            segment = signal[start:start + self.segment_length, :]
            # Transpose to (channels, samples) format for PyTorch
            segment = segment.T
            segments.append(segment)
            start += step_size
        
        return segments
    
    def _normalize_data(self):
        """Apply z-score normalization across all segments"""
        # Reshape for normalization: (n_segments * n_channels, n_samples)
        original_shape = self.data.shape
        reshaped = self.data.reshape(-1, self.data.shape[-1])
        
        # Normalize
        scaler = StandardScaler()
        normalized = scaler.fit_transform(reshaped.T).T
        
        # Reshape back
        self.data = normalized.reshape(original_shape)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
        --------
        data : torch.Tensor
            EEG segment of shape (channels, samples)
        label : torch.Tensor
            Emotion label
        """
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return data, label


def get_dreamer_dataloaders(data_path, task='valence', segment_length=384, 
                            batch_size=32, num_workers=4, **kwargs):
    """
    Create train and test dataloaders for DREAMER dataset
    
    Parameters:
    -----------
    data_path : str
        Path to DREAMER.mat file
    task : str
        Classification task: 'valence', 'arousal', or 'dominance'
    segment_length : int
        Length of each EEG segment in samples
    batch_size : int
        Batch size for dataloaders
    num_workers : int
        Number of workers for data loading
    **kwargs : 
        Additional arguments for DREAMERDataset
        
    Returns:
    --------
    train_loader : DataLoader
        Training data loader
    test_loader : DataLoader
        Test data loader
    """
    
    train_dataset = DREAMERDataset(
        data_path=data_path,
        task=task,
        segment_length=segment_length,
        split='train',
        **kwargs
    )
    
    test_dataset = DREAMERDataset(
        data_path=data_path,
        task=task,
        segment_length=segment_length,
        split='test',
        **kwargs
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test the dataset loader
    data_path = "path/to/DREAMER.mat"
    
    try:
        train_loader, test_loader = get_dreamer_dataloaders(
            data_path=data_path,
            task='valence',
            segment_length=384,
            batch_size=32
        )
        
        print(f"\nTrain loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Test a batch
        for data, labels in train_loader:
            print(f"\nBatch shape: {data.shape}")  # Should be (batch_size, 14, 384)
            print(f"Labels shape: {labels.shape}")
            print(f"Labels: {labels}")
            break
            
    except Exception as e:
        print(f"Error: {e}")
        print("Please provide the correct path to DREAMER.mat file")