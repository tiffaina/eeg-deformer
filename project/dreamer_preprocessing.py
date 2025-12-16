"""
DREAMER Database Preprocessing for EEG-Deformer
This script loads and preprocesses the DREAMER dataset for use with EEG-Deformer model
"""

import numpy as np
import scipy.io as sio
from scipy import signal
import os
from sklearn.preprocessing import StandardScaler
import pickle

class DREAMERPreprocessor:
    def __init__(self, 
                 dreamer_file_path='DREAMER.mat',
                 target_fs=128,
                 window_size=4,
                 overlap=0.5,
                 baseline_length=1):
        """
        Initialize DREAMER preprocessor
        
        Args:
            dreamer_file_path: Path to DREAMER.mat file
            target_fs: Target sampling rate (default: 128 Hz)
            window_size: Window size in seconds (default: 4 seconds)
            overlap: Overlap ratio between windows (default: 0.5)
            baseline_length: Baseline period length in seconds (default: 1 second)
        """
        self.dreamer_file_path = dreamer_file_path
        self.target_fs = target_fs
        self.window_size = window_size
        self.overlap = overlap
        self.baseline_length = baseline_length
        
        # DREAMER electrode mapping
        # The dataset uses 14 EEG channels
        self.eeg_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 
                             'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        print(f"Initialized DREAMER Preprocessor")
        print(f"Target sampling rate: {target_fs} Hz")
        print(f"Window size: {window_size} seconds")
        print(f"Overlap: {overlap}")
        
    def load_dreamer_data(self):
        """Load DREAMER.mat file"""
        print(f"\nLoading DREAMER data from {self.dreamer_file_path}...")
        
        mat_data = sio.loadmat(self.dreamer_file_path)
        dreamer = mat_data['DREAMER']
        
        print(f"Data loaded successfully!")
        print(f"Number of subjects: {dreamer['Data'][0, 0].shape[1]}")
        
        return dreamer
    
    def extract_subject_data(self, dreamer, subject_idx):
        """
        Extract data for a specific subject
        
        Args:
            dreamer: Loaded DREAMER structure
            subject_idx: Subject index (0-based)
            
        Returns:
            subject_data: Dictionary containing subject's EEG data and labels
        """
        data_cell = dreamer['Data'][0, 0]
        subject = data_cell[0, subject_idx]
        
        # Extract EEG data and baseline
        eeg_data = subject['EEG'][0, 0]  # Shape: (num_videos, num_channels, num_samples)
        baseline = eeg_data['baseline'][0, 0]  # Baseline data
        stimuli = eeg_data['stimuli'][0, 0]    # Stimuli data
        
        # Extract labels
        score_valence = subject['ScoreValence'][0, 0].flatten()
        score_arousal = subject['ScoreArousal'][0, 0].flatten()
        score_dominance = subject['ScoreDominance'][0, 0].flatten()
        
        # Get age and gender
        age = subject['Age'][0, 0][0, 0] if subject['Age'][0, 0].size > 0 else None
        gender = subject['Gender'][0, 0][0] if subject['Gender'][0, 0].size > 0 else None
        
        subject_data = {
            'subject_idx': subject_idx,
            'age': age,
            'gender': gender,
            'baseline': baseline,
            'stimuli': stimuli,
            'valence': score_valence,
            'arousal': score_arousal,
            'dominance': score_dominance
        }
        
        return subject_data
    
    def resample_signal(self, data, original_fs):
        """
        Resample signal to target sampling rate
        
        Args:
            data: Signal data (channels, samples)
            original_fs: Original sampling rate
            
        Returns:
            resampled_data: Resampled signal
        """
        if original_fs == self.target_fs:
            return data
        
        num_samples = int(data.shape[1] * self.target_fs / original_fs)
        resampled_data = signal.resample(data, num_samples, axis=1)
        
        return resampled_data
    
    def apply_baseline_correction(self, trial_data, baseline_data, original_fs):
        """
        Apply baseline correction to trial data
        
        Args:
            trial_data: Trial EEG data (channels, samples)
            baseline_data: Baseline EEG data (channels, samples)
            original_fs: Original sampling rate
            
        Returns:
            corrected_data: Baseline-corrected data
        """
        # Calculate baseline mean from first baseline_length seconds
        baseline_samples = int(self.baseline_length * original_fs)
        baseline_mean = np.mean(baseline_data[:, :baseline_samples], axis=1, keepdims=True)
        
        # Subtract baseline from trial data
        corrected_data = trial_data - baseline_mean
        
        return corrected_data
    
    def create_windows(self, data, fs):
        """
        Create sliding windows from continuous data
        
        Args:
            data: EEG data (channels, samples)
            fs: Sampling rate
            
        Returns:
            windows: List of windowed data (num_windows, channels, window_samples)
        """
        window_samples = int(self.window_size * fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        num_channels, num_samples = data.shape
        windows = []
        
        for start in range(0, num_samples - window_samples + 1, step_samples):
            end = start + window_samples
            window = data[:, start:end]
            windows.append(window)
        
        return np.array(windows)
    
    def binarize_labels(self, scores, threshold=3):
        """
        Binarize continuous scores (1-5 scale)
        
        Args:
            scores: Continuous scores
            threshold: Threshold for binarization (default: 3)
            
        Returns:
            binary_labels: Binary labels (0: low, 1: high)
        """
        return (scores > threshold).astype(int)
    
    def preprocess_all_subjects(self, output_dir='./data_processed/DREAMER', 
                                task='valence', original_fs=128):
        """
        Preprocess all subjects in DREAMER dataset
        
        Args:
            output_dir: Directory to save preprocessed data
            task: Task type ('valence', 'arousal', or 'dominance')
            original_fs: Original sampling rate in DREAMER (128 Hz for EEG)
            
        Returns:
            None (saves data to disk)
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing DREAMER dataset for {task.upper()} task")
        print(f"{'='*60}\n")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load DREAMER data
        dreamer = self.load_dreamer_data()
        
        num_subjects = dreamer['Data'][0, 0].shape[1]
        
        all_data = []
        all_labels = []
        subject_ids = []
        
        for subject_idx in range(num_subjects):
            print(f"\nProcessing Subject {subject_idx + 1}/{num_subjects}...")
            
            try:
                # Extract subject data
                subject_data = self.extract_subject_data(dreamer, subject_idx)
                
                baseline = subject_data['baseline']
                stimuli = subject_data['stimuli']
                
                # Get labels based on task
                if task == 'valence':
                    labels = subject_data['valence']
                elif task == 'arousal':
                    labels = subject_data['arousal']
                elif task == 'dominance':
                    labels = subject_data['dominance']
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                # Process each video/trial
                num_trials = stimuli.shape[0]
                
                for trial_idx in range(num_trials):
                    trial_data = stimuli[trial_idx, :, :]  # (channels, samples)
                    baseline_data = baseline[trial_idx, :, :]
                    trial_label = labels[trial_idx]
                    
                    # Apply baseline correction
                    corrected_data = self.apply_baseline_correction(
                        trial_data, baseline_data, original_fs
                    )
                    
                    # Resample if needed
                    resampled_data = self.resample_signal(corrected_data, original_fs)
                    
                    # Create windows
                    windows = self.create_windows(resampled_data, self.target_fs)
                    
                    # Add windows to dataset
                    for window in windows:
                        all_data.append(window)
                        all_labels.append(trial_label)
                        subject_ids.append(subject_idx)
                
                print(f"  Subject {subject_idx + 1} processed: {num_trials} trials")
                
            except Exception as e:
                print(f"  Error processing subject {subject_idx + 1}: {str(e)}")
                continue
        
        # Convert to numpy arrays
        all_data = np.array(all_data)  # (num_samples, channels, time_points)
        all_labels = np.array(all_labels)
        subject_ids = np.array(subject_ids)
        
        # Binarize labels
        binary_labels = self.binarize_labels(all_labels)
        
        print(f"\n{'='*60}")
        print(f"Preprocessing completed!")
        print(f"{'='*60}")
        print(f"Total samples: {all_data.shape[0]}")
        print(f"Data shape: {all_data.shape}")
        print(f"Labels shape: {binary_labels.shape}")
        print(f"Class distribution: Low={np.sum(binary_labels==0)}, High={np.sum(binary_labels==1)}")
        
        # Save preprocessed data
        output_data = {
            'data': all_data,
            'labels': binary_labels,
            'continuous_labels': all_labels,
            'subject_ids': subject_ids,
            'channel_names': self.eeg_channels,
            'sampling_rate': self.target_fs,
            'window_size': self.window_size,
            'task': task
        }
        
        output_file = os.path.join(output_dir, f'DREAMER_{task}_preprocessed.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        
        print(f"\nData saved to: {output_file}")
        
        return output_data


def main():
    """Main function to run preprocessing"""
    
    # Configuration
    DREAMER_FILE = 'DREAMER.mat'  # Update this path to your DREAMER.mat file
    OUTPUT_DIR = './data_processed/DREAMER'
    TASK = 'valence'  # Options: 'valence', 'arousal', 'dominance'
    
    # Initialize preprocessor
    preprocessor = DREAMERPreprocessor(
        dreamer_file_path=DREAMER_FILE,
        target_fs=128,
        window_size=4,
        overlap=0.5,
        baseline_length=1
    )
    
    # Preprocess data
    preprocessor.preprocess_all_subjects(
        output_dir=OUTPUT_DIR,
        task=TASK,
        original_fs=128  # DREAMER EEG sampling rate
    )
    
    print("\nPreprocessing complete!")


if __name__ == '__main__':
    main()