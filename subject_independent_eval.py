"""
Subject-Independent Cross-Validation for DREAMER
This is the gold standard evaluation method for EEG emotion recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys
import scipy.io as sio
from sklearn.preprocessing import StandardScaler

sys.path.append('./EEG-Deformer')
from models.EEGDeformer import Deformer


class SubjectIndependentEvaluator:
    """
    Leave-One-Subject-Out Cross-Validation (LOSO-CV)
    This evaluates how well the model generalizes to completely new subjects
    """
    
    def __init__(self, data_path, task='valence', segment_length=384, config=None):
        self.data_path = data_path
        self.task = task
        self.segment_length = segment_length
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load and organize data by subject
        self.subject_data = self._load_data_by_subject()
        self.num_subjects = len(self.subject_data)
        
        print(f"Loaded {self.num_subjects} subjects for {task} task")
        
    def _default_config(self):
        return {
            'segment_length': self.segment_length,
            'overlap': 0.5,
            'normalize': True,
            'binary_classification': True,
            'threshold': 3,
            'batch_size': 32,
            'num_epochs': 30,  # Reduced for cross-validation
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'temporal_kernel': 13,
            'num_kernel': 64,
            'num_classes': 2,
            'depth': 4,
            'heads': 16,
            'mlp_dim': 16,
            'dim_head': 16,
            'dropout': 0.5,
        }
    
    def _load_data_by_subject(self):
        """Load data organized by subject for LOSO-CV"""
        print(f"Loading DREAMER data by subject...")
        
        mat_data = sio.loadmat(self.data_path)
        dreamer = mat_data['DREAMER']
        data_field = dreamer['Data'][0, 0]
        num_subjects = data_field.shape[1]
        
        subject_data = []
        
        for subject_idx in range(num_subjects):
            subject = data_field[0, subject_idx]
            eeg_struct = subject['EEG'][0, 0]
            
            # Get scores based on task
            if self.task == 'valence':
                scores = subject['ScoreValence'][0, 0].flatten()
            elif self.task == 'arousal':
                scores = subject['ScoreArousal'][0, 0].flatten()
            elif self.task == 'dominance':
                scores = subject['ScoreDominance'][0, 0].flatten()
            
            stimuli_data = eeg_struct['stimuli'][0, 0]
            
            segments = []
            labels = []
            
            for video_idx in range(stimuli_data.shape[0]):
                eeg_recording = stimuli_data[video_idx, 0]
                
                if eeg_recording.size == 0:
                    continue
                
                label = scores[video_idx]
                
                # Binary classification
                if self.config['binary_classification']:
                    label = 1 if label > self.config['threshold'] else 0
                else:
                    label = int(label) - 1
                
                # Segment the signal
                subject_segments = self._segment_signal(eeg_recording)
                segments.extend(subject_segments)
                labels.extend([label] * len(subject_segments))
            
            subject_data.append({
                'segments': np.array(segments),
                'labels': np.array(labels),
                'subject_id': subject_idx
            })
            
            print(f"  Subject {subject_idx+1}: {len(segments)} segments, "
                  f"labels: {np.bincount(labels)}")
        
        return subject_data
    
    def _segment_signal(self, signal):
        """Segment EEG signal into fixed-length windows"""
        num_samples = signal.shape[0]
        step_size = int(self.segment_length * (1 - self.config['overlap']))
        
        segments = []
        start = 0
        
        while start + self.segment_length <= num_samples:
            segment = signal[start:start + self.segment_length, :]
            segment = segment.T  # (channels, samples)
            segments.append(segment)
            start += step_size
        
        return segments
    
    def _normalize_data(self, train_data, test_data):
        """Apply z-score normalization using train statistics"""
        # Reshape for normalization
        train_shape = train_data.shape
        test_shape = test_data.shape
        
        train_reshaped = train_data.reshape(-1, train_data.shape[-1])
        test_reshaped = test_data.reshape(-1, test_data.shape[-1])
        
        # Fit scaler on train data only
        scaler = StandardScaler()
        train_normalized = scaler.fit_transform(train_reshaped.T).T
        test_normalized = scaler.transform(test_reshaped.T).T
        
        return (train_normalized.reshape(train_shape), 
                test_normalized.reshape(test_shape))
    
    def train_and_evaluate_fold(self, test_subject_idx):
        """Train on all subjects except one, test on that subject"""
        print(f"\n{'='*60}")
        print(f"Testing on Subject {test_subject_idx + 1}")
        print(f"{'='*60}")
        
        # Split data
        train_segments = []
        train_labels = []
        test_segments = []
        test_labels = []
        
        for idx, subject in enumerate(self.subject_data):
            if idx == test_subject_idx:
                test_segments = subject['segments']
                test_labels = subject['labels']
            else:
                train_segments.append(subject['segments'])
                train_labels.append(subject['labels'])
        
        # Concatenate training data from multiple subjects
        train_segments = np.concatenate(train_segments, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        print(f"Train: {len(train_segments)} segments from {self.num_subjects-1} subjects")
        print(f"Test: {len(test_segments)} segments from 1 subject")
        print(f"Train labels: {np.bincount(train_labels)}")
        print(f"Test labels: {np.bincount(test_labels)}")
        
        # Normalize
        if self.config['normalize']:
            train_segments, test_segments = self._normalize_data(
                train_segments, test_segments
            )
        
        # Convert to PyTorch tensors
        train_data = torch.FloatTensor(train_segments)
        train_labels_tensor = torch.LongTensor(train_labels)
        test_data = torch.FloatTensor(test_segments)
        test_labels_tensor = torch.LongTensor(test_labels)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(train_data, train_labels_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_labels_tensor)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], 
            shuffle=True, num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.config['batch_size'], 
            shuffle=False, num_workers=2
        )
        
        # Initialize model
        model = Deformer(
            num_chan=14,
            num_time=self.config['segment_length'],
            temporal_kernel=self.config['temporal_kernel'],
            num_kernel=self.config['num_kernel'],
            num_classes=self.config['num_classes'],
            depth=self.config['depth'],
            heads=self.config['heads'],
            mlp_dim=self.config['mlp_dim'],
            dim_head=self.config['dim_head'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        best_acc = 0.0
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Train
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for data, labels in train_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * correct / total
            
            # Validate
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    
                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            
            test_acc = 100. * correct / total
            
            if test_acc > best_acc:
                best_acc = test_acc
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['num_epochs']}: "
                      f"Train={train_acc:.2f}%, Test={test_acc:.2f}%")
        
        print(f"\nBest accuracy for Subject {test_subject_idx + 1}: {best_acc:.2f}%")
        
        return best_acc
    
    def run_cross_validation(self):
        """Run leave-one-subject-out cross-validation"""
        print(f"\n{'='*80}")
        print(f"SUBJECT-INDEPENDENT CROSS-VALIDATION - {self.task.upper()}")
        print(f"{'='*80}\n")
        
        all_accuracies = []
        
        for subject_idx in range(self.num_subjects):
            accuracy = self.train_and_evaluate_fold(subject_idx)
            all_accuracies.append(accuracy)
        
        # Results
        mean_acc = np.mean(all_accuracies)
        std_acc = np.std(all_accuracies)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS - {self.task.upper()}")
        print(f"{'='*80}")
        print(f"Individual Subject Accuracies:")
        for i, acc in enumerate(all_accuracies):
            print(f"  Subject {i+1:2d}: {acc:.2f}%")
        print(f"\nMean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"Min Accuracy:  {min(all_accuracies):.2f}%")
        print(f"Max Accuracy:  {max(all_accuracies):.2f}%")
        print(f"{'='*80}\n")
        
        # Save results
        results = {
            'task': self.task,
            'accuracies': all_accuracies,
            'mean': mean_acc,
            'std': std_acc,
            'min': min(all_accuracies),
            'max': max(all_accuracies)
        }
        
        import json
        with open(f'subject_independent_{self.task}.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        return results


def main():
    """Run subject-independent evaluation"""
    
    data_path = 'DREAMER.mat'  # Update this path
    
    # Configuration
    config = {
        'segment_length': 384,
        'batch_size': 32,
        'num_epochs': 30,  # Can increase to 50-100 for better results
        'learning_rate': 1e-3,
    }
    
    # Run for all three tasks
    tasks = ['valence', 'arousal', 'dominance']
    all_results = {}
    
    for task in tasks:
        print(f"\n\n{'#'*80}")
        print(f"# TASK: {task.upper()}")
        print(f"{'#'*80}\n")
        
        evaluator = SubjectIndependentEvaluator(
            data_path=data_path,
            task=task,
            segment_length=config['segment_length'],
            config=config
        )
        
        results = evaluator.run_cross_validation()
        all_results[task] = results
    
    # Summary of all tasks
    print(f"\n{'='*80}")
    print("SUMMARY - ALL TASKS")
    print(f"{'='*80}")
    for task, results in all_results.items():
        print(f"{task.capitalize():12} - {results['mean']:.2f}% ± {results['std']:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()