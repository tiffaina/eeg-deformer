"""
Inference script for trained EEGDeformer models on DREAMER dataset
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append('./EEG-Deformer')
from models.EEGDeformer import Deformer
from dreamer_dataset import get_dreamer_dataloaders


class EEGDeformerTester:
    """Testing and inference class for EEGDeformer on DREAMER"""
    
    def __init__(self, checkpoint_path, config=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if config is None:
            config = checkpoint['config']
        self.config = config
        
        # Initialize model
        self.model = Deformer(
            num_chan=14,
            num_time=config['segment_length'],
            temporal_kernel=config.get('temporal_kernel', 13),
            num_kernel=config.get('num_kernel', 64),
            num_classes=config['num_classes'],
            depth=config.get('depth', 4),
            heads=config.get('heads', 16),
            mlp_dim=config.get('mlp_dim', 16),
            dim_head=config.get('dim_head', 16),
            dropout=config.get('dropout', 0.5)
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Best accuracy: {checkpoint.get('best_acc', 'N/A')}")
        
    def predict(self, data_loader):
        """
        Make predictions on a dataset
        
        Returns:
        --------
        predictions : np.array
            Predicted labels
        probabilities : np.array
            Prediction probabilities
        true_labels : np.array
            True labels
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return (np.array(all_predictions), 
                np.array(all_probabilities), 
                np.array(all_labels))
    
    def evaluate(self, data_loader, verbose=True):
        """
        Evaluate model on a dataset
        
        Returns:
        --------
        metrics : dict
            Dictionary containing various evaluation metrics
        """
        predictions, probabilities, true_labels = self.predict(data_loader)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted')
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'probabilities': probabilities,
            'true_labels': true_labels
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Accuracy:  {accuracy*100:.2f}%")
            print(f"Precision: {precision*100:.2f}%")
            print(f"Recall:    {recall*100:.2f}%")
            print(f"F1-Score:  {f1*100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(true_labels, predictions, 
                                       target_names=['Low', 'High']))
            print("="*60 + "\n")
        
        return metrics
    
    def plot_confusion_matrix(self, true_labels, predictions, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Low', 'High'],
                   yticklabels=['Low', 'High'])
        plt.title(f'Confusion Matrix - {self.config["task"].capitalize()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def predict_single_sample(self, eeg_segment):
        """
        Predict emotion for a single EEG segment
        
        Parameters:
        -----------
        eeg_segment : np.array or torch.Tensor
            EEG segment of shape (14, segment_length) or (segment_length, 14)
            
        Returns:
        --------
        prediction : int
            Predicted class (0 or 1)
        probability : np.array
            Prediction probabilities for each class
        """
        self.model.eval()
        
        # Convert to torch tensor if needed
        if isinstance(eeg_segment, np.ndarray):
            eeg_segment = torch.FloatTensor(eeg_segment)
        
        # Ensure correct shape: (channels, samples)
        if eeg_segment.shape[0] != 14:
            eeg_segment = eeg_segment.T
        
        # Add batch dimension
        eeg_segment = eeg_segment.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(eeg_segment)
            probability = torch.softmax(output, dim=1).cpu().numpy()[0]
            prediction = output.argmax(dim=1).cpu().numpy()[0]
        
        return prediction, probability


def test_all_tasks():
    """Test models for all three tasks"""
    
    tasks = ['valence', 'arousal', 'dominance']
    data_path = 'DREAMER.mat'  # Update this
    
    results = {}
    
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Testing {task.upper()} model")
        print(f"{'='*60}")
        
        checkpoint_path = f'checkpoints/dreamer/best_model_{task}.pth'
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create test dataloader
        _, test_loader = get_dreamer_dataloaders(
            data_path=data_path,
            task=task,
            segment_length=config['segment_length'],
            batch_size=config['batch_size'],
            overlap=config.get('overlap', 0.5),
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42),
            normalize=config.get('normalize', True),
            binary_classification=config.get('binary_classification', True),
            threshold=config.get('threshold', 3)
        )
        
        # Initialize tester
        tester = EEGDeformerTester(checkpoint_path)
        
        # Evaluate
        metrics = tester.evaluate(test_loader)
        results[task] = metrics
        
        # Plot confusion matrix
        tester.plot_confusion_matrix(
            metrics['true_labels'],
            metrics['predictions'],
            save_path=f'confusion_matrix_{task}.png'
        )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY OF ALL TASKS")
    print("="*60)
    for task, metrics in results.items():
        print(f"{task.capitalize():12} - Accuracy: {metrics['accuracy']*100:.2f}%")
    print("="*60)
    
    return results


def main():
    """Main testing function"""
    
    # Example 1: Test a specific model
    checkpoint_path = 'checkpoints/dreamer/best_model_valence.pth'
    data_path = 'DREAMER.mat'
    
    if os.path.exists(checkpoint_path):
        # Load checkpoint to get config
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create test dataloader
        _, test_loader = get_dreamer_dataloaders(
            data_path=data_path,
            task=config['task'],
            segment_length=config['segment_length'],
            batch_size=config['batch_size']
        )
        
        # Test
        tester = EEGDeformerTester(checkpoint_path)
        metrics = tester.evaluate(test_loader)
        
        # Plot confusion matrix
        tester.plot_confusion_matrix(
            metrics['true_labels'],
            metrics['predictions'],
            save_path=f'confusion_matrix_{config["task"]}.png'
        )
    
    # Example 2: Test all tasks
    # results = test_all_tasks()


if __name__ == "__main__":
    main()