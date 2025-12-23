"""
Training script for EEGDeformer on DREAMER dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import your EEGDeformer model
# Assuming the model is in EEG-Deformer/models/EEGDeformer.py
import sys
sys.path.append('./EEG-Deformer')
from models.EEGDeformer import Deformer

from dreamer_dataset import get_dreamer_dataloaders


class EEGDeformerTrainer:
    """Trainer class for EEGDeformer on DREAMER dataset"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create dataloaders
        self.train_loader, self.test_loader = get_dreamer_dataloaders(
            data_path=config['data_path'],
            task=config['task'],
            segment_length=config['segment_length'],
            batch_size=config['batch_size'],
            overlap=config.get('overlap', 0.5),
            test_size=config.get('test_size', 0.2),
            random_state=config.get('random_state', 42),
            normalize=config.get('normalize', True),
            binary_classification=config.get('binary_classification', True),
            threshold=config.get('threshold', 3)
        )
        
        # Initialize model
        self.model = Deformer(
            num_chan=14,  # DREAMER has 14 EEG channels
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
        
        print(f"\nModel parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
        
        # Tracking
        self.best_acc = 0.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        # Create save directory
        self.save_dir = config.get('save_dir', 'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Validation')
            for data, labels in pbar:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Starting training for task: {self.config['task'].upper()}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print('-' * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            test_loss, test_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            # Print results
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ New best accuracy: {test_acc:.2f}%")
            
            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best test accuracy: {self.best_acc:.2f}%")
        print(f"{'='*60}\n")
        
        # Save final results
        self.save_results()
        
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'config': self.config,
            'history': self.history
        }
        
        if is_best:
            path = os.path.join(self.save_dir, f'best_model_{self.config["task"]}.pth')
            torch.save(checkpoint, path)
        else:
            path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, path)
    
    def save_results(self):
        """Save training results and config"""
        results = {
            'config': self.config,
            'best_accuracy': self.best_acc,
            'history': self.history,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        path = os.path.join(self.save_dir, f'results_{self.config["task"]}.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {path}")


def main():
    """Main training function"""
    
    # Configuration for DREAMER dataset
    config = {
        # Data parameters
        'data_path': 'DREAMER.mat',  # Update this path
        'task': 'valence',  # Options: 'valence', 'arousal', 'dominance'
        'segment_length': 384,  # 3 seconds at 128 Hz (DREAMER sampling rate)
        'overlap': 0.5,
        'test_size': 0.2,
        'random_state': 42,
        'normalize': True,
        'binary_classification': True,  # High/Low classification
        'threshold': 3,  # For binary classification
        
        # Model parameters (optimized for DREAMER)
        'num_classes': 2,  # Binary classification
        'temporal_kernel': 13,  # Odd[0.1 * 128] = 13
        'num_kernel': 64,
        'depth': 4,
        'heads': 16,
        'mlp_dim': 16,
        'dim_head': 16,
        'dropout': 0.5,
        
        # Training parameters
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 1e-3,
        'weight_decay': 0.01,
        'min_lr': 1e-6,
        
        # Other
        'save_dir': 'checkpoints/dreamer',
    }
    
    # Create trainer and start training
    trainer = EEGDeformerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()