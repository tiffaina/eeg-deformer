"""
Training Script for DREAMER Dataset with EEG-Deformer
This script trains the EEG-Deformer model on DREAMER dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import os
import argparse
import time
from tqdm import tqdm
import json

# Import custom modules
from dreamer_dataset import get_dreamer_dataloader


class EEGDeformer(nn.Module):
    """
    EEG-Deformer model architecture
    Adapted for DREAMER dataset (14 channels)
    """
    
    def __init__(self, num_chan=14, num_time=512, temporal_kernel=13, 
                 num_kernel=64, num_classes=2, depth=4, heads=16, 
                 mlp_dim=16, dim_head=16, dropout=0.5):
        super(EEGDeformer, self).__init__()
        
        self.num_chan = num_chan
        self.num_time = num_time
        self.num_classes = num_classes
        
        # Shallow feature encoder (Temporal Convolution)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, num_kernel, (1, temporal_kernel), padding=(0, temporal_kernel//2)),
            nn.BatchNorm2d(num_kernel),
            nn.ELU(),
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(num_kernel, num_kernel, (num_chan, 1)),
            nn.BatchNorm2d(num_kernel),
            nn.ELU(),
        )
        
        # Separable convolution
        self.separable_conv = nn.Sequential(
            nn.Conv2d(num_kernel, num_kernel, (1, 16), groups=num_kernel, padding=(0, 7)),
            nn.Conv2d(num_kernel, num_kernel, (1, 1)),
            nn.BatchNorm2d(num_kernel),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )
        
        # Calculate the flattened dimension after convolutions
        # This needs to be calculated based on num_time and pooling operations
        self.flat_dim = num_kernel * (num_time // 8)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.flat_dim,
                nhead=heads,
                dim_feedforward=mlp_dim * self.flat_dim,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(depth)
        ])
        
        # Information purification units (IP-Units) with dense connections
        self.ip_units = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.flat_dim * (i + 1), self.flat_dim),
                nn.LayerNorm(self.flat_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            for i in range(depth)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim * depth, self.flat_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(self.flat_dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, num_channels, num_timepoints)
        
        # Add channel dimension for Conv2d
        x = x.unsqueeze(1)  # (batch_size, 1, num_channels, num_timepoints)
        
        # Shallow feature encoding
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable_conv(x)
        
        # Flatten for transformer
        batch_size = x.size(0)
        x = x.view(batch_size, -1).unsqueeze(1)  # (batch_size, 1, flat_dim)
        
        # Hierarchical Coarse-to-Fine Transformer with Dense IP
        transformer_outputs = []
        for i, (transformer_layer, ip_unit) in enumerate(zip(self.transformer_layers, self.ip_units)):
            x = transformer_layer(x)
            
            # Dense connection: concatenate all previous transformer outputs
            if i == 0:
                dense_input = x.squeeze(1)
            else:
                dense_input = torch.cat(transformer_outputs + [x.squeeze(1)], dim=1)
            
            # Information purification
            ip_output = ip_unit(dense_input)
            transformer_outputs.append(ip_output)
        
        # Concatenate all IP outputs
        final_embedding = torch.cat(transformer_outputs, dim=1)
        
        # Classification
        output = self.classifier(final_embedding)
        
        return output


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for data, labels in pbar:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, labels in tqdm(dataloader, desc='Evaluating'):
            data, labels = data.to(device), labels.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Confusion matrix and classification report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, output_dict=True)
    
    return epoch_loss, epoch_acc, epoch_f1, cm, report


def train_model(model, dataloaders, criterion, optimizer, scheduler, 
                num_epochs, device, save_dir, patience=10):
    """
    Complete training loop with early stopping
    """
    best_acc = 0.0
    best_f1 = 0.0
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(num_epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'{"="*60}')
        
        # Training phase
        train_loss, train_acc, train_f1 = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        
        # Validation phase
        if 'val' in dataloaders:
            val_loss, val_acc, val_f1, _, _ = evaluate(
                model, dataloaders['val'], criterion, device
            )
            
            print(f'Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}')
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            
            # Check for improvement
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1 = val_f1
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'best_f1': best_f1,
                }, os.path.join(save_dir, 'best_model.pth'))
                
                print(f'*** New best model saved! Acc: {best_acc:.4f} | F1: {best_f1:.4f} ***')
            else:
                patience_counter += 1
                print(f'No improvement. Patience: {patience_counter}/{patience}')
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return history


def main(args):
    """Main training function"""
    
    print(f"\n{'='*60}")
    print(f"EEG-Deformer Training on DREAMER Dataset")
    print(f"{'='*60}\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    dataloaders, data_info = get_dreamer_dataloader(
        data_path=args.data_path,
        split_type=args.split_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = EEGDeformer(
        num_chan=data_info['num_channels'],
        num_time=data_info['num_timepoints'],
        temporal_kernel=args.temporal_kernel,
        num_kernel=args.num_kernel,
        num_classes=data_info['num_classes'],
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        dropout=args.dropout
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Train model
    print("\nStarting training...")
    start_time = time.time()
    
    history = train_model(
        model, dataloaders, criterion, optimizer, scheduler,
        args.num_epochs, device, args.save_dir, args.patience
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_acc, test_f1, cm, report = evaluate(
        model, dataloaders['test'], criterion, device
    )
    
    print(f"\n{'='*60}")
    print(f"Final Test Results")
    print(f"{'='*60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:")
    print(classification_report(
        [0, 1] * len(dataloaders['test']),  # dummy data for formatting
        [0, 1] * len(dataloaders['test']),
        target_names=['Low', 'High']
    ))
    
    # Save final results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_loss': float(test_loss),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'training_time': training_time,
        'best_val_acc': float(checkpoint['best_acc']),
        'best_val_f1': float(checkpoint['best_f1']),
        'args': vars(args)
    }
    
    with open(os.path.join(args.save_dir, 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EEG-Deformer on DREAMER Dataset')
    
    # Data parameters
    parser.add_argument('--data-path', type=str, 
                       default='./data_processed/DREAMER/DREAMER_valence_preprocessed.pkl',
                       help='Path to preprocessed data file')
    parser.add_argument('--split-type', type=str, default='subject_independent',
                       choices=['subject_independent', 'subject_dependent'],
                       help='Type of data split')
    
    # Model parameters
    parser.add_argument('--temporal-kernel', type=int, default=13,
                       help='Temporal convolution kernel size')
    parser.add_argument('--num-kernel', type=int, default=64,
                       help='Number of convolutional kernels')
    parser.add_argument('--depth', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=16,
                       help='Number of attention heads')
    parser.add_argument('--mlp-dim', type=int, default=16,
                       help='MLP dimension multiplier')
    parser.add_argument('--dim-head', type=int, default=16,
                       help='Dimension per attention head')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Other parameters
    parser.add_argument('--save-dir', type=str, default='./results/dreamer_valence',
                       help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    main(args)