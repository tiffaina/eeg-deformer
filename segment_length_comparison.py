"""
Standalone Segment Length Comparison for DREAMER
Tests different time window lengths (2s, 3s, 4s, 5s)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import sys
import json

sys.path.append('./EEG-Deformer')
from models.EEGDeformer import Deformer
from dreamer_dataset import get_dreamer_dataloaders


def train_model(data_path, task, segment_length, num_epochs=30):
    """Train model with specific segment length"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data with segment_length={segment_length}...")
    train_loader, test_loader = get_dreamer_dataloaders(
        data_path=data_path,
        task=task,
        segment_length=segment_length,
        batch_size=32,
        overlap=0.5,
        test_size=0.2,
        random_state=42,
        normalize=True,
        binary_classification=True,
        threshold=3
    )
    
    # Initialize model
    model = Deformer(
        num_chan=14,
        num_time=segment_length,
        temporal_kernel=13,
        num_kernel=64,
        num_classes=2,
        depth=4,
        heads=16,
        mlp_dim=16,
        dim_head=16,
        dropout=0.5
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0.0
    history = {'train_acc': [], 'test_acc': []}
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Test
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * test_correct / test_total
        
        scheduler.step()
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Print every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Train={train_acc:.2f}%, Test={test_acc:.2f}%, "
                  f"Best={best_acc:.2f}%")
    
    print(f"\nBest accuracy: {best_acc:.2f}%\n")
    
    return best_acc, history


def compare_segment_lengths(data_path, task='valence', num_epochs=30):
    """Compare different segment lengths"""
    
    segment_lengths = [256, 384, 512, 640]  # 2, 3, 4, 5 seconds
    results = []
    
    print(f"\n{'='*80}")
    print(f"SEGMENT LENGTH COMPARISON FOR {task.upper()}")
    print(f"{'='*80}\n")
    
    for seg_len in segment_lengths:
        print(f"\n{'='*60}")
        print(f"Testing segment length: {seg_len} samples ({seg_len/128:.1f}s)")
        print(f"{'='*60}")
        
        best_acc, history = train_model(
            data_path=data_path,
            task=task,
            segment_length=seg_len,
            num_epochs=num_epochs
        )
        
        results.append({
            'segment_length': seg_len,
            'seconds': seg_len / 128,
            'best_acc': best_acc,
            'final_train_acc': history['train_acc'][-1],
            'final_test_acc': history['test_acc'][-1],
            'history': history
        })
    
    # Print comparison table
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY - {task.upper()}")
    print(f"{'='*80}")
    print(f"{'Samples':>8} {'Seconds':>8} {'Best Acc':>10} {'Final Test':>12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['segment_length']:>8} {r['seconds']:>8.1f} "
              f"{r['best_acc']:>10.2f}% {r['final_test_acc']:>12.2f}%")
    
    # Find best
    best = max(results, key=lambda x: x['best_acc'])
    print(f"{'-'*80}")
    print(f"Best: {best['segment_length']} samples "
          f"({best['seconds']:.1f}s) - {best['best_acc']:.2f}%")
    print(f"{'='*80}\n")
    
    # Save results
    output_file = f'segment_length_comparison_{task}.json'
    with open(output_file, 'w') as f:
        json.dump({
            'task': task,
            'num_epochs': num_epochs,
            'results': results,
            'best': best
        }, f, indent=4)
    
    print(f"Results saved to {output_file}\n")
    
    return results


def compare_all_tasks(data_path='DREAMER.mat', num_epochs=30):
    """Compare segment lengths for all three tasks"""
    
    all_results = {}
    
    for task in ['valence', 'arousal', 'dominance']:
        results = compare_segment_lengths(data_path, task, num_epochs)
        all_results[task] = results
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY - ALL TASKS")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        best = max(results, key=lambda x: x['best_acc'])
        print(f"\n{task.capitalize():12} - Best: {best['segment_length']} samples "
              f"({best['seconds']:.1f}s) - {best['best_acc']:.2f}%")
    
    print(f"\n{'='*80}\n")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='DREAMER.mat')
    parser.add_argument('--task', type=str, default='all', 
                       choices=['valence', 'arousal', 'dominance', 'all'])
    parser.add_argument('--epochs', type=int, default=30)
    
    args = parser.parse_args()
    
    if args.task == 'all':
        compare_all_tasks(args.data_path, args.epochs)
    else:
        compare_segment_lengths(args.data_path, args.task, args.epochs)