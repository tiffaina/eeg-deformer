"""
Simple example demonstrating how to use EEGDeformer with DREAMER dataset
"""

import sys
sys.path.append('./EEG-Deformer')

import torch
from models.EEGDeformer import Deformer
from dreamer_dataset import get_dreamer_dataloaders


def example_1_basic_usage():
    """Example 1: Basic model instantiation and forward pass"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # Create a sample EEG segment
    # DREAMER: 14 channels, 384 samples (3 seconds at 128 Hz)
    batch_size = 4
    num_channels = 14
    segment_length = 384
    
    sample_data = torch.randn(batch_size, num_channels, segment_length)
    print(f"Input shape: {sample_data.shape}")
    
    # Initialize model
    model = Deformer(
        num_chan=14,           # DREAMER has 14 EEG channels
        num_time=384,          # 3 seconds at 128 Hz
        temporal_kernel=13,    # Odd[0.1 * 128] = 13
        num_kernel=64,
        num_classes=2,         # Binary classification (high/low)
        depth=4,
        heads=16,
        mlp_dim=16,
        dim_head=16,
        dropout=0.5
    )
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(sample_data)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output values (logits): {outputs}")
    
    # Get predictions
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1)
    
    print(f"\nProbabilities:\n{probabilities}")
    print(f"Predictions: {predictions}")
    print(f"  (0 = Low emotion, 1 = High emotion)")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")


def example_2_load_dreamer_data():
    """Example 2: Loading actual DREAMER data"""
    print("\n" + "="*60)
    print("Example 2: Loading DREAMER Data")
    print("="*60)
    
    data_path = 'DREAMER.mat'  # Update this path
    
    try:
        # Create dataloaders
        train_loader, test_loader = get_dreamer_dataloaders(
            data_path=data_path,
            task='valence',
            segment_length=384,
            batch_size=8,
            overlap=0.5,
            test_size=0.2,
            random_state=42,
            normalize=True,
            binary_classification=True,
            threshold=3
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        # Examine first batch
        for data, labels in train_loader:
            print(f"\nFirst batch:")
            print(f"  Data shape: {data.shape}")  # (batch_size, 14, 384)
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels: {labels}")
            print(f"  Label distribution: {torch.bincount(labels)}")
            
            # Show data statistics
            print(f"\nData statistics:")
            print(f"  Mean: {data.mean():.4f}")
            print(f"  Std: {data.std():.4f}")
            print(f"  Min: {data.min():.4f}")
            print(f"  Max: {data.max():.4f}")
            break
            
    except FileNotFoundError:
        print(f"\nError: Could not find {data_path}")
        print("Please download DREAMER.mat and update the path in this script")
    except Exception as e:
        print(f"\nError loading data: {e}")


def example_3_training_loop():
    """Example 3: Simple training loop"""
    print("\n" + "="*60)
    print("Example 3: Simple Training Loop")
    print("="*60)
    
    data_path = 'DREAMER.mat'
    
    try:
        # Setup
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load data
        train_loader, test_loader = get_dreamer_dataloaders(
            data_path=data_path,
            task='valence',
            segment_length=384,
            batch_size=16
        )
        
        # Initialize model
        model = Deformer(
            num_chan=14,
            num_time=384,
            temporal_kernel=13,
            num_kernel=64,
            num_classes=2,
            depth=4,
            heads=16,
            mlp_dim=16,
            dim_head=16,
            dropout=0.5
        ).to(device)
        
        # Loss and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Train for a few batches (demo only)
        model.train()
        print("\nTraining for 5 batches (demo)...")
        
        for i, (data, labels) in enumerate(train_loader):
            if i >= 5:  # Only train on 5 batches for demo
                break
            
            data, labels = data.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = predicted.eq(labels).sum().item() / labels.size(0)
            
            print(f"  Batch {i+1}/5 - Loss: {loss.item():.4f}, Acc: {accuracy*100:.2f}%")
        
        print("\nDemo training completed!")
        
    except FileNotFoundError:
        print(f"\nError: Could not find {data_path}")
        print("Please download DREAMER.mat and update the path in this script")
    except Exception as e:
        print(f"\nError: {e}")


def example_4_model_comparison():
    """Example 4: Compare different model configurations"""
    print("\n" + "="*60)
    print("Example 4: Model Configuration Comparison")
    print("="*60)
    
    configs = [
        {'depth': 2, 'heads': 8, 'name': 'Small'},
        {'depth': 4, 'heads': 16, 'name': 'Medium (Default)'},
        {'depth': 6, 'heads': 24, 'name': 'Large'},
    ]
    
    for config in configs:
        model = Deformer(
            num_chan=14,
            num_time=384,
            temporal_kernel=13,
            num_kernel=64,
            num_classes=2,
            depth=config['depth'],
            heads=config['heads'],
            mlp_dim=16,
            dim_head=16,
            dropout=0.5
        )
        
        params = sum(p.numel() for p in model.parameters())
        
        print(f"\n{config['name']} Model:")
        print(f"  Depth: {config['depth']}, Heads: {config['heads']}")
        print(f"  Parameters: {params:,}")


def example_5_different_tasks():
    """Example 5: Training on different tasks"""
    print("\n" + "="*60)
    print("Example 5: Different Emotion Recognition Tasks")
    print("="*60)
    
    tasks = ['valence', 'arousal', 'dominance']
    data_path = 'DREAMER.mat'
    
    try:
        for task in tasks:
            print(f"\n{task.upper()}:")
            
            # Load data for this task
            train_loader, test_loader = get_dreamer_dataloaders(
                data_path=data_path,
                task=task,
                segment_length=384,
                batch_size=16
            )
            
            print(f"  Train batches: {len(train_loader)}")
            print(f"  Test batches: {len(test_loader)}")
            
            # Check label distribution
            all_labels = []
            for _, labels in train_loader:
                all_labels.extend(labels.numpy())
            
            import numpy as np
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"  Label distribution: {dict(zip(unique, counts))}")
            
    except FileNotFoundError:
        print(f"\nError: Could not find {data_path}")
        print("Please download DREAMER.mat and update the path in this script")
    except Exception as e:
        print(f"\nError: {e}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" EEGDeformer + DREAMER: Usage Examples")
    print("="*80)
    
    # Example 1: Basic usage (no data required)
    example_1_basic_usage()
    
    # Example 2-5 require DREAMER.mat file
    # Uncomment to run if you have the data
    
    # example_2_load_dreamer_data()
    # example_3_training_loop()
    # example_4_model_comparison()
    # example_5_different_tasks()
    
    print("\n" + "="*80)
    print("Examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()