"""
Hyperparameter Tuning for EEGDeformer on DREAMER
Grid search or random search for optimal parameters
"""

import torch
import itertools
import json
from datetime import datetime
from train_dreamer import EEGDeformerTrainer


def grid_search(data_path, task='valence'):
    """
    Grid search over hyperparameters
    
    This will take a while! Consider using a subset for testing.
    """
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [1e-3, 5e-4, 1e-4],
        'depth': [2, 4, 6],
        'heads': [8, 16],
        'dropout': [0.3, 0.5],
        'batch_size': [32, 64],
        'num_kernel': [32, 64],
    }
    
    # Base configuration
    base_config = {
        'data_path': data_path,
        'task': task,
        'segment_length': 384,
        'num_epochs': 30,  # Reduced for tuning
        'temporal_kernel': 13,
        'num_classes': 2,
        'mlp_dim': 16,
        'dim_head': 16,
        'weight_decay': 0.01,
        'save_dir': f'tuning_results/{task}',
    }
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Testing {len(combinations)} configurations...")
    print(f"This will take approximately {len(combinations) * 30} minutes\n")
    
    results = []
    
    for idx, params in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"Configuration {idx+1}/{len(combinations)}")
        print(f"{'='*60}")
        print(f"Parameters: {params}")
        
        # Create config
        config = {**base_config, **params}
        
        # Train
        try:
            trainer = EEGDeformerTrainer(config)
            trainer.train()
            
            # Record results
            result = {
                'params': params,
                'best_acc': trainer.best_acc,
                'final_train_acc': trainer.history['train_acc'][-1],
                'final_test_acc': trainer.history['test_acc'][-1],
            }
            results.append(result)
            
            print(f"Result: {trainer.best_acc:.2f}%")
            
        except Exception as e:
            print(f"Error with config {params}: {e}")
            continue
    
    # Sort by best accuracy
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    # Save results
    output = {
        'task': task,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_configs': len(combinations),
        'results': results,
        'best_config': results[0] if results else None
    }
    
    with open(f'hyperparameter_tuning_{task}.json', 'w') as f:
        json.dump(output, f, indent=4)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TOP 5 CONFIGURATIONS")
    print(f"{'='*60}")
    for i, result in enumerate(results[:5]):
        print(f"\n{i+1}. Accuracy: {result['best_acc']:.2f}%")
        print(f"   Parameters: {result['params']}")
    
    return results


def quick_search(data_path, task='valence', num_trials=10):
    """
    Random search - faster alternative to grid search
    Randomly samples configurations
    """
    import random
    
    # Parameter ranges
    param_ranges = {
        'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        'depth': [2, 3, 4, 5, 6],
        'heads': [4, 8, 12, 16, 24],
        'dropout': [0.2, 0.3, 0.4, 0.5, 0.6],
        'batch_size': [16, 32, 64],
        'num_kernel': [32, 48, 64, 96, 128],
    }
    
    base_config = {
        'data_path': data_path,
        'task': task,
        'segment_length': 384,
        'num_epochs': 30,
        'temporal_kernel': 13,
        'num_classes': 2,
        'mlp_dim': 16,
        'dim_head': 16,
        'weight_decay': 0.01,
        'save_dir': f'tuning_results/{task}',
    }
    
    print(f"Random search with {num_trials} trials...")
    results = []
    
    for trial in range(num_trials):
        # Random configuration
        params = {
            key: random.choice(values) 
            for key, values in param_ranges.items()
        }
        
        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{num_trials}")
        print(f"{'='*60}")
        print(f"Parameters: {params}")
        
        config = {**base_config, **params}
        
        try:
            trainer = EEGDeformerTrainer(config)
            trainer.train()
            
            result = {
                'params': params,
                'best_acc': trainer.best_acc,
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    results.sort(key=lambda x: x['best_acc'], reverse=True)
    
    # Save
    with open(f'random_search_{task}.json', 'w') as f:
        json.dump({
            'task': task,
            'num_trials': num_trials,
            'results': results,
            'best': results[0] if results else None
        }, f, indent=4)
    
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Accuracy: {results[0]['best_acc']:.2f}%")
    print(f"Parameters: {results[0]['params']}")
    
    return results


def compare_segment_lengths(data_path, task='valence'):
    """Test different time window lengths"""
    
    segment_lengths = [256, 384, 512, 640]  # 2, 3, 4, 5 seconds
    results = []
    
    for seg_len in segment_lengths:
        print(f"\n{'='*60}")
        print(f"Testing segment length: {seg_len} samples ({seg_len/128:.1f}s)")
        print(f"{'='*60}")
        
        config = {
            'data_path': data_path,
            'task': task,
            'segment_length': seg_len,
            'num_time': seg_len,
            'overlap': 0.5,
            'test_size': 0.2,
            'random_state': 42,
            'normalize': True,
            'binary_classification': True,
            'threshold': 3,
            'num_classes': 2,
            'temporal_kernel': 13,
            'num_kernel': 64,
            'depth': 4,
            'heads': 16,
            'mlp_dim': 16,
            'dim_head': 16,
            'dropout': 0.5,
            'batch_size': 32,
            'num_epochs': 30,  # Reduced for quick comparison
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            'min_lr': 1e-6,
            'save_dir': f'checkpoints/segment_comparison/{task}',
        }
        
        trainer = EEGDeformerTrainer(config)
        trainer.train()
        
        results.append({
            'segment_length': seg_len,
            'seconds': seg_len / 128,
            'best_acc': trainer.best_acc
        })
    
    # Print comparison
    print(f"\n{'='*60}")
    print("SEGMENT LENGTH COMPARISON")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['segment_length']:4d} samples ({r['seconds']:.1f}s): {r['best_acc']:.2f}%")
    
    # Save results
    import json
    with open(f'segment_length_comparison_{task}.json', 'w') as f:
        json.dump({
            'task': task,
            'results': results,
            'best': max(results, key=lambda x: x['best_acc'])
        }, f, indent=4)
    
    return results


if __name__ == "__main__":
    data_path = 'DREAMER.mat'
    task = 'valence'
    
    # Choose one:
    
    # Option 1: Full grid search (slow but thorough)
    # results = grid_search(data_path, task)
    
    # Option 2: Random search (faster)
    results = quick_search(data_path, task, num_trials=10)
    
    # Option 3: Test segment lengths
    # results = compare_segment_lengths(data_path, task)