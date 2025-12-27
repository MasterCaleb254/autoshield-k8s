# scripts/train_model.py
#!/usr/bin/env python3
"""
Train the Optimized CNN-LSTM model for AutoShield-K8s.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
from datetime import datetime
from pathlib import Path

from autoshield.detector.trainer import ModelTrainer

def main():
    """Main training function"""
    
    # Configuration
    config = {
        'model_type': 'optimized',  # Using the optimized model
        'model_config': {
            'input_features': 12,
            'sequence_length': 20,  # Updated sequence length
            'num_classes': 4,
            'hidden_size': 32,      # Reduced hidden size
            'dropout_rate': 0.2,    # Adjusted dropout
            'use_batch_norm': True
        },
        'training': {
            'num_epochs': 50,
            'batch_size': 64,       # Increased batch size
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 5,          # Reduced patience for early stopping
            'min_delta': 0.001      # Minimum improvement for early stopping
        },
        'output_dir': f"data/models/optimized/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    print("AutoShield-K8s: Optimized CNN-LSTM Model Training")
    print("=" * 50)
    
    # Load dataset
    print("Loading dataset...")
    try:
        X_train = np.load('data/processed/train/X_train.npy')
        y_train = np.load('data/processed/train/y_train.npy')
        X_val = np.load('data/processed/validation/X_val.npy')
        y_val = np.load('data/processed/validation/y_val.npy')
        
        print(f"Dataset loaded:")
        print(f"  Training:   {X_train.shape[0]} samples")
        print(f"  Validation: {X_val.shape[0]} samples")
        print(f"  Features:   {X_train.shape[1]}")
        print(f"  Classes:    {len(np.unique(y_train))}")
        
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please run generate_training_data.py first.")
        return 1
    
    # Initialize trainer
    print("\nInitializing model...")
    trainer = ModelTrainer(
        model_type=config['model_type'],
        model_config=config['model_config']
    )
    
    # Train model
    print("\nStarting training...")
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_epochs=config['training']['num_epochs'],
        batch_size=config['training']['batch_size'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        output_dir=config['output_dir']
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test = np.load('data/processed/test/X_test.npy')
    y_test = np.load('data/processed/test/y_test.npy')
    
    test_metrics = trainer.evaluate(X_test, y_test)
    
    # Save final results
    results['test_metrics'] = test_metrics
    
    results_path = Path(config['output_dir']) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best validation accuracy: {results['best_accuracy']:.2f}%")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    if 'f1_weighted' in test_metrics:
        print(f"Test F1-score: {test_metrics['f1_weighted']:.4f}")
    print(f"Average inference latency: {test_metrics['latency_ms']['mean']:.2f} ms")
    print(f"P95 inference latency: {test_metrics['latency_ms']['p95']:.2f} ms")
    print(f"\nModel saved to: {config['output_dir']}")
    
    # Check if latency meets spec (<1ms P95)
    if test_metrics['latency_ms']['p95'] <= 1.0:
        print("\n✅ SUCCESS: Model meets latency requirement (<1ms P95)")
    else:
        print(f"\n⚠️ WARNING: Model P95 latency ({test_metrics['latency_ms']['p95']:.2f} ms) exceeds 1ms target")
        print("Consider using model optimization techniques or hardware acceleration.")
    
    return 0

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    exit(main())