# scripts/evaluate_model.py
#!/usr/bin/env python3
"""
Evaluate trained model on test data and generate performance report.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

from autoshield.detector.trainer import ModelTrainer

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_training_history(history, output_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def main(model_path: str = None):
    """Evaluate model and generate reports"""
    
    # If no model path provided, find the latest
    if model_path is None:
        model_dirs = sorted(Path("data/models/cnn-lstm").glob("*"))
        if not model_dirs:
            print("No trained models found. Please train a model first.")
            return 1
        
        latest_dir = model_dirs[-1]
        model_path = latest_dir / "final_model.pth"
        print(f"Using latest model: {model_path}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    trainer = ModelTrainer.load_model(str(model_path))
    
    # Load test data
    print("Loading test data...")
    X_test = np.load('data/processed/test/X_test.npy')
    y_test = np.load('data/processed/test/y_test.npy')
    
    # Evaluate
    print("Evaluating model...")
    results = trainer.evaluate(X_test, y_test)
    
    # Create output directory
    output_dir = Path("reports/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate confusion matrix plot
    cm = np.array(results['confusion_matrix'])
    class_names = ['NORMAL', 'LATERAL_MOVEMENT', 'PORT_SCAN', 'SYN_FLOOD']
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")
    
    # Plot training history if available
    if hasattr(trainer, 'history') and trainer.history:
        plot_training_history(trainer.history, output_dir / "training_history.png")
    
    # Generate markdown report
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, 'w') as f:
        f.write("# AutoShield-K8s Model Evaluation Report\n\n")
        
        f.write("## Performance Summary\n\n")
        f.write(f"- **Overall Accuracy**: {results['accuracy']:.4f}\n")
        f.write(f"- **Weighted Precision**: {results['precision_weighted']:.4f}\n")
        f.write(f"- **Weighted Recall**: {results['recall_weighted']:.4f}\n")
        f.write(f"- **Weighted F1-Score**: {results['f1_weighted']:.4f}\n\n")
        
        f.write("## Inference Latency\n\n")
        latency = results['latency_ms']
        f.write(f"- **Mean**: {latency['mean']:.2f} ms\n")
        f.write(f"- **P50 (Median)**: {latency['p50']:.2f} ms\n")
        f.write(f"- **P95**: {latency['p95']:.2f} ms\n")
        f.write(f"- **P99**: {latency['p99']:.2f} ms\n")
        f.write(f"- **Min**: {latency['min']:.2f} ms\n")
        f.write(f"- **Max**: {latency['max']:.2f} ms\n\n")
        
        # Check latency requirement
        if latency['p95'] <= 1.0:
            f.write("✅ **LATENCY TARGET ACHIEVED**: P95 < 1ms\n\n")
        else:
            f.write(f"⚠️ **LATENCY TARGET NOT MET**: P95 = {latency['p95']:.2f} ms (> 1ms)\n\n")
        
        f.write("## Per-Class Performance\n\n")
        report = results['classification_report']
        
        f.write("| Class | Precision | Recall | F1-Score | Support |\n")
        f.write("|-------|-----------|--------|----------|---------|\n")
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                f.write(f"| {class_name} | {metrics['precision']:.4f} | "
                       f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                       f"{metrics['support']} |\n")
        
        f.write("\n## Recommendations\n\n")
        
        if latency['p95'] > 1.0:
            f.write("1. **Optimize for latency**: Consider using the lightweight model variant\n")
            f.write("2. **Quantize model**: Use PyTorch quantization for faster inference\n")
            f.write("3. **Batch inference**: Process multiple windows together\n")
        
        if report['PORT_SCAN']['recall'] < 0.9:
            f.write("4. **Improve port scan detection**: Add more port scan variations to training data\n")
        
        if report['SYN_FLOOD']['recall'] < 0.9:
            f.write("5. **Improve SYN flood detection**: Increase SYN flood samples in training\n")
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey Metrics:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1-Score: {results['f1_weighted']:.4f}")
    print(f"  P95 Latency: {results['latency_ms']['p95']:.2f} ms")
    
    if results['latency_ms']['p95'] <= 1.0:
        print("\n✅ Model meets SPEC requirement: <1ms P95 inference latency!")
    else:
        print(f"\n⚠️  Model exceeds latency target by {results['latency_ms']['p95'] - 1.0:.2f} ms")
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    exit(main(args.model))