# src/autoshield/detector/trainer.py
"""
Training pipeline for CNN-LSTM model.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging

from .model import create_model
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ModelTrainer:
    """Handles model training, validation, and evaluation"""
    
    def __init__(self, 
                 model_type: str = "cnn_lstm",
                 model_config: Optional[Dict] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = torch.device(device)
        self.model_type = model_type
        self.model_config = model_config or {}
        
        # Create model
        self.model = create_model(model_type=model_type, **self.model_config)
        self.model.to(self.device)
        
        # Training state
        self.best_accuracy = 0.0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def prepare_data(self, 
                     X: np.ndarray, 
                     y: np.ndarray,
                     batch_size: int = 32,
                     sequence_length: int = 20) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            X: Features array [n_samples, n_features]
            y: Labels array [n_samples]
            batch_size: Batch size for training
            sequence_length: Sequence length for LSTM (create sequences from flat data)
            
        Returns:
            train_loader, val_loader
        """
        # For now, we treat each sample as a sequence of length 1
        # TODO: Implement proper sequence creation from time-series data
        X_reshaped = X.reshape(-1, 1, X.shape[1])  # [n_samples, 1, n_features]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_reshaped)
        y_tensor = torch.LongTensor(y)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, 
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    epoch: int) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, _, _ = self.model(data)
            loss = criterion(logits, target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            if batch_idx % 50 == 0:
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                           f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, 
                 val_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float, Dict]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        # For per-class metrics
        class_correct = [0] * self.model.num_classes
        class_total = [0] * self.model.num_classes
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                logits, _, _ = self.model(data)
                loss = criterion(logits, target)
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                for i in range(self.model.num_classes):
                    mask = target == i
                    if mask.any():
                        class_total[i] += mask.sum().item()
                        class_correct[i] += (predicted[mask] == target[mask]).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Per-class accuracy
        class_accuracy = []
        for i in range(self.model.num_classes):
            if class_total[i] > 0:
                class_accuracy.append(100. * class_correct[i] / class_total[i])
            else:
                class_accuracy.append(0.0)
        
        metrics = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': class_accuracy,
            'confusion_matrix': None  # TODO: Add confusion matrix
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              num_epochs: int = 50,
              batch_size: int = 32,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              patience: int = 10,
              output_dir: str = "data/models/cnn-lstm") -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training history and best model metrics
        """
        logger.info(f"Starting training for {self.model_type} model")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            X_train, y_train, batch_size=batch_size
        )
        
        # Criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True
        )
        
        # Early stopping
        early_stop_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            logger.info(f'Epoch {epoch}/{num_epochs}: '
                       f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                early_stop_counter = 0
                
                # Save model
                self.save_model(
                    output_dir, 
                    f"best_epoch_{epoch}_acc_{val_acc:.2f}.pth"
                )
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self.evaluate(X_val, y_val)
        
        # Save final model
        self.save_model(output_dir, "final_model.pth")
        
        # Save training history
        self.save_training_history(output_dir)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_accuracy:.2f}%")
        
        return {
            'history': self.history,
            'best_accuracy': self.best_accuracy,
            'final_metrics': final_metrics,
            'model_config': self.model_config
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        from sklearn.metrics import classification_report, confusion_matrix
        
        self.model.eval()
        
        # Prepare data
        X_tensor = torch.FloatTensor(X.reshape(-1, 1, X.shape[1]))
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_targets = []
        all_confidences = []
        
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                
                predictions = self.model.predict(data)
                all_preds.extend(predictions['class'])
                all_targets.extend(target.cpu().numpy())
                all_confidences.extend(predictions['confidence'])
        
        # Calculate metrics
        report = classification_report(
            all_targets, 
            all_preds,
            target_names=['NORMAL', 'LATERAL_MOVEMENT', 'PORT_SCAN', 'SYN_FLOOD'],
            output_dict=True
        )
        
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate inference latency
        sample_input = X_tensor[:1].to(self.device)
        latency_stats = self.model.get_latency(sample_input)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'precision_weighted': report['weighted avg']['precision'],
            'recall_weighted': report['weighted avg']['recall'],
            'f1_weighted': report['weighted avg']['f1-score'],
            'latency_ms': latency_stats,
            'avg_confidence': float(np.mean(all_confidences))
        }
    
    def save_model(self, output_dir: str, filename: str):
        """Save model to disk"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = Path(output_dir) / filename
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'model_type': self.model_type,
            'input_features': self.model.input_features,
            'num_classes': self.model.num_classes,
            'history': self.history,
            'best_accuracy': self.best_accuracy
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def save_training_history(self, output_dir: str):
        """Save training history for visualization"""
        history_path = Path(output_dir) / "training_history.json"
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")
    
    @classmethod
    def load_model(cls, model_path: str, device: str = "cpu"):
        """Load a trained model"""
        checkpoint = torch.load(model_path, map_location=device)
        
        trainer = cls(
            model_type=checkpoint['model_type'],
            model_config=checkpoint['model_config'],
            device=device
        )
        
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.history = checkpoint.get('history', {})
        trainer.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Best accuracy: {trainer.best_accuracy:.2f}%")
        
        return trainer