# src/autoshield/detector/trainer.py
"""
Training pipeline for CNN-LSTM model with advanced training techniques.
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

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss

class ModelTrainer:
    """Handles model training, validation, and evaluation with advanced techniques"""
    
    def __init__(self, 
                 model_type: str = "optimized",
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
                     batch_size: int = 64,
                     sequence_length: int = 20) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training with efficient data loading.
        
        Args:
            X: Features array [n_samples, n_features]
            y: Labels array [n_samples]
            batch_size: Batch size for training
            sequence_length: Sequence length for LSTM
            
        Returns:
            train_loader, val_loader
        """
        # Create sequences of specified length
        n_samples = X.shape[0] - sequence_length + 1
        X_sequences = np.zeros((n_samples, sequence_length, X.shape[1]))
        y_sequences = np.zeros(n_samples, dtype=np.int64)
        
        for i in range(n_samples):
            X_sequences[i] = X[i:i+sequence_length]
            y_sequences[i] = y[i+sequence_length-1]
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences)
        y_tensor = torch.LongTensor(y_sequences)
        
        # Create dataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into train and validation (90/10 split)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders with multiple workers and pinned memory
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def mixup_data(self, x, y, alpha=0.2):
        """Apply mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, 
                    train_loader: DataLoader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    epoch: int = 0,
                    alpha: float = 0.2) -> Tuple[float, float]:
        """Train for one epoch with mixup augmentation and gradient clipping"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply mixup
            mixed_x, y_a, y_b, lam = self.mixup_data(data, target, alpha=alpha)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed data
            logits = self.model(mixed_x)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Calculate accuracy on original (non-mixed) data for monitoring
            with torch.no_grad():
                logits = self.model(data)
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                          f'Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
        
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
                
                logits = self.model(data)
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
            'loss': avg_loss
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              num_epochs: int = 50,
              batch_size: int = 64,
              learning_rate: float = 1e-3,
              weight_decay: float = 1e-4,
              patience: int = 5,
              min_delta: float = 0.001,
              output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the model with improved training loop.
        """
        # Create output directory
        if output_dir is None:
            output_dir = f"data/models/{self.model_type}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data loaders
        if X_val is None or y_val is None:
            train_loader, val_loader = self.prepare_data(
                X_train, y_train, 
                batch_size=batch_size,
                sequence_length=self.model.sequence_length
            )
        else:
            train_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_train),
                    torch.LongTensor(y_train)
                ),
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val)
                ),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Loss function with class weights
        class_weights = torch.tensor(
            [1.0 / (y_train == i).sum() for i in range(self.model.num_classes)],
            device=self.device
        )
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer with weight decay
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        logger.info(f"Starting training for {self.model_type} model")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val) if X_val is not None else 'N/A'}")
        
        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, scheduler, epoch
            )
            
            # Validate
            val_loss, val_acc, val_metrics = self.validate(val_loader, criterion)
            
            # Update learning rate scheduler
            scheduler.step(val_acc)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Check for early stopping
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save best model
            if val_acc > self.best_accuracy + min_delta:
                self.best_accuracy = val_acc
                self.best_loss = val_loss
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'model_type': self.model_type,
                    'model_config': self.model_config
                }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save final model
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': self.best_accuracy,
            'model_type': self.model_type,
            'model_config': self.model_config
        }, os.path.join(output_dir, 'final_model.pth'))
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"Training completed. Best validation accuracy: {self.best_accuracy:.2f}%")
        
        return {
            'best_accuracy': self.best_accuracy,
            'history': history,
            'output_dir': output_dir
        }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> Dict[str, Any]:
        """Evaluate model on test set with latency measurement"""
        self.model.eval()
        dataset = TensorDataset(
            torch.FloatTensor(X),
            torch.LongTensor(y)
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Get criterion with class weights
        class_weights = torch.tensor(
            [1.0 / (y == i).sum() for i in range(self.model.num_classes)],
            device=self.device
        )
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Evaluate
        loss, accuracy, metrics = self.validate(loader, criterion)
        
        # Measure latency
        sample_input = torch.randn(1, self.model.sequence_length, X.shape[1]).to(self.device)
        latency = self.model.get_latency(sample_input)
        
        return {
            'accuracy': accuracy / 100.0,  # Convert to 0-1 range
            'loss': loss,
            'latency_ms': latency,
            'per_class_accuracy': metrics['per_class_accuracy']
        }