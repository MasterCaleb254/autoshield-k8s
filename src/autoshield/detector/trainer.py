# src/autoshield/detector/trainer.py
"""
Training pipeline for CNN-LSTM model with advanced training techniques.
"""
import torch
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW

from autoshield.detector.model import create_model
from autoshield.utils.logging import setup_logger

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
        Prepare data for training with proper sequence creation.
        
        Args:
            X: Input features array of shape (n_samples, n_features)
            y: Target labels array of shape (n_samples,)
            batch_size: Batch size for DataLoader
            sequence_length: Length of sequences to create
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to numpy arrays if they're not already
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Calculate number of sequences
        n_sequences = len(X) - sequence_length + 1
        
        # Initialize arrays for sequences
        X_sequences = np.zeros((n_sequences, sequence_length, X.shape[1]))
        y_sequences = np.zeros(n_sequences, dtype=int)
        
        # Create sequences
        for i in range(n_sequences):
            X_sequences[i] = X[i:i+sequence_length]
            y_sequences[i] = y[i+sequence_length-1]  # Predict next step after sequence
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_sequences)
        y_tensor = torch.LongTensor(y_sequences)
        
        # Create dataset and split into train/val (90/10 split)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader

    def mixup_data(self, x, y, alpha=0.2):
        """Apply mixup data augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        lam_t = torch.tensor(lam, device=x.device, dtype=x.dtype)
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam_t * x + (1.0 - lam_t) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, float(lam)
    
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
                criterion: nn.Module) -> Tuple[float, float, Dict[str, Any]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
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
                
                # Calculate per-class accuracy
                for i in range(self.model.num_classes):
                    idx = (target == i)
                    class_correct[i] += (predicted[idx] == target[idx]).sum().item()
                    class_total[i] += idx.sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        class_accuracy = [100 * class_correct[i] / max(1, class_total[i]) 
                         for i in range(self.model.num_classes)]
        
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
                X_train, y_train, batch_size, self.model.sequence_length
            )
        else:
            train_loader, _ = self.prepare_data(
                X_train, y_train, batch_size, self.model.sequence_length
            )
            _, val_loader = self.prepare_data(
                X_val, y_val, batch_size, self.model.sequence_length
            )
        
        # Loss function with class weights
        class_weights = torch.tensor(
            [1.0 / (y_train == i).sum() for i in range(self.model.num_classes)],
            device=self.device,
            dtype=torch.float32
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
            patience=2
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
            
            print(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
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

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 64,
        optimized_model_path: Optional[str] = None,
        num_latency_iterations: int = 200
    ) -> Dict[str, Any]:
        """Evaluate model on test set with latency measurement."""
        self.model.eval()
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 2:
            seq_len = int(self.model.sequence_length)
            n_sequences = len(X) - seq_len + 1
            if n_sequences <= 0:
                raise ValueError(
                    f"Not enough samples ({len(X)}) to form sequences of length {seq_len}"
                )

            X_sequences = np.zeros((n_sequences, seq_len, X.shape[1]), dtype=np.float32)
            for i in range(n_sequences):
                X_sequences[i] = X[i:i + seq_len]
            y_sequences = y[seq_len - 1:]

            X_tensor = torch.FloatTensor(X_sequences)
            y_tensor = torch.LongTensor(y_sequences)
        else:
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )

        class_weights = torch.tensor(
            [1.0 / (y_tensor.cpu().numpy() == i).sum() for i in range(self.model.num_classes)],
            device=self.device,
            dtype=torch.float32
        )
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        loss, accuracy, metrics = self.validate(loader, criterion)

        # Predictions for F1
        all_preds: List[int] = []
        all_targets: List[int] = []
        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.device)
                logits = self.model(data)
                preds = torch.argmax(logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_targets.extend(target.cpu().tolist())

        f1_weighted: Optional[float] = None
        try:
            from sklearn.metrics import f1_score
            f1_weighted = float(f1_score(all_targets, all_preds, average='weighted'))
        except Exception:
            f1_weighted = None

        sample_input = torch.randn(1, self.model.sequence_length, X_tensor.shape[-1]).to(self.device)

        # Prefer measuring latency from optimized artifact when provided
        latency = None
        if optimized_model_path is not None and os.path.exists(optimized_model_path):
            try:
                ts = torch.jit.load(optimized_model_path, map_location=self.device)
                ts.eval()

                # Warmup
                with torch.no_grad():
                    for _ in range(10):
                        _ = ts(sample_input)

                latencies = []
                with torch.no_grad():
                    for _ in range(num_latency_iterations):
                        if torch.cuda.is_available():
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            _ = ts(sample_input)
                            end.record()
                            torch.cuda.synchronize()
                            latencies.append(start.elapsed_time(end))
                        else:
                            import time
                            t0 = time.perf_counter()
                            _ = ts(sample_input)
                            latencies.append((time.perf_counter() - t0) * 1000)

                latencies = np.array(latencies)
                latency = {
                    'mean': float(np.mean(latencies)),
                    'std': float(np.std(latencies)),
                    'p50': float(np.percentile(latencies, 50)),
                    'p95': float(np.percentile(latencies, 95)),
                    'p99': float(np.percentile(latencies, 99)),
                    'min': float(np.min(latencies)),
                    'max': float(np.max(latencies)),
                    'artifact_path': str(optimized_model_path)
                }
            except Exception:
                latency = None

        if latency is None:
            latency = self.model.get_latency(sample_input, num_iterations=num_latency_iterations)

        return {
            'accuracy': accuracy / 100.0,
            'accuracy_percent': accuracy,
            'loss': loss,
            'latency_ms': latency,
            'per_class_accuracy': metrics['per_class_accuracy'],
            'f1_weighted': f1_weighted
        }