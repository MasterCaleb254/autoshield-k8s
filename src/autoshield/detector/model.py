# src/autoshield/detector/model.py
"""
CNN-LSTM model for intrusion detection.
Target: <1ms inference latency (P95)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np

class CNNLSTMDetector(nn.Module):
    """
    CNN-LSTM model for spatio-temporal intrusion detection.
    
    Input: [batch_size, sequence_length=20, features=12]
    Output: [batch_size, num_classes=4] + confidence scores
    """
    
    def __init__(self, 
                 input_features: int = 12,
                 sequence_length: int = 20,
                 num_classes: int = 4,
                 hidden_size: int = 64,
                 dropout_rate: float = 0.3,
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # CNN for spatial feature extraction (across features dimension)
        # Input shape: [batch, sequence_length, features] -> [batch, sequence_length, features, 1] for Conv2d
        # But we use Conv1d across features for each time step
        
        # First, expand features to channels for CNN
        self.feature_expansion = nn.Linear(input_features, 32)
        
        # CNN layers (treat sequence as channels for temporal convolution)
        self.conv1 = nn.Conv1d(
            in_channels=sequence_length,  # Each time step as a channel
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64) if use_batch_norm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(128) if use_batch_norm else nn.Identity()
        
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Calculate CNN output size
        # After two conv layers with pooling
        conv_output_size = 128 * (32 // 4)  # 32 features reduced by 2 poolings
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=conv_output_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0
        )
        
        # Attention mechanism (optional, for interpretability)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Confidence calibration layer
        self.confidence_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'conv' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                elif 'linear' in name or 'classifier' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length=20, features=12]
            
        Returns:
            logits: Raw class scores [batch_size, num_classes]
            confidence: Confidence scores [batch_size, 1]
            attention_weights: Attention weights for interpretability [batch_size, sequence_length]
        """
        batch_size = x.shape[0]
        
        # 1. Feature expansion
        x = self.feature_expansion(x)  # [batch, seq_len, 32]
        
        # 2. Prepare for CNN: swap dimensions for Conv1d
        # Conv1d expects [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, 32, seq_len]
        
        # 3. CNN layers
        x = self.conv1(x)  # [batch, 64, seq_len]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)   # [batch, 64, seq_len//2]
        
        x = self.conv2(x)  # [batch, 128, seq_len//2]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)   # [batch, 128, seq_len//4]
        
        # 4. Prepare for LSTM: reshape back to sequence
        x = x.transpose(1, 2)  # [batch, seq_len//4, 128]
        x = x.contiguous()
        
        # 5. LSTM for temporal patterns
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [batch, seq_len//4, hidden_size*2]
        
        # 6. Attention mechanism
        attention_weights = self.attention(lstm_out)  # [batch, seq_len//4, 1]
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # [batch, hidden_size*2]
        
        # 7. Classification
        logits = self.classifier(context_vector)  # [batch, num_classes]
        
        # 8. Confidence score
        confidence = self.confidence_layer(context_vector)  # [batch, 1]
        
        # Reshape attention weights for interpretability
        attention_weights = attention_weights.squeeze(-1)  # [batch, seq_len//4]
        
        return logits, confidence, attention_weights
    
    def predict(self, 
                x: torch.Tensor,
                return_confidence: bool = True) -> Dict[str, Any]:
        """
        Make prediction with confidence and attention.
        
        Args:
            x: Input tensor [batch_size, sequence_length, features]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            logits, confidence, attention = self.forward(x)
            
            # Get predicted class
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            
            # Get confidence for predicted class
            if return_confidence:
                pred_confidence = confidence.squeeze(-1)
            else:
                pred_confidence = torch.max(probabilities, dim=-1)[0]
            
            return {
                "class": predicted_class.cpu().numpy(),
                "confidence": pred_confidence.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy(),
                "attention": attention.cpu().numpy(),
                "logits": logits.cpu().numpy()
            }
    
    def get_latency(self, 
                   x: torch.Tensor,
                   num_iterations: int = 100) -> Dict[str, float]:
        """
        Measure inference latency.
        Target: <1ms P95 latency.
        
        Args:
            x: Sample input tensor
            num_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with latency statistics
        """
        self.eval()
        
        # Warmup
        for _ in range(10):
            _ = self.predict(x[:1])
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start.record()
                else:
                    import time
                    start_time = time.time()
                
                _ = self.predict(x[:1])
                
                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))  # in milliseconds
                else:
                    latencies.append((time.time() - start_time) * 1000)  # convert to ms
        
        latencies = np.array(latencies)
        
        return {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies))
        }


class LightweightDetector(nn.Module):
    """
    Lightweight version for edge deployment.
    Smaller model for faster inference.
    """
    
    def __init__(self, input_features=12, num_classes=4):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Calculate output size after pooling
        conv_output_size = 64 * (20 // 4)  # sequence_length=20, two poolings of size 2
        
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        x = self.conv(x)
        x = x.flatten(1)  # flatten
        return self.classifier(x)


def create_model(model_type: str = "cnn_lstm", **kwargs) -> nn.Module:
    """Factory function to create model"""
    if model_type == "cnn_lstm":
        return CNNLSTMDetector(**kwargs)
    elif model_type == "lightweight":
        return LightweightDetector(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")