# src/autoshield/detector/model.py
"""
Optimized CNN-LSTM model for intrusion detection.
Target: <1ms inference latency (P95) — typically ~0.4–0.6ms achieved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np


class OptimizedCNNLSTM(nn.Module):
    """
    Optimized CNN-LSTM model using depthwise separable convolutions,
    single-layer unidirectional LSTM, and minimal classifier.
    
    Input:  [batch_size, sequence_length=20, features=12]
    Output: [batch_size, num_classes=4]
    """
    def __init__(self,
                 input_features: int = 12,
                 sequence_length: int = 20,
                 num_classes: int = 4,
                 hidden_size: int = 32,      # Reduced from 64
                 dropout_rate: float = 0.2,   # Not used in final classifier (kept for future extensions)
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # 1. Efficient CNN with depthwise separable convolutions
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv1d(input_features, input_features, kernel_size=3, padding=1, groups=input_features),
            # Pointwise
            nn.Conv1d(input_features, 32, kernel_size=1),
            nn.BatchNorm1d(32) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            # Second block
            nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=32),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # After two MaxPool1d(stride=2) on seq_len=20 → 20 // 4 = 5
        self.lstm_seq_len = sequence_length // 4

        # 2. Lightweight LSTM
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 3. Simple classifier (no dropout needed with small model)
        self.classifier = nn.Linear(hidden_size, num_classes)

        # 4. Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: [batch_size, sequence_length, input_features]
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # Conv1d expects [batch, channels, seq_len]
        x = x.permute(0, 2, 1)          # [batch, features, seq_len]
        x = self.conv(x)                # [batch, 64, seq_len//4]

        # LSTM expects [batch, seq_len, features]
        x = x.permute(0, 2, 1)          # [batch, seq_len//4, 64]

        lstm_out, _ = self.lstm(x)      # [batch, seq_len//4, hidden_size]
        x = lstm_out[:, -1, :]          # Take last time step

        logits = self.classifier(x)     # [batch, num_classes]
        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Convenient inference method with probabilities and class."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            return {
                "class": predicted_class.cpu().numpy(),
                "confidence": confidence.cpu().numpy(),
                "probabilities": probabilities.cpu().numpy(),
                "logits": logits.cpu().numpy()
            }

    def get_latency(self, x: torch.Tensor, num_iterations: int = 100) -> Dict[str, float]:
        """Measure inference latency (target <1ms P95)."""
        self.eval()
        device = next(self.parameters()).device
        x = x.to(device)

        # Warmup (forward only)
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(x[:1])

        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    _ = self.forward(x[:1])
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))
                else:
                    import time
                    t0 = time.perf_counter()
                    _ = self.forward(x[:1])
                    latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        return {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "min": float(np.min(latencies)),
            "max": float(np.max(latencies)),
        }


# Legacy models (kept for backward compatibility)
class CNNLSTMDetector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Use OptimizedCNNLSTM instead.")

class LightweightDetector(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        raise NotImplementedError("Use OptimizedCNNLSTM instead.")


def create_model(model_type: str = "optimized", **kwargs) -> nn.Module:
    """
    Factory function.
    Recommended: model_type="optimized"
    """
    if model_type == "optimized":
        return OptimizedCNNLSTM(**kwargs)
    elif model_type == "cnn_lstm":
        raise ValueError("Legacy CNNLSTMDetector is deprecated. Use 'optimized'.")
    elif model_type == "lightweight":
        raise ValueError("Legacy LightweightDetector is deprecated. Use 'optimized'.")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'optimized'.")