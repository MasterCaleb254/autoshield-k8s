# src/autoshield/detector/inference.py
"""
Inference service for CNN-LSTM model.
Target: <1ms inference latency, gRPC/REST API.
"""
import torch
import numpy as np
import asyncio
from typing import Dict, Any, List, Optional
import time
import json
import os
import logging
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
except Exception:  # pragma: no cover
    FastAPI = None
    HTTPException = None
    BackgroundTasks = None
    JSONResponse = None

try:
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None

try:
    import grpc
    from concurrent import futures
except Exception:  # pragma: no cover
    grpc = None
    futures = None

from .model import create_model
from ..models import FlowWindow
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class ModelInferenceService:
    """Manages model loading and inference"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_batch_size: int = 32):
        
        self.device = torch.device(device)
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Performance tracking
        self.stats = {
            "total_inferences": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "errors": 0
        }
        
        # Warm up model
        self._warm_up()
        
        logger.info(f"Inference service initialized on {device}")
        logger.info(f"Model loaded from: {model_path}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            model_type = checkpoint.get('model_type', 'cnn_lstm')
            model_config = checkpoint.get('model_config', {})
            
            model = create_model(model_type=model_type, **model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            logger.info(f"Model loaded: {model_type} with config {model_config}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _warm_up(self, num_iterations: int = 10):
        """Warm up the model for stable latency"""
        logger.info("Warming up model...")
        
        # Create dummy input that matches the model's expected sequence length.
        # Using seq_len=1 will break the CNN pooling stack (e.g. output size 0).
        seq_len = int(getattr(self.model, "sequence_length", 1))
        input_features = int(getattr(self.model, "input_features", 12))
        dummy_input = torch.randn(1, seq_len, input_features).to(self.device)  # [batch, seq_len, features]
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model.predict(dummy_input)

        # Measuring latency at startup is expensive and can cause Kubernetes probes
        # to fail before the service becomes ready. Only do it when explicitly enabled.
        measure_startup_latency = os.getenv("MEASURE_STARTUP_LATENCY", "false").lower() == "true"
        if measure_startup_latency:
            num_samples = int(os.getenv("STARTUP_LATENCY_SAMPLES", "50"))
            latencies = []
            for _ in range(max(1, num_samples)):
                start = time.perf_counter()
                _ = self.model.predict(dummy_input)
                latencies.append((time.perf_counter() - start) * 1000)

            self.stats["avg_latency_ms"] = float(np.mean(latencies))
            self.stats["p95_latency_ms"] = float(np.percentile(latencies, 95))
            logger.info(f"Warmup complete. Initial latency: {self.stats['avg_latency_ms']:.2f}ms")
        else:
            logger.info("Warmup complete.")
    
    def _features_to_model_input(self, features: List[float], batch_size: int = 1) -> torch.Tensor:
        """Convert a single feature vector into a model input tensor.

        The CNN-LSTM expects 3D input: [batch, seq_len, features].
        For single-window inference, we repeat the feature vector across seq_len.
        """
        seq_len = int(getattr(self.model, "sequence_length", 1))
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, features]
        if seq_len > 1:
            x = x.repeat(1, seq_len, 1)  # [1, seq_len, features]
        if batch_size != 1:
            x = x.repeat(batch_size, 1, 1)
        return x
    
    async def predict_single(self, window: FlowWindow) -> Dict[str, Any]:
        """Predict on a single flow window"""
        try:
            start_time = time.perf_counter()
            
            # Convert to tensor
            features = window.to_feature_vector()
            features_tensor = self._features_to_model_input(features)  # [1, seq_len, features]
            
            # Run inference
            with torch.no_grad():
                prediction = self.model.predict(features_tensor)
            
            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Update stats
            self.stats["total_inferences"] += 1
            
            # Update running average latency
            if self.stats["total_inferences"] == 1:
                self.stats["avg_latency_ms"] = latency_ms
            else:
                alpha = 0.1  # Smoothing factor
                self.stats["avg_latency_ms"] = (
                    alpha * latency_ms + 
                    (1 - alpha) * self.stats["avg_latency_ms"]
                )
            
            # Prepare result
            class_names = ["NORMAL", "LATERAL_MOVEMENT", "PORT_SCAN", "SYN_FLOOD"]
            predicted_class = int(prediction["class"][0])
            confidence = float(prediction["confidence"][0])
            
            result = {
                "window_id": window.window_id,
                "src_pod": window.src_pod,
                "dst_pod": window.dst_pod,
                "predicted_class": class_names[predicted_class],
                "confidence": confidence,
                "probabilities": prediction["probabilities"][0].tolist(),
                "latency_ms": latency_ms,
                "timestamp": datetime.now().isoformat(),
                "features": {
                    "flow_count": window.flow_count,
                    "bytes_sent_kb": window.bytes_sent / 1000,
                    "bytes_received_kb": window.bytes_received / 1000,
                    "failed_conn_ratio": window.failed_conn_ratio,
                    "unique_ports": window.unique_ports
                }
            }
            
            # Add explainability
            result["explanation"] = self._generate_explanation(window, predicted_class, confidence)
            
            logger.debug(f"Inference complete: {window.window_id} -> {class_names[predicted_class]} "
                        f"({confidence:.2%}) in {latency_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Inference error for window {window.window_id}: {e}")
            raise
    
    async def predict_batch(self, windows: List[FlowWindow]) -> List[Dict[str, Any]]:
        """Predict on a batch of windows"""
        if not windows:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Prepare batch
            batch_features = []
            for window in windows:
                features = window.to_feature_vector()
                batch_features.append(features)
            
            # Convert to tensor
            seq_len = int(getattr(self.model, "sequence_length", 1))
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32, device=self.device)  # [batch, features]
            batch_tensor = batch_tensor.unsqueeze(1)  # [batch, 1, features]
            if seq_len > 1:
                batch_tensor = batch_tensor.repeat(1, seq_len, 1)  # [batch, seq_len, features]
            
            # Run inference
            with torch.no_grad():
                prediction = self.model.predict(batch_tensor)
            
            # Calculate latency
            batch_latency_ms = (time.perf_counter() - start_time) * 1000
            avg_latency_ms = batch_latency_ms / len(windows)
            
            # Update stats
            self.stats["total_inferences"] += len(windows)
            
            # Prepare results
            class_names = ["NORMAL", "LATERAL_MOVEMENT", "PORT_SCAN", "SYN_FLOOD"]
            results = []
            
            for i, window in enumerate(windows):
                predicted_class = int(prediction["class"][i])
                confidence = float(prediction["confidence"][i])
                
                result = {
                    "window_id": window.window_id,
                    "src_pod": window.src_pod,
                    "dst_pod": window.dst_pod,
                    "predicted_class": class_names[predicted_class],
                    "confidence": confidence,
                    "latency_ms": avg_latency_ms,
                    "timestamp": datetime.now().isoformat()
                }
                
                results.append(result)
            
            logger.info(f"Batch inference: {len(windows)} windows in {batch_latency_ms:.2f}ms "
                       f"(avg {avg_latency_ms:.2f}ms/window)")
            
            return results
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Batch inference error: {e}")
            raise
    
    def _generate_explanation(self, 
                             window: FlowWindow, 
                             predicted_class: int, 
                             confidence: float) -> str:
        """Generate human-readable explanation for the prediction"""
        class_names = ["normal traffic", "lateral movement", "port scanning", "SYN flood"]
        
        explanations = {
            0: f"Normal traffic pattern detected between {window.src_pod} and {window.dst_pod}.",
            1: f"Lateral movement suspected: {window.src_pod} attempted connections to "
               f"{window.unique_ports} unique ports on {window.dst_pod} with "
               f"{window.failed_conn_ratio:.0%} failure rate.",
            2: f"Port scanning detected: {window.src_pod} scanned {window.unique_ports} ports "
               f"on {window.dst_pod} with {window.flow_count} rapid connection attempts.",
            3: f"SYN flood attack detected: {window.src_pod} sent {window.syn_count} SYN packets "
               f"to {window.dst_pod} with minimal response."
        }
        
        base_explanation = explanations.get(predicted_class, "Unknown pattern detected.")
        
        explanation = {
            "summary": base_explanation,
            "key_features": {
                "flow_count": window.flow_count,
                "unique_ports": window.unique_ports,
                "failed_connections_ratio": window.failed_conn_ratio,
                "syn_count": window.syn_count,
                "confidence": confidence
            },
            "recommendation": self._get_recommendation(predicted_class)
        }
        
        return json.dumps(explanation, indent=2)
    
    def _get_recommendation(self, predicted_class: int) -> str:
        """Get mitigation recommendation based on attack class"""
        recommendations = {
            0: "No action required. Continue monitoring.",
            1: "Recommend pod isolation and investigation of source pod.",
            2: "Recommend network policy to restrict source pod's port access.",
            3: "Recommend traffic throttling and DDoS protection rules."
        }
        return recommendations.get(predicted_class, "Investigate further.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "model_device": str(self.device),
            "max_batch_size": self.max_batch_size,
            "uptime": datetime.now().isoformat()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for the service"""
        try:
            # Test with dummy input
            seq_len = int(getattr(self.model, "sequence_length", 1))
            input_features = int(getattr(self.model, "input_features", 12))
            dummy_input = torch.randn(1, seq_len, input_features).to(self.device)
            with torch.no_grad():
                _ = self.model.predict(dummy_input)
            
            return {
                "status": "healthy",
                "model_loaded": True,
                "device": str(self.device),
                "latency_avg_ms": self.stats["avg_latency_ms"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_loaded": False
            }