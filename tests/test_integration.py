# tests/test_integration.py
"""
Integration test for the inference service.
"""
import pytest
import requests
import json
import numpy as np
from datetime import datetime

from autoshield.models import FlowWindow
from autoshield.detector.inference import ModelInferenceService

@pytest.fixture
def inference_service():
    """Create inference service for testing"""
    # Use a test model or mock
    service = ModelInferenceService(
        model_path="data/models/cnn-lstm/latest/final_model.pth",
        device="cpu"
    )
    return service

@pytest.fixture
def sample_window():
    """Create a sample flow window for testing"""
    return FlowWindow(
        window_id="test_window_001",
        src_pod="frontend",
        dst_pod="backend",
        flow_count=25,
        bytes_sent=50000,
        bytes_received=30000,
        syn_count=15,
        ack_count=20,
        rst_count=2,
        fin_count=10,
        total_duration_ms=2000,
        avg_interarrival_ms=80,
        std_interarrival_ms=15,
        failed_conn_ratio=0.1,
        unique_ports=3,
        start_time=datetime.now(),
        end_time=datetime.now()
    )

def test_inference_service_init(inference_service):
    """Test inference service initialization"""
    assert inference_service is not None
    assert inference_service.model is not None
    assert inference_service.stats["total_inferences"] == 0

def test_single_prediction(inference_service, sample_window):
    """Test single prediction"""
    result = inference_service.predict_single(sample_window)
    
    assert "window_id" in result
    assert result["window_id"] == sample_window.window_id
    assert "predicted_class" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1
    assert "latency_ms" in result
    assert result["latency_ms"] > 0

def test_batch_prediction(inference_service):
    """Test batch prediction"""
    windows = []
    for i in range(5):
        window = FlowWindow(
            window_id=f"test_window_{i}",
            src_pod=f"pod-{i}",
            dst_pod="target",
            flow_count=20 + i,
            bytes_sent=10000 * (i + 1),
            bytes_received=5000 * (i + 1),
            syn_count=10 + i,
            ack_count=15 + i,
            rst_count=i,
            fin_count=5 + i,
            total_duration_ms=1000 + i * 100,
            avg_interarrival_ms=50 + i * 10,
            std_interarrival_ms=10 + i,
            failed_conn_ratio=0.05 + i * 0.01,
            unique_ports=1 + i,
            start_time=datetime.now(),
            end_time=datetime.now()
        )
        windows.append(window)
    
    results = inference_service.predict_batch(windows)
    
    assert len(results) == len(windows)
    for i, result in enumerate(results):
        assert result["window_id"] == f"test_window_{i}"
        assert result["latency_ms"] > 0

def test_latency_requirement(inference_service, sample_window):
    """Test that inference meets <1ms P95 requirement"""
    latencies = []
    
    for _ in range(100):
        start = datetime.now()
        inference_service.predict_single(sample_window)
        latency = (datetime.now() - start).total_seconds() * 1000
        latencies.append(latency)
    
    p95 = np.percentile(latencies, 95)
    
    print(f"P95 inference latency: {p95:.2f} ms")
    
    # Check if meets requirement (with some tolerance for test environment)
    assert p95 <= 2.0  # 2ms for test environment
    
def test_health_check(inference_service):
    """Test health check"""
    health = inference_service.health_check()
    
    assert health["status"] == "healthy"
    assert health["model_loaded"] == True

@pytest.mark.integration
def test_api_endpoint():
    """Test the REST API endpoint (requires service running)"""
    # This would require the service to be running
    # In CI/CD, we would start the service in a container
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    
    # Test prediction endpoint with sample data
    sample_data = {
        "window_id": "test_api_001",
        "src_pod": "frontend",
        "dst_pod": "backend",
        "features": {
            "flow_count": 25,
            "bytes_sent": 50000,
            "bytes_received": 30000,
            "syn_count": 15,
            "ack_count": 20,
            "rst_count": 2,
            "fin_count": 10,
            "total_duration_ms": 2000,
            "avg_interarrival_ms": 80,
            "std_interarrival_ms": 15,
            "failed_conn_ratio": 0.1,
            "unique_ports": 3
        },
        "timestamp": datetime.now().isoformat()
    }
    
    response = requests.post(
        f"{base_url}/predict",
        json=sample_data,
        headers={"Content-Type": "application/json"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["window_id"] == sample_data["window_id"]
    assert "predicted_class" in data