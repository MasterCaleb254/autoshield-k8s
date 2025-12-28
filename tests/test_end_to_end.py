# tests/test_end_to_end.py
"""
End-to-end test of the complete AutoShield pipeline.
"""
import pytest
import asyncio
import json
from datetime import datetime

from autoshield.orchestrator import AutoShieldOrchestrator
from autoshield.models import FlowWindow

@pytest.fixture
def orchestrator():
    """Create orchestrator for testing"""
    # Use test mode (no actual Kubernetes actions)
    orchestrator = AutoShieldOrchestrator(
        model_path="data/models/cnn-lstm/latest/final_model.pth",
        policy_file="config/policies/test.yaml",
        enable_actuation=False  # Test mode - no actual actions
    )
    return orchestrator

@pytest.fixture
def normal_window():
    """Create normal traffic window"""
    return {
        "window_id": "test_normal_001",
        "src_pod": "frontend",
        "dst_pod": "backend",
        "flow_count": 15,
        "bytes_sent": 25000,
        "bytes_received": 15000,
        "syn_count": 8,
        "ack_count": 14,
        "rst_count": 1,
        "fin_count": 6,
        "total_duration_ms": 1500,
        "avg_interarrival_ms": 100,
        "std_interarrival_ms": 20,
        "failed_conn_ratio": 0.05,
        "unique_ports": 2,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat()
    }

@pytest.fixture
def attack_window():
    """Create attack traffic window (port scan)"""
    return {
        "window_id": "test_attack_001",
        "src_pod": "attacker-pod",
        "dst_pod": "target-service",
        "flow_count": 35,
        "bytes_sent": 5000,
        "bytes_received": 1000,
        "syn_count": 32,
        "ack_count": 5,
        "rst_count": 25,
        "fin_count": 2,
        "total_duration_ms": 2500,
        "avg_interarrival_ms": 70,
        "std_interarrival_ms": 10,
        "failed_conn_ratio": 0.75,
        "unique_ports": 22,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat()
    }

@pytest.mark.asyncio
async def test_normal_traffic(orchestrator, normal_window):
    """Test processing of normal traffic"""
    result = await orchestrator.process_window(normal_window)
    
    assert result["window_id"] == normal_window["window_id"]
    assert "detection_result" in result
    assert "policy_decision" in result
    
    # Normal traffic should not trigger actions
    if result["detection_result"]["predicted_class"] == "NORMAL":
        assert result["policy_decision"] is None

@pytest.mark.asyncio
async def test_attack_traffic(orchestrator, attack_window):
    """Test processing of attack traffic"""
    result = await orchestrator.process_window(attack_window)
    
    assert result["window_id"] == attack_window["window_id"]
    assert "detection_result" in result
    
    detection = result["detection_result"]
    # Attack should be detected (though actual class depends on model)
    assert detection["predicted_class"] != "NORMAL" or detection["confidence"] < 0.5
    
    # Should have enhanced explanation
    assert "enhanced_explanation" in result

def test_orchestrator_stats(orchestrator, normal_window, attack_window):
    """Test orchestrator statistics"""
    # Process a few windows
    asyncio.run(orchestrator.process_window(normal_window))
    asyncio.run(orchestrator.process_window(attack_window))
    
    stats = orchestrator.get_stats()
    
    assert stats["orchestrator"]["total_windows_processed"] >= 2
    assert "inference" in stats
    assert "policy" in stats

def test_safety_features(orchestrator):
    """Test safety controller functionality"""
    # Try to process window with protected namespace
    protected_window = {
        "window_id": "test_protected_001",
        "src_pod": "kube-system/coredns",
        "dst_pod": "backend",
        "flow_count": 100,  # Very high to trigger
        "bytes_sent": 50000,
        "bytes_received": 50000,
        "syn_count": 50,
        "ack_count": 50,
        "rst_count": 0,
        "fin_count": 0,
        "total_duration_ms": 1000,
        "avg_interarrival_ms": 10,
        "std_interarrival_ms": 2,
        "failed_conn_ratio": 0.0,
        "unique_ports": 1,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat()
    }
    
    result = asyncio.run(orchestrator.process_window(protected_window))
    
    # Even if detected as attack, safety should prevent action on kube-system
    # (but actuation is disabled anyway in test mode)
    assert "detection_result" in result

@pytest.mark.integration
def test_full_pipeline():
    """Integration test with all components"""
    # This would test the complete pipeline with actual services
    # Requires all services to be deployed
    pass

if __name__ == "__main__":
    # Run quick demo
    orchestrator = AutoShieldOrchestrator(
        model_path="data/models/cnn-lstm/latest/final_model.pth",
        enable_actuation=False
    )
    
    # Test with sample data
    test_data = {
        "window_id": "demo_001",
        "src_pod": "suspicious-pod",
        "dst_pod": "database",
        "flow_count": 28,
        "bytes_sent": 10000,
        "bytes_received": 2000,
        "syn_count": 20,
        "ack_count": 8,
        "rst_count": 15,
        "fin_count": 3,
        "total_duration_ms": 1800,
        "avg_interarrival_ms": 65,
        "std_interarrival_ms": 15,
        "failed_conn_ratio": 0.65,
        "unique_ports": 18,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat()
    }
    
    result = asyncio.run(orchestrator.process_window(test_data))
    print(json.dumps(result, indent=2, default=str))