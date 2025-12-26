"""
tests/test_dataset.py

Test the generated dataset for quality and consistency.
"""
import numpy as np
import pytest
from autoshield.detector.simulator import AttackSimulator


def test_dataset_shapes():
    """Test that dataset has correct shapes"""
    simulator = AttackSimulator()
    X, y = simulator.generate_dataset(samples_per_class=100)

    assert X.shape[0] == 400  # 4 classes  100 samples
    assert X.shape[1] == 12   # 12 features
    assert y.shape[0] == 400

    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    assert len(unique) == 4
    assert all(count == 100 for count in counts)


def test_feature_ranges():
    """Test that features are within reasonable ranges"""
    simulator = AttackSimulator()
    X, _ = simulator.generate_dataset(samples_per_class=50)

    # Flow count should be positive
    assert np.all(X[:, 0] > 0)  # flow_count

    # Bytes should be positive
    assert np.all(X[:, 1] >= 0)  # bytes_sent_kb
    assert np.all(X[:, 2] >= 0)  # bytes_received_kb

    # TCP flags should be non-negative
    assert np.all(X[:, 3:7] >= 0)  # syn, ack, rst, fin counts

    # Failed ratio between 0 and 1
    assert np.all((X[:, 10] >= 0) & (X[:, 10] <= 1))


def test_attack_patterns():
    """Test that attack patterns have expected characteristics"""
    simulator = AttackSimulator()

    # Generate one of each attack type
    lateral = simulator.generate_lateral_movement_window("test")
    port_scan = simulator.generate_port_scan_window("test")
    syn_flood = simulator.generate_syn_flood_window("test")

    # Lateral movement: high unique ports, high failed ratio
    assert lateral.unique_ports >= 10
    assert lateral.failed_conn_ratio >= 0.3

    # Port scan: very high unique ports
    assert port_scan.unique_ports >= 15

    # SYN flood: very high SYN count, very low ACK
    syn_ack_ratio = syn_flood.syn_count / max(1, syn_flood.ack_count)
    assert syn_ack_ratio > 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
