# scripts/generate_training_data.py
#!/usr/bin/env python3
"""
Script to generate training data for AutoShield-K8s CNN-LSTM model.
Generates both normal and attack traffic patterns.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from autoshield.detector.simulator import AttackSimulator

def main():
    """Generate and save training dataset"""
    
    # Create output directories
    os.makedirs("data/raw/training", exist_ok=True)
    os.makedirs("data/processed/train", exist_ok=True)
    os.makedirs("data/processed/validation", exist_ok=True)
    os.makedirs("data/processed/test", exist_ok=True)
    
    # Initialize simulator
    simulator = AttackSimulator()
    
    # Generate dataset (4000 samples: 1000 per class × 4 classes)
    X, y = simulator.generate_dataset(samples_per_class=1000)
    
    # Split into train/validation/test (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )  # 0.25 × 0.8 = 0.2
    
    print(f"\nDataset split:")
    print(f"  Training:   {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test:       {X_test.shape[0]} samples")
    
    # Save splits
    np.save("data/processed/train/X_train.npy", X_train)
    np.save("data/processed/train/y_train.npy", y_train)
    
    np.save("data/processed/validation/X_val.npy", X_val)
    np.save("data/processed/validation/y_val.npy", y_val)
    
    np.save("data/processed/test/X_test.npy", X_test)
    np.save("data/processed/test/y_test.npy", y_test)
    
    # Save metadata
    metadata = {
        "num_classes": 4,
        "class_names": ["NORMAL", "LATERAL_MOVEMENT", "PORT_SCAN", "SYN_FLOOD"],
        "feature_names": [
            "flow_count", "bytes_sent_kb", "bytes_received_kb",
            "syn_count", "ack_count", "rst_count", "fin_count",
            "total_duration_s", "avg_interarrival_ms", "std_interarrival_ms",
            "failed_conn_ratio", "unique_ports"
        ],
        "num_features": 12,
        "samples_per_class": 1000,
        "train_size": X_train.shape[0],
        "val_size": X_val.shape[0],
        "test_size": X_test.shape[0],
    }
    
    import json
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Dataset generated successfully!")
    print(f"   Saved to: data/processed/")
    
    # Generate a live attack stream for testing
    print("\nGenerating live attack stream for demo...")
    from autoshield.detector.simulator import AttackClass
    
    lateral_stream = simulator.generate_live_attack_stream(
        AttackClass.LATERAL_MOVEMENT, duration_seconds=30
    )
    
    # Save live stream
    live_data = []
    for window in lateral_stream:
        live_data.append({
            "window": window.to_dict(),
            "features": window.to_feature_vector(),
            "label": 1 if "lateral" in window.window_id else 0
        })
    
    with open("data/processed/test/live_attack_stream.json", "w") as f:
        json.dump(live_data, f, default=str, indent=2)
    
    print(f"   Live stream saved: data/processed/test/live_attack_stream.json")

if __name__ == "__main__":
    main()