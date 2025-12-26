# src/autoshield/detector/simulator.py
"""
Attack simulation for generating labeled training data.
Simulates various attack patterns to train the CNN-LSTM detector.
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

from ..models import FlowMetadata, FlowWindow

class AttackClass(Enum):
    """Attack types we want to detect"""
    NORMAL = "normal"
    LATERAL_MOVEMENT = "lateral_movement"
    PORT_SCAN = "port_scan"
    SYN_FLOOD = "syn_flood"
    BRUTE_FORCE = "brute_force"
    DATA_EXFILTRATION = "data_exfiltration"
    DDoS = "ddos"

@dataclass
class AttackPattern:
    """Definition of an attack pattern"""
    name: str
    class_type: AttackClass
    description: str
    parameters: Dict
    feature_signature: Dict  # Expected feature patterns

class AttackSimulator:
    """Generates simulated flow windows with attack patterns"""
    
    def __init__(self, base_traffic_rate: float = 1.0):
        self.base_traffic_rate = base_traffic_rate
        self.pods = ["frontend", "backend", "database", "payment", "auth", "api-gateway"]
        self.services = ["redis", "mysql", "elasticsearch", "kafka", "rabbitmq"]
        self.normal_patterns = self._create_normal_patterns()
        self.attack_patterns = self._create_attack_patterns()
    
    def _create_normal_patterns(self) -> Dict[str, Dict]:
        """Define normal communication patterns"""
        return {
            "frontend-backend": {
                "src_pod": "frontend",
                "dst_pod": "backend",
                "flow_rate": 2.0,  # flows per second
                "bytes_per_flow": (1000, 5000),
                "success_rate": 0.98,
                "port_range": (80, 90),
            },
            "backend-database": {
                "src_pod": "backend",
                "dst_pod": "database",
                "flow_rate": 0.5,
                "bytes_per_flow": (200, 1000),
                "success_rate": 0.99,
                "port_range": (5432, 5432),
            },
            "api-gateway-auth": {
                "src_pod": "api-gateway",
                "dst_pod": "auth",
                "flow_rate": 1.5,
                "bytes_per_flow": (500, 2000),
                "success_rate": 0.97,
                "port_range": (443, 443),
            }
        }
    
    def _create_attack_patterns(self) -> List[AttackPattern]:
        """Define attack patterns based on SPEC requirements"""
        return [
            AttackPattern(
                name="Lateral Movement",
                class_type=AttackClass.LATERAL_MOVEMENT,
                description="Pod accessing multiple services outside normal patterns",
                parameters={
                    "flow_rate_multiplier": 3.0,
                    "unique_ports_min": 10,
                    "failed_ratio": 0.4,
                    "target_count": 5,
                },
                feature_signature={
                    "flow_count": "high",
                    "unique_ports": "very_high",
                    "failed_conn_ratio": "high",
                    "syn_count": "high",
                    "ack_count": "low",
                }
            ),
            AttackPattern(
                name="Port Scan",
                class_type=AttackClass.PORT_SCAN,
                description="Sequential port scanning of a target",
                parameters={
                    "ports_per_window": 15,
                    "interarrival_ms": 50,
                    "syn_ratio": 0.95,
                    "ack_ratio": 0.05,
                },
                feature_signature={
                    "flow_count": "very_high",
                    "unique_ports": "extremely_high",
                    "failed_conn_ratio": "very_high",
                    "syn_count": "very_high",
                    "rst_count": "high",
                }
            ),
            AttackPattern(
                name="SYN Flood",
                class_type=AttackClass.SYN_FLOOD,
                description="SYN packets without ACK completion",
                parameters={
                    "syn_per_second": 100,
                    "ack_ratio": 0.1,
                    "duration_ms": 10,
                    "target_port": 80,
                },
                feature_signature={
                    "syn_count": "extremely_high",
                    "ack_count": "very_low",
                    "flow_count": "very_high",
                    "failed_conn_ratio": "very_high",
                    "interarrival_ms": "very_low",
                }
            ),
            AttackPattern(
                name="Brute Force",
                class_type=AttackClass.BRUTE_FORCE,
                description="Repeated authentication attempts",
                parameters={
                    "attempts_per_second": 5,
                    "success_ratio": 0.01,
                    "fixed_port": True,
                    "duration_variance": 0.1,
                },
                feature_signature={
                    "flow_count": "high",
                    "failed_conn_ratio": "very_high",
                    "unique_ports": "very_low",
                    "interarrival_ms": "regular",
                    "rst_count": "moderate",
                }
            ),
            AttackPattern(
                name="Data Exfiltration",
                class_type=AttackClass.DATA_EXFILTRATION,
                description="Large data transfers to external services",
                parameters={
                    "bytes_multiplier": 10.0,
                    "duration_multiplier": 5.0,
                    "target_external": True,
                    "port": 443,
                },
                feature_signature={
                    "bytes_sent": "extremely_high",
                    "bytes_received": "low",
                    "flow_count": "moderate",
                    "total_duration_ms": "high",
                    "unique_ports": "low",
                }
            ),
            AttackPattern(
                name="DDoS",
                class_type=AttackClass.DDoS,
                description="Distributed denial of service",
                parameters={
                    "source_count": 20,
                    "flow_rate_multiplier": 50.0,
                    "bytes_per_flow": (50, 200),
                    "target_single": True,
                },
                feature_signature={
                    "flow_count": "extremely_high",
                    "bytes_sent": "high",
                    "bytes_received": "very_low",
                    "interarrival_ms": "extremely_low",
                    "unique_ports": "low",
                }
            ),
        ]
    
    def generate_normal_window(self, window_id: str) -> FlowWindow:
        """Generate a window of normal traffic"""
        pattern_key = random.choice(list(self.normal_patterns.keys()))
        pattern = self.normal_patterns[pattern_key]
        
        flow_count = np.random.poisson(lam=20)  # Average 20 flows per window
        
        # Generate flows with normal characteristics
        syn_count = int(flow_count * 0.5)  # About half are new connections
        ack_count = int(flow_count * 0.9)  # Most have ACK
        rst_count = int(flow_count * 0.02)  # Few resets
        fin_count = int(flow_count * 0.4)  # Graceful closures
        
        bytes_sent = np.random.randint(5000, 50000)
        bytes_received = np.random.randint(5000, 30000)
        
        # Normal inter-arrival times (100-500ms)
        avg_interarrival = np.random.uniform(100, 500)
        std_interarrival = avg_interarrival * 0.2
        
        # Normal failed connection ratio (< 5%)
        failed_ratio = np.random.beta(1, 20)  # Skewed toward 0
        
        # Unique ports (1-3 in normal traffic)
        unique_ports = np.random.randint(1, 4)
        
        # Total duration (1-5 seconds)
        total_duration = np.random.uniform(1000, 5000)
        
        return FlowWindow(
            window_id=window_id,
            src_pod=pattern["src_pod"],
            dst_pod=pattern["dst_pod"],
            flow_count=flow_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=rst_count,
            fin_count=fin_count,
            total_duration_ms=total_duration,
            avg_interarrival_ms=avg_interarrival,
            std_interarrival_ms=std_interarrival,
            failed_conn_ratio=failed_ratio,
            unique_ports=unique_ports,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=total_duration),
        )
    
    def generate_lateral_movement_window(self, window_id: str) -> FlowWindow:
        """Generate lateral movement attack pattern"""
        # High flow count with many unique ports
        flow_count = np.random.poisson(lam=30)
        unique_ports = np.random.randint(10, 25)
        
        # Many SYN, few ACK (failed connections)
        syn_count = int(flow_count * 0.8)
        ack_count = int(flow_count * 0.3)
        rst_count = int(flow_count * 0.4)  # Many resets (access denied)
        
        # Moderate bytes
        bytes_sent = np.random.randint(1000, 10000)
        bytes_received = np.random.randint(500, 5000)
        
        # Quick attempts
        total_duration = np.random.uniform(500, 2000)
        avg_interarrival = np.random.uniform(50, 150)  # Fast attempts
        std_interarrival = avg_interarrival * 0.3
        
        # High failed ratio
        failed_ratio = np.random.beta(5, 2)  # Skewed toward 0.7
        
        return FlowWindow(
            window_id=window_id,
            src_pod="compromised-pod",  # Simulated attacker
            dst_pod="multiple-targets",
            flow_count=flow_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=rst_count,
            fin_count=int(flow_count * 0.1),
            total_duration_ms=total_duration,
            avg_interarrival_ms=avg_interarrival,
            std_interarrival_ms=std_interarrival,
            failed_conn_ratio=failed_ratio,
            unique_ports=unique_ports,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=total_duration),
        )
    
    def generate_port_scan_window(self, window_id: str) -> FlowWindow:
        """Generate port scanning attack pattern"""
        # Very high flow count
        flow_count = np.random.poisson(lam=40)
        unique_ports = np.random.randint(15, 100)  # Scanning many ports
        
        # Almost all SYN, very few ACK
        syn_count = int(flow_count * 0.95)
        ack_count = int(flow_count * 0.05)
        rst_count = int(flow_count * 0.7)  # Many closed ports
        
        # Very small packets
        bytes_sent = np.random.randint(50, 500) * flow_count
        bytes_received = np.random.randint(20, 200) * flow_count
        
        # Very regular, quick scanning
        total_duration = np.random.uniform(1000, 3000)
        avg_interarrival = np.random.uniform(20, 100)  # Very fast
        std_interarrival = avg_interarrival * 0.1  # Very regular
        
        # Very high failed ratio (ports closed)
        failed_ratio = np.random.beta(8, 1)  # Skewed toward 0.9
        
        return FlowWindow(
            window_id=window_id,
            src_pod="scanner-pod",
            dst_pod="target-service",
            flow_count=flow_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=rst_count,
            fin_count=int(flow_count * 0.02),
            total_duration_ms=total_duration,
            avg_interarrival_ms=avg_interarrival,
            std_interarrival_ms=std_interarrival,
            failed_conn_ratio=failed_ratio,
            unique_ports=unique_ports,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=total_duration),
        )
    
    def generate_syn_flood_window(self, window_id: str) -> FlowWindow:
        """Generate SYN flood attack pattern"""
        # Extremely high flow count
        flow_count = np.random.poisson(lam=100)
        
        # Almost all SYN, almost no ACK
        syn_count = int(flow_count * 0.99)
        ack_count = int(flow_count * 0.01)
        
        # Very small SYN packets
        bytes_sent = np.random.randint(40, 60) * flow_count
        bytes_received = np.random.randint(0, 10) * flow_count  # No responses
        
        # Extremely short duration, very fast rate
        total_duration = np.random.uniform(100, 500)
        avg_interarrival = np.random.uniform(1, 10)  # Extremely fast
        std_interarrival = avg_interarrival * 0.5
        
        # Almost all failed (no ACK back)
        failed_ratio = np.random.beta(9, 1)  # ~0.95
        
        return FlowWindow(
            window_id=window_id,
            src_pod="bot-1",  # Multiple sources in real attack
            dst_pod="web-server",
            flow_count=flow_count,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=int(flow_count * 0.05),
            fin_count=0,
            total_duration_ms=total_duration,
            avg_interarrival_ms=avg_interarrival,
            std_interarrival_ms=std_interarrival,
            failed_conn_ratio=failed_ratio,
            unique_ports=1,  # All targeting same port
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(milliseconds=total_duration),
        )
    
    def generate_dataset(self, 
                         samples_per_class: int = 1000,
                         output_file: str = "data/processed/train/dataset.npz") -> Tuple[np.ndarray, np.ndarray]:
        """Generate complete labeled dataset"""
        
        print(f"Generating dataset with {samples_per_class} samples per class...")
        
        X = []  # Feature vectors
        y = []  # Labels
        
        # Generate normal traffic
        print("Generating normal traffic samples...")
        for i in range(samples_per_class):
            window = self.generate_normal_window(f"normal_{i}")
            X.append(window.to_feature_vector())
            y.append(0)  # Class 0: NORMAL
        
        # Generate attack samples
        attack_generators = [
            (1, self.generate_lateral_movement_window, "Lateral Movement"),
            (2, self.generate_port_scan_window, "Port Scan"),
            (3, self.generate_syn_flood_window, "SYN Flood"),
        ]
        
        for class_id, generator, name in attack_generators:
            print(f"Generating {name} samples...")
            for i in range(samples_per_class):
                window = generator(f"{name.lower()}_{i}")
                X.append(window.to_feature_vector())
                y.append(class_id)
        
        # Convert to numpy arrays
        X_array = np.array(X, dtype=np.float32)
        y_array = np.array(y, dtype=np.int64)
        
        # Shuffle the dataset
        indices = np.arange(len(X_array))
        np.random.shuffle(indices)
        X_array = X_array[indices]
        y_array = y_array[indices]
        
        # Save dataset
        print(f"Saving dataset to {output_file}...")
        np.savez(
            output_file,
            X_train=X_array,
            y_train=y_array,
            feature_names=[
                "flow_count", "bytes_sent_kb", "bytes_received_kb",
                "syn_count", "ack_count", "rst_count", "fin_count",
                "total_duration_s", "avg_interarrival_ms", "std_interarrival_ms",
                "failed_conn_ratio", "unique_ports"
            ],
            class_names=["NORMAL", "LATERAL_MOVEMENT", "PORT_SCAN", "SYN_FLOOD"]
        )
        
        print(f"Dataset generated: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        print(f"Class distribution: {np.bincount(y_array)}")
        
        return X_array, y_array
    
    def generate_live_attack_stream(self, 
                                   attack_type: AttackClass,
                                   duration_seconds: int = 60) -> List[FlowWindow]:
        """Generate a stream of attack windows for testing"""
        print(f"Generating live {attack_type.value} attack stream for {duration_seconds} seconds...")
        
        windows = []
        start_time = datetime.now()
        
        if attack_type == AttackClass.LATERAL_MOVEMENT:
            generator = self.generate_lateral_movement_window
        elif attack_type == AttackClass.PORT_SCAN:
            generator = self.generate_port_scan_window
        elif attack_type == AttackClass.SYN_FLOOD:
            generator = self.generate_syn_flood_window
        else:
            generator = self.generate_normal_window
        
        # Generate one window per second
        for i in range(duration_seconds):
            window = generator(f"live_{attack_type.value}_{i}")
            windows.append(window)
            
            # Add some normal traffic too (realistic mix)
            if i % 3 == 0:  # Every 3 seconds, add normal traffic
                normal_window = self.generate_normal_window(f"normal_mix_{i}")
                windows.append(normal_window)
        
        return windows