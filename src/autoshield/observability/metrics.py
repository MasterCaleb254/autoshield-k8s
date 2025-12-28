# src/autoshield/observability/metrics.py
"""
Prometheus metrics exporter for AutoShield-K8s.
"""
from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
import time
from typing import Dict, Any
from datetime import datetime

class AutoShieldMetrics:
    """Metrics collector for Prometheus"""
    
    def __init__(self, port: int = 9090):
        self.port = port
        
        # Counters
        self.windows_processed = Counter(
            'autoshield_windows_processed_total',
            'Total number of flow windows processed'
        )
        
        self.attacks_detected = Counter(
            'autoshield_attacks_detected_total',
            'Total number of attacks detected',
            ['attack_class']
        )
        
        self.actions_executed = Counter(
            'autoshield_actions_executed_total',
            'Total number of mitigation actions executed',
            ['action_type', 'status']
        )
        
        self.false_positives = Counter(
            'autoshield_false_positives_total',
            'Total number of false positives'
        )
        
        # Gauges
        self.processing_queue_size = Gauge(
            'autoshield_processing_queue_size',
            'Current size of processing queue'
        )
        
        self.attack_rate = Gauge(
            'autoshield_attack_rate_per_minute',
            'Attack detection rate per minute'
        )
        
        self.system_status = Gauge(
            'autoshield_system_status',
            'System health status (1=healthy, 0=unhealthy)',
            ['component']
        )
        
        # Histograms
        self.inference_latency = Histogram(
            'autoshield_inference_latency_seconds',
            'Inference latency distribution',
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
        )
        
        self.processing_latency = Histogram(
            'autoshield_processing_latency_seconds',
            'Total processing latency distribution',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        )
        
        # Summaries
        self.confidence_scores = Summary(
            'autoshield_detection_confidence',
            'Confidence score distribution'
        )
        
        # Attack class distribution
        self.attack_class_distribution = Gauge(
            'autoshield_attack_class_distribution',
            'Distribution of detected attack classes',
            ['attack_class']
        )
        
        # Start metrics server
        start_http_server(self.port)
        print(f"Metrics server started on port {self.port}")
    
    def record_window_processed(self):
        """Record a window processed"""
        self.windows_processed.inc()
    
    def record_attack_detected(self, attack_class: str, confidence: float):
        """Record an attack detection"""
        self.attacks_detected.labels(attack_class=attack_class).inc()
        self.confidence_scores.observe(confidence)
    
    def record_action_executed(self, action_type: str, status: str):
        """Record an action execution"""
        self.actions_executed.labels(action_type=action_type, status=status).inc()
    
    def record_false_positive(self):
        """Record a false positive"""
        self.false_positives.inc()
    
    def record_inference_latency(self, latency_seconds: float):
        """Record inference latency"""
        self.inference_latency.observe(latency_seconds)
    
    def record_processing_latency(self, latency_seconds: float):
        """Record total processing latency"""
        self.processing_latency.observe(latency_seconds)
    
    def update_queue_size(self, size: int):
        """Update processing queue size"""
        self.processing_queue_size.set(size)
    
    def update_attack_rate(self, rate: float):
        """Update attack detection rate"""
        self.attack_rate.set(rate)
    
    def update_system_status(self, component: str, healthy: bool):
        """Update system component status"""
        self.system_status.labels(component=component).set(1 if healthy else 0)
    
    def update_attack_distribution(self, distribution: Dict[str, int]):
        """Update attack class distribution"""
        for attack_class, count in distribution.items():
            self.attack_class_distribution.labels(attack_class=attack_class).set(count)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        # This would collect metrics from Prometheus registry
        # For simplicity, return a summary
        
        return {
            'windows_processed': self.windows_processed._value.get(),
            'attacks_detected': sum(label._value.get() for label in self.attacks_detected._metrics.values()),
            'actions_executed': sum(label._value.get() for label in self.actions_executed._metrics.values()),
            'false_positives': self.false_positives._value.get(),
            'queue_size': self.processing_queue_size._value.get(),
            'attack_rate': self.attack_rate._value.get()
        }

# Global metrics instance
metrics = None

def init_metrics(port: int = 9090):
    """Initialize metrics exporter"""
    global metrics
    metrics = AutoShieldMetrics(port)
    return metrics