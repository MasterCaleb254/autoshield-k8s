# autoshield/aggregator.py
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Deque, Set, Optional
import hashlib
import json
import logging

from .models import FlowMetadata, FlowWindow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SlidingWindowAggregator:
    """Maintains sliding windows of flows per pod-pair"""
    
    def __init__(self, window_size: int = 20, emit_interval_ms: int = 1000):
        self.window_size = window_size
        self.emit_interval_ms = emit_interval_ms
        
        # Store flows per pod-pair
        self.flow_windows: Dict[str, Deque[FlowMetadata]] = defaultdict(deque)
        
        # Store timing information
        self.window_timestamps: Dict[str, Deque[datetime]] = defaultdict(deque)
        
        # Store port information
        self.unique_ports: Dict[str, Set[int]] = defaultdict(set)
        
        # Callback for completed windows
        self.on_window_complete = None
        
        # Track failed connections
        self.failed_counts: Dict[str, int] = defaultdict(int)
        self.total_counts: Dict[str, int] = defaultdict(int)
    
    def _get_window_key(self, src_pod: str, dst_pod: str) -> str:
        """Generate unique key for pod-pair"""
        return f"{src_pod}->{dst_pod}"
    
    def add_flow(self, flow: FlowMetadata) -> Optional[FlowWindow]:
        """Add flow to appropriate window and emit if window is complete"""
        if not flow.src_pod or not flow.dst_pod:
            logger.warning(f"Flow missing pod info: {flow}")
            return None
        
        key = self._get_window_key(flow.src_pod, flow.dst_pod)
        
        # Add flow to window
        self.flow_windows[key].append(flow)
        self.window_timestamps[key].append(flow.timestamp)
        
        # Track ports
        self.unique_ports[key].add(flow.dst_port)
        
        # Track failed connections
        self.total_counts[key] += 1
        if flow.verdict == "DROPPED":
            self.failed_counts[key] += 1
        
        # Check if window is complete
        if len(self.flow_windows[key]) >= self.window_size:
            return self._emit_window(key)
        
        return None
    
    def _emit_window(self, key: str) -> FlowWindow:
        """Create FlowWindow from completed window and reset"""
        flows = list(self.flow_windows[key])
        timestamps = list(self.window_timestamps[key])
        
        # Calculate features
        window_id = hashlib.md5(f"{key}-{timestamps[0]}".encode()).hexdigest()[:8]
        
        # Count TCP flags
        syn_count = sum(1 for f in flows if f.tcp_flags.get('SYN', False))
        ack_count = sum(1 for f in flows if f.tcp_flags.get('ACK', False))
        rst_count = sum(1 for f in flows if f.tcp_flags.get('RST', False))
        fin_count = sum(1 for f in flows if f.tcp_flags.get('FIN', False))
        
        # Calculate durations and inter-arrival times
        durations = []
        interarrivals = []
        
        for i in range(len(timestamps)):
            if flows[i].latency_ns:
                durations.append(flows[i].latency_ns / 1_000_000.0)  # Convert to ms
            
            if i > 0:
                delta = (timestamps[i] - timestamps[i-1]).total_seconds() * 1000  # Convert to ms
                interarrivals.append(delta)
        
        # Calculate statistics
        total_duration = sum(durations) if durations else 0
        avg_interarrival = sum(interarrivals) / len(interarrivals) if interarrivals else 0
        std_interarrival = 0
        if len(interarrivals) > 1:
            mean = avg_interarrival
            variance = sum((x - mean) ** 2 for x in interarrivals) / (len(interarrivals) - 1)
            std_interarrival = variance ** 0.5
        
        # Failed connection ratio
        total = self.total_counts[key]
        failed = self.failed_counts[key]
        failed_ratio = failed / total if total > 0 else 0
        
        # Parse src and dst from key
        src_pod, dst_pod = key.split('->')
        
        # Create window
        window = FlowWindow(
            window_id=window_id,
            src_pod=src_pod,
            dst_pod=dst_pod,
            flow_count=len(flows),
            bytes_sent=sum(f.bytes_sent for f in flows),
            bytes_received=sum(f.bytes_received for f in flows),
            syn_count=syn_count,
            ack_count=ack_count,
            rst_count=rst_count,
            fin_count=fin_count,
            total_duration_ms=total_duration,
            avg_interarrival_ms=avg_interarrival,
            std_interarrival_ms=std_interarrival,
            failed_conn_ratio=failed_ratio,
            unique_ports=len(self.unique_ports[key]),
            start_time=timestamps[0],
            end_time=timestamps[-1]
        )
        
        # Reset window
        self.flow_windows[key].clear()
        self.window_timestamps[key].clear()
        self.unique_ports[key].clear()
        self.failed_counts[key] = 0
        self.total_counts[key] = 0
        
        logger.info(f"Emitted window {window_id}: {src_pod} -> {dst_pod}, {len(flows)} flows")
        
        return window