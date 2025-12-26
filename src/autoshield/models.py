# autoshield/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import json

@dataclass
class FlowMetadata:
    """Represents a single flow event from Hubble/Cilium"""
    flow_id: str
    src_pod: str
    dst_pod: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # TCP=6, UDP=17
    verdict: str  # FORWARDED, DROPPED, etc
    bytes_sent: int
    bytes_received: int
    tcp_flags: Dict[str, bool] = field(default_factory=dict)
    latency_ns: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @classmethod
    def from_hubble_event(cls, event: Dict) -> 'FlowMetadata':
        """Parse Hubble flow event"""
        source = event.get('source', {})
        destination = event.get('destination', {})
        ip = event.get('IP', {})
        l4 = event.get('l4', {})
        
        # Extract TCP flags if present
        tcp_flags = {}
        if 'TCP' in l4:
            tcp = l4['TCP']
            tcp_flags = {
                'SYN': tcp.get('flags', {}).get('SYN', False),
                'ACK': tcp.get('flags', {}).get('ACK', False),
                'RST': tcp.get('flags', {}).get('RST', False),
                'FIN': tcp.get('flags', {}).get('FIN', False),
            }
        
        return cls(
            flow_id=event.get('uuid', ''),
            src_pod=source.get('labels', [''])[0] if source.get('labels') else '',
            dst_pod=destination.get('labels', [''])[0] if destination.get('labels') else '',
            src_ip=ip.get('source', ''),
            dst_ip=ip.get('destination', ''),
            src_port=ip.get('sourcePort', 0),
            dst_port=ip.get('destinationPort', 0),
            protocol=ip.get('ipVersion', 4),  # Simplified
            verdict=event.get('verdict', 'UNKNOWN'),
            bytes_sent=event.get('Bytes', 0),
            bytes_received=event.get('Bytes', 0),  # Hubble provides total bytes
            tcp_flags=tcp_flags,
            latency_ns=event.get('latency', {}).get('nanoseconds', None),
            timestamp=datetime.fromisoformat(event.get('time', datetime.now().isoformat()))
        )

@dataclass
class FlowWindow:
    """Aggregated features for a sliding window of flows"""
    window_id: str
    src_pod: str
    dst_pod: str
    flow_count: int
    bytes_sent: int
    bytes_received: int
    syn_count: int
    ack_count: int
    rst_count: int
    fin_count: int
    total_duration_ms: float
    avg_interarrival_ms: float
    std_interarrival_ms: float
    failed_conn_ratio: float  # DROPPED verdicts / total
    unique_ports: int
    start_time: datetime
    end_time: datetime
    
    def to_feature_vector(self) -> List[float]:
        """Convert to ML feature vector"""
        return [
            self.flow_count,
            self.bytes_sent / 1000.0,  # Normalize to KB
            self.bytes_received / 1000.0,
            self.syn_count,
            self.ack_count,
            self.rst_count,
            self.fin_count,
            self.total_duration_ms / 1000.0,  # Convert to seconds
            self.avg_interarrival_ms,
            self.std_interarrival_ms,
            self.failed_conn_ratio,
            self.unique_ports,
        ]
    
    def to_dict(self) -> Dict:
        return {
            "window_id": self.window_id,
            "src_pod": self.src_pod,
            "dst_pod": self.dst_pod,
            "features": self.to_feature_vector(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }