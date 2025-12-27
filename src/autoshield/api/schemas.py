# src/autoshield/api/schemas.py
"""
Pydantic schemas for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class FlowFeatures(BaseModel):
    """Flow window features"""
    flow_count: int = Field(..., ge=0, description="Number of flows in window")
    bytes_sent: int = Field(..., ge=0, description="Bytes sent")
    bytes_received: int = Field(..., ge=0, description="Bytes received")
    syn_count: int = Field(..., ge=0, description="SYN packet count")
    ack_count: int = Field(..., ge=0, description="ACK packet count")
    rst_count: int = Field(..., ge=0, description="RST packet count")
    fin_count: int = Field(..., ge=0, description="FIN packet count")
    total_duration_ms: float = Field(..., ge=0, description="Total duration in ms")
    avg_interarrival_ms: float = Field(..., ge=0, description="Average inter-arrival time")
    std_interarrival_ms: float = Field(..., ge=0, description="Std dev of inter-arrival time")
    failed_conn_ratio: float = Field(..., ge=0, le=1, description="Failed connection ratio")
    unique_ports: int = Field(..., ge=0, description="Number of unique destination ports")

class FlowWindowRequest(BaseModel):
    """Single flow window prediction request"""
    window_id: str = Field(..., description="Unique window identifier")
    src_pod: str = Field(..., description="Source pod name")
    dst_pod: str = Field(..., description="Destination pod name")
    features: FlowFeatures = Field(..., description="Flow window features")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    windows: List[FlowWindowRequest] = Field(..., description="List of flow windows")

class PredictionResponse(BaseModel):
    """Prediction response"""
    window_id: str
    src_pod: str
    dst_pod: str
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    probabilities: List[float]
    latency_ms: float
    timestamp: str
    features: Dict[str, Any]
    explanation: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    latency_avg_ms: Optional[float] = None
    error: Optional[str] = None

class StatsResponse(BaseModel):
    """Statistics response"""
    total_inferences: int
    avg_latency_ms: float
    p95_latency_ms: float
    errors: int
    model_device: str
    max_batch_size: int
    uptime: str