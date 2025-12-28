# autoshield/feature_extractor.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Optional
import sys
import signal
import time
import os

from .models import FlowMetadata
from .aggregator import SlidingWindowAggregator
from .collector import HubbleFlowCollector

logger = logging.getLogger(__name__)

class FeatureExtractionService:
    """Main service that orchestrates flow collection and feature extraction"""
    
    def __init__(self, output_file: Optional[str] = None, kafka_brokers: Optional[str] = None):
        self.aggregator = SlidingWindowAggregator(window_size=20, emit_interval_ms=1000)
        self.collector = HubbleFlowCollector()
        self.output_file = output_file
        self.kafka_brokers = kafka_brokers
        
        # Set up callbacks
        self.collector.add_callback(self._process_flow_event)
        self.aggregator.on_window_complete = self._process_completed_window
        
        # Statistics
        self.stats = {
            "flows_processed": 0,
            "windows_emitted": 0,
            "last_emission": None
        }
    
    def _process_flow_event(self, event: dict):
        """Process incoming flow event from Hubble"""
        try:
            flow = FlowMetadata.from_hubble_event(event)
            self.stats["flows_processed"] += 1
            
            # Add to aggregator
            window = self.aggregator.add_flow(flow)
            if window:
                self._process_completed_window(window)
                
            # Log periodically
            if self.stats["flows_processed"] % 100 == 0:
                logger.info(f"Processed {self.stats['flows_processed']} flows, "
                           f"Emitted {self.stats['windows_emitted']} windows")
                
        except Exception as e:
            logger.error(f"Error processing flow event: {e}")
    
    def _process_completed_window(self, window):
        """Handle completed flow window"""
        self.stats["windows_emitted"] += 1
        self.stats["last_emission"] = datetime.now().isoformat()
        
        # Convert to feature vector
        features = window.to_feature_vector()
        window_dict = window.to_dict()
        
        # Output to file if specified
        if self.output_file:
            with open(self.output_file, 'a') as f:
                f.write(json.dumps(window_dict) + '\n')
        
        # TODO: Send to Kafka if configured
        if self.kafka_brokers:
            self._send_to_kafka(window_dict)
        
        # TODO: Send to ML inference service
        try:
            asyncio.get_running_loop().create_task(self._send_to_detection_service(window_dict))
        except RuntimeError:
            # No running event loop in this thread/context.
            # Avoid blocking the flow pipeline.
            pass
        
        logger.debug(f"Window {window.window_id}: {features}")
    
    def _send_to_kafka(self, window_dict: dict):
        """Send window to Kafka topic"""
        # Implementation for Kafka producer
        pass
    
    async def _send_to_detection_service(self, window_dict: dict):
        """Send window to ML detection service via REST API"""
        try:
            import aiohttp

            endpoint = os.getenv(
                "INFERENCE_URL",
                "http://inference-service.autoshield-system:8000/predict"
            )

            feats = window_dict.get("features")
            if feats is None:
                return

            payload = {
                "window_id": window_dict.get("window_id"),
                "src_pod": window_dict.get("src_pod"),
                "dst_pod": window_dict.get("dst_pod"),
                "features": {
                    "flow_count": int(feats[0]),
                    "bytes_sent": int(float(feats[1]) * 1000),
                    "bytes_received": int(float(feats[2]) * 1000),
                    "syn_count": int(float(feats[3])),
                    "ack_count": int(float(feats[4])),
                    "rst_count": int(float(feats[5])),
                    "fin_count": int(float(feats[6])),
                    "total_duration_ms": float(feats[7]) * 1000,
                    "avg_interarrival_ms": float(feats[8]),
                    "std_interarrival_ms": float(feats[9]),
                    "failed_conn_ratio": float(feats[10]),
                    "unique_ports": int(float(feats[11])),
                },
                "timestamp": window_dict.get("start_time"),
            }

            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(
                            "Detection result: %s (confidence: %.2f%%)",
                            result.get("predicted_class"),
                            float(result.get("confidence", 0.0)) * 100.0,
                        )
                        await self._send_to_policy_engine(result)
                    else:
                        logger.error(f"Detection service error: {response.status}")

        except Exception as e:
            logger.error(f"Failed to send to detection service: {e}")

    async def _send_to_policy_engine(self, detection_result: dict):
        """Send detection output to policy engine (placeholder)."""
        return
    
    def start(self):
        """Start the feature extraction service"""
        logger.info("Starting Feature Extraction Service...")
        self.collector.start()
        
        # Keep service running
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
        try:
            while True:
                # Print stats every 30 seconds
                logger.info(f"Stats: {json.dumps(self.stats, indent=2)}")
                time.sleep(30)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the service"""
        logger.info("Stopping Feature Extraction Service...")
        self.collector.stop()
        logger.info("Service stopped")

if __name__ == "__main__":
    import time
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Start service
    service = FeatureExtractionService(output_file="flow_windows.jsonl")
    service.start()