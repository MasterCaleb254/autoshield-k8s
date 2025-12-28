# src/autoshield/orchestrator.py
"""
Main orchestrator that ties everything together:
Feature extraction → Detection → Policy → Action
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import time

from .utils.logging import setup_logger
from .detector.inference import ModelInferenceService
from .policy.engine import PolicyEngine
from .actuator.kubernetes import KubernetesActuator
from .observability.explainer import ExplainabilityEngine
from .observability.auditor import AuditLogger

logger = setup_logger(__name__)

class AutoShieldOrchestrator:
    """Main orchestrator for the closed-loop defense system"""
    
    def __init__(self, 
                 model_path: str,
                 policy_file: str = "config/policies/default.yaml",
                 enable_actuation: bool = True):
        
        # Initialize components
        logger.info("Initializing AutoShield orchestrator...")
        
        # Inference service
        self.inference_service = ModelInferenceService(model_path=model_path)
        logger.info("✓ Inference service initialized")
        
        # Policy engine
        self.policy_engine = PolicyEngine(policy_file=policy_file)
        logger.info("✓ Policy engine initialized")
        
        # Actuator (if enabled)
        self.enable_actuation = enable_actuation
        if enable_actuation:
            self.actuator = KubernetesActuator()
            logger.info("✓ Kubernetes actuator initialized")
        else:
            self.actuator = None
            logger.info("⚠ Actuation disabled (monitor mode)")
        
        # Observability
        self.explainer = ExplainabilityEngine()
        self.audit_logger = AuditLogger()
        logger.info("✓ Observability components initialized")
        
        # Statistics
        self.stats = {
            "total_windows_processed": 0,
            "attacks_detected": 0,
            "actions_executed": 0,
            "false_positives": 0,
            "avg_processing_time_ms": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Processing queue
        self.processing_queue = asyncio.Queue(maxsize=1000)
        
        logger.info("✅ AutoShield orchestrator ready")
    
    async def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single flow window through the entire pipeline.
        
        Returns:
            Processing result with detection and action info
        """
        start_time = time.time()
        
        try:
            self.stats["total_windows_processed"] += 1
            
            # Step 1: Create FlowWindow object
            from .models import FlowWindow
            window = FlowWindow(**window_data)
            
            # Step 2: Run inference
            detection_result = await self.inference_service.predict_single(window)
            
            # Step 3: Evaluate policy
            policy_decision = self.policy_engine.evaluate(detection_result)
            
            result = {
                "window_id": window.window_id,
                "src_pod": window.src_pod,
                "dst_pod": window.dst_pod,
                "detection_result": detection_result,
                "policy_decision": policy_decision,
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 4: Execute action if policy says so
            if policy_decision and self.actuator:
                action_result = self.actuator.execute_action(
                    action_config=policy_decision['action_config'],
                    detection_result=detection_result
                )
                
                result["action_result"] = action_result
                
                if action_result.get('status') == 'success':
                    self.stats["actions_executed"] += 1
                    self.stats["attacks_detected"] += 1
                elif action_result.get('status') == 'blocked':
                    logger.info(f"Action blocked by safety controller: {window.window_id}")
            
            # Step 5: Generate enhanced explainability
            enhanced_explanation = self.explainer.enhance_explanation(
                detection_result, 
                policy_decision,
                result.get('action_result')
            )
            result["enhanced_explanation"] = enhanced_explanation
            
            # Step 6: Audit log
            self.audit_logger.log_processing_result(result)
            
            # Update statistics
            processing_time_ms = (time.time() - start_time) * 1000
            if self.stats["total_windows_processed"] == 1:
                self.stats["avg_processing_time_ms"] = processing_time_ms
            else:
                alpha = 0.1
                self.stats["avg_processing_time_ms"] = (
                    alpha * processing_time_ms + 
                    (1 - alpha) * self.stats["avg_processing_time_ms"]
                )
            
            logger.debug(f"Processed window {window.window_id} in {processing_time_ms:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process window {window_data.get('window_id')}: {e}")
            
            # Log error
            error_result = {
                "window_id": window_data.get('window_id'),
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
            self.audit_logger.log_error(error_result)
            
            return error_result
    
    async def start_processing_loop(self):
        """Start processing loop from queue"""
        logger.info("Starting processing loop...")
        
        while True:
            try:
                window_data = await self.processing_queue.get()
                
                # Process the window
                result = await self.process_window(window_data)
                
                # Log result if it was an attack
                if result.get('detection_result', {}).get('predicted_class') != 'NORMAL':
                    logger.info(f"Attack detected: {result['detection_result']['predicted_class']}")
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Processing loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1)  # Avoid tight loop on error
    
    async def add_window_to_queue(self, window_data: Dict[str, Any]):
        """Add window to processing queue"""
        await self.processing_queue.put(window_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        inference_stats = self.inference_service.get_stats()
        policy_stats = self.policy_engine.get_stats()
        
        return {
            "orchestrator": self.stats,
            "inference": inference_stats,
            "policy": policy_stats,
            "queue_size": self.processing_queue.qsize(),
            "enable_actuation": self.enable_actuation,
            "uptime": (datetime.now() - datetime.fromisoformat(self.stats["start_time"])).total_seconds()
        }
    
    def set_actuation_mode(self, enabled: bool):
        """Enable or disable actuation"""
        self.enable_actuation = enabled
        if enabled and not self.actuator:
            self.actuator = KubernetesActuator()
        logger.info(f"Actuation mode: {'enabled' if enabled else 'disabled'}")
    
    def rollback_action(self, action_id: str) -> Dict[str, Any]:
        """Rollback a specific action"""
        if not self.actuator:
            return {"status": "error", "reason": "actuator_not_initialized"}
        
        return self.actuator.rollback_action(action_id)
    
    def get_recent_actions(self, limit: int = 20) -> list:
        """Get recent actions taken"""
        if self.actuator:
            return self.actuator.action_history[-limit:]
        return []