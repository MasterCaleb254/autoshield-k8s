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
from .dashboard.api import broadcast_detection_event, broadcast_action_event
from .observability.metrics import init_metrics
from .observability.alerts import AlertManager, AlertSeverity

logger = setup_logger(__name__)

class AutoShieldOrchestrator:
    """Main orchestrator for the closed-loop defense system"""
    
    def __init__(self, 
                 model_path: str,
                 policy_file: str = "config/policies/default.yaml",
                 enable_actuation: bool = True,
                 enable_dashboard: bool = True):
        
        # Initialize components
        logger.info("Initializing AutoShield orchestrator...")
        
        # Track attack distribution
        self.attack_distribution = defaultdict(int)
        self.stats = {
            "total_windows_processed": 0,
            "attacks_detected": 0,
            "actions_executed": 0,
            "false_positives": 0,
            "avg_processing_time_ms": 0,
            "start_time": datetime.now().isoformat()
        }
        
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
        
        # Initialize observability
        if enable_dashboard:
            self.metrics = init_metrics(port=9090)
            self.alert_manager = AlertManager()
            
            self._broadcast_detection = broadcast_detection_event
            self._broadcast_action = broadcast_action_event
            logger.info("✓ Dashboard and metrics initialized")
        
        # Observability
        self.explainer = ExplainabilityEngine()
        self.audit_logger = AuditLogger()
        logger.info("✓ Observability components initialized")
        
        # Statistics
        self.processing_queue = asyncio.Queue(maxsize=1000)
        
        logger.info("✅ AutoShield orchestrator ready")
    
    async def process_window(self, window_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process window with observability integration"""
        start_time = time.time()
        result = {"timestamp": datetime.utcnow().isoformat()}
        
        try:
            # 1. Run inference
            detection_result = await self.inference_service.predict_single(window_data)
            result["detection"] = detection_result
            
            # Update metrics
            if hasattr(self, 'metrics'):
                self.metrics.record_window_processed()
                self.metrics.record_processing_latency(
                    (time.time() - start_time)
                )
                
                if detection_result.get('predicted_class') != 'NORMAL':
                    self.metrics.record_attack_detected(
                        detection_result['predicted_class'],
                        detection_result['confidence']
                    )
                    
                    # Update attack distribution
                    attack_class = detection_result['predicted_class']
                    self.attack_distribution[attack_class] += 1
                    if hasattr(self.metrics, 'update_attack_distribution'):
                        self.metrics.update_attack_distribution(
                            dict(self.attack_distribution)
                        )
            
            # Broadcast to dashboard
            if hasattr(self, '_broadcast_detection'):
                self._broadcast_detection(detection_result)
            
            # 2. Apply policy
            policy_decision = self.policy_engine.evaluate(detection_result)
            if policy_decision:
                result["policy_decision"] = policy_decision
                
                # 3. Execute action if enabled
                if self.enable_actuation and self.actuator:
                    action_result = self.actuator.execute_action(
                        action_config=policy_decision["action_config"],
                        detection_result=detection_result
                    )
                    result["action_result"] = action_result
                    
                    # Update metrics for action
                    if hasattr(self, 'metrics'):
                        self.metrics.record_action_executed(
                            action_result.get('action_type', 'unknown'),
                            action_result.get('status', 'unknown')
                        )
                    
                    # Broadcast action to dashboard
                    if hasattr(self, '_broadcast_action'):
                        self._broadcast_action(action_result)
                    
                    # Create alert for critical actions
                    if hasattr(self, 'alert_manager'):
                        if action_result.get('status') == 'success':
                            severity = {
                                'network_policy': 'WARNING',
                                'pod_isolation': 'ERROR',
                                'pod_termination': 'CRITICAL'
                            }.get(action_result.get('action_type'), 'INFO')
                            
                            self.alert_manager.create_alert(
                                title=f"Mitigation Action: {action_result.get('action_type')}",
                                message=f"Action executed on {action_result.get('target')}",
                                severity=severity,
                                metadata=action_result
                            )
            
            # Evaluate alert rules
            if hasattr(self, 'alert_manager'):
                context = {
                    'confidence': detection_result.get('confidence'),
                    'attack_class': detection_result.get('predicted_class'),
                    'false_positives': self.stats.get('false_positives', 0),
                    'system_status': 'healthy'  # Simplified
                }
                self.alert_manager.evaluate_alert_rules(context)
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