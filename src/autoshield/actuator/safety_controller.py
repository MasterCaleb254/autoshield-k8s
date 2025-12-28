# src/autoshield/actuator/safety_controller.py
"""
Safety controller to prevent harmful actions.
Implements circuit breakers, rate limiting, and critical resource protection.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from collections import defaultdict

from ...utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class SafetyRule:
    """Safety rule definition"""
    name: str
    condition: str  # e.g., "max_actions_per_minute", "protected_namespace"
    threshold: Any
    action: str  # "block", "alert", "require_approval"
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

class SafetyController:
    """Controls action execution to ensure safety"""
    
    def __init__(self):
        self.rules: List[SafetyRule] = []
        self.action_history: List[Dict[str, Any]] = []
        self.circuit_breakers: Dict[str, bool] = defaultdict(bool)
        
        # Protected resources (should not be acted upon)
        self.protected_namespaces = {'kube-system', 'autoshield-system'}
        self.protected_pod_prefixes = {'autoshield-', 'cilium-', 'kube-apiserver'}
        
        # Rate limiting
        self.action_counts = defaultdict(int)
        self.last_reset = datetime.now()
        
        # Load safety rules
        self._load_default_rules()
        
        logger.info("Safety controller initialized")
    
    def _load_default_rules(self):
        """Load default safety rules"""
        self.rules = [
            SafetyRule(
                name="protected_namespace",
                condition="protected_namespace",
                threshold=None,
                action="block",
                parameters={
                    "message": "Action blocked: target is in protected namespace"
                }
            ),
            SafetyRule(
                name="max_actions_per_minute",
                condition="max_actions_per_minute",
                threshold=10,
                action="block",
                parameters={
                    "message": "Action blocked: rate limit exceeded"
                }
            ),
            SafetyRule(
                name="circuit_breaker",
                condition="circuit_breaker_tripped",
                threshold=None,
                action="block",
                parameters={
                    "message": "Action blocked: circuit breaker is tripped"
                }
            ),
            SafetyRule(
                name="critical_pod",
                condition="critical_pod",
                threshold=None,
                action="require_approval",
                parameters={
                    "message": "Action requires manual approval: target is critical pod"
                }
            ),
            SafetyRule(
                name="consecutive_false_positives",
                condition="consecutive_false_positives",
                threshold=3,
                action="block",
                parameters={
                    "message": "Action blocked: too many consecutive false positives"
                }
            )
        ]
    
    def is_action_allowed(self, 
                         action_type: str,
                         target_pod: str,
                         parameters: Dict[str, Any]) -> bool:
        """Check if action is allowed by safety rules"""
        
        # Check rate limiting
        self._reset_counts_if_needed()
        
        # Check each rule
        context = {
            'action_type': action_type,
            'target_pod': target_pod,
            'parameters': parameters,
            'action_history': self.action_history[-100:],  # Last 100 actions
            'action_counts': self.action_counts
        }
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if self._evaluate_condition(rule.condition, rule.threshold, context):
                logger.warning(f"Safety rule triggered: {rule.name}")
                
                if rule.action == "block":
                    return False
                elif rule.action == "require_approval":
                    # In production, this would trigger an approval workflow
                    logger.info(f"Action requires approval: {rule.parameters.get('message')}")
                    # For MVP, we allow but log
                    return True
        
        return True
    
    def _evaluate_condition(self, 
                           condition: str, 
                           threshold: Any,
                           context: Dict[str, Any]) -> bool:
        """Evaluate safety condition"""
        
        if condition == "protected_namespace":
            target_pod = context.get('target_pod', '')
            # Check if pod is in protected namespace
            for namespace in self.protected_namespaces:
                if f"{namespace}/" in target_pod or target_pod.startswith(namespace):
                    return True
        
        elif condition == "max_actions_per_minute":
            current_minute = datetime.now().minute
            minute_key = f"{datetime.now().hour}:{current_minute}"
            if self.action_counts.get(minute_key, 0) >= threshold:
                return True
        
        elif condition == "circuit_breaker_tripped":
            action_type = context.get('action_type')
            if self.circuit_breakers.get(action_type, False):
                return True
        
        elif condition == "critical_pod":
            target_pod = context.get('target_pod', '')
            # Check if pod name starts with protected prefix
            for prefix in self.protected_pod_prefixes:
                if target_pod.startswith(prefix):
                    return True
        
        elif condition == "consecutive_false_positives":
            # Check last N actions for this pod
            target_pod = context.get('target_pod', '')
            recent_actions = [
                a for a in context.get('action_history', [])
                if a.get('target_pod') == target_pod
            ][-threshold:]  # Last N actions for this pod
            
            if len(recent_actions) >= threshold:
                # Check if all were false positives (manually marked or rolled back)
                all_false_positives = all(
                    a.get('false_positive', False) or a.get('rolled_back', False)
                    for a in recent_actions
                )
                if all_false_positives:
                    return True
        
        return False
    
    def record_action(self, action_record: Dict[str, Any]):
        """Record an executed action for safety analysis"""
        self.action_history.append(action_record)
        
        # Update rate limiting counts
        current_minute = datetime.now().minute
        minute_key = f"{datetime.now().hour}:{current_minute}"
        self.action_counts[minute_key] = self.action_counts.get(minute_key, 0) + 1
        
        # Check for circuit breaker conditions
        self._check_circuit_breakers()
    
    def _check_circuit_breakers(self):
        """Check if circuit breakers should be tripped"""
        # Trip circuit breaker if too many actions in short time
        recent_actions = self.action_history[-20:]  # Last 20 actions
        if len(recent_actions) >= 15:
            # Check time span
            if recent_actions:
                first_time = datetime.fromisoformat(recent_actions[0].get('timestamp', ''))
                last_time = datetime.fromisoformat(recent_actions[-1].get('timestamp', ''))
                time_span = (last_time - first_time).total_seconds()
                
                if time_span < 60:  # 15+ actions in less than 60 seconds
                    # Trip circuit breaker for all action types in recent actions
                    for action in recent_actions:
                        action_type = action.get('action_type')
                        self.circuit_breakers[action_type] = True
                    
                    logger.critical("Circuit breaker tripped: too many actions in short time")
    
    def _reset_counts_if_needed(self):
        """Reset action counts if new minute"""
        now = datetime.now()
        if now.minute != self.last_reset.minute:
            self.action_counts.clear()
            self.last_reset = now
    
    def mark_false_positive(self, action_id: str):
        """Mark an action as false positive"""
        for action in self.action_history:
            if action.get('action_id') == action_id:
                action['false_positive'] = True
                action['false_positive_time'] = datetime.now().isoformat()
                logger.info(f"Marked action {action_id} as false positive")
                break
    
    def reset_circuit_breaker(self, action_type: Optional[str] = None):
        """Reset circuit breaker"""
        if action_type:
            self.circuit_breakers[action_type] = False
            logger.info(f"Circuit breaker reset for {action_type}")
        else:
            self.circuit_breakers.clear()
            logger.info("All circuit breakers reset")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get safety controller status"""
        return {
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules if r.enabled),
            'total_actions_recorded': len(self.action_history),
            'circuit_breakers_tripped': sum(1 for v in self.circuit_breakers.values() if v),
            'actions_this_minute': self.action_counts.get(
                f"{datetime.now().hour}:{datetime.now().minute}", 0
            )
        }