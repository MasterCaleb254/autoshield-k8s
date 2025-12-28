# src/autoshield/policy/engine.py
"""
Policy engine for autonomous response decisions.
Evaluates detection results against configurable policies.
"""
import yaml
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from pydantic import BaseModel, Field, validator
import hashlib

from ...utils.logging import setup_logger

logger = setup_logger(__name__)

class ActionType(Enum):
    """Types of mitigation actions"""
    NETWORK_POLICY = "network_policy"
    POD_ISOLATION = "pod_isolation"
    POD_TERMINATION = "pod_termination"
    TRAFFIC_THROTTLE = "traffic_throttle"
    NODE_QUARANTINE = "node_quarantine"
    ALERT_ONLY = "alert_only"
    NO_ACTION = "no_action"

class AttackSeverity(Enum):
    """Attack severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PolicyRule:
    """Individual policy rule"""
    id: str
    name: str
    description: str
    attack_class: str  # e.g., "LATERAL_MOVEMENT", "PORT_SCAN"
    severity: AttackSeverity
    confidence_threshold: float  # Minimum confidence for action (0.0-1.0)
    actions: List[Dict[str, Any]]  # List of actions with parameters
    cooldown_seconds: int = 300  # Minimum time between same-rule actions
    blast_radius_limit: int = 3  # Max affected pods before escalation
    enabled: bool = True
    
    def should_trigger(self, 
                      detection_result: Dict[str, Any],
                      recent_actions: List[Dict[str, Any]]) -> bool:
        """Check if rule should trigger based on detection and context"""
        if not self.enabled:
            return False
        
        # Check attack class match
        if detection_result.get('predicted_class') != self.attack_class:
            return False
        
        # Check confidence threshold
        confidence = detection_result.get('confidence', 0.0)
        if confidence < self.confidence_threshold:
            return False
        
        # Check cooldown period
        if self._is_in_cooldown(recent_actions):
            logger.debug(f"Rule {self.id} in cooldown")
            return False
        
        # Check blast radius
        if self._exceeds_blast_radius(recent_actions):
            logger.warning(f"Rule {self.id} exceeds blast radius limit")
            # Return True anyway, but this will trigger escalation
            return True
        
        return True
    
    def _is_in_cooldown(self, recent_actions: List[Dict[str, Any]]) -> bool:
        """Check if rule was recently triggered"""
        if not recent_actions:
            return False
        
        now = datetime.now()
        for action in recent_actions[-10:]:  # Check last 10 actions
            if action.get('rule_id') == self.id:
                action_time = datetime.fromisoformat(action.get('timestamp', ''))
                if (now - action_time).seconds < self.cooldown_seconds:
                    return True
        return False
    
    def _exceeds_blast_radius(self, recent_actions: List[Dict[str, Any]]) -> bool:
        """Check if too many pods have been affected recently"""
        if not recent_actions:
            return False
        
        # Count unique pods affected by this rule in last hour
        hour_ago = datetime.now() - timedelta(hours=1)
        affected_pods = set()
        
        for action in recent_actions:
            if action.get('rule_id') == self.id:
                action_time = datetime.fromisoformat(action.get('timestamp', ''))
                if action_time > hour_ago:
                    affected_pods.add(action.get('target_pod', ''))
        
        return len(affected_pods) >= self.blast_radius_limit
    
    def select_action(self, 
                     detection_result: Dict[str, Any],
                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate action based on context"""
        # In a real system, this would be more sophisticated
        # For MVP, we return the first action
        if not self.actions:
            return {
                'action_type': ActionType.ALERT_ONLY.value,
                'parameters': {}
            }
        
        # Choose action based on severity and confidence
        confidence = detection_result.get('confidence', 0.0)
        
        # If confidence is very high and severity is high, use more aggressive actions
        if (confidence > 0.95 and 
            self.severity in [AttackSeverity.HIGH, AttackSeverity.CRITICAL]):
            # Find the most aggressive action
            aggressive_actions = [a for a in self.actions 
                                if a.get('aggressiveness', 0) > 7]
            if aggressive_actions:
                return aggressive_actions[0]
        
        # Default to first action
        action_config = self.actions[0].copy()
        
        # Add dynamic parameters
        action_config['parameters']['confidence'] = confidence
        action_config['parameters']['severity'] = self.severity.value
        action_config['parameters']['source_pod'] = detection_result.get('src_pod', '')
        
        return action_config

class PolicySchema(BaseModel):
    """Pydantic schema for policy validation"""
    version: str = Field(default="1.0.0")
    name: str
    description: str
    default_action: Dict[str, Any] = Field(
        default={"action_type": "alert_only", "parameters": {}}
    )
    rules: List[Dict[str, Any]]
    
    @validator('rules')
    def validate_rules(cls, v):
        for rule in v:
            if 'confidence_threshold' in rule:
                threshold = rule['confidence_threshold']
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"Invalid confidence threshold: {threshold}")
        return v

class PolicyEngine:
    """Main policy evaluation engine"""
    
    def __init__(self, policy_file: str = "config/policies/default.yaml"):
        self.policy_file = policy_file
        self.rules: Dict[str, PolicyRule] = {}
        self.recent_actions: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
        # Load policies
        self.load_policies()
        
        # Statistics
        self.stats = {
            "total_evaluations": 0,
            "actions_triggered": 0,
            "false_positives": 0,
            "avg_evaluation_time_ms": 0
        }
        
        logger.info(f"Policy engine initialized with {len(self.rules)} rules")
    
    def load_policies(self, policy_file: Optional[str] = None):
        """Load policies from YAML file"""
        file_path = policy_file or self.policy_file
        
        try:
            with open(file_path, 'r') as f:
                policy_data = yaml.safe_load(f)
            
            # Validate schema
            policy = PolicySchema(**policy_data)
            
            # Parse rules
            self.rules.clear()
            for rule_data in policy.rules:
                rule_id = rule_data.get('id') or hashlib.md5(
                    json.dumps(rule_data, sort_keys=True).encode()
                ).hexdigest()[:8]
                
                rule = PolicyRule(
                    id=rule_id,
                    name=rule_data.get('name', 'Unnamed Rule'),
                    description=rule_data.get('description', ''),
                    attack_class=rule_data.get('attack_class'),
                    severity=AttackSeverity(rule_data.get('severity', 'medium')),
                    confidence_threshold=rule_data.get('confidence_threshold', 0.85),
                    actions=rule_data.get('actions', []),
                    cooldown_seconds=rule_data.get('cooldown_seconds', 300),
                    blast_radius_limit=rule_data.get('blast_radius_limit', 3),
                    enabled=rule_data.get('enabled', True)
                )
                
                self.rules[rule_id] = rule
            
            logger.info(f"Loaded {len(self.rules)} rules from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to load policies from {file_path}: {e}")
            raise
    
    def evaluate(self, 
                detection_result: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Evaluate detection result against policies.
        
        Returns:
            Action to take, or None if no action needed
        """
        import time
        start_time = time.time()
        
        self.stats["total_evaluations"] += 1
        
        context = context or {}
        attack_class = detection_result.get('predicted_class')
        
        if not attack_class or attack_class == "NORMAL":
            # No action needed for normal traffic
            return None
        
        # Find matching rules
        matching_rules = []
        for rule in self.rules.values():
            if rule.attack_class == attack_class:
                matching_rules.append(rule)
        
        if not matching_rules:
            logger.warning(f"No policy rules found for attack class: {attack_class}")
            return None
        
        # Sort rules by severity (most severe first)
        matching_rules.sort(key=lambda r: r.severity.value, reverse=True)
        
        # Evaluate each rule
        for rule in matching_rules:
            if rule.should_trigger(detection_result, self.recent_actions):
                # Rule triggered, select action
                action_config = rule.select_action(detection_result, context)
                
                # Create action record
                action_record = {
                    'action_id': hashlib.md5(
                        f"{rule.id}_{detection_result.get('window_id')}".encode()
                    ).hexdigest()[:8],
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                    'detection_result': detection_result,
                    'action_config': action_config,
                    'timestamp': datetime.now().isoformat(),
                    'target_pod': detection_result.get('src_pod'),
                    'severity': rule.severity.value,
                    'confidence': detection_result.get('confidence')
                }
                
                # Add to recent actions (FIFO, max 100)
                self.recent_actions.append(action_record)
                if len(self.recent_actions) > 100:
                    self.recent_actions.pop(0)
                
                # Add to history
                self.action_history.append(action_record)
                if len(self.action_history) > self.max_history_size:
                    self.action_history.pop(0)
                
                # Update statistics
                self.stats["actions_triggered"] += 1
                
                # Calculate evaluation time
                eval_time_ms = (time.time() - start_time) * 1000
                if self.stats["total_evaluations"] == 1:
                    self.stats["avg_evaluation_time_ms"] = eval_time_ms
                else:
                    alpha = 0.1
                    self.stats["avg_evaluation_time_ms"] = (
                        alpha * eval_time_ms + 
                        (1 - alpha) * self.stats["avg_evaluation_time_ms"]
                    )
                
                logger.info(f"Policy triggered: {rule.name} -> {action_config.get('action_type')}")
                logger.info(f"  Target: {detection_result.get('src_pod')}")
                logger.info(f"  Confidence: {detection_result.get('confidence'):.2%}")
                logger.info(f"  Rule severity: {rule.severity.value}")
                
                return action_record
        
        # No rules triggered
        logger.debug(f"No policy rules triggered for {attack_class}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get policy engine statistics"""
        return {
            **self.stats,
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for r in self.rules.values() if r.enabled),
            'recent_actions_count': len(self.recent_actions),
            'total_action_history': len(self.action_history)
        }
    
    def get_action_history(self, 
                          limit: int = 50,
                          filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get recent action history with optional filtering"""
        history = self.action_history.copy()
        
        if filter_by:
            filtered = []
            for action in history:
                match = True
                for key, value in filter_by.items():
                    if action.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(action)
            history = filtered
        
        return history[-limit:] if limit else history
    
    def reset_cooldown(self, rule_id: Optional[str] = None):
        """Reset cooldown for a specific rule or all rules"""
        if rule_id:
            # Remove recent actions for this rule
            self.recent_actions = [
                a for a in self.recent_actions 
                if a.get('rule_id') != rule_id
            ]
            logger.info(f"Cooldown reset for rule: {rule_id}")
        else:
            # Reset all
            self.recent_actions.clear()
            logger.info("Cooldown reset for all rules")