# src/autoshield/observability/explainer.py
"""
Explainability engine for human-readable explanations.
"""
import json
from typing import Dict, Any, Optional
from datetime import datetime

class ExplainabilityEngine:
    """Generates human-readable explanations for decisions"""
    
    def __init__(self):
        self.template_repository = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """Load explanation templates"""
        return {
            "lateral_movement": """
**Lateral Movement Detected**

**Summary**: Pod `{src_pod}` is attempting to access multiple services it doesn't normally communicate with.

**Key Indicators**:
- Attempted connections to `{unique_ports}` unique ports
- Connection failure rate: `{failed_ratio:.0%}`
- Unusual traffic pattern detected with `{confidence:.0%}` confidence

**Risk Assessment**: HIGH - This behavior is consistent with lateral movement attacks where compromised pods scan the network for vulnerable services.

**Recommended Investigation**:
1. Check if `{src_pod}` is running expected workloads
2. Review recent deployments to this pod
3. Check for suspicious processes in the pod
""",
            "port_scan": """
**Port Scanning Detected**

**Summary**: Pod `{src_pod}` is scanning ports on `{dst_pod}`.

**Key Indicators**:
- Scanned `{unique_ports}` different ports
- Rapid connection attempts: `{flow_count}` in `{duration:.1f}s`
- Low connection success rate: `{failed_ratio:.0%}`

**Risk Assessment**: MEDIUM - This could be reconnaissance for future attacks.

**Recommended Actions**:
1. Review if this is legitimate service discovery
2. Consider implementing network policies to restrict port access
3. Monitor for follow-up attack patterns
""",
            "syn_flood": """
**SYN Flood (DoS) Detected**

**Summary**: Potential denial-of-service attack targeting `{dst_pod}`.

**Key Indicators**:
- Excessive SYN packets: `{syn_count}` with minimal ACK responses
- High packet rate: `{avg_interarrival:.1f}ms` between packets
- Targeted port: `{target_port}`

**Risk Assessment**: CRITICAL - This can degrade service availability.

**Immediate Actions**:
1. Enable DDoS protection if available
2. Consider rate limiting traffic to the target
3. Investigate source for compromise
"""
        }
    
    def enhance_explanation(self,
                          detection_result: Dict[str, Any],
                          policy_decision: Optional[Dict[str, Any]],
                          action_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create enhanced explanation with context"""
        
        attack_class = detection_result.get('predicted_class', 'UNKNOWN')
        confidence = detection_result.get('confidence', 0.0)
        features = detection_result.get('features', {})
        
        # Base explanation from detection
        base_explanation = json.loads(detection_result.get('explanation', '{}'))
        
        # Add policy context
        if policy_decision:
            base_explanation['policy'] = {
                'rule_name': policy_decision.get('rule_name'),
                'severity': policy_decision.get('severity'),
                'confidence_threshold': policy_decision.get('confidence')
            }
        
        # Add action context
        if action_result:
            base_explanation['action'] = {
                'type': action_result.get('action_type'),
                'status': action_result.get('status'),
                'target': action_result.get('target'),
                'timestamp': action_result.get('timestamp')
            }
        
        # Generate human-readable summary
        template_key = attack_class.lower().replace(' ', '_')
        template = self.template_repository.get(template_key, "")
        
        if template:
            try:
                summary = template.format(
                    src_pod=detection_result.get('src_pod', 'unknown'),
                    dst_pod=detection_result.get('dst_pod', 'unknown'),
                    unique_ports=features.get('unique_ports', 0),
                    failed_ratio=features.get('failed_conn_ratio', 0.0),
                    confidence=confidence,
                    flow_count=features.get('flow_count', 0),
                    duration=features.get('total_duration_s', 0),
                    syn_count=features.get('syn_count', 0),
                    avg_interarrival=features.get('avg_interarrival_ms', 0),
                    target_port="multiple"  # Simplified
                )
                base_explanation['human_summary'] = summary
            except Exception as e:
                base_explanation['human_summary'] = f"Error generating summary: {e}"
        
        # Add risk scoring
        base_explanation['risk_score'] = self._calculate_risk_score(
            attack_class, confidence, features
        )
        
        # Add recommendations
        base_explanation['recommendations'] = self._generate_recommendations(
            attack_class, confidence, action_result
        )
        
        return base_explanation
    
    def _calculate_risk_score(self, 
                            attack_class: str, 
                            confidence: float,
                            features: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk score (0-100)"""
        
        # Base score based on attack class
        class_scores = {
            'NORMAL': 0,
            'LATERAL_MOVEMENT': 70,
            'PORT_SCAN': 50,
            'SYN_FLOOD': 90,
            'BRUTE_FORCE': 60,
            'DATA_EXFILTRATION': 80
        }
        
        base_score = class_scores.get(attack_class, 50)
        
        # Adjust based on confidence
        adjusted_score = base_score * confidence
        
        # Adjust based on features
        feature_adjustments = 0
        
        if features.get('failed_conn_ratio', 0) > 0.5:
            feature_adjustments += 10
        
        if features.get('unique_ports', 0) > 10:
            feature_adjustments += 15
        
        if features.get('syn_count', 0) > features.get('ack_count', 1) * 5:
            feature_adjustments += 20
        
        final_score = min(100, adjusted_score + feature_adjustments)
        
        return {
            'score': final_score,
            'level': self._risk_level(final_score),
            'factors': {
                'attack_class': base_score,
                'confidence_multiplier': confidence,
                'feature_adjustments': feature_adjustments
            }
        }
    
    def _risk_level(self, score: float) -> str:
        """Convert score to risk level"""
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 20:
            return "LOW"
        else:
            return "INFO"
    
    def _generate_recommendations(self,
                                attack_class: str,
                                confidence: float,
                                action_result: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations"""
        
        recommendations = []
        
        # Base recommendations
        if attack_class == 'LATERAL_MOVEMENT':
            recommendations.extend([
                "Investigate source pod for compromise",
                "Review network policies for pod communication",
                "Consider implementing service mesh for finer control"
            ])
        elif attack_class == 'PORT_SCAN':
            recommendations.extend([
                "Validate if this is legitimate service discovery",
                "Implement network policies to restrict port access",
                "Monitor for follow-up attack patterns"
            ])
        elif attack_class == 'SYN_FLOOD':
            recommendations.extend([
                "Enable DDoS protection mechanisms",
                "Consider rate limiting at network layer",
                "Scale target service if under attack"
            ])
        
        # Action-specific recommendations
        if action_result:
            if action_result.get('action_type') == 'network_policy':
                recommendations.append(
                    "Review NetworkPolicy periodically and remove if no longer needed"
                )
            elif action_result.get('action_type') == 'pod_termination':
                recommendations.append(
                    "Check if terminated pod was recreated and if behavior persists"
                )
        
        # Confidence-based recommendations
        if confidence < 0.8:
            recommendations.append(
                "Low confidence detection - consider manual verification before taking action"
            )
        
        return recommendations