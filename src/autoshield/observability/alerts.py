# src/autoshield/observability/alerts.py
"""
Alert manager for sending notifications.
"""
import json
import requests
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertChannel(Enum):
    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"
    LOG = "log"

class Alert:
    """Alert definition"""
    
    def __init__(self, 
                 title: str,
                 message: str,
                 severity: AlertSeverity = AlertSeverity.INFO,
                 source: str = "autoshield",
                 metadata: Optional[Dict] = None):
        
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
        self.acknowledged = False
        self.resolved = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "source": self.source,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }

class AlertManager:
    """Manages alert generation and delivery"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        self.alert_rules = self._load_alert_rules()
        
        # Alert channels
        self.channels = {
            AlertChannel.LOG: self._send_to_log,
            AlertChannel.WEBHOOK: self._send_to_webhook,
            # Add more channels as needed
        }
        
        logger.info("Alert manager initialized")
    
    def _load_alert_rules(self) -> List[Dict]:
        """Load alert rules from configuration"""
        default_rules = [
            {
                "name": "high_confidence_attack",
                "condition": "confidence > 0.95 AND attack_class != 'NORMAL'",
                "severity": AlertSeverity.CRITICAL,
                "channels": [AlertChannel.LOG, AlertChannel.WEBHOOK]
            },
            {
                "name": "multiple_false_positives",
                "condition": "false_positives > 5 IN last_hour",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.LOG]
            },
            {
                "name": "system_unhealthy",
                "condition": "system_status != 'healthy' FOR 5_minutes",
                "severity": AlertSeverity.ERROR,
                "channels": [AlertChannel.LOG, AlertChannel.WEBHOOK]
            },
            {
                "name": "high_latency",
                "condition": "p95_latency > 2.0 FOR 10_minutes",
                "severity": AlertSeverity.WARNING,
                "channels": [AlertChannel.LOG]
            }
        ]
        
        return default_rules
    
    def create_alert(self, 
                    title: str,
                    message: str,
                    severity: AlertSeverity = AlertSeverity.INFO,
                    channels: Optional[List[AlertChannel]] = None,
                    metadata: Optional[Dict] = None) -> Alert:
        """Create and send an alert"""
        
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            metadata=metadata
        )
        
        # Store alert
        self.alerts.append(alert)
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop(0)
        
        # Determine channels
        if channels is None:
            # Use default channels based on severity
            if severity == AlertSeverity.CRITICAL:
                channels = [AlertChannel.LOG, AlertChannel.WEBHOOK]
            elif severity == AlertSeverity.ERROR:
                channels = [AlertChannel.LOG, AlertChannel.WEBHOOK]
            elif severity == AlertSeverity.WARNING:
                channels = [AlertChannel.LOG]
            else:
                channels = [AlertChannel.LOG]
        
        # Send to channels
        for channel in channels:
            if channel in self.channels:
                try:
                    self.channels[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel}: {e}")
        
        logger.info(f"Alert created: {title} ({severity.value})")
        
        return alert
    
    def _send_to_log(self, alert: Alert):
        """Send alert to log"""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical
        }.get(alert.severity, logger.info)
        
        log_method(f"ALERT [{alert.severity.value}]: {alert.title} - {alert.message}")
    
    def _send_to_webhook(self, alert: Alert):
        """Send alert to webhook"""
        webhook_url = self.config.get("webhook_url")
        if not webhook_url:
            return
        
        payload = {
            "text": f"🚨 *{alert.title}*",
            "attachments": [{
                "color": self._get_slack_color(alert.severity),
                "fields": [
                    {"title": "Message", "value": alert.message, "short": False},
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Source", "value": alert.source, "short": True},
                    {"title": "Time", "value": alert.timestamp, "short": True}
                ]
            }]
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _get_slack_color(self, severity: AlertSeverity) -> str:
        """Get Slack attachment color for severity"""
        colors = {
            AlertSeverity.INFO: "#3498db",
            AlertSeverity.WARNING: "#f39c12",
            AlertSeverity.ERROR: "#e74c3c",
            AlertSeverity.CRITICAL: "#8b0000"
        }
        return colors.get(severity, "#3498db")
    
    def evaluate_alert_rules(self, context: Dict[str, Any]):
        """Evaluate alert rules based on current context"""
        for rule in self.alert_rules:
            try:
                # Simplified rule evaluation
                # In production, this would use a proper rule engine
                if self._evaluate_condition(rule["condition"], context):
                    self.create_alert(
                        title=f"Rule triggered: {rule['name']}",
                        message=f"Alert rule '{rule['name']}' was triggered",
                        severity=rule["severity"],
                        channels=rule.get("channels", [AlertChannel.LOG]),
                        metadata={"rule": rule, "context": context}
                    )
            except Exception as e:
                logger.error(f"Failed to evaluate alert rule {rule['name']}: {e}")
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string against context"""
        # Very simplified condition evaluation
        # In production, use a proper expression evaluator
        
        if "confidence > 0.95" in condition:
            confidence = context.get("confidence", 0)
            attack_class = context.get("attack_class", "NORMAL")
            if confidence > 0.95 and attack_class != "NORMAL":
                return True
        
        if "false_positives > 5" in condition:
            false_positives = context.get("false_positives", 0)
            if false_positives > 5:
                return True
        
        if "system_status != 'healthy'" in condition:
            system_status = context.get("system_status", "healthy")
            if system_status != "healthy":
                return True
        
        if "p95_latency > 2.0" in condition:
            p95_latency = context.get("p95_latency", 0)
            if p95_latency > 2.0:
                return True
        
        return False
    
    def get_alerts(self, 
                  limit: int = 50,
                  severity: Optional[AlertSeverity] = None,
                  resolved: Optional[bool] = None) -> List[Dict]:
        """Get alerts with optional filtering"""
        alerts = self.alerts.copy()
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return [a.to_dict() for a in alerts[-limit:]]
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
    
    def resolve_alert(self, alert_index: int):
        """Resolve an alert"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
    
    def clear_resolved_alerts(self):
        """Clear resolved alerts"""
        self.alerts = [a for a in self.alerts if not a.resolved]