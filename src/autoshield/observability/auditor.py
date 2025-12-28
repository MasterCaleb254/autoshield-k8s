# src/autoshield/observability/auditor.py
"""
Audit logging for compliance and forensics.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Any
import csv
import os
from pathlib import Path

class AuditLogger:
    """Structured audit logging for all system actions"""
    
    def __init__(self, log_dir: str = "logs/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up structured logging
        self.logger = logging.getLogger("autoshield_audit")
        self.logger.setLevel(logging.INFO)
        
        # File handler for JSON logs
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(file_handler)
        
        # CSV handler for easy analysis
        self.csv_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_csv_logger()
        
        logger.info(f"Audit logging initialized: {log_dir}")
    
    def _init_csv_logger(self):
        """Initialize CSV logging with headers"""
        if not self.csv_file.exists():
            with open(self.csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'event_type', 'window_id', 'src_pod', 'dst_pod',
                    'attack_class', 'confidence', 'action_taken', 'action_result',
                    'risk_score', 'explanation_summary'
                ])
    
    def log_processing_result(self, result: Dict[str, Any]):
        """Log complete processing result"""
        
        # Extract key information
        detection = result.get('detection_result', {})
        policy = result.get('policy_decision', {})
        action = result.get('action_result', {})
        
        # Create audit record
        audit_record = {
            'timestamp': result.get('timestamp', datetime.now().isoformat()),
            'event_type': 'processing_result',
            'window_id': result.get('window_id'),
            'src_pod': result.get('src_pod'),
            'dst_pod': result.get('dst_pod'),
            'detection': detection,
            'policy_decision': policy,
            'action_result': action,
            'enhanced_explanation': result.get('enhanced_explanation'),
            'full_context': result
        }
        
        # Log to JSONL
        self.logger.info(json.dumps(audit_record, default=str))
        
        # Log to CSV
        self._log_to_csv(
            timestamp=audit_record['timestamp'],
            window_id=audit_record['window_id'],
            src_pod=audit_record['src_pod'],
            dst_pod=audit_record['dst_pod'],
            attack_class=detection.get('predicted_class', 'NORMAL'),
            confidence=detection.get('confidence', 0.0),
            action_taken=policy.get('rule_name') if policy else 'none',
            action_result=action.get('status') if action else 'none',
            risk_score=result.get('enhanced_explanation', {}).get('risk_score', {}).get('score', 0),
            explanation_summary=result.get('enhanced_explanation', {}).get('human_summary', '')[:200]
        )
    
    def log_error(self, error_result: Dict[str, Any]):
        """Log error events"""
        error_record = {
            'timestamp': error_result.get('timestamp', datetime.now().isoformat()),
            'event_type': 'error',
            'window_id': error_result.get('window_id'),
            'error': error_result.get('error'),
            'context': error_result
        }
        
        self.logger.error(json.dumps(error_record, default=str))
    
    def log_action(self, action_type: str, target: str, result: Dict[str, Any]):
        """Log individual action"""
        action_record = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'action',
            'action_type': action_type,
            'target': target,
            'result': result
        }
        
        self.logger.info(json.dumps(action_record, default=str))
    
    def _log_to_csv(self, **kwargs):
        """Log to CSV file"""
        try:
            with open(self.csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    kwargs.get('timestamp', ''),
                    kwargs.get('window_id', ''),
                    kwargs.get('src_pod', ''),
                    kwargs.get('dst_pod', ''),
                    kwargs.get('attack_class', ''),
                    kwargs.get('confidence', 0.0),
                    kwargs.get('action_taken', ''),
                    kwargs.get('action_result', ''),
                    kwargs.get('risk_score', 0),
                    kwargs.get('explanation_summary', '')
                ])
        except Exception as e:
            logging.error(f"Failed to write to CSV: {e}")
    
    def get_recent_logs(self, limit: int = 100, event_type: Optional[str] = None) -> list:
        """Get recent audit logs"""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        if not log_file.exists():
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    log = json.loads(line.strip())
                    if not event_type or log.get('event_type') == event_type:
                        logs.append(log)
                except:
                    continue
        
        return logs[-limit:] if limit else logs