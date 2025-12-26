# autoshield/collector.py
import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Optional
import subprocess
import threading

logger = logging.getLogger(__name__)

class HubbleFlowCollector:
    """Collects flow events from Hubble Relay"""
    
    def __init__(self, kubeconfig: Optional[str] = None, namespace: str = "kube-system"):
        self.kubeconfig = kubeconfig
        self.namespace = namespace
        self.running = False
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """Add callback for flow events"""
        self.callbacks.append(callback)
    
    def _run_hubble_stream(self):
        """Run Hubble observe command and stream output"""
        cmd = [
            "kubectl", "exec", "-n", self.namespace, "-it",
            "deploy/cilium-operator", "--",
            "hubble", "observe",
            "--output", "json",
            "--follow"
        ]
        
        if self.kubeconfig:
            cmd = ["kubectl", "--kubeconfig", self.kubeconfig] + cmd[1:]
        
        logger.info(f"Starting Hubble stream: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            while self.running and process.poll() is None:
                line = process.stdout.readline()
                if line:
                    try:
                        event = json.loads(line.strip())
                        # Notify all callbacks
                        for callback in self.callbacks:
                            try:
                                callback(event)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse JSON: {e}")
                
                # Check for errors
                err_line = process.stderr.readline()
                if err_line:
                    logger.error(f"Hubble error: {err_line}")
            
            if process.returncode and process.returncode != 0:
                logger.error(f"Hubble process exited with code: {process.returncode}")
                
        except Exception as e:
            logger.error(f"Hubble stream error: {e}")
    
    def start(self):
        """Start collecting flows"""
        self.running = True
        thread = threading.Thread(target=self._run_hubble_stream, daemon=True)
        thread.start()
        logger.info("Hubble collector started")
    
    def stop(self):
        """Stop collecting flows"""
        self.running = False
        logger.info("Hubble collector stopped")