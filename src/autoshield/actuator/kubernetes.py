# src/autoshield/actuator/kubernetes.py
"""
Kubernetes actuator for executing mitigation actions.
"""
import yaml
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from kubernetes.stream import stream

from ...utils.logging import setup_logger
from .safety_controller import SafetyController

logger = setup_logger(__name__)

class KubernetesActuator:
    """Executes Kubernetes-native mitigation actions"""
    
    def __init__(self, 
                 kubeconfig: Optional[str] = None,
                 namespace: str = "autoshield-system"):
        
        # Load kubeconfig
        try:
            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                config.load_incluster_config()  # For running inside cluster
        except Exception as e:
            logger.error(f"Failed to load kubeconfig: {e}")
            # Fallback to default
            config.load_kube_config()
        
        # Initialize Kubernetes clients
        self.core_v1 = client.CoreV1Api()
        self.apps_v1 = client.AppsV1Api()
        self.networking_v1 = client.NetworkingV1Api()
        self.custom_objects_api = client.CustomObjectsApi()  # For Cilium CRDs
        
        self.namespace = namespace
        
        # Safety controller
        self.safety_controller = SafetyController()
        
        # Action history
        self.action_history: List[Dict[str, Any]] = []
        
        logger.info("Kubernetes actuator initialized")
    
    def execute_action(self, 
                      action_config: Dict[str, Any],
                      detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a mitigation action.
        
        Returns:
            Action result with status and metadata
        """
        action_type = action_config.get('action_type')
        parameters = action_config.get('parameters', {})
        
        # Get target pod info
        target_pod = detection_result.get('src_pod')
        target_namespace = self._extract_namespace(target_pod) or "default"
        
        # Safety check
        if not self.safety_controller.is_action_allowed(
            action_type, target_pod, parameters
        ):
            logger.warning(f"Safety check failed for action: {action_type} on {target_pod}")
            return {
                'status': 'blocked',
                'reason': 'safety_check_failed',
                'action_type': action_type,
                'target': target_pod,
                'timestamp': datetime.now().isoformat()
            }
        
        # Execute action based on type
        action_methods = {
            'network_policy': self._apply_network_policy,
            'pod_isolation': self._isolate_pod,
            'pod_termination': self._terminate_pod,
            'traffic_throttle': self._throttle_traffic,
            'node_quarantine': self._quarantine_node,
            'alert_only': self._send_alert
        }
        
        if action_type not in action_methods:
            logger.error(f"Unknown action type: {action_type}")
            return {
                'status': 'error',
                'reason': 'unknown_action_type',
                'action_type': action_type
            }
        
        try:
            # Execute the action
            result = action_methods[action_type](
                target_pod=target_pod,
                target_namespace=target_namespace,
                parameters=parameters,
                detection_result=detection_result
            )
            
            # Record action
            action_record = {
                'action_type': action_type,
                'target_pod': target_pod,
                'target_namespace': target_namespace,
                'parameters': parameters,
                'detection_result': detection_result,
                'execution_result': result,
                'timestamp': datetime.now().isoformat(),
                'status': 'executed'
            }
            
            self.action_history.append(action_record)
            self.safety_controller.record_action(action_record)
            
            logger.info(f"Action executed: {action_type} on {target_pod}")
            
            return {
                'status': 'success',
                'action_type': action_type,
                'target': target_pod,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute action {action_type}: {e}")
            
            return {
                'status': 'error',
                'action_type': action_type,
                'target': target_pod,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _apply_network_policy(self, 
                            target_pod: str,
                            target_namespace: str,
                            parameters: Dict[str, Any],
                            detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Apply NetworkPolicy to isolate pod"""
        
        # Extract pod labels
        pod_labels = self._get_pod_labels(target_pod, target_namespace)
        if not pod_labels:
            raise ValueError(f"Pod {target_pod} not found in namespace {target_namespace}")
        
        # Create NetworkPolicy name
        policy_name = f"autoshield-block-{target_pod}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Determine direction
        direction = parameters.get('direction', 'egress')
        
        if direction == 'egress':
            # Block all egress from pod
            network_policy = {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'NetworkPolicy',
                'metadata': {
                    'name': policy_name,
                    'namespace': target_namespace,
                    'labels': {
                        'autoshield-managed': 'true',
                        'action': 'block-egress',
                        'target-pod': target_pod
                    }
                },
                'spec': {
                    'podSelector': {
                        'matchLabels': pod_labels
                    },
                    'policyTypes': ['Egress'],
                    'egress': []  # Empty array blocks all egress
                }
            }
        else:  # ingress
            # Block all ingress to pod
            network_policy = {
                'apiVersion': 'networking.k8s.io/v1',
                'kind': 'NetworkPolicy',
                'metadata': {
                    'name': policy_name,
                    'namespace': target_namespace,
                    'labels': {
                        'autoshield-managed': 'true',
                        'action': 'block-ingress',
                        'target-pod': target_pod
                    }
                },
                'spec': {
                    'podSelector': {
                        'matchLabels': pod_labels
                    },
                    'policyTypes': ['Ingress'],
                    'ingress': []  # Empty array blocks all ingress
                }
            }
        
        # Apply the NetworkPolicy
        try:
            existing_policies = self.networking_v1.list_namespaced_network_policy(
                namespace=target_namespace,
                label_selector=f"autoshield-managed=true,target-pod={target_pod}"
            )
            
            # Delete existing policies for this pod
            for policy in existing_policies.items:
                self.networking_v1.delete_namespaced_network_policy(
                    name=policy.metadata.name,
                    namespace=target_namespace
                )
                logger.info(f"Deleted old policy: {policy.metadata.name}")
            
            # Create new policy
            created_policy = self.networking_v1.create_namespaced_network_policy(
                namespace=target_namespace,
                body=network_policy
            )
            
            return {
                'policy_name': created_policy.metadata.name,
                'direction': direction,
                'namespace': target_namespace,
                'pod_labels': pod_labels,
                'yaml': yaml.dump(network_policy)
            }
            
        except ApiException as e:
            logger.error(f"Failed to apply NetworkPolicy: {e}")
            raise
    
    def _isolate_pod(self,
                    target_pod: str,
                    target_namespace: str,
                    parameters: Dict[str, Any],
                    detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Isolate pod by applying both ingress and egress blocking policies"""
        
        # Apply ingress block
        ingress_result = self._apply_network_policy(
            target_pod=target_pod,
            target_namespace=target_namespace,
            parameters={'direction': 'ingress'},
            detection_result=detection_result
        )
        
        # Apply egress block
        egress_result = self._apply_network_policy(
            target_pod=target_pod,
            target_namespace=target_namespace,
            parameters={'direction': 'egress'},
            detection_result=detection_result
        )
        
        # Add isolation annotation to pod
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=target_pod,
                namespace=target_namespace
            )
            
            # Add annotation
            if not pod.metadata.annotations:
                pod.metadata.annotations = {}
            
            pod.metadata.annotations['autoshield-isolated'] = datetime.now().isoformat()
            pod.metadata.annotations['autoshield-isolation-reason'] = detection_result.get('explanation', '')
            
            self.core_v1.patch_namespaced_pod(
                name=target_pod,
                namespace=target_namespace,
                body=pod
            )
            
        except ApiException as e:
            logger.warning(f"Could not annotate pod {target_pod}: {e}")
        
        return {
            'status': 'isolated',
            'ingress_policy': ingress_result.get('policy_name'),
            'egress_policy': egress_result.get('policy_name'),
            'namespace': target_namespace,
            'isolation_timestamp': datetime.now().isoformat()
        }
    
    def _terminate_pod(self,
                      target_pod: str,
                      target_namespace: str,
                      parameters: Dict[str, Any],
                      detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Terminate a pod"""
        
        grace_period = parameters.get('grace_period_seconds', 0)
        reason = parameters.get('reason', 'autoshield_mitigation')
        
        try:
            # Get pod details before deletion
            pod = self.core_v1.read_namespaced_pod(
                name=target_pod,
                namespace=target_namespace
            )
            
            # Check if pod is part of a deployment
            owner_refs = pod.metadata.owner_references or []
            deployment_name = None
            
            for ref in owner_refs:
                if ref.kind == 'ReplicaSet':
                    # Get the ReplicaSet to find its Deployment
                    try:
                        rs = self.apps_v1.read_namespaced_replica_set(
                            name=ref.name,
                            namespace=target_namespace
                        )
                        rs_owner_refs = rs.metadata.owner_references or []
                        for rs_ref in rs_owner_refs:
                            if rs_ref.kind == 'Deployment':
                                deployment_name = rs_ref.name
                    except:
                        pass
            
            # Delete the pod
            self.core_v1.delete_namespaced_pod(
                name=target_pod,
                namespace=target_namespace,
                grace_period_seconds=grace_period,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground'
                )
            )
            
            result = {
                'status': 'terminated',
                'pod_name': target_pod,
                'namespace': target_namespace,
                'grace_period': grace_period,
                'reason': reason,
                'termination_time': datetime.now().isoformat()
            }
            
            if deployment_name:
                result['deployment'] = deployment_name
                # The deployment will automatically recreate the pod
            
            logger.warning(f"Pod terminated: {target_pod} in {target_namespace}")
            
            return result
            
        except ApiException as e:
            logger.error(f"Failed to terminate pod {target_pod}: {e}")
            raise
    
    def _throttle_traffic(self,
                         target_pod: str,
                         target_namespace: str,
                         parameters: Dict[str, Any],
                         detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Throttle traffic using Cilium NetworkPolicy"""
        
        # This requires Cilium to be installed
        rate_limit = parameters.get('rate_limit', '100kbps')
        burst_limit = parameters.get('burst_limit', '200kbps')
        direction = parameters.get('direction', 'egress')
        
        # Create CiliumNetworkPolicy
        policy_name = f"autoshield-throttle-{target_pod}-{datetime.now().strftime('%H%M%S')}"
        
        cilium_policy = {
            'apiVersion': 'cilium.io/v2',
            'kind': 'CiliumNetworkPolicy',
            'metadata': {
                'name': policy_name,
                'namespace': target_namespace,
                'labels': {
                    'autoshield-managed': 'true',
                    'action': 'throttle',
                    'target-pod': target_pod
                }
            },
            'spec': {
                'endpointSelector': {
                    'matchLabels': self._get_pod_labels(target_pod, target_namespace)
                },
                'egress': [{
                    'toPorts': [{
                        'ports': [{'port': '0', 'protocol': 'ANY'}],
                        'rules': {
                            'l7': [{
                                'rateLimit': {
                                    'rateLimit': rate_limit,
                                    'burst': burst_limit
                                }
                            }]
                        }
                    }]
                }]
            }
        }
        
        try:
            # Apply CiliumNetworkPolicy
            created_policy = self.custom_objects_api.create_namespaced_custom_object(
                group='cilium.io',
                version='v2',
                namespace=target_namespace,
                plural='ciliumnetworkpolicies',
                body=cilium_policy
            )
            
            return {
                'policy_name': policy_name,
                'policy_type': 'CiliumNetworkPolicy',
                'rate_limit': rate_limit,
                'burst_limit': burst_limit,
                'direction': direction,
                'namespace': target_namespace
            }
            
        except ApiException as e:
            logger.error(f"Failed to create CiliumNetworkPolicy: {e}")
            
            # Fallback to regular annotation-based throttling if Cilium not available
            return self._fallback_throttle(target_pod, target_namespace, parameters)
    
    def _fallback_throttle(self,
                          target_pod: str,
                          target_namespace: str,
                          parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback throttling method using pod annotations"""
        
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=target_pod,
                namespace=target_namespace
            )
            
            # Add throttling annotation
            if not pod.metadata.annotations:
                pod.metadata.annotations = {}
            
            pod.metadata.annotations['autoshield-throttled'] = datetime.now().isoformat()
            pod.metadata.annotations['autoshield-throttle-rate'] = parameters.get('rate_limit', '100kbps')
            
            self.core_v1.patch_namespaced_pod(
                name=target_pod,
                namespace=target_namespace,
                body=pod
            )
            
            return {
                'status': 'annotated',
                'method': 'pod_annotation',
                'rate_limit': parameters.get('rate_limit', '100kbps'),
                'note': 'Actual throttling requires Cilium or network plugin support'
            }
            
        except ApiException as e:
            logger.error(f"Fallback throttling failed: {e}")
            raise
    
    def _quarantine_node(self,
                        target_pod: str,
                        target_namespace: str,
                        parameters: Dict[str, Any],
                        detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Quarantine a node by tainting it"""
        
        # Get node name from pod
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=target_pod,
                namespace=target_namespace
            )
            node_name = pod.spec.node_name
            
            if not node_name:
                raise ValueError(f"Pod {target_pod} not scheduled on any node")
            
            # Get the node
            node = self.core_v1.read_node(name=node_name)
            
            # Add taint
            taint = client.V1Taint(
                key='autoshield-quarantine',
                value='true',
                effect='NoSchedule',
                time_added=datetime.now()
            )
            
            if not node.spec.taints:
                node.spec.taints = []
            
            # Check if taint already exists
            taint_exists = any(
                t.key == 'autoshield-quarantine' 
                for t in node.spec.taints
            )
            
            if not taint_exists:
                node.spec.taints.append(taint)
                
                # Add label
                if not node.metadata.labels:
                    node.metadata.labels = {}
                node.metadata.labels['autoshield-quarantined'] = 'true'
                
                # Update node
                self.core_v1.patch_node(name=node_name, body=node)
                
                # Cordon the node (prevent new pods)
                node.spec.unschedulable = True
                self.core_v1.patch_node(name=node_name, body=node)
                
                logger.warning(f"Node {node_name} quarantined and cordoned")
                
                return {
                    'status': 'quarantined',
                    'node_name': node_name,
                    'taint_added': 'autoshield-quarantine',
                    'cordoned': True,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'already_quarantined',
                    'node_name': node_name,
                    'note': 'Node already has quarantine taint'
                }
            
        except ApiException as e:
            logger.error(f"Failed to quarantine node: {e}")
            raise
    
    def _send_alert(self,
                   target_pod: str,
                   target_namespace: str,
                   parameters: Dict[str, Any],
                   detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Send alert without taking action"""
        
        channel = parameters.get('channel', 'slack')
        priority = parameters.get('priority', 'medium')
        
        # In production, this would integrate with actual alerting systems
        alert_message = {
            'severity': priority,
            'channel': channel,
            'target': target_pod,
            'namespace': target_namespace,
            'attack_type': detection_result.get('predicted_class'),
            'confidence': detection_result.get('confidence'),
            'timestamp': datetime.now().isoformat(),
            'explanation': detection_result.get('explanation', '')
        }
        
        logger.warning(f"ALERT: {json.dumps(alert_message, indent=2)}")
        
        return {
            'status': 'alert_sent',
            'channel': channel,
            'priority': priority,
            'message': alert_message
        }
    
    def _get_pod_labels(self, pod_name: str, namespace: str) -> Dict[str, str]:
        """Get labels of a pod"""
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            return pod.metadata.labels or {}
        except ApiException as e:
            logger.error(f"Failed to get pod labels for {pod_name}: {e}")
            return {}
    
    def _extract_namespace(self, pod_name: str) -> Optional[str]:
        """Extract namespace from pod name if in format 'namespace/pod-name'"""
        if '/' in pod_name:
            return pod_name.split('/')[0]
        return None
    
    def rollback_action(self, action_id: str) -> Dict[str, Any]:
        """Rollback a previously taken action"""
        # Find action in history
        action_to_rollback = None
        for action in self.action_history:
            if action.get('action_id') == action_id:
                action_to_rollback = action
                break
        
        if not action_to_rollback:
            return {'status': 'error', 'reason': 'action_not_found'}
        
        action_type = action_to_rollback.get('action_type')
        target_pod = action_to_rollback.get('target_pod')
        target_namespace = action_to_rollback.get('target_namespace')
        
        try:
            if action_type == 'network_policy':
                # Delete NetworkPolicies
                policies = self.networking_v1.list_namespaced_network_policy(
                    namespace=target_namespace,
                    label_selector=f"autoshield-managed=true,target-pod={target_pod}"
                )
                for policy in policies.items:
                    self.networking_v1.delete_namespaced_network_policy(
                        name=policy.metadata.name,
                        namespace=target_namespace
                    )
                
                # Remove pod annotations
                self._remove_pod_annotation(target_pod, target_namespace, 'autoshield-isolated')
            
            elif action_type == 'pod_termination':
                # For pod termination, we can't rollback, but we can note it
                logger.info(f"Rollback note: Pod {target_pod} was terminated")
            
            elif action_type == 'traffic_throttle':
                # Delete CiliumNetworkPolicies
                try:
                    policies = self.custom_objects_api.list_namespaced_custom_object(
                        group='cilium.io',
                        version='v2',
                        namespace=target_namespace,
                        plural='ciliumnetworkpolicies',
                        label_selector=f"autoshield-managed=true,target-pod={target_pod}"
                    )
                    for policy in policies.get('items', []):
                        self.custom_objects_api.delete_namespaced_custom_object(
                            group='cilium.io',
                            version='v2',
                            namespace=target_namespace,
                            plural='ciliumnetworkpolicies',
                            name=policy['metadata']['name']
                        )
                except:
                    pass
                
                # Remove pod annotations
                self._remove_pod_annotation(target_pod, target_namespace, 'autoshield-throttled')
            
            elif action_type == 'node_quarantine':
                # Remove taint and uncordon
                node_name = action_to_rollback.get('execution_result', {}).get('node_name')
                if node_name:
                    node = self.core_v1.read_node(name=node_name)
                    # Remove taint
                    if node.spec.taints:
                        node.spec.taints = [
                            t for t in node.spec.taints 
                            if t.key != 'autoshield-quarantine'
                        ]
                    # Uncordon
                    node.spec.unschedulable = False
                    # Remove label
                    if node.metadata.labels and 'autoshield-quarantined' in node.metadata.labels:
                        del node.metadata.labels['autoshield-quarantined']
                    
                    self.core_v1.patch_node(name=node_name, body=node)
            
            # Mark as rolled back
            action_to_rollback['rolled_back'] = True
            action_to_rollback['rollback_time'] = datetime.now().isoformat()
            
            logger.info(f"Action rolled back: {action_id}")
            
            return {
                'status': 'rolled_back',
                'action_id': action_id,
                'action_type': action_type,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback action {action_id}: {e}")
            return {
                'status': 'error',
                'action_id': action_id,
                'error': str(e)
            }
    
    def _remove_pod_annotation(self, pod_name: str, namespace: str, annotation_key: str):
        """Remove annotation from pod"""
        try:
            pod = self.core_v1.read_namespaced_pod(
                name=pod_name,
                namespace=namespace
            )
            
            if pod.metadata.annotations and annotation_key in pod.metadata.annotations:
                del pod.metadata.annotations[annotation_key]
                self.core_v1.patch_namespaced_pod(
                    name=pod_name,
                    namespace=namespace,
                    body=pod
                )
        except ApiException as e:
            logger.warning(f"Could not remove annotation from pod {pod_name}: {e}")