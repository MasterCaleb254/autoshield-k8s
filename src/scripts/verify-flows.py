#!/usr/bin/env python3
# verify-flows.py
import subprocess
import json
import time

def test_hubble_flows():
    """Test that Hubble is collecting flow data"""
    print("Testing Hubble flow collection...")
    
    # Get Hubble UI service
    hubble_ui_cmd = "kubectl get svc -n kube-system hubble-ui -o jsonpath='{.spec.ports[0].nodePort}'"
    result = subprocess.run(hubble_ui_cmd, shell=True, capture_output=True, text=True)
    hubble_port = result.stdout.strip()
    
    if hubble_port:
        print(f"✓ Hubble UI available on nodePort: {hubble_port}")
    else:
        print("✗ Hubble UI not found")
        return False
    
    # Get Hubble Relay endpoint
    hubble_relay_cmd = "kubectl get svc -n kube-system hubble-relay -o jsonpath='{.spec.clusterIP}'"
    result = subprocess.run(hubble_relay_cmd, shell=True, capture_output=True, text=True)
    hubble_ip = result.stdout.strip()
    
    if hubble_ip:
        print(f"✓ Hubble Relay available at: {hubble_ip}:80")
    else:
        print("✗ Hubble Relay not found")
        return False
    
    # Test flow observation
    print("\nGenerating test traffic...")
    
    # Generate some traffic
    test_commands = [
        "kubectl run -it --rm --image=curlimages/curl:8.4.0 test-curl --restart=Never -- curl -s http://frontend-service.test-apps > /dev/null 2>&1",
        "kubectl run -it --rm --image=curlimages/curl:8.4.0 test-curl2 --restart=Never -- curl -s http://backend-service.test-apps > /dev/null 2>&1"
    ]
    
    for cmd in test_commands:
        try:
            subprocess.run(cmd, shell=True, timeout=10)
        except:
            pass
    
    # Check flows via Hubble CLI
    print("\nChecking captured flows...")
    time.sleep(2)
    
    hubble_cmd = "kubectl exec -n kube-system -t ds/cilium -- hubble observe --since=1m --output json | jq -c 'select(.verdict == \"FORWARDED\")' | head -5"
    result = subprocess.run(hubble_cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        flows = result.stdout.strip().split('\n')
        print(f"✓ Captured {len(flows)} flows")
        for flow in flows[:3]:  # Show first 3 flows
            try:
                flow_data = json.loads(flow)
                src = flow_data.get('source', {}).get('labels', [])
                dst = flow_data.get('destination', {}).get('labels', [])
                print(f"  Flow: {src} → {dst}")
            except:
                pass
        return True
    else:
        print("✗ No flows captured")
        return False

if __name__ == "__main__":
    success = test_hubble_flows()
    if success:
        print("\n✅ Phase 1 verification complete: eBPF flow visibility is working!")
    else:
        print("\n❌ Phase 1 verification failed. Check Cilium installation.")