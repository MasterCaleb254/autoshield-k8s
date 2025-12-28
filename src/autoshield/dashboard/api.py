# src/autoshield/dashboard/api.py
"""
Dashboard API for real-time monitoring and visualization.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Dict, List, Optional
import asyncio
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from ..orchestrator import AutoShieldOrchestrator
from ..observability.auditor import AuditLogger
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

# Create dashboard app
dashboard_app = FastAPI(
    title="AutoShield-K8s Dashboard",
    description="Real-time monitoring dashboard for autonomous intrusion defense",
    version="1.0.0"
)

# Add CORS middleware
dashboard_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class DashboardState:
    def __init__(self):
        self.websocket_connections: List[WebSocket] = []
        self.recent_events: List[Dict] = []
        self.system_stats: Dict = {}
        self.attack_timeline: List[Dict] = []
        self.metrics_history: Dict[str, List] = defaultdict(list)
        self.max_history = 1000

state = DashboardState()

@dashboard_app.on_event("startup")
async def startup_event():
    """Initialize dashboard state"""
    # Start background tasks
    asyncio.create_task(update_system_stats())
    asyncio.create_task(cleanup_old_data())

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected
        for connection in disconnected:
            self.disconnect(connection)

manager = ConnectionManager()

@dashboard_app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve dashboard HTML"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AutoShield-K8s Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/luxon@3.0.0/build/global/luxon.min.js"></script>
        <style>
            :root {
                --primary: #3498db;
                --success: #2ecc71;
                --warning: #f39c12;
                --danger: #e74c3c;
                --dark: #2c3e50;
                --light: #ecf0f1;
            }
            body {
                background-color: #f8f9fa;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .navbar {
                background: linear-gradient(135deg, var(--dark) 0%, #34495e 100%);
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .card {
                border: none;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                transition: transform 0.2s;
                height: 100%;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .card-header {
                border-radius: 10px 10px 0 0 !important;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                font-weight: 600;
            }
            .stat-card {
                border-left: 4px solid var(--primary);
            }
            .stat-number {
                font-size: 2.5rem;
                font-weight: 700;
                color: var(--dark);
            }
            .attack-card {
                border-left: 4px solid var(--danger);
            }
            .normal-card {
                border-left: 4px solid var(--success);
            }
            .event-item {
                border-left: 3px solid var(--primary);
                padding: 10px;
                margin-bottom: 5px;
                background: white;
                border-radius: 5px;
                transition: all 0.3s;
            }
            .event-item:hover {
                background: #f8f9fa;
                transform: translateX(5px);
            }
            .severity-low { border-left-color: var(--success) !important; }
            .severity-medium { border-left-color: var(--warning) !important; }
            .severity-high { border-left-color: var(--danger) !important; }
            .severity-critical { border-left-color: #8b0000 !important; }
            .status-indicator {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                display: inline-block;
                margin-right: 5px;
            }
            .status-healthy { background-color: var(--success); }
            .status-warning { background-color: var(--warning); }
            .status-critical { background-color: var(--danger); }
            .chart-container {
                position: relative;
                height: 300px;
                width: 100%;
            }
            .timeline {
                max-height: 400px;
                overflow-y: auto;
            }
            .realtime-badge {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <!-- Navbar -->
        <nav class="navbar navbar-expand-lg navbar-dark mb-4">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">
                    <i class="fas fa-shield-alt me-2"></i>
                    <strong>AutoShield-K8s</strong> Dashboard
                </a>
                <div class="d-flex">
                    <span class="navbar-text me-3">
                        <span class="status-indicator status-healthy"></span>
                        <span id="connection-status">Connected</span>
                    </span>
                    <span class="navbar-text">
                        <i class="fas fa-clock me-1"></i>
                        <span id="current-time">--:--:--</span>
                    </span>
                </div>
            </div>
        </nav>

        <div class="container-fluid">
            <!-- Top Row: Statistics -->
            <div class="row mb-4">
                <!-- System Status -->
                <div class="col-xl-3 col-md-6 mb-4">
                    <div class="card stat-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title text-muted">System Status</h5>
                                    <h2 class="stat-number" id="total-processed">0</h2>
                                    <p class="card-text">Windows Processed</p>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-server fa-3x text-primary"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Attacks Detected -->
                <div class="col-xl-3 col-md-6 mb-4">
                    <div class="card attack-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title text-muted">Attacks Detected</h5>
                                    <h2 class="stat-number" id="attacks-detected">0</h2>
                                    <p class="card-text">Threats Identified</p>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-bug fa-3x text-danger"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Actions Taken -->
                <div class="col-xl-3 col-md-6 mb-4">
                    <div class="card stat-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title text-muted">Actions Taken</h5>
                                    <h2 class="stat-number" id="actions-taken">0</h2>
                                    <p class="card-text">Mitigations Executed</p>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-robot fa-3x text-warning"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Average Latency -->
                <div class="col-xl-3 col-md-6 mb-4">
                    <div class="card normal-card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h5 class="card-title text-muted">Avg Latency</h5>
                                    <h2 class="stat-number" id="avg-latency">0</h2>
                                    <p class="card-text">Milliseconds</p>
                                </div>
                                <div class="align-self-center">
                                    <i class="fas fa-bolt fa-3x text-success"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Middle Row: Charts -->
            <div class="row mb-4">
                <!-- Attack Distribution Chart -->
                <div class="col-lg-8 mb-4">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">
                                <i class="fas fa-chart-pie me-2"></i>
                                Attack Distribution
                            </h5>
                            <span class="badge bg-primary realtime-badge">Live</span>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="attackChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Latency Trend -->
                <div class="col-lg-4 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">
                                <i class="fas fa-chart-line me-2"></i>
                                Latency Trend
                            </h5>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="latencyChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Bottom Row: Timeline & Details -->
            <div class="row">
                <!-- Real-time Events -->
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5 class="mb-0">
                                <i class="fas fa-stream me-2"></i>
                                Real-time Events
                            </h5>
                            <span class="badge bg-danger realtime-badge">Live</span>
                        </div>
                        <div class="card-body p-0">
                            <div class="timeline p-3" id="event-timeline">
                                <!-- Events will be inserted here -->
                                <div class="text-center text-muted py-5">
                                    <i class="fas fa-sync fa-spin fa-2x mb-3"></i>
                                    <p>Waiting for events...</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- System Details -->
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header">
                            <h5 class="mb-0">
                                <i class="fas fa-info-circle me-2"></i>
                                System Details
                            </h5>
                        </div>
                        <div class="card-body">
                            <div id="system-details">
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <small class="text-muted">Inference Service</small>
                                            <div class="d-flex align-items-center">
                                                <span class="status-indicator status-healthy" id="inference-status"></span>
                                                <span id="inference-info">Loading...</span>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <small class="text-muted">Policy Engine</small>
                                            <div class="d-flex align-items-center">
                                                <span class="status-indicator status-healthy" id="policy-status"></span>
                                                <span id="policy-info">Loading...</span>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <small class="text-muted">Actuator</small>
                                            <div class="d-flex align-items-center">
                                                <span class="status-indicator status-healthy" id="actuator-status"></span>
                                                <span id="actuator-info">Loading...</span>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <small class="text-muted">Uptime</small>
                                            <div>
                                                <span id="uptime">00:00:00</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <hr>
                                
                                <h6 class="mt-3 mb-3">Recent Actions</h6>
                                <div class="table-responsive">
                                    <table class="table table-sm" id="actions-table">
                                        <thead>
                                            <tr>
                                                <th>Time</th>
                                                <th>Action</th>
                                                <th>Target</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <!-- Actions will be inserted here -->
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- JavaScript -->
        <script>
            // WebSocket connection
            let ws = null;
            let reconnectAttempts = 0;
            const maxReconnectAttempts = 5;
            
            // Charts
            let attackChart = null;
            let latencyChart = null;
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                connectWebSocket();
                initializeCharts();
                updateClock();
                setInterval(updateClock, 1000);
            });
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log('WebSocket connected');
                    reconnectAttempts = 0;
                    updateConnectionStatus(true);
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                ws.onclose = function() {
                    console.log('WebSocket disconnected');
                    updateConnectionStatus(false);
                    
                    // Attempt reconnection
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        setTimeout(connectWebSocket, 1000 * reconnectAttempts);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function handleWebSocketMessage(data) {
                const eventType = data.event_type;
                
                switch(eventType) {
                    case 'system_stats':
                        updateSystemStats(data.data);
                        break;
                    case 'new_event':
                        addEventToTimeline(data.data);
                        break;
                    case 'attack_detected':
                        updateAttackChart(data.data);
                        addEventToTimeline(data.data);
                        break;
                    case 'action_taken':
                        addActionToTable(data.data);
                        break;
                    case 'latency_update':
                        updateLatencyChart(data.data);
                        break;
                }
            }
            
            function updateSystemStats(stats) {
                // Update stat cards
                document.getElementById('total-processed').textContent = stats.total_windows_processed.toLocaleString();
                document.getElementById('attacks-detected').textContent = stats.attacks_detected.toLocaleString();
                document.getElementById('actions-taken').textContent = stats.actions_executed.toLocaleString();
                document.getElementById('avg-latency').textContent = stats.avg_processing_time_ms.toFixed(1);
                
                // Update system details
                document.getElementById('inference-info').textContent = 
                    `${stats.inference.total_inferences} inferences`;
                document.getElementById('policy-info').textContent = 
                    `${stats.policy.total_rules} rules`;
                document.getElementById('actuator-info').textContent = 
                    stats.enable_actuation ? 'Active' : 'Monitor Mode';
                document.getElementById('uptime').textContent = 
                    formatDuration(stats.uptime);
                
                // Update status indicators
                updateStatusIndicator('inference-status', stats.inference.avg_latency_ms < 2);
                updateStatusIndicator('policy-status', stats.policy.avg_evaluation_time_ms < 10);
                updateStatusIndicator('actuator-status', true);
            }
            
            function addEventToTimeline(event) {
                const timeline = document.getElementById('event-timeline');
                
                // Remove loading placeholder
                const placeholder = timeline.querySelector('.text-center');
                if (placeholder) {
                    placeholder.remove();
                }
                
                // Create event element
                const eventDiv = document.createElement('div');
                eventDiv.className = `event-item severity-${event.severity || 'medium'}`;
                
                const time = new Date(event.timestamp).toLocaleTimeString();
                const icon = getEventIcon(event.event_type);
                const severityBadge = event.severity ? 
                    `<span class="badge bg-${getSeverityColor(event.severity)}">${event.severity}</span>` : '';
                
                eventDiv.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <div>
                            <i class="${icon} me-2"></i>
                            <strong>${event.title || event.event_type}</strong>
                        </div>
                        <small class="text-muted">${time}</small>
                    </div>
                    <div class="mt-1">
                        ${severityBadge}
                        <small>${event.description || ''}</small>
                    </div>
                    ${event.confidence ? `<div><small>Confidence: ${(event.confidence * 100).toFixed(1)}%</small></div>` : ''}
                `;
                
                // Add to top
                timeline.insertBefore(eventDiv, timeline.firstChild);
                
                // Limit to 20 events
                const events = timeline.querySelectorAll('.event-item');
                if (events.length > 20) {
                    events[events.length - 1].remove();
                }
                
                // Auto-scroll to show new events
                timeline.scrollTop = 0;
            }
            
            function addActionToTable(action) {
                const tbody = document.querySelector('#actions-table tbody');
                const time = new Date(action.timestamp).toLocaleTimeString();
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${time}</td>
                    <td>
                        <span class="badge bg-${getActionColor(action.action_type)}">
                            ${action.action_type}
                        </span>
                    </td>
                    <td><code>${action.target_pod}</code></td>
                    <td>
                        <span class="badge bg-${action.status === 'success' ? 'success' : 'danger'}">
                            ${action.status}
                        </span>
                    </td>
                `;
                
                tbody.insertBefore(row, tbody.firstChild);
                
                // Limit to 10 rows
                const rows = tbody.querySelectorAll('tr');
                if (rows.length > 10) {
                    rows[rows.length - 1].remove();
                }
            }
            
            function initializeCharts() {
                // Attack distribution chart
                const attackCtx = document.getElementById('attackChart').getContext('2d');
                attackChart = new Chart(attackCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Normal', 'Lateral Movement', 'Port Scan', 'SYN Flood'],
                        datasets: [{
                            data: [0, 0, 0, 0],
                            backgroundColor: [
                                '#2ecc71',
                                '#e74c3c',
                                '#f39c12',
                                '#9b59b6'
                            ],
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'bottom'
                            }
                        }
                    }
                });
                
                // Latency trend chart
                const latencyCtx = document.getElementById('latencyChart').getContext('2d');
                latencyChart = new Chart(latencyCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Processing Latency (ms)',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            y: {
                                beginAtZero: true,
                                title: {
                                    display: true,
                                    text: 'Milliseconds'
                                }
                            }
                        }
                    }
                });
            }
            
            function updateAttackChart(data) {
                if (!attackChart) return;
                
                // Update chart data
                attackChart.data.datasets[0].data = [
                    data.normal || 0,
                    data.lateral_movement || 0,
                    data.port_scan || 0,
                    data.syn_flood || 0
                ];
                attackChart.update();
            }
            
            function updateLatencyChart(data) {
                if (!latencyChart) return;
                
                const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                
                // Add new data point
                latencyChart.data.labels.push(now);
                latencyChart.data.datasets[0].data.push(data.latency_ms);
                
                // Keep only last 20 points
                if (latencyChart.data.labels.length > 20) {
                    latencyChart.data.labels.shift();
                    latencyChart.data.datasets[0].data.shift();
                }
                
                latencyChart.update();
            }
            
            function updateConnectionStatus(connected) {
                const statusEl = document.getElementById('connection-status');
                const indicator = document.querySelector('#connection-status').previousElementSibling;
                
                if (connected) {
                    statusEl.textContent = 'Connected';
                    indicator.className = 'status-indicator status-healthy';
                } else {
                    statusEl.textContent = 'Disconnected';
                    indicator.className = 'status-indicator status-critical';
                }
            }
            
            function updateClock() {
                const now = new Date();
                document.getElementById('current-time').textContent = 
                    now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
            }
            
            function formatDuration(seconds) {
                const hrs = Math.floor(seconds / 3600);
                const mins = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
            }
            
            function getEventIcon(eventType) {
                const icons = {
                    'attack_detected': 'fas fa-bug',
                    'action_taken': 'fas fa-robot',
                    'system_alert': 'fas fa-exclamation-triangle',
                    'normal_traffic': 'fas fa-check-circle'
                };
                return icons[eventType] || 'fas fa-info-circle';
            }
            
            function getSeverityColor(severity) {
                const colors = {
                    'low': 'success',
                    'medium': 'warning',
                    'high': 'danger',
                    'critical': 'dark'
                };
                return colors[severity.toLowerCase()] || 'secondary';
            }
            
            function getActionColor(actionType) {
                const colors = {
                    'network_policy': 'primary',
                    'pod_isolation': 'warning',
                    'pod_termination': 'danger',
                    'traffic_throttle': 'info',
                    'alert_only': 'secondary'
                };
                return colors[actionType] || 'secondary';
            }
            
            function updateStatusIndicator(elementId, healthy) {
                const indicator = document.getElementById(elementId);
                indicator.className = `status-indicator ${healthy ? 'status-healthy' : 'status-critical'}`;
            }
        </script>
    </body>
    </html>
    """

@dashboard_app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        # Send initial data
        await websocket.send_json({
            "event_type": "system_stats",
            "data": state.system_stats
        })
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)  # Just keep connection alive
            # Client can also send requests if needed
            # data = await websocket.receive_text()
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@dashboard_app.get("/api/stats")
async def get_stats():
    """Get current system statistics"""
    return state.system_stats

@dashboard_app.get("/api/events")
async def get_events(limit: int = 50, event_type: Optional[str] = None):
    """Get recent events"""
    events = state.recent_events
    
    if event_type:
        events = [e for e in events if e.get("event_type") == event_type]
    
    return events[-limit:] if limit else events

@dashboard_app.get("/api/attack-timeline")
async def get_attack_timeline(hours: int = 24):
    """Get attack timeline for the specified hours"""
    cutoff = datetime.now() - timedelta(hours=hours)
    timeline = [e for e in state.attack_timeline 
                if datetime.fromisoformat(e.get("timestamp", "")) > cutoff]
    return timeline

@dashboard_app.get("/api/metrics")
async def get_metrics(metric_name: str, limit: int = 100):
    """Get historical metrics"""
    metrics = state.metrics_history.get(metric_name, [])
    return metrics[-limit:] if limit else metrics

@dashboard_app.post("/api/broadcast")
async def broadcast_event(event: Dict, background_tasks: BackgroundTasks):
    """Broadcast an event to all connected clients"""
    event["timestamp"] = datetime.now().isoformat()
    
    # Store in recent events
    state.recent_events.append(event)
    if len(state.recent_events) > state.max_history:
        state.recent_events.pop(0)
    
    # Store in attack timeline if relevant
    if event.get("event_type") in ["attack_detected", "action_taken"]:
        state.attack_timeline.append(event)
        if len(state.attack_timeline) > state.max_history:
            state.attack_timeline.pop(0)
    
    # Broadcast via WebSocket
    background_tasks.add_task(manager.broadcast, event)
    
    return {"status": "broadcasted"}

async def update_system_stats():
    """Periodically update system statistics"""
    while True:
        try:
            # In production, this would fetch from orchestrator
            stats = {
                "total_windows_processed": 1234,
                "attacks_detected": 45,
                "actions_executed": 23,
                "avg_processing_time_ms": 1.2,
                "inference": {
                    "total_inferences": 1234,
                    "avg_latency_ms": 0.8,
                    "p95_latency_ms": 1.1
                },
                "policy": {
                    "total_rules": 5,
                    "avg_evaluation_time_ms": 0.2
                },
                "enable_actuation": True,
                "uptime": 3600  # seconds
            }
            
            state.system_stats = stats
            
            # Broadcast update
            await manager.broadcast({
                "event_type": "system_stats",
                "data": stats
            })
            
            # Update latency chart
            state.metrics_history["latency"].append({
                "timestamp": datetime.now().isoformat(),
                "latency_ms": stats["avg_processing_time_ms"]
            })
            
            await manager.broadcast({
                "event_type": "latency_update",
                "data": {
                    "latency_ms": stats["avg_processing_time_ms"],
                    "timestamp": datetime.now().isoformat()
                }
            })
            
        except Exception as e:
            logger.error(f"Failed to update system stats: {e}")
        
        await asyncio.sleep(5)  # Update every 5 seconds

async def cleanup_old_data():
    """Clean up old data periodically"""
    while True:
        try:
            # Keep only last 24 hours of metrics
            cutoff = datetime.now() - timedelta(hours=24)
            
            for metric_name in list(state.metrics_history.keys()):
                state.metrics_history[metric_name] = [
                    m for m in state.metrics_history[metric_name]
                    if datetime.fromisoformat(m.get("timestamp", "")) > cutoff
                ]
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
        
        await asyncio.sleep(3600)  # Cleanup every hour

def broadcast_detection_event(detection_result: Dict):
    """Broadcast detection event to dashboard"""
    event = {
        "event_type": "attack_detected",
        "title": f"{detection_result.get('predicted_class', 'Unknown')} Detected",
        "description": f"From {detection_result.get('src_pod')} to {detection_result.get('dst_pod')}",
        "severity": "high",
        "confidence": detection_result.get("confidence"),
        "timestamp": datetime.now().isoformat()
    }
    
    # This would be called from the orchestrator
    asyncio.create_task(manager.broadcast(event))

def broadcast_action_event(action_result: Dict):
    """Broadcast action event to dashboard"""
    event = {
        "event_type": "action_taken",
        "title": f"{action_result.get('action_type', 'Unknown')} Action",
        "description": f"Target: {action_result.get('target', 'Unknown')}",
        "severity": "medium",
        "status": action_result.get("status"),
        "timestamp": datetime.now().isoformat()
    }
    
    # This would be called from the orchestrator
    asyncio.create_task(manager.broadcast(event))