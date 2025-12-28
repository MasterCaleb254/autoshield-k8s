# src/autoshield/api/server.py
"""
REST API server for AutoShield-K8s inference service.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List, Optional
import asyncio
import json

from ..detector.inference import ModelInferenceService
from ..models import FlowWindow
from .schemas import (
    FlowWindowRequest,
    BatchPredictionRequest,
    PredictionResponse,
    HealthResponse,
    StatsResponse
)

# Create FastAPI app
app = FastAPI(
    title="AutoShield-K8s Inference API",
    description="CNN-LSTM intrusion detection service for Kubernetes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference service instance
_inference_service = None

def get_inference_service():
    """Dependency to get inference service instance"""
    if _inference_service is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    return _inference_service

@app.on_event("startup")
async def startup_event():
    """Initialize inference service on startup"""
    global _inference_service
    try:
        # Load from latest model
        import os
        from pathlib import Path
        
        # Prefer explicit model path when provided (e.g. via Kubernetes env var)
        model_path_env = os.getenv("MODEL_PATH")
        latest_model: Path
        if model_path_env:
            candidate = Path(model_path_env)
            if candidate.exists() and candidate.is_file():
                latest_model = candidate
            else:
                raise FileNotFoundError(f"MODEL_PATH not found: {model_path_env}")
        else:
            # Fall back to scanning baked-in model directory
            model_dir = Path("data/models")
            model_files = list(model_dir.glob("**/final_model.pth"))
            
            if not model_files:
                raise FileNotFoundError("No trained model found under data/models/**/final_model.pth")
            
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        _inference_service = ModelInferenceService(
            model_path=str(latest_model),
            device="cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
        )
        
        print(f"✅ Inference service initialized with model: {latest_model}")
        
    except Exception as e:
        print(f"❌ Failed to initialize inference service: {e}")
        raise

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "service": "AutoShield-K8s Inference API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Service health check",
            "/predict": "Single prediction",
            "/predict/batch": "Batch prediction",
            "/stats": "Service statistics"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(service: ModelInferenceService = Depends(get_inference_service)):
    """Health check endpoint"""
    return service.health_check()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: FlowWindowRequest,
    background_tasks: BackgroundTasks,
    service: ModelInferenceService = Depends(get_inference_service)
):
    """
    Make a single prediction on a flow window.
    
    - **window_id**: Unique identifier for the window
    - **src_pod**: Source pod name
    - **dst_pod**: Destination pod name
    - **features**: Flow window features
    """
    try:
        # Convert request to FlowWindow
        window = FlowWindow(
            window_id=request.window_id,
            src_pod=request.src_pod,
            dst_pod=request.dst_pod,
            flow_count=request.features.flow_count,
            bytes_sent=request.features.bytes_sent,
            bytes_received=request.features.bytes_received,
            syn_count=request.features.syn_count,
            ack_count=request.features.ack_count,
            rst_count=request.features.rst_count,
            fin_count=request.features.fin_count,
            total_duration_ms=request.features.total_duration_ms,
            avg_interarrival_ms=request.features.avg_interarrival_ms,
            std_interarrival_ms=request.features.std_interarrival_ms,
            failed_conn_ratio=request.features.failed_conn_ratio,
            unique_ports=request.features.unique_ports,
            start_time=request.timestamp,
            end_time=request.timestamp  # Adjust as needed
        )
        
        # Make prediction
        prediction = await service.predict_single(window)
        
        # Log to background task (for audit trail)
        background_tasks.add_task(log_prediction, prediction)
        
        return prediction
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse], tags=["Prediction"])
async def predict_batch(
    request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    service: ModelInferenceService = Depends(get_inference_service)
):
    """
    Make batch predictions on multiple flow windows.
    
    - **windows**: List of flow window requests
    """
    try:
        # Convert requests to FlowWindows
        windows = []
        for req in request.windows:
            window = FlowWindow(
                window_id=req.window_id,
                src_pod=req.src_pod,
                dst_pod=req.dst_pod,
                flow_count=req.features.flow_count,
                bytes_sent=req.features.bytes_sent,
                bytes_received=req.features.bytes_received,
                syn_count=req.features.syn_count,
                ack_count=req.features.ack_count,
                rst_count=req.features.rst_count,
                fin_count=req.features.fin_count,
                total_duration_ms=req.features.total_duration_ms,
                avg_interarrival_ms=req.features.avg_interarrival_ms,
                std_interarrival_ms=req.features.std_interarrival_ms,
                failed_conn_ratio=req.features.failed_conn_ratio,
                unique_ports=req.features.unique_ports,
                start_time=req.timestamp,
                end_time=req.timestamp
            )
            windows.append(window)
        
        # Make batch prediction
        predictions = await service.predict_batch(windows)
        
        # Log to background task
        background_tasks.add_task(log_batch_predictions, predictions)
        
        return predictions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
async def get_stats(service: ModelInferenceService = Depends(get_inference_service)):
    """Get service statistics"""
    return service.get_stats()

async def log_prediction(prediction: dict):
    """Log prediction for audit trail"""
    # In production, this would write to a database or log aggregator
    import logging
    logger = logging.getLogger("audit")
    logger.info(f"Prediction: {json.dumps(prediction)}")

async def log_batch_predictions(predictions: List[dict]):
    """Log batch predictions"""
    import logging
    logger = logging.getLogger("audit")
    logger.info(f"Batch predictions: {len(predictions)} windows processed")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )