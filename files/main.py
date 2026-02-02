"""
AgriSense - Smart Agriculture IoT Platform
Main FastAPI Application Entry Point
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from datetime import datetime
from typing import List
import logging

# Import routes
from api.sensor_api import router as sensor_router
from api.ai_routes import router as ai_router
from routes.ml_predictions import router as ml_router
from routes.health_routes import router as health_router
from routes.admin_routes import router as admin_router

# Import core modules
from core.engine import RecoEngine
from core.data_store import init_sensor_db, get_latest_sensor_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AgriSense API",
    description="Smart Agriculture IoT Platform with 18+ ML Models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")

manager = ConnectionManager()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting AgriSense Backend...")
    
    # Initialize database
    init_sensor_db()
    logger.info("✅ Database initialized")
    
    # Initialize ML models (optional based on env var)
    import os
    if os.getenv("AGRISENSE_DISABLE_ML", "0") == "0":
        try:
            from smart_farming_ml import SmartFarmingRecommendationSystem
            system = SmartFarmingRecommendationSystem()
            logger.info("✅ ML models loaded")
        except Exception as e:
            logger.warning(f"⚠️ ML models not loaded: {e}")
    
    logger.info("🌾 AgriSense Backend is ready!")

# Include routers
app.include_router(sensor_router, prefix="/sensors", tags=["Sensors"])
app.include_router(ai_router, prefix="/ai", tags=["AI"])
app.include_router(ml_router, tags=["ML Predictions"])
app.include_router(health_router, tags=["Health"])
app.include_router(admin_router, prefix="/admin", tags=["Admin"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to AgriSense API",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "features": [
            "Real-time IoT Monitoring",
            "18+ ML Models",
            "AI Chatbot",
            "Disease Detection",
            "Weed Management",
            "Yield Prediction",
            "Smart Irrigation"
        ]
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "operational"
    }

# WebSocket endpoint for real-time sensor data
@app.websocket("/ws/sensors")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sensor streaming"""
    await manager.connect(websocket)
    try:
        while True:
            # Send sensor data every 3 seconds
            sensor_data = get_latest_sensor_data()
            await websocket.send_json({
                "type": "sensor_update",
                "data": sensor_data,
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Dashboard summary endpoint
@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary with live metrics"""
    try:
        sensor_data = get_latest_sensor_data()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "soil_moisture": sensor_data.get("soil_moisture", 45),
                "temperature": sensor_data.get("temperature", 25),
                "humidity": sensor_data.get("humidity", 60),
                "ph_level": sensor_data.get("ph", 6.5)
            },
            "system": {
                "status": "operational",
                "active_devices": 3,
                "ml_models_loaded": 18
            }
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# Alerts endpoint
@app.get("/alerts")
async def get_alerts():
    """Get system alerts and activity log"""
    return {
        "success": True,
        "alerts": [
            {
                "id": 1,
                "type": "info",
                "message": "System initialized successfully",
                "timestamp": datetime.now().isoformat()
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
