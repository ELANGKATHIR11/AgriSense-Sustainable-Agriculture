"""
AgriSense Health Check Routes
System status and monitoring endpoints
"""

from fastapi import APIRouter
from datetime import datetime
import psutil
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

START_TIME = datetime.now()

@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - START_TIME).total_seconds()
    }

@router.get("/status")
async def system_status():
    """Detailed system status"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - START_TIME).total_seconds(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            }
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/version")
async def get_version():
    """Get API version information"""
    return {
        "name": "AgriSense API",
        "version": "2.0.0",
        "build_date": "2026-01-27",
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
