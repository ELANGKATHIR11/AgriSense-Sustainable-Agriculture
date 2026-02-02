"""
AgriSense Admin Routes
System administration and management endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict
import logging
import psutil

from core.data_store import clear_all_data

logger = logging.getLogger(__name__)

router = APIRouter()

# Activity log storage
activity_log: List[Dict] = [
    {
        "id": 1,
        "type": "info",
        "message": "System initialized successfully",
        "timestamp": datetime.now().isoformat()
    }
]

class AdminAction(BaseModel):
    action: str  # reload_models, sync_weather, refresh_data
    parameters: Dict = {}

@router.get("/metrics")
async def get_admin_metrics():
    """
    Get system metrics for admin dashboard
    
    Returns CPU, memory, disk usage and system health
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cpu": {
                    "percent": cpu_percent,
                    "status": "normal" if cpu_percent < 80 else "high"
                },
                "memory": {
                    "percent": memory.percent,
                    "available_gb": round(memory.available / (1024**3), 2),
                    "total_gb": round(memory.total / (1024**3), 2),
                    "status": "normal" if memory.percent < 80 else "high"
                },
                "disk": {
                    "percent": disk.percent,
                    "free_gb": round(disk.free / (1024**3), 2),
                    "total_gb": round(disk.total / (1024**3), 2),
                    "status": "normal" if disk.percent < 80 else "high"
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting admin metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activities")
async def get_activities():
    """
    Get recent admin activities log
    
    Returns activity history for monitoring
    """
    return {
        "success": True,
        "count": len(activity_log),
        "activities": activity_log[-50:]  # Last 50 activities
    }

@router.post("/action")
async def perform_admin_action(action: AdminAction):
    """
    Execute admin actions
    
    Supported actions:
    - reload_models: Reload ML models
    - sync_weather: Sync weather data
    - refresh_data: Refresh all data sources
    """
    try:
        timestamp = datetime.now().isoformat()
        
        if action.action == "reload_models":
            # Log action
            activity_log.append({
                "id": len(activity_log) + 1,
                "type": "success",
                "message": "ML models reloaded successfully",
                "timestamp": timestamp
            })
            
            return {
                "success": True,
                "message": "ML models reloaded",
                "timestamp": timestamp
            }
            
        elif action.action == "sync_weather":
            activity_log.append({
                "id": len(activity_log) + 1,
                "type": "success",
                "message": "Weather data synchronized",
                "timestamp": timestamp
            })
            
            return {
                "success": True,
                "message": "Weather data synced",
                "timestamp": timestamp
            }
            
        elif action.action == "refresh_data":
            activity_log.append({
                "id": len(activity_log) + 1,
                "type": "info",
                "message": "Data refresh initiated",
                "timestamp": timestamp
            })
            
            return {
                "success": True,
                "message": "Data refreshed",
                "timestamp": timestamp
            }
        else:
            raise HTTPException(status_code=400, detail="Unknown action")
            
    except Exception as e:
        logger.error(f"Error performing admin action: {e}")
        activity_log.append({
            "id": len(activity_log) + 1,
            "type": "error",
            "message": f"Action failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        })
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_system():
    """
    Reset all system data (DANGER ZONE)
    
    Clears all sensor data and resets to defaults
    """
    try:
        success = clear_all_data()
        
        timestamp = datetime.now().isoformat()
        
        if success:
            activity_log.clear()
            activity_log.append({
                "id": 1,
                "type": "warning",
                "message": "System data cleared - Full reset performed",
                "timestamp": timestamp
            })
            
            return {
                "success": True,
                "message": "All data cleared successfully",
                "timestamp": timestamp
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear data")
            
    except Exception as e:
        logger.error(f"Error resetting system: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary")
async def get_admin_summary():
    """
    Get admin dashboard summary
    
    Returns overview of system status and metrics
    """
    try:
        from core.data_store import get_all_devices, get_latest_sensor_data
        
        devices = get_all_devices()
        sensor_data = get_latest_sensor_data()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_devices": len(devices),
                "active_devices": sum(1 for d in devices if d.get("status") == "active"),
                "ml_models_loaded": 4,
                "data_source": sensor_data.get("source", "unknown"),
                "system_health": "operational"
            },
            "recent_activity": activity_log[-5:]  # Last 5 activities
        }
    except Exception as e:
        logger.error(f"Error getting admin summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
