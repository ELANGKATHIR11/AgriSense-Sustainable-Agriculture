"""
AgriSense Sensor API Routes
Handles IoT device data and real-time monitoring
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import logging

from core.data_store import (
    get_latest_sensor_data,
    insert_sensor_data,
    get_sensor_history,
    get_all_devices,
    update_device_status
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class SensorReading(BaseModel):
    device_id: str
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    soil_moisture: Optional[float] = None
    ph_level: Optional[float] = None
    nitrogen: Optional[float] = None
    phosphorus: Optional[float] = None
    potassium: Optional[float] = None
    light_intensity: Optional[float] = None

class DeviceStatus(BaseModel):
    device_id: str
    status: str  # active, inactive, maintenance, error

@router.get("/live")
async def get_live_sensors():
    """
    Get latest sensor readings
    
    Returns real-time data from IoT devices or mock data if unavailable
    """
    try:
        data = get_latest_sensor_data()
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "data": data,
            "units": {
                "temperature": "°C",
                "humidity": "%",
                "soil_moisture": "%",
                "ph": "pH",
                "nitrogen": "mg/kg",
                "phosphorus": "mg/kg",
                "potassium": "mg/kg",
                "light_intensity": "lux"
            }
        }
    except Exception as e:
        logger.error(f"Error getting live sensors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data")
async def post_sensor_data(reading: SensorReading):
    """
    Receive sensor data from IoT devices
    
    Accepts data from ESP32, Arduino, or other sensors
    """
    try:
        # Convert to dictionary
        data_dict = {
            "temperature": reading.temperature,
            "humidity": reading.humidity,
            "soil_moisture": reading.soil_moisture,
            "ph_level": reading.ph_level,
            "nitrogen": reading.nitrogen,
            "phosphorus": reading.phosphorus,
            "potassium": reading.potassium,
            "light_intensity": reading.light_intensity
        }
        
        # Insert into database
        success = insert_sensor_data(reading.device_id, data_dict)
        
        if success:
            return {
                "success": True,
                "message": "Sensor data recorded",
                "timestamp": datetime.now().isoformat(),
                "device_id": reading.device_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to insert data")
            
    except Exception as e:
        logger.error(f"Error posting sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_history(hours: int = Query(24, ge=1, le=168)):
    """
    Get sensor data history
    
    Args:
        hours: Number of hours to retrieve (1-168)
    """
    try:
        history = get_sensor_history(hours)
        
        return {
            "success": True,
            "hours": hours,
            "count": len(history),
            "data": history
        }
    except Exception as e:
        logger.error(f"Error getting sensor history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices")
async def list_devices():
    """Get all registered IoT devices"""
    try:
        devices = get_all_devices()
        
        return {
            "success": True,
            "count": len(devices),
            "devices": devices
        }
    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/devices/status")
async def get_devices_status():
    """Get device status summary"""
    try:
        devices = get_all_devices()
        
        status_summary = {
            "active": sum(1 for d in devices if d["status"] == "active"),
            "inactive": sum(1 for d in devices if d["status"] == "inactive"),
            "maintenance": sum(1 for d in devices if d["status"] == "maintenance"),
            "error": sum(1 for d in devices if d["status"] == "error")
        }
        
        return {
            "success": True,
            "total_devices": len(devices),
            "status_summary": status_summary,
            "devices": devices
        }
    except Exception as e:
        logger.error(f"Error getting device status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/devices/status")
async def update_status(status_update: DeviceStatus):
    """Update device status"""
    try:
        success = update_device_status(status_update.device_id, status_update.status)
        
        if success:
            return {
                "success": True,
                "message": f"Device {status_update.device_id} status updated to {status_update.status}"
            }
        else:
            raise HTTPException(status_code=404, detail="Device not found")
            
    except Exception as e:
        logger.error(f"Error updating device status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def iot_health():
    """IoT system health check"""
    try:
        devices = get_all_devices()
        latest_data = get_latest_sensor_data()
        
        return {
            "success": True,
            "status": "operational",
            "active_devices": sum(1 for d in devices if d["status"] == "active"),
            "data_source": latest_data.get("source", "unknown"),
            "last_reading": latest_data.get("timestamp")
        }
    except Exception as e:
        logger.error(f"IoT health check failed: {e}")
        return {
            "success": False,
            "status": "degraded",
            "error": str(e)
        }
