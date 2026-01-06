"""
Example FastAPI routes for PocketDB integration
Demonstrates how to use the database module in API endpoints
"""

from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from agrisense_app.backend.database import get_database_manager, DatabaseManager


# ============= Pydantic Models =============

class SensorReadingCreate(BaseModel):
    zone_id: str
    device_id: str
    plant: Optional[str] = None
    temperature_c: float
    humidity: float
    soil_moisture_pct: Optional[float] = None
    ph: Optional[float] = None
    ec_dS_m: Optional[float] = None


class SensorReadingResponse(SensorReadingCreate):
    id: str
    timestamp: str


class RecommendationCreate(BaseModel):
    zone_id: str
    plant: str
    water_liters: float
    expected_savings_liters: Optional[float] = None
    fert_n_g: Optional[float] = None
    fert_p_g: Optional[float] = None
    fert_k_g: Optional[float] = None
    yield_potential: Optional[float] = None


class AlertCreate(BaseModel):
    zone_id: str
    category: str
    message: str
    sent: Optional[bool] = False


class DatabaseHealthResponse(BaseModel):
    status: str
    backend: str
    connected: bool
    collections: dict


# ============= Router Setup =============

router = APIRouter(prefix="/api/v1", tags=["database"])


async def get_db() -> DatabaseManager:
    """Get database manager from app state."""
    # This assumes database is initialized in app startup
    # Adjust based on your FastAPI app setup
    db = get_database_manager()
    if not db._is_initialized:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


# ============= Sensor Reading Endpoints =============

@router.post("/sensor-readings", response_model=SensorReadingResponse)
async def create_sensor_reading(
    reading: SensorReadingCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Create a new sensor reading.
    
    - **zone_id**: Agricultural zone identifier
    - **device_id**: Sensor device identifier
    - **temperature_c**: Temperature in Celsius
    - **humidity**: Humidity percentage
    """
    try:
        data = {
            **reading.dict(),
            "ts": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        result = await db.insert_reading(data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create reading: {str(e)}")


@router.get("/sensor-readings", response_model=List[SensorReadingResponse])
async def get_sensor_readings(
    zone_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    db: DatabaseManager = Depends(get_db)
):
    """
    Get sensor readings for a zone.
    
    - **zone_id**: Filter by zone (optional)
    - **limit**: Maximum number of readings to return
    """
    try:
        readings = await db.get_readings(zone_id, limit)
        return readings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch readings: {str(e)}")


@router.get("/sensor-readings/{zone_id}/latest")
async def get_latest_reading(
    zone_id: str,
    db: DatabaseManager = Depends(get_db)
):
    """Get the most recent sensor reading for a zone."""
    try:
        readings = await db.get_readings(zone_id, limit=1)
        if not readings:
            raise HTTPException(status_code=404, detail="No readings found for zone")
        return readings[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch reading: {str(e)}")


# ============= Recommendation Endpoints =============

@router.post("/recommendations", response_model=dict)
async def create_recommendation(
    rec: RecommendationCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Create a new recommendation.
    
    - **zone_id**: Target agricultural zone
    - **plant**: Crop type
    - **water_liters**: Recommended water amount
    """
    try:
        data = {
            **rec.dict(),
            "ts": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        if db.backend.value == "pocketdb":
            result = await db._adapter.insert_recommendation(data)
        else:
            result = data  # Adapt for other backends as needed
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create recommendation: {str(e)}")


@router.get("/recommendations")
async def get_recommendations(
    zone_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseManager = Depends(get_db)
):
    """Get recommendations for a zone."""
    try:
        if db.backend.value == "pocketdb":
            recommendations = await db._adapter.get_recommendations(zone_id, limit)
        else:
            recommendations = []
        return {"data": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recommendations: {str(e)}")


# ============= Alert Endpoints =============

@router.post("/alerts", response_model=dict)
async def create_alert(
    alert: AlertCreate,
    db: DatabaseManager = Depends(get_db)
):
    """
    Create a new alert.
    
    - **zone_id**: Affected zone
    - **category**: Alert category (e.g., 'disease', 'water', 'pest')
    - **message**: Alert message
    """
    try:
        data = {
            **alert.dict(),
            "ts": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        if db.backend.value == "pocketdb":
            result = await db._adapter.insert_alert(data)
        else:
            result = data
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create alert: {str(e)}")


@router.get("/alerts")
async def get_alerts(
    zone_id: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: DatabaseManager = Depends(get_db)
):
    """Get alerts for a zone."""
    try:
        if db.backend.value == "pocketdb":
            alerts = await db._adapter.get_alerts(zone_id, limit)
            if category:
                alerts = [a for a in alerts if a.get("category") == category]
        else:
            alerts = []
        return {"data": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts: {str(e)}")


# ============= Health and Statistics Endpoints =============

@router.get("/health/database", response_model=DatabaseHealthResponse)
async def database_health(db: DatabaseManager = Depends(get_db)):
    """Check database health and get statistics."""
    try:
        is_healthy = await db.health_check()
        stats = await db.get_stats()
        
        return DatabaseHealthResponse(
            status="healthy" if is_healthy else "unhealthy",
            backend=db.backend.value,
            connected=db._is_initialized,
            collections=stats.get("collections", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/statistics/database")
async def database_statistics(db: DatabaseManager = Depends(get_db)):
    """Get detailed database statistics."""
    try:
        stats = await db.get_stats()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "backend": db.backend.value,
            "statistics": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


# ============= Maintenance Endpoints =============

@router.post("/maintenance/cleanup-old-data")
async def cleanup_old_data(
    collection: str = Query("sensor_readings"),
    days_to_keep: int = Query(90, ge=1, le=365),
    db: DatabaseManager = Depends(get_db)
):
    """
    Clean up old records from specified collection.
    
    - **collection**: Collection to clean (sensor_readings, recommendations, alerts)
    - **days_to_keep**: Keep records from last N days
    """
    try:
        if db.backend.value == "pocketdb":
            deleted_count = await db._adapter.clear_old_records(collection, days_to_keep)
            return {
                "status": "success",
                "collection": collection,
                "deleted_records": deleted_count,
                "days_kept": days_to_keep
            }
        else:
            return {"status": "not_supported", "backend": db.backend.value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


# ============= Data Export Endpoints =============

@router.get("/export/sensor-readings")
async def export_sensor_readings(
    zone_id: str = Query(...),
    format: str = Query("json", regex="^(json|csv)$"),
    db: DatabaseManager = Depends(get_db)
):
    """
    Export sensor readings in specified format.
    
    - **zone_id**: Zone to export
    - **format**: json or csv
    """
    try:
        readings = await db.get_readings(zone_id, limit=10000)
        
        if format == "csv":
            import csv
            from io import StringIO
            
            if not readings:
                raise HTTPException(status_code=404, detail="No readings found")
            
            output = StringIO()
            writer = csv.DictWriter(output, fieldnames=readings[0].keys())
            writer.writeheader()
            writer.writerows(readings)
            
            return {
                "format": "csv",
                "data": output.getvalue(),
                "count": len(readings)
            }
        else:
            return {
                "format": "json",
                "data": readings,
                "count": len(readings)
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ============= FastAPI App Integration Example =============

"""
To integrate these routes into your FastAPI app:

from fastapi import FastAPI
from agrisense_app.backend.database import init_database

app = FastAPI(title="AgriSense with PocketDB")

# Initialize database at startup
@app.on_event("startup")
async def startup():
    app.state.db = await init_database("pocketdb")
    print("Database initialized")

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown():
    await app.state.db.close()
    print("Database closed")

# Include router
from .example_routes import router
app.include_router(router)

# Run with: uvicorn main:app --reload
"""
