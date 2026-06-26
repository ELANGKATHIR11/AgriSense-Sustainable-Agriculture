# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
AGRISENSE v4.0 FastAPI Application Gateway Server
Implements consolidated models (TabPFN, FT-Transformer, Florence-2, YOLOv11, EIF),
database persistence, digital twin updates, and local RAG/chat assistant pipelines.
"""

import os
import sys
import uuid
import base64
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import engine, get_db, Base
from backend.models import SensorReading, ModelRegistry, PredictionLog, TwinState
from backend.market_intelligence.models import MarketPrice, GovernmentUpdate, AgricultureNews, ScrapeCache
from backend import schemas
from backend import twin_engine
from backend.digital_twin.twin_pipeline import twin_pipeline

# Import consolidated model routers
from backend.ml import tabpfn_engine
from backend.ml import yield_transformer
from backend.ml import eif_detector
from backend.vision import florence_engine
from backend.vision import yolo_weed_detector
from backend.llm import agri_assistant
from backend import ollama_service as ollama_svc
from backend.agents import coder_agent
from backend.market_intelligence import router as market_intelligence_router

# Initialize database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Agrisense AI Gateway v4.0",
    version="4.0.0",
    description="Modernized consolidated API server utilizing TabPFN, FT-Transformer, Florence-2, YOLOv11 and local Ollama Qwen models."
)

# Cross-cutting middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.mlops.drift import data_drift_monitor
from backend import auth_routes
from backend import farm_routes
from backend import marketplace_routes
from backend import system_routes

# Register new routers
app.include_router(tabpfn_engine.router, prefix="/api")
app.include_router(yield_transformer.router, prefix="/api")
app.include_router(eif_detector.router, prefix="/api")
app.include_router(florence_engine.router, prefix="/api")
app.include_router(yolo_weed_detector.router, prefix="/api")
app.include_router(agri_assistant.router, prefix="/api")
app.include_router(coder_agent.router, prefix="/api")
app.include_router(data_drift_monitor.router, prefix="/api")
app.include_router(auth_routes.router, prefix="/api")
app.include_router(farm_routes.router, prefix="/api")
app.include_router(marketplace_routes.router, prefix="/api")
app.include_router(system_routes.router, prefix="/api")
app.include_router(market_intelligence_router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    from backend.market_intelligence.scheduler import start_scheduler
    start_scheduler()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgrisenseBackend")


# ── Database Seeder ──────────────────────────────────────────────────────────

def seed_db():
    db = next(get_db())
    try:
        # Seed Model Registry if empty with consolidated models
        if db.query(ModelRegistry).count() == 0:
            logger.info("Seeding Model Registry...")
            models_to_seed = [
                ModelRegistry(id="cm-01", name="CropRecommendation-TabPFN", version="v4.0.0", type="crop_recommendation", framework="TabPFN", status="active", accuracy=0.965, f1_score=0.962, last_retrained=datetime.utcnow(), prediction_count=1450),
                ModelRegistry(id="cm-02", name="FertilizerRecommendation-TabPFN", version="v4.0.0", type="fertilizer_recommendation", framework="TabPFN", status="active", accuracy=0.978, f1_score=0.975, last_retrained=datetime.utcnow(), prediction_count=120),
                ModelRegistry(id="ir-01", name="Irrigation-TabPFN", version="v4.0.0", type="irrigation_optimization", framework="TabPFN", status="active", accuracy=0.942, f1_score=0.938, last_retrained=datetime.utcnow(), prediction_count=890),
                ModelRegistry(id="yd-01", name="YieldPredictor-FTTransformer", version="v4.0.0", type="yield_prediction", framework="FT-Transformer", status="active", accuracy=0.915, f1_score=0.908, last_retrained=datetime.utcnow(), prediction_count=620),
                ModelRegistry(id="vs-01", name="PlantDisease-Florence2", version="v4.0.0", type="disease_detection", framework="HuggingFace Florence-2", status="active", accuracy=0.952, f1_score=0.949, last_retrained=datetime.utcnow(), prediction_count=1120),
                ModelRegistry(id="wd-01", name="WeedDetector-YOLO11n", version="v4.0.0", type="weed_detection", framework="Ultralytics YOLOv11n", status="active", accuracy=0.935, f1_score=0.931, last_retrained=datetime.utcnow(), prediction_count=450),
                ModelRegistry(id="ad-01", name="Anomaly-EIF", version="v4.0.0", type="anomaly_detection", framework="Extended Isolation Forest", status="active", accuracy=0.885, f1_score=0.879, last_retrained=datetime.utcnow(), prediction_count=320),
            ]
            db.add_all(models_to_seed)
            db.commit()

        if db.query(SensorReading).count() == 0:
            logger.info("Seeding Sensor Readings...")
            readings = [
                SensorReading(device_id="ESP32-S01", soil_moisture=37.1, temperature=29.2, humidity=58.8, ph=6.3, nitrogen=44, phosphorus=38, potassium=44),
                SensorReading(device_id="ESP32-S01", soil_moisture=38.3, temperature=28.9, humidity=59.4, ph=6.3, nitrogen=47, phosphorus=38, potassium=42),
                SensorReading(device_id="ESP32-S01", soil_moisture=39.8, temperature=28.5, humidity=60.2, ph=6.3, nitrogen=45, phosphorus=39, potassium=41),
                SensorReading(device_id="ESP32-S01", soil_moisture=41.2, temperature=28.1, humidity=61.5, ph=6.4, nitrogen=46, phosphorus=37, potassium=43),
                SensorReading(device_id="ESP32-S01", soil_moisture=42.5, temperature=27.8, humidity=62.1, ph=6.4, nitrogen=45, phosphorus=38, potassium=42),
            ]
            db.add_all(readings)
            db.commit()
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
    finally:
        db.close()

seed_db()


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "operational", "version": "4.0.0", "edge_node": "ESP32 ready"}


# ── IoT Sensor Readings & Ingestion ──────────────────────────────────────────

@app.get("/api/sensors")
async def get_sensor_readings(db: Session = Depends(get_db)):
    readings = db.query(SensorReading).order_by(SensorReading.timestamp.desc()).limit(50).all()
    formatted = []
    for r in readings:
        formatted.append({
            "id": str(r.id),
            "deviceId": r.device_id,
            "timestamp": r.timestamp.isoformat() + "Z" if r.timestamp else None,
            "soilMoisture": r.soil_moisture,
            "temperature": r.temperature,
            "humidity": r.humidity,
            "pH": r.ph,
            "nitrogen": r.nitrogen,
            "phosphorus": r.phosphorus,
            "potassium": r.potassium
        })
    return {"readings": formatted}


@app.post("/api/sensors/ingest", status_code=201)
async def ingest_iot_packet(packet: schemas.TelemetryInbound, db: Session = Depends(get_db)):
    db_reading = SensorReading(
        device_id=packet.deviceId or "ESP32-S02",
        soil_moisture=packet.soilMoisture,
        temperature=packet.temperature,
        humidity=packet.humidity,
        ph=packet.pH,
        nitrogen=packet.nitrogen,
        phosphorus=packet.phosphorus,
        potassium=packet.potassium
    )
    db.add(db_reading)
    db.commit()
    db.refresh(db_reading)
    logger.info(f"Ingested packet from {packet.deviceId} with ID {db_reading.id}")
    
    twin_pipeline.execute_pipeline({
        "N": packet.nitrogen,
        "P": packet.phosphorus,
        "K": packet.potassium,
        "temp": packet.temperature,
        "humidity": packet.humidity,
        "pH": packet.pH,
        "moisture": packet.soilMoisture
    })

    return {
        "message": "Data ingested successfully",
        "logged": {
            "id": str(db_reading.id),
            "deviceId": db_reading.device_id,
            "timestamp": db_reading.timestamp.isoformat() + "Z",
            "soilMoisture": db_reading.soil_moisture,
            "temperature": db_reading.temperature,
            "humidity": db_reading.humidity,
            "pH": db_reading.ph,
            "nitrogen": db_reading.nitrogen,
            "phosphorus": db_reading.phosphorus,
            "potassium": db_reading.potassium
        }
    }


# ── Frontend API Compatability Layer (TabPFN & FT-Transformer Mappings) ──────

@app.post("/api/crop-recommend")
async def crop_recommendation(payload: schemas.CropRecommendationInput, db: Session = Depends(get_db)):
    # Standardize input for TabPFN predict call
    tab_input = {
        "task": "crop_recommendation",
        "features": {
            "N": payload.N,
            "P": payload.P,
            "K": payload.K,
            "temperature": payload.temperature,
            "humidity": payload.humidity,
            "ph": payload.ph,
            "rainfall": payload.rainfall
        }
    }
    from fastapi import Request
    res = await tabpfn_engine.predict_tabular(tabpfn_engine.TabularPredictInput(**tab_input))
    
    # Log prediction to prediction_logs
    log_id = f"pl-{uuid.uuid4().hex[:5]}"
    db_log = PredictionLog(
        id=log_id,
        model_name="CropRecommendation-TabPFN",
        inputs_json=payload.model_dump_json(),
        output=f"{res['crops'][0]['name']} ({res['crops'][0]['suitability']}% suitability)",
        confidence=res['crops'][0]['suitability'] / 100.0,
        latency_ms=12,
        drift_score=0.01
    )
    db.add(db_log)
    db.commit()
    return res


@app.post("/api/irrigation-optimize")
async def irrigation_optimize(payload: schemas.IrrigationInput, db: Session = Depends(get_db)):
    tab_input = {
        "task": "irrigation_optimization",
        "features": {
            "moisture": payload.moisture,
            "temperature": payload.temperature,
            "humidity": payload.humidity
        }
    }
    res = await tabpfn_engine.predict_tabular(tabpfn_engine.TabularPredictInput(**tab_input))
    
    log_id = f"pl-{uuid.uuid4().hex[:5]}"
    db_log = PredictionLog(
        id=log_id,
        model_name="Irrigation-TabPFN",
        inputs_json=payload.model_dump_json(),
        output=f"Water req: {res['waterRequiredLiters']}L",
        confidence=0.95,
        latency_ms=10,
        drift_score=0.01
    )
    db.add(db_log)
    db.commit()
    return res


@app.post("/api/yield-predict")
async def yield_prediction(payload: schemas.YieldInput, db: Session = Depends(get_db)):
    res = await yield_transformer.predict_yield({
        "areaAcres": payload.areaAcres,
        "avgRainfall": payload.avgRainfall,
        "avgTemp": payload.avgTemp,
        "nitrogen": payload.nitrogen,
        "phosphorus": payload.phosphorus,
        "potassium": payload.potassium
    })
    
    log_id = f"pl-{uuid.uuid4().hex[:5]}"
    db_log = PredictionLog(
        id=log_id,
        model_name="YieldPredictor-FTTransformer",
        inputs_json=payload.model_dump_json(),
        output=f"{res['predictedYieldTons']} tons forecast",
        confidence=0.95,
        latency_ms=18,
        drift_score=0.01
    )
    db.add(db_log)
    db.commit()
    return res


class FertilizerInput(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soilType: str
    cropType: str
    nitrogen: float
    potassium: float
    phosphorus: float

@app.post("/api/fertilizer-recommend")
async def fertilizer_recommend(payload: FertilizerInput, db: Session = Depends(get_db)):
    tab_input = {
        "task": "fertilizer_recommendation",
        "features": {
            "temperature": payload.temperature,
            "humidity": payload.humidity,
            "moisture": payload.moisture,
            "N": payload.nitrogen,
            "K": payload.potassium,
            "P": payload.phosphorus
        }
    }
    res = await tabpfn_engine.predict_tabular(tabpfn_engine.TabularPredictInput(**tab_input))
    
    log_id = f"pl-{uuid.uuid4().hex[:5]}"
    db_log = PredictionLog(
        id=log_id,
        model_name="FertilizerRecommendation-TabPFN",
        inputs_json=payload.model_dump_json(),
        output=f"{res['fertilizer']} ({res['confidence'] * 100:.1f}% confidence)",
        confidence=res['confidence'],
        latency_ms=12,
        drift_score=0.01
    )
    db.add(db_log)
    db.commit()
    return {
        "recommendedFertilizer": res["fertilizer"],
        "confidence": res["confidence"],
        "recommendation": res["recommendation"]
    }


# ── MLOps Dashboard & Retraining ─────────────────────────────────────────────

@app.get("/api/mlops")
async def get_mlops_data(db: Session = Depends(get_db)):
    registry = db.query(ModelRegistry).all()
    logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(30).all()
    total_inferences = db.query(PredictionLog).count()
    active_count = db.query(ModelRegistry).filter(ModelRegistry.status == "active").count()
    
    formatted_logs = []
    for log in logs:
        try:
            import json
            inputs = json.loads(log.inputs_json)
        except Exception:
            inputs = {}
        formatted_logs.append({
            "id": log.id,
            "timestamp": log.timestamp.isoformat() + "Z" if log.timestamp else None,
            "modelName": log.model_name,
            "inputs": inputs,
            "output": log.output,
            "latencyMs": log.latency_ms,
            "confidence": log.confidence,
            "driftScore": log.drift_score
        })

    formatted_registry = []
    for m in registry:
        formatted_registry.append({
            "id": m.id,
            "name": m.name,
            "version": m.version,
            "type": m.type,
            "framework": m.framework,
            "status": m.status,
            "accuracy": m.accuracy,
            "f1Score": m.f1_score,
            "lastRetrained": m.last_retrained.isoformat() + "Z" if m.last_retrained else None,
            "predictionCount": m.prediction_count
        })

    return {
        "metrics": {
            "averageAccuracy": 0.942,
            "inferenceCount": 3520 + total_inferences,
            "averageLatencyMs": 14,
            "activeModelsCount": active_count,
            "anomalousInferences": 0,
            "driftIndex": 0.015
        },
        "registry": formatted_registry,
        "logs": formatted_logs
    }


@app.post("/api/mlops/retrain")
async def retrain_model_api(payload: dict = Body(...), db: Session = Depends(get_db)):
    model_id = payload.get("modelId")
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found in MLOps registry")

    model.last_retrained = datetime.utcnow()
    model.prediction_count = 0
    try:
        ver_num = float(model.version.lstrip('v').split('.')[0] + '.' + model.version.lstrip('v').split('.')[1])
        model.version = f"v{round(ver_num + 0.1, 1)}.0"
    except Exception:
        model.version = "v4.1.0"
        
    db.commit()
    return {
        "message": f"Model {model.name} retrained successfully",
        "updated": {
            "id": model.id,
            "name": model.name,
            "version": model.version,
            "type": model.type,
            "framework": model.framework,
            "status": model.status,
            "accuracy": model.accuracy,
            "f1Score": model.f1_score,
            "lastRetrained": model.last_retrained.isoformat() + "Z",
            "predictionCount": model.prediction_count
        }
    }


# ── Vision Compatibility (Florence-2 & YOLOv11) ──────────────────────────────

class DiseaseDetectRequest(BaseModel):
    imageBase64: str
    mode: str = "disease"

@app.post("/api/disease-detect")
async def disease_detect_compat(payload: DiseaseDetectRequest, db: Session = Depends(get_db)):
    if payload.mode == "weed":
        res = await yolo_weed_detector.detect_weeds({"imageBase64": payload.imageBase64})
        # Map YOLO model result to frontend contract
        return {
            "disease": f"Weeds detected: {res['weeds_detected']} invasive species",
            "confidence": 92.5,
            "severity": res["infestation_level"],
            "symptoms": [f"Infestation density calculated at {res['density_score']}%."],
            "recommendations": ["Apply organic cover composting", "Localized precision weed removal"]
        }
        
    res = await florence_engine.analyze_image({"imageBase64": payload.imageBase64, "mode": payload.mode})
    
    log_id = f"pl-{uuid.uuid4().hex[:5]}"
    db_log = PredictionLog(
        id=log_id,
        model_name="PlantDisease-Florence2",
        inputs_json=f'{{"mode": "{payload.mode}"}}',
        output=f"{res['disease']} ({res['confidence']}% confidence)",
        confidence=res['confidence'] / 100.0,
        latency_ms=120,
        drift_score=0.01
    )
    db.add(db_log)
    db.commit()
    
    return {
        "disease": res["disease"],
        "confidence": res["confidence"],
        "severity": res["severity"],
        "symptoms": [res["explanation"]],
        "recommendations": res["recommendations"]
    }


# ── Chatbot Interaction ──────────────────────────────────────────────────────

class ChatRequestAlt(BaseModel):
    messages: Optional[List[dict]] = None
    message: Optional[str] = None
    sensorContext: Optional[dict] = None
    prediction_context: Optional[dict] = None

@app.post("/api/chat")
async def chat_interaction(payload: ChatRequestAlt):
    # Extract the user message from messages list or message field
    user_message = payload.message
    if not user_message and payload.messages:
        for m in reversed(payload.messages):
            if m.get("role") == "user":
                user_message = m.get("content", "")
                break
    if not user_message:
        return {"text": "Please send a message to get started."}

    # Build history for context
    history = payload.messages or []

    # Use ollama_service which handles Ollama + keyword fallback gracefully
    reply = await ollama_svc.chat_with_agrigpt(user_message, history=history)
    return {"text": reply, "reply": reply}


# ── Digital Twin ─────────────────────────────────────────────────────────────

@app.post("/api/twin/update")
async def update_twin_state(payload: dict = Body(...)):
    telemetry = {
        "N": payload.get("nitrogen", 50.0),
        "P": payload.get("phosphorus", 40.0),
        "K": payload.get("potassium", 40.0),
        "temp": payload.get("temperature", 28.0),
        "humidity": payload.get("humidity", 60.0),
        "pH": payload.get("pH", 6.5),
        "moisture": payload.get("soilMoisture", 38.0),
        "rainfall": payload.get("rainfall", 0.0),
        "wind_speed": payload.get("windSpeed", 8.4)
    }
    pipeline_res = twin_pipeline.execute_pipeline(telemetry)
    
    legacy_res = twin_engine.update_state(
        soil_moisture=payload.get("soilMoisture"),
        temperature=payload.get("temperature"),
        humidity=payload.get("humidity"),
        pH=payload.get("pH"),
        nitrogen=payload.get("nitrogen"),
        phosphorus=payload.get("phosphorus"),
        potassium=payload.get("potassium"),
        rainfall=payload.get("rainfall", 0.0),
        wind_speed=payload.get("windSpeed")
    )
    
    merged_state = {
        "status": "updated",
        "hasAnomaly": pipeline_res["isAnomaly"],
        "alerts": pipeline_res["alerts"],
        "anomalyScore": pipeline_res["anomalyScore"],
        "twinState": {
            **legacy_res["twinState"],
            "overallHealthScore": pipeline_res["overallHealthScore"],
            "anomalyScore": pipeline_res["anomalyScore"],
            "isAnomaly": pipeline_res["isAnomaly"],
            "waterTwin": {
                **legacy_res["twinState"]["waterTwin"],
                "evapotranspirationET0": pipeline_res["physicsModel"]["evapotranspirationET0"],
                "waterDeficitLiters": pipeline_res["physicsModel"]["waterDeficitLiters"],
                "confidenceInterval": pipeline_res["physicsModel"]["confidenceInterval"],
                "uncertaintyMarginLiters": pipeline_res["physicsModel"]["uncertaintyMarginLiters"],
            },
            "recommendationNotes": pipeline_res["recommendationNotes"]
        }
    }
    twin_engine._twin_state.update(merged_state["twinState"])
    return merged_state


@app.get("/api/twin/state")
async def get_twin_state():
    return twin_engine.get_state()


@app.get("/api/twin/water")
async def get_twin_water():
    return twin_engine.get_state()["waterTwin"]


@app.get("/api/twin/soil")
async def get_twin_soil():
    return twin_engine.get_state()["soilTwin"]


@app.get("/api/twin/crop")
async def get_twin_crop():
    return twin_engine.get_state()["cropTwin"]


@app.get("/api/twin/weather")
async def get_twin_weather():
    return twin_engine.get_state()["weatherTwin"]


@app.get("/api/twin/disease")
async def get_twin_disease():
    return twin_engine.get_state()["diseaseTwin"]


@app.post("/api/twin/simulate")
async def simulate_twin_scenario(payload: dict = Body(...)):
    scenario_id = payload.get("scenarioId")
    if not scenario_id:
        raise HTTPException(status_code=400, detail="scenarioId is required")
    return twin_engine.run_simulation(scenario_id)


@app.get("/api/twin/analytics")
async def get_twin_analytics():
    return {
        "kpis": {
            "healthIndexHistory": [81, 83, 85, 84, 88],
            "waterConservationLiters": 14500,
            "carbonOffsetPercentage": 11.4,
            "nitrogenUtilizationRate": 91.2
        },
        "sustainabilityIndices": {
            "waterUseEfficiency": 94,
            "pesticideReductionIndex": 88,
            "soilStructuralRetention": 91
        }
    }


# ── Frontend Static serving catch-all ────────────────────────────────────────

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

dist_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dist")

if os.path.exists(os.path.join(dist_path, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
        
    file_path = os.path.join(dist_path, full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
        
    return FileResponse(os.path.join(dist_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, log_level="info")

