from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
from datetime import datetime

# Telemetry
class TelemetryInbound(BaseModel):
    deviceId: str = Field(..., min_length=1, max_length=50)
    soilMoisture: float = Field(..., ge=0.0, le=100.0)
    temperature: float = Field(..., ge=-20.0, le=60.0)
    humidity: float = Field(..., ge=0.0, le=100.0)
    pH: float = Field(..., ge=0.0, le=14.0)
    nitrogen: int = Field(..., ge=0, le=500)
    phosphorus: int = Field(..., ge=0, le=500)
    potassium: int = Field(..., ge=0, le=500)

# Chat
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

# ML Inputs
class CropRecommendationInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class IrrigationInput(BaseModel):
    moisture: float
    temperature: float
    humidity: float
    cropType: Optional[str] = None

class YieldInput(BaseModel):
    areaAcres: float
    avgRainfall: float
    avgTemp: float
    cropType: str
    nitrogen: Optional[float] = 45.0
    phosphorus: Optional[float] = 38.0
    potassium: Optional[float] = 42.0

# Outputs
class CropPrediction(BaseModel):
    crops: List[dict]
    optimalPH: str
    nutritionStatus: str

class IrrigationPrediction(BaseModel):
    waterRequiredLiters: float
    moistureStatus: str
    advice: str
    durationMinutes: int
    irrigationSchedule: str

class YieldPrediction(BaseModel):
    predictedYieldTons: float
    confidenceMin: float
    confidenceMax: float
    marketValueEstimate: int
    yieldBreakdown: str
