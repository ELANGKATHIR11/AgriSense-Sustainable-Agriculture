from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class DeviceReadingsSchema(BaseModel):
    soilMoisturePct: float
    soilTempC: float
    airTempC: float
    airHumidityPct: float
    soilPh: float
    nitrogenMgKg: float
    phosphorusMgKg: float
    potassiumMgKg: float
    ecDsM: float
    solarLux: float

class Coordinates3D(BaseModel):
    x: float
    y: float
    z: float

class DeviceTelemetrySchema(BaseModel):
    id: str
    name: str
    type: str
    protocol: str
    zoneId: str
    zoneName: str
    cropType: str
    status: str
    firmware: str
    batteryPct: float
    signalRssi: int
    lastSeenMinutesAgo: int
    samplingIntervalSec: int
    moistureThresholdPct: float
    calibrationOffsetPct: Optional[float] = 0.0
    coordinates3D: Optional[Coordinates3D] = None
    readings: DeviceReadingsSchema
    isSimulated: bool

class IrrigationZoneStateSchema(BaseModel):
    zoneId: str
    zoneName: str
    cropType: str
    valveStatus: str
    soilMoisturePct: float
    targetMoisturePct: float
    flowRateLpm: float
    lastIrrigatedIso: str
    recommendedDurationMin: int
    waterSavedLitersToday: float
    autoScheduleEnabled: bool

class FarmEventSchema(BaseModel):
    id: str
    timestamp: str
    type: str
    zoneId: str
    code: str
    value: str
    threshold: str

class IrrigationStateSchema(BaseModel):
    tankLevelPct: float
    tankCapacityLiters: float
    currentVolumeLiters: float
    pumpStatus: str
    pumpPressureBar: float
    waterQualityPh: float
    waterQualityTdsPpm: float
    dailyWaterSavedPct: float
    zones: List[IrrigationZoneStateSchema]
    recentEvents: List[FarmEventSchema]

class SensorHistoryPointSchema(BaseModel):
    timestamp: str
    hourLabel: str
    soilMoisturePct: float
    soilTempC: float
    airHumidityPct: float
    evapotranspirationMm: float

class WeatherSummarySchema(BaseModel):
    conditionCode: str
    tempC: float
    humidityPct: float
    windSpeedKmh: float
    rainProbabilityPct: float
    solarRadiationWm2: float
    source: str

class CropHealthSummarySchema(BaseModel):
    overallStatus: str
    healthyPlotsCount: int
    attentionPlotsCount: int
    activeAlertCode: str

class SensorsSummarySchema(BaseModel):
    totalSensors: int
    onlineSensors: int
    staleSensors: int
    avgSoilMoisturePct: float
    avgSoilTempC: float
    avgAirHumidityPct: float
    avgAirTempC: float
    weatherSummary: WeatherSummarySchema
    cropHealthSummary: CropHealthSummarySchema

class SensorsResponseSchema(BaseModel):
    summary: SensorsSummarySchema
    devices: List[DeviceTelemetrySchema]
    history24h: List[SensorHistoryPointSchema]
    irrigation: IrrigationStateSchema
    timestamp: str

class SystemStatusResponseSchema(BaseModel):
    status: str
    backendVersion: str
    mqttBroker: str
    edgeGateway: str
    aiService: str
    aiConfigured: bool
    totalDevices: int
    onlineDevices: int
    staleDevices: int
    timestamp: str
