from pydantic import BaseModel, Field
from typing import List, Optional

class CropRecommendationItemSchema(BaseModel):
    cropKey: str
    cropName: str
    scientificName: str
    rank: int
    confidencePct: float
    waterRequirementMm: float
    growthDurationDays: int
    expectedYieldTonHa: float
    dripSavingsPct: float
    limitingFactor: str

class CropRecommendationRequestSchema(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class CropRecommendationResponseSchema(BaseModel):
    recommendations: List[CropRecommendationItemSchema]
    modelSource: str

class FertilizerScheduleItemSchema(BaseModel):
    productKey: str
    productName: str
    dosageKgHa: float
    timingKey: str
    applicationNote: str

class NutrientDeficitsSchema(BaseModel):
    nitrogen: float
    phosphorus: float
    potassium: float

class FertilizerRequestSchema(BaseModel):
    cropType: str
    nitrogen: float
    phosphorus: float
    potassium: float
    soilPh: float
    soilMoisture: float

class FertilizerResponseSchema(BaseModel):
    cropType: str
    deficitsKgHa: NutrientDeficitsSchema
    schedule: List[FertilizerScheduleItemSchema]
    organicAmendment: str
    moistureWarning: Optional[str] = None
    timestamp: str

class DiseaseAnalyzeRequestSchema(BaseModel):
    imageBase64: Optional[str] = None
    mimeType: Optional[str] = None
    cropHint: Optional[str] = None
    samplePreset: Optional[str] = None
    language: str = "en"

class DiseaseDiagnosisResultSchema(BaseModel):
    isHealthy: bool
    cropIdentified: str
    diseaseName: str
    scientificName: str
    confidencePct: float
    severity: str
    symptomsObserved: List[str]
    organicTreatment: List[str]
    chemicalTreatment: List[str]
    preventionSteps: List[str]
    source: str
    isModelPrediction: bool
    analyzedAt: str

class DigitalTwinSeriesPointSchema(BaseModel):
    dateIso: str
    dateShort: str
    observedMoisturePct: float
    predictedMoisturePct: float
    soilTempC: float
    irrigationMm: float
    cropNdvi: float
    evapotranspirationMm: float

class WaterTwinSchema(BaseModel):
    modelStatus: str
    rootZoneDepletionMm: float
    fieldCapacityPct: float
    wiltingPointPct: float
    dailyEt0Mm: float
    nextIrrigationHours: float
    seepageLossPct: float
    confidencePct: float
    isSimulatedForecast: bool

class SoilTwinSchema(BaseModel):
    modelStatus: str
    cationExchangeCapacityMeq: float
    organicCarbonPct: float
    microbialActivityIndex: float
    salinityRisk: str
    npkMineralizationRate: str
    confidencePct: float
    isSimulatedForecast: bool

class CropTwinSchema(BaseModel):
    modelStatus: str
    phenologyStageKey: str
    daysAfterSowing: int
    growingDegreeDays: float
    leafAreaIndex: float
    canopyCoverPct: float
    projectedHarvestDays: int
    projectedYieldTonHa: float
    confidencePct: float
    isSimulatedForecast: bool

class DigitalTwinResponseSchema(BaseModel):
    zoneId: str
    days: int
    waterTwin: WaterTwinSchema
    soilTwin: SoilTwinSchema
    cropTwin: CropTwinSchema
    series: List[DigitalTwinSeriesPointSchema]
    timestamp: str

class AssistantChatRequestSchema(BaseModel):
    message: str
    language: str = "en"

class AssistantChatResponseSchema(BaseModel):
    reply: str
    languageUsed: str
    isFallbackLanguage: bool
    sources: List[str]
    timestamp: str
