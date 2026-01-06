from pydantic import BaseModel
from typing import List, Optional

class SensorData(BaseModel):
    device_id: str
    timestamp: str
    temperature: float
    humidity: float
    soil_moisture: float

class CropData(BaseModel):
    crop_id: str
    crop_name: str
    growth_stage: str
    yield_estimate: float
    area_ha: float

class WeatherData(BaseModel):
    location: str
    timestamp: str
    temperature: float
    precipitation: float
    humidity: float

class SoilData(BaseModel):
    location: str
    soil_type: str
    ph_level: float
    organic_matter: float
    nutrient_levels: dict

class SyntheticDataset(BaseModel):
    sensor_data: List[SensorData]
    crop_data: List[CropData]
    weather_data: List[WeatherData]
    soil_data: List[SoilData]