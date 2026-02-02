from typing import List, Optional
from pydantic import BaseModel, Field


class Cure(BaseModel):
    immediate_actions: List[str]
    chemical_treatments: List[str]
    biological_treatments: Optional[List[str]] = []


class Prevention(BaseModel):
    cultural_practices: List[str] = Field(
        ..., description="Farming practices to prevent recurrence"
    )
    long_term_strategy: List[str] = Field(
        ..., description="Long term protection strategy"
    )


class VLMAnalysisResult(BaseModel):
    crop_identified: str
    scientific_name: Optional[str] = None
    diagnosis: str
    confidence: float
    severity: Optional[str] = "Medium"
    detected_concepts: Optional[str] = None
    cure: Cure
    prevention: Prevention

    class Config:
        schema_extra = {
            "example": {
                "crop_identified": "Paddy",
                "diagnosis": "Leaf Blast",
                "confidence": 91.5,
                "severity": "High",
                "cure": {
                    "immediate_actions": ["Remove infected plants"],
                    "chemical_treatments": ["Spray Tricyclazole as per label"],
                    "biological_treatments": ["Use Pseudomonas fluorescens"],
                },
                "prevention": {
                    "cultural_practices": [
                        "Avoid excess nitrogen",
                        "Maintain proper spacing",
                    ],
                    "long_term_strategy": ["Use resistant varieties like IR64"],
                },
            }
        }
