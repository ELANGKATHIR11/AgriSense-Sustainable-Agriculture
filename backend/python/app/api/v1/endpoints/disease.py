import os
import base64
from fastapi import APIRouter, HTTPException
from app.ml.disease_classifier import disease_classifier
from app.schemas.recommendations import DiseaseAnalyzeRequestSchema, DiseaseDiagnosisResultSchema
from app.core.config import settings
from app.api.v1.endpoints.operations import increment_vision_count

router = APIRouter()

PRESET_SAMPLE_DIR = r"D:\Agrisense-A samart Agriculture Solution\Agrisense\dataset\diseases\test\test"

@router.post("/analyze", response_model=DiseaseDiagnosisResultSchema)
def analyze_disease(payload: DiseaseAnalyzeRequestSchema):
    increment_vision_count()
    image_bytes = None

    if payload.imageBase64:
        raw_b64 = payload.imageBase64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        try:
            image_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")
    elif payload.samplePreset:
        # Check if sample preset file exists in dataset/diseases/test/test
        preset_clean = payload.samplePreset.strip().lower()
        matched_file = None
        if os.path.exists(PRESET_SAMPLE_DIR):
            for fname in os.listdir(PRESET_SAMPLE_DIR):
                if preset_clean in fname.lower() or fname.lower().startswith(preset_clean):
                    matched_file = os.path.join(PRESET_SAMPLE_DIR, fname)
                    break
            if not matched_file:
                # Pick first sample
                files = os.listdir(PRESET_SAMPLE_DIR)
                if files:
                    matched_file = os.path.join(PRESET_SAMPLE_DIR, files[0])

        if matched_file and os.path.exists(matched_file):
            with open(matched_file, "rb") as f:
                image_bytes = f.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Either valid 'imageBase64' or a valid 'samplePreset' must be supplied."
        )

    try:
        result = disease_classifier.predict_image(image_bytes, crop_hint=payload.cropHint)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plant disease inference failed: {e}")
