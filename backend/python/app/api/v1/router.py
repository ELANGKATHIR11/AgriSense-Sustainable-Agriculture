from fastapi import APIRouter
from app.api.v1.endpoints import (
    sensors, irrigation, recommendations, disease, digital_twins, assistant, operations
)

api_router = APIRouter()

api_router.include_router(sensors.router, prefix="", tags=["sensors"])
api_router.include_router(irrigation.router, prefix="/irrigation", tags=["irrigation"])
api_router.include_router(recommendations.router, prefix="/recommendations", tags=["recommendations"])
api_router.add_api_route("/crops", recommendations.get_all_crops, methods=["GET"], tags=["crops"])
api_router.include_router(disease.router, prefix="/disease", tags=["disease"])
api_router.include_router(digital_twins.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(assistant.router, prefix="/assistant", tags=["assistant"])
api_router.include_router(operations.router, prefix="/operations", tags=["operations"])
