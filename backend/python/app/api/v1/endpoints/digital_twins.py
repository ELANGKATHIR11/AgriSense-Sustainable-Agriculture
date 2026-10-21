from fastapi import APIRouter, Query
from app.ml.digital_twin_engine import digital_twin_engine
from app.schemas.recommendations import DigitalTwinResponseSchema

router = APIRouter()

@router.get("/digital-twin", response_model=DigitalTwinResponseSchema)
def get_digital_twin(
    zoneId: str = Query("ZONE-A", description="Farm zone identifier"),
    days: int = Query(7, ge=1, le=30, description="Forecast timeline window in days"),
    rainDeltaPct: float = Query(0.0, description="Precipitation delta percentage"),
    tempDeltaC: float = Query(0.0, description="Canopy temperature anomaly in Celsius"),
    irrigationEffPct: float = Query(85.0, description="Drip delivery efficiency percentage"),
    fertilizerEfficiencyPct: float = Query(88.0, description="NPK uptake efficiency percentage")
):
    result = digital_twin_engine.simulate(
        zone_id=zoneId,
        days=days,
        rain_delta_pct=rainDeltaPct,
        temp_delta_c=tempDeltaC,
        irrigation_eff_pct=irrigationEffPct,
        fertilizer_eff_pct=fertilizerEfficiencyPct
    )
    return result
