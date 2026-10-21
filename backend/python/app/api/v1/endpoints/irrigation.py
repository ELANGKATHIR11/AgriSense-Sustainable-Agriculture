from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.entities import IrrigationZone, IrrigationStateRecord, FarmEventRecord, AuditLogRecord
from app.schemas.telemetry import IrrigationZoneStateSchema, IrrigationStateSchema, FarmEventSchema

router = APIRouter()

@router.post("/control")
def control_irrigation(payload: dict = Body(...), db: Session = Depends(get_db)):
    zone_id = payload.get("zoneId")
    action = payload.get("action")
    duration = payload.get("durationMin", 30)

    zone = db.query(IrrigationZone).filter(IrrigationZone.zone_id == zone_id).first()
    if not zone:
        raise HTTPException(status_code=404, detail="Zone not found")

    ir_state = db.query(IrrigationStateRecord).first()
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    if action == "start":
        zone.valve_status = "active"
        zone.flow_rate_lpm = 45.0
        zone.last_irrigated_iso = now_iso
        if ir_state:
            ir_state.pump_status = "active"
            ir_state.pump_pressure_bar = 3.6
    elif action == "stop":
        zone.valve_status = "idle"
        zone.flow_rate_lpm = 0.0
        # If no other zone is active, idle the pump
        active_zones = db.query(IrrigationZone).filter(IrrigationZone.valve_status == "active", IrrigationZone.zone_id != zone_id).count()
        if active_zones == 0 and ir_state:
            ir_state.pump_status = "idle"
            ir_state.pump_pressure_bar = 3.2
    elif action == "toggle_auto":
        zone.auto_schedule_enabled = not zone.auto_schedule_enabled

    import uuid
    # Log farm event
    event_id = f"EVT-{datetime.utcnow().strftime('%M%S')}-{uuid.uuid4().hex[:6]}"
    event = FarmEventRecord(
        id=event_id,
        timestamp=datetime.utcnow(),
        type="info",
        zone_id=zone_id,
        code=f"IRRIGATION_{action.upper()}",
        value=f"Action {action} processed for {zone.zone_name}",
        threshold="NOMINAL"
    )
    db.add(event)

    # Log audit event
    audit_id = f"AUD-{datetime.utcnow().strftime('%H%M%S')}-{uuid.uuid4().hex[:6]}"
    audit = AuditLogRecord(
        id=audit_id,
        timestamp=datetime.utcnow(),
        actor="operator",
        role="Field Agronomist",
        category="IRRIGATION",
        action=f"VALVE_{action.upper()}",
        target=f"{zone.zone_id} ({zone.zone_name})",
        status="CONFIRMED",
        correlation_id=f"req-{datetime.utcnow().strftime('%f')[:8]}",
        details=f"Irrigation command '{action}' acknowledged and dispatched by local valve controller."
    )
    db.add(audit)

    db.commit()
    db.refresh(zone)
    if ir_state:
        db.refresh(ir_state)

    zones = db.query(IrrigationZone).all()
    recent_events = db.query(FarmEventRecord).order_by(FarmEventRecord.timestamp.desc()).limit(5).all()

    zone_obj = {
        "zoneId": zone.zone_id,
        "zoneName": zone.zone_name,
        "cropType": zone.crop_type,
        "valveStatus": zone.valve_status,
        "soilMoisturePct": zone.soil_moisture_pct,
        "targetMoisturePct": zone.target_moisture_pct,
        "flowRateLpm": zone.flow_rate_lpm,
        "lastIrrigatedIso": zone.last_irrigated_iso,
        "recommendedDurationMin": zone.recommended_duration_min,
        "waterSavedLitersToday": zone.water_saved_liters_today,
        "autoScheduleEnabled": zone.auto_schedule_enabled
    }

    irrigation_obj = {
        "tankLevelPct": ir_state.tank_level_pct if ir_state else 78.5,
        "tankCapacityLiters": ir_state.tank_capacity_liters if ir_state else 50000.0,
        "currentVolumeLiters": ir_state.current_volume_liters if ir_state else 39250.0,
        "pumpStatus": ir_state.pump_status if ir_state else "idle",
        "pumpPressureBar": ir_state.pump_pressure_bar if ir_state else 3.2,
        "waterQualityPh": ir_state.water_quality_ph if ir_state else 7.1,
        "waterQualityTdsPpm": ir_state.water_quality_tds_ppm if ir_state else 340.0,
        "dailyWaterSavedPct": ir_state.daily_water_saved_pct if ir_state else 34.2,
        "zones": [{
            "zoneId": z.zone_id, "zoneName": z.zone_name, "cropType": z.crop_type, "valveStatus": z.valve_status,
            "soilMoisturePct": z.soil_moisture_pct, "targetMoisturePct": z.target_moisture_pct, "flowRateLpm": z.flow_rate_lpm,
            "lastIrrigatedIso": z.last_irrigated_iso, "recommendedDurationMin": z.recommended_duration_min,
            "waterSavedLitersToday": z.water_saved_liters_today, "autoScheduleEnabled": z.auto_schedule_enabled
        } for z in zones],
        "recentEvents": [{
            "id": e.id, "timestamp": e.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"), "type": e.type, "zoneId": e.zone_id,
            "code": e.code, "value": e.value, "threshold": e.threshold
        } for e in recent_events]
    }

    return {
        "confirmed": True,
        "zone": zone_obj,
        "irrigation": irrigation_obj
    }
