import asyncio
import random
from datetime import datetime
from app.db.session import SessionLocal
from app.models.entities import Device, SensorReading, IrrigationZone, IrrigationStateRecord, FarmEventRecord, AuditLogRecord

class LiveTelemetryService:
    def __init__(self):
        self._running = False
        self._task = None

    async def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._run_loop())
            print("[LiveTelemetryService] Real-time sensor telemetry engine started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            print("[LiveTelemetryService] Telemetry engine stopped.")

    async def _run_loop(self):
        while self._running:
            try:
                self.tick_once()
            except Exception as e:
                print(f"[LiveTelemetryService Error]: {e}")
            await asyncio.sleep(4)  # 4-second real-time live IoT streaming cycle

    def tick_once(self):
        db = SessionLocal()
        try:
            devices = db.query(Device).all()
            now = datetime.utcnow()
            hour = now.hour

            # Natural diurnal baseline
            is_day = 6 <= hour <= 18
            ambient_temp = 28.0 + (4.0 if 11 <= hour <= 15 else (-2.0 if not is_day else 1.0))
            ambient_hum = 65.0 - (12.0 if 11 <= hour <= 15 else (-8.0 if not is_day else 0.0))
            solar_base = 65000.0 * max(0.0, 1.0 - abs(hour - 12) / 6.0) if is_day else 0.0

            for dev in devices:
                # Keep ESP32-NODE-104 as a stale node if intended, or update online nodes
                if "104" in dev.id:
                    # Periodic stale node check
                    continue

                latest_r = db.query(SensorReading).filter(
                    SensorReading.device_id == dev.id
                ).order_by(SensorReading.timestamp.desc()).first()

                curr_moist = latest_r.soil_moisture_pct if latest_r else dev.moisture_threshold_pct + 5.0
                curr_ph = latest_r.soil_ph if latest_r else 6.6
                curr_n = latest_r.nitrogen_mg_kg if latest_r else 135.0
                curr_p = latest_r.phosphorus_mg_kg if latest_r else 42.0
                curr_k = latest_r.potassium_mg_kg if latest_r else 175.0

                # Check if zone valve is running
                zone = db.query(IrrigationZone).filter(IrrigationZone.zone_id == dev.zone_id).first()
                is_irrigating = zone and zone.valve_status == "active"

                if is_irrigating:
                    # Soil absorbs water
                    new_moist = min(zone.target_moisture_pct + 2.0, curr_moist + round(random.uniform(0.4, 0.8), 2))
                    if new_moist >= zone.target_moisture_pct and zone.auto_schedule_enabled:
                        # Auto shutoff
                        zone.valve_status = "idle"
                        zone.flow_rate_lpm = 0.0
                        db.add(FarmEventRecord(
                            id=f"EVT-{now.strftime('%H%M%S')}",
                            timestamp=now,
                            type="info",
                            zone_id=zone.zone_id,
                            code="AUTO_IRRIGATION_COMPLETE",
                            value=f"Target {zone.target_moisture_pct}% moisture reached; valve closed.",
                            threshold=f"{zone.target_moisture_pct}%"
                        ))
                        db.add(AuditLogRecord(
                            id=f"AUD-{now.strftime('%H%M%S')}",
                            timestamp=now,
                            actor="system-scheduler",
                            role="Automation Agent",
                            category="IRRIGATION",
                            action="AUTO_VALVE_STOP",
                            target=zone.zone_name,
                            status="CONFIRMED",
                            correlation_id=f"auto-{now.strftime('%f')[:6]}",
                            details=f"Closed zone valve after reaching target moisture level {zone.target_moisture_pct}%."
                        ))
                else:
                    # Natural soil evaporation / root absorption decay
                    decay = round(random.uniform(0.02, 0.08), 2)
                    new_moist = max(18.0, curr_moist - decay)

                    # Auto start if below threshold
                    if zone and zone.auto_schedule_enabled and new_moist < dev.moisture_threshold_pct and zone.valve_status == "idle":
                        zone.valve_status = "active"
                        zone.flow_rate_lpm = 45.0
                        zone.last_irrigated_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
                        db.add(FarmEventRecord(
                            id=f"EVT-{now.strftime('%H%M%S')}",
                            timestamp=now,
                            type="warning",
                            zone_id=zone.zone_id,
                            code="AUTO_IRRIGATION_START",
                            value=f"Moisture {new_moist:.1f}% below threshold {dev.moisture_threshold_pct}%; auto-valve open.",
                            threshold=f"{dev.moisture_threshold_pct}%"
                        ))

                if zone:
                    zone.soil_moisture_pct = round(new_moist, 1)

                # Micro jitter on readings for realistic physical telemetry
                new_temp = round(ambient_temp + random.uniform(-0.3, 0.3), 1)
                new_soil_temp = round(new_temp - 4.2 + random.uniform(-0.2, 0.2), 1)
                new_hum = round(max(30.0, min(98.0, ambient_hum + random.uniform(-0.8, 0.8))), 1)
                new_lux = round(max(0.0, solar_base + random.uniform(-200.0, 200.0)), 1) if is_day else 0.0

                new_reading = SensorReading(
                    device_id=dev.id,
                    timestamp=now,
                    soil_moisture_pct=round(new_moist, 1),
                    soil_temp_c=new_soil_temp,
                    air_temp_c=new_temp,
                    air_humidity_pct=new_hum,
                    soil_ph=round(curr_ph + random.uniform(-0.01, 0.01), 2),
                    nitrogen_mg_kg=round(curr_n + random.uniform(-0.1, 0.1), 1),
                    phosphorus_mg_kg=round(curr_p + random.uniform(-0.05, 0.05), 1),
                    potassium_mg_kg=round(curr_k + random.uniform(-0.1, 0.1), 1),
                    ec_ds_m=1.22,
                    solar_lux=new_lux
                )
                db.add(new_reading)

                dev.last_seen = now
                dev.battery_pct = max(10.0, dev.battery_pct - 0.001)

            db.commit()
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

telemetry_service = LiveTelemetryService()
