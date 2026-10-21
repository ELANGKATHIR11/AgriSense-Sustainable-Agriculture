from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from datetime import datetime
from app.db.base import Base

class Device(Base):
    __tablename__ = "devices"
    
    id = Column(String(64), primary_key=True)
    name = Column(String(128), nullable=False)
    type = Column(String(64), default="ESP32_SOIL_NODE")
    protocol = Column(String(64), default="MQTT/TLS")
    zone_id = Column(String(64), default="ZONE-A")
    zone_name = Column(String(128), default="Cauvery Block A")
    crop_type = Column(String(64), default="Paddy (CR 1009)")
    status = Column(String(32), default="online")
    firmware = Column(String(32), default="v2.4.1-rc3")
    battery_pct = Column(Float, default=92.0)
    signal_rssi = Column(Integer, default=-64)
    sampling_interval_sec = Column(Integer, default=60)
    moisture_threshold_pct = Column(Float, default=45.0)
    calibration_offset_pct = Column(Float, default=0.0)
    coord_x = Column(Float, default=0.0)
    coord_y = Column(Float, default=0.0)
    coord_z = Column(Float, default=0.0)
    is_simulated = Column(Boolean, default=False)
    last_seen = Column(DateTime, default=datetime.utcnow)

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    device_id = Column(String(64), ForeignKey("devices.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    soil_moisture_pct = Column(Float, default=0.0)
    soil_temp_c = Column(Float, default=0.0)
    air_temp_c = Column(Float, default=0.0)
    air_humidity_pct = Column(Float, default=0.0)
    soil_ph = Column(Float, default=6.5)
    nitrogen_mg_kg = Column(Float, default=0.0)
    phosphorus_mg_kg = Column(Float, default=0.0)
    potassium_mg_kg = Column(Float, default=0.0)
    ec_ds_m = Column(Float, default=0.0)
    solar_lux = Column(Float, default=0.0)

class IrrigationZone(Base):
    __tablename__ = "irrigation_zones"
    
    zone_id = Column(String(64), primary_key=True)
    zone_name = Column(String(128), nullable=False)
    crop_type = Column(String(64), nullable=False)
    valve_status = Column(String(32), default="idle")
    soil_moisture_pct = Column(Float, default=52.0)
    target_moisture_pct = Column(Float, default=55.0)
    flow_rate_lpm = Column(Float, default=0.0)
    last_irrigated_iso = Column(String(64), default=datetime.utcnow().isoformat())
    recommended_duration_min = Column(Integer, default=25)
    water_saved_liters_today = Column(Float, default=1240.0)
    auto_schedule_enabled = Column(Boolean, default=True)

class IrrigationStateRecord(Base):
    __tablename__ = "irrigation_state"
    
    id = Column(Integer, primary_key=True, default=1)
    tank_level_pct = Column(Float, default=78.5)
    tank_capacity_liters = Column(Float, default=50000.0)
    current_volume_liters = Column(Float, default=39250.0)
    pump_status = Column(String(32), default="idle")
    pump_pressure_bar = Column(Float, default=3.2)
    water_quality_ph = Column(Float, default=7.1)
    water_quality_tds_ppm = Column(Float, default=340.0)
    daily_water_saved_pct = Column(Float, default=34.2)

class FarmEventRecord(Base):
    __tablename__ = "farm_events"
    
    id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    type = Column(String(32), default="info")
    zone_id = Column(String(64), default="ZONE-A")
    code = Column(String(64), default="VALVE_CYCLE")
    value = Column(String(128), default="")
    threshold = Column(String(128), default="")

class AuditLogRecord(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    actor = Column(String(64), default="system")
    role = Column(String(64), default="Automated Controller")
    category = Column(String(32), default="SETTINGS")
    action = Column(String(64), default="INIT")
    target = Column(String(128), default="")
    status = Column(String(32), default="CONFIRMED")
    correlation_id = Column(String(64), default="")
    details = Column(Text, default="")

class ServiceLogRecord(Base):
    __tablename__ = "service_logs"
    
    id = Column(String(64), primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(16), default="INFO")
    service = Column(String(64), default="backend")
    correlation_id = Column(String(64), default="")
    message = Column(Text, default="")

class PlatformSettingsRecord(Base):
    __tablename__ = "platform_settings"
    
    id = Column(Integer, primary_key=True, default=1)
    timezone = Column(String(64), default="Asia/Kolkata")
    moisture_critical_alert_pct = Column(Float, default=32.0)
    stale_node_timeout_min = Column(Integer, default=15)
    max_continuous_irrigation_min = Column(Integer, default=45)
    safety_interlock_required = Column(Boolean, default=True)
    quarantine_stale_nodes = Column(Boolean, default=True)
    enable_3d_shadows = Column(Boolean, default=True)
    telemetry_retention_days = Column(Integer, default=90)
    log_level = Column(String(16), default="INFO")
