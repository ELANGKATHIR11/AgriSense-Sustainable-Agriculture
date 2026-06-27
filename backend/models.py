from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    role = Column(String, default="farmer") # admin, farmer, researcher, student
    is_active = Column(Boolean, default=True)
    preferred_language = Column(String, default="en")

class SensorReading(Base):
    __tablename__ = "sensor_readings"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    soil_moisture = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    ph = Column(Float)
    nitrogen = Column(Integer)
    phosphorus = Column(Integer)
    potassium = Column(Integer)

class TwinState(Base):
    __tablename__ = "twin_state"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    overall_health_score = Column(Float)
    risk_index = Column(Float)
    yield_index = Column(Float)
    sustainability_index = Column(Float)

class ModelRegistry(Base):
    __tablename__ = "model_registry"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    version = Column(String)
    type = Column(String)
    framework = Column(String)
    status = Column(String) # active, staging, archived
    accuracy = Column(Float)
    f1_score = Column(Float)
    last_retrained = Column(DateTime(timezone=True))
    prediction_count = Column(Integer, default=0)

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(String, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    model_name = Column(String)
    inputs_json = Column(Text)
    output = Column(String)
    confidence = Column(Float)
    latency_ms = Column(Integer)
    drift_score = Column(Float)

# ── Commercial Extensions ──

class Farm(Base):
    __tablename__ = "farms"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    location = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Field(Base):
    __tablename__ = "fields"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    area_acres = Column(Float)
    crop_type = Column(String)
    farm_id = Column(Integer, ForeignKey("farms.id"))

class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, unique=True, index=True)
    device_type = Column(String, default="ESP32") # Gateway, Node, Actuator
    status = Column(String, default="active") # active, offline
    field_id = Column(Integer, ForeignKey("fields.id"))
    last_seen = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

class MarketProduct(Base):
    __tablename__ = "marketplace_products"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    category = Column(String) # seed, fertilizer, pesticide, tool
    price = Column(Float)
    supplier = Column(String)
    buy_url = Column(String)
    description = Column(Text)

class Vendor(Base):
    __tablename__ = "marketplace_vendors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    contact = Column(String)
    location = Column(String)
    rating = Column(Float, default=5.0)

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    plan_name = Column(String, default="Free") # Free, Professional, Enterprise
    status = Column(String, default="active")
    start_date = Column(DateTime(timezone=True), server_default=func.now())
    end_date = Column(DateTime(timezone=True))

class LicenseKey(Base):
    __tablename__ = "licenses"
    id = Column(Integer, primary_key=True, index=True)
    key_string = Column(String, unique=True, index=True)
    is_valid = Column(Boolean, default=True)
    plan = Column(String, default="Professional")
    activated_at = Column(DateTime(timezone=True), server_default=func.now())

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    action = Column(String)
    user_email = Column(String)
    details = Column(Text)

class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    title = Column(String)
    message = Column(Text)
    severity = Column(String, default="info") # info, warning, critical
    is_read = Column(Boolean, default=False)
