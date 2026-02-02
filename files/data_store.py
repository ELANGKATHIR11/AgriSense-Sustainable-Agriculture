"""
AgriSense Data Store
Hybrid database layer with SQLite fallback and mock data support
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random
import logging

logger = logging.getLogger(__name__)

# Database connection
DB_PATH = "sensors.db"

def init_sensor_db():
    """Initialize SQLite database for sensor data"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create devices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'active',
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                configuration TEXT
            )
        """)
        
        # Create sensor_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                temperature REAL,
                humidity REAL,
                pressure REAL,
                soil_moisture REAL,
                soil_temperature REAL,
                ph_level REAL,
                nitrogen REAL,
                phosphorus REAL,
                potassium REAL,
                light_intensity REAL,
                FOREIGN KEY (device_id) REFERENCES devices(device_id)
            )
        """)
        
        # Create index for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sensor_timestamp 
            ON sensor_data(device_id, timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
        
        # Insert default device if not exists
        _insert_default_devices()
        
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

def _insert_default_devices():
    """Insert default IoT devices"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        default_devices = [
            ("ESP32-001", "Main Sensor Hub", "hybrid", "Field-A", "active"),
            ("NANO-001", "Temperature Module", "sensor", "Field-A", "active"),
            ("ENV-001", "Weather Station", "environmental", "Field-A", "active")
        ]
        
        for device in default_devices:
            cursor.execute("""
                INSERT OR IGNORE INTO devices (device_id, name, type, location, status)
                VALUES (?, ?, ?, ?, ?)
            """, device)
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error inserting default devices: {e}")

def get_latest_sensor_data() -> Dict:
    """
    Get latest sensor readings with fallback to mock data
    
    Returns:
        Dictionary with current sensor values
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                temperature, humidity, soil_moisture, ph_level,
                nitrogen, phosphorus, potassium, light_intensity,
                timestamp
            FROM sensor_data
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "temperature": row[0],
                "humidity": row[1],
                "soil_moisture": row[2],
                "ph": row[3],
                "nitrogen": row[4],
                "phosphorus": row[5],
                "potassium": row[6],
                "light_intensity": row[7],
                "timestamp": row[8],
                "source": "database"
            }
    except Exception as e:
        logger.warning(f"Database read failed, using mock data: {e}")
    
    # Fallback to mock data
    return _generate_mock_sensor_data()

def _generate_mock_sensor_data() -> Dict:
    """Generate realistic mock sensor data"""
    base_time = datetime.now()
    
    # Generate realistic sensor values with slight variations
    return {
        "temperature": round(random.uniform(22, 32), 1),
        "humidity": round(random.uniform(50, 75), 1),
        "soil_moisture": round(random.uniform(35, 55), 1),
        "ph": round(random.uniform(6.0, 7.5), 1),
        "nitrogen": round(random.uniform(30, 60), 1),
        "phosphorus": round(random.uniform(20, 45), 1),
        "potassium": round(random.uniform(25, 50), 1),
        "light_intensity": round(random.uniform(300, 800), 0),
        "timestamp": base_time.isoformat(),
        "source": "mock"
    }

def insert_sensor_data(device_id: str, data: Dict) -> bool:
    """
    Insert sensor reading into database
    
    Args:
        device_id: Device identifier
        data: Sensor readings dictionary
    
    Returns:
        Success status
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO sensor_data (
                device_id, temperature, humidity, soil_moisture,
                ph_level, nitrogen, phosphorus, potassium, light_intensity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            device_id,
            data.get("temperature"),
            data.get("humidity"),
            data.get("soil_moisture"),
            data.get("ph_level"),
            data.get("nitrogen"),
            data.get("phosphorus"),
            data.get("potassium"),
            data.get("light_intensity")
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Sensor data inserted for device {device_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to insert sensor data: {e}")
        return False

def get_sensor_history(hours: int = 24) -> List[Dict]:
    """
    Get sensor data history
    
    Args:
        hours: Number of hours to retrieve
    
    Returns:
        List of sensor readings
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        cursor.execute("""
            SELECT 
                temperature, humidity, soil_moisture, ph_level,
                nitrogen, phosphorus, potassium, light_intensity,
                timestamp
            FROM sensor_data
            WHERE timestamp > ?
            ORDER BY timestamp ASC
        """, (cutoff_time.isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "temperature": row[0],
                "humidity": row[1],
                "soil_moisture": row[2],
                "ph": row[3],
                "nitrogen": row[4],
                "phosphorus": row[5],
                "potassium": row[6],
                "light_intensity": row[7],
                "timestamp": row[8]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error retrieving sensor history: {e}")
        return []

def get_all_devices() -> List[Dict]:
    """Get all registered devices"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT device_id, name, type, location, status, last_active
            FROM devices
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "device_id": row[0],
                "name": row[1],
                "type": row[2],
                "location": row[3],
                "status": row[4],
                "last_active": row[5]
            }
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error retrieving devices: {e}")
        return []

def update_device_status(device_id: str, status: str) -> bool:
    """Update device status"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE devices
            SET status = ?, last_active = CURRENT_TIMESTAMP
            WHERE device_id = ?
        """, (status, device_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error updating device status: {e}")
        return False

def clear_all_data() -> bool:
    """Clear all sensor data (admin function)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM sensor_data")
        conn.commit()
        conn.close()
        
        logger.info("✅ All sensor data cleared")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to clear data: {e}")
        return False

# Initialize database on module import
try:
    init_sensor_db()
except Exception as e:
    logger.warning(f"Could not initialize database: {e}")
