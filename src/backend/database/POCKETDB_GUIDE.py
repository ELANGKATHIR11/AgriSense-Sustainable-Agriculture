"""
PocketDB Integration Guide for AgriSense

Quick Start:
    1. Install dependencies:
       pip install pocketbase-client

    2. Set environment variables:
       export AGRISENSE_DB_BACKEND=pocketdb
       export POCKETDB_URL=http://localhost:8090
       export POCKETDB_DATA_DIR=/var/lib/agrisense/pb_data

    3. Use in FastAPI application:
       from agrisense_app.backend.database import init_database
       
       @app.on_event("startup")
       async def startup():
           global db
           db = await init_database("pocketdb")
       
       @app.on_event("shutdown")
       async def shutdown():
           await db.close()
       
       @app.get("/readings")
       async def get_readings(zone_id: str):
           return await db.get_readings(zone_id)

    4. Migrate from SQLite:
       from agrisense_app.backend.database import migrate_database
       
       result = await migrate_database("sqlite", "pocketdb")
"""

# Example 1: Basic Usage
EXAMPLE_BASIC_USAGE = """
import asyncio
from agrisense_app.backend.database import init_database

async def main():
    # Initialize PocketDB
    db = await init_database("pocketdb")
    
    # Insert sensor reading
    reading = {
        "ts": "2024-01-04T10:00:00Z",
        "zone_id": "field_1",
        "device_id": "sensor_001",
        "plant": "rice",
        "temperature_c": 25.5,
        "humidity": 60.0,
        "soil_moisture_pct": 45.0
    }
    await db.insert_reading(reading)
    
    # Get readings
    readings = await db.get_readings(zone_id="field_1")
    print(f"Found {len(readings)} readings")
    
    # Get database stats
    stats = await db.get_stats()
    print(f"Database stats: {stats}")
    
    # Cleanup
    await db.close()

asyncio.run(main())
"""

# Example 2: FastAPI Integration
EXAMPLE_FASTAPI_INTEGRATION = """
from fastapi import FastAPI, HTTPException
from agrisense_app.backend.database import init_database, get_database_manager

app = FastAPI(title="AgriSense with PocketDB")

# Initialize database at startup
@app.on_event("startup")
async def startup_event():
    db = await init_database("pocketdb")
    app.state.db = db

# Cleanup at shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db.close()

# API endpoints
@app.post("/api/sensor-readings")
async def create_sensor_reading(zone_id: str, temperature: float, humidity: float):
    try:
        reading = {
            "zone_id": zone_id,
            "temperature_c": temperature,
            "humidity": humidity,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = await app.state.db.insert_reading(reading)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/readings")
async def get_readings(zone_id: str = None, limit: int = 100):
    readings = await app.state.db.get_readings(zone_id, limit)
    return {"data": readings}

@app.get("/api/health/db")
async def db_health():
    is_healthy = await app.state.db.health_check()
    stats = await app.state.db.get_stats()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "stats": stats
    }
"""

# Example 3: Migration from SQLite
EXAMPLE_MIGRATION = """
import asyncio
from agrisense_app.backend.database import migrate_database, DatabaseConfig, DatabaseBackend

async def main():
    # Migrate from SQLite to PocketDB
    print("Starting migration: SQLite → PocketDB")
    
    result = await migrate_database(
        from_backend="sqlite",
        to_backend="pocketdb"
    )
    
    print(f"Migration result: {result}")
    
    if result["status"] == "success":
        print(f"Successfully migrated {result['migration_stats']} records")
        print(f"Validation: {result['validation']}")

asyncio.run(main())
"""

# Example 4: Environment Variables Setup
EXAMPLE_ENV_SETUP = """
# .env file

# Database Backend Selection
AGRISENSE_DB_BACKEND=pocketdb          # Options: sqlite, pocketdb, mongodb

# PocketDB Configuration
POCKETDB_URL=http://localhost:8090     # PocketDB server URL
POCKETDB_DATA_DIR=/var/lib/agrisense/pb_data  # Data directory
POCKETDB_ADMIN_EMAIL=admin@agrisense.local
POCKETDB_ADMIN_PASSWORD=SecurePassword123!

# Database Initialization
AGRISENSE_DB_AUTO_INIT=1               # Auto-create tables/collections
AGRISENSE_DB_MIGRATIONS=1              # Enable migration support

# Optional: MongoDB for production
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/agrisense
"""

# Example 5: Docker Compose Setup
EXAMPLE_DOCKER_COMPOSE = '''
version: "3.9"

services:
  # PocketDB instance
  pocketdb:
    image: ghcr.io/pocketbase/pocketbase:latest
    ports:
      - "8090:8090"
    volumes:
      - pocketdb_data:/pb_data
    environment:
      POCKETBASE_PB_ALLOW_CORS: "true"
    networks:
      - agrisense

  # AgriSense Backend
  backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      AGRISENSE_DB_BACKEND: pocketdb
      POCKETDB_URL: http://pocketdb:8090
      POCKETDB_DATA_DIR: /app/data
    depends_on:
      - pocketdb
    volumes:
      - ./src:/app/src
      - agrisense_data:/app/data
    networks:
      - agrisense

volumes:
  pocketdb_data:
  agrisense_data:

networks:
  agrisense:
    driver: bridge
'''

# Example 6: Configuration Management
EXAMPLE_CONFIG_MANAGEMENT = """
from agrisense_app.backend.database import DatabaseConfig, DatabaseBackend, get_database_config

# Load from environment
config = get_database_config()
print(f"Backend: {config.backend.value}")
print(f"PocketDB URL: {config.pocketdb_url}")

# Or create custom config
custom_config = DatabaseConfig(
    backend=DatabaseBackend.POCKETDB,
    pocketdb_url="http://prod-pocketdb:8090",
    pocketdb_data_dir="/mnt/storage/agrisense",
    auto_init=True
)

# Use in database manager
from agrisense_app.backend.database import get_database_manager
db = get_database_manager(custom_config)
await db.init()
"""

print(__doc__)
