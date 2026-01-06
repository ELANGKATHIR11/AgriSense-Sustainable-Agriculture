# AgriSense Database Module

Multi-backend database support for AgriSense with seamless switching between SQLite, PocketDB, and MongoDB.

## Features

✅ **Multi-Backend Support**
- SQLite (default, development)
- PocketDB (lightweight, embedded)
- MongoDB (production-scale)

✅ **Easy Backend Switching**
- Simple environment variable configuration
- Runtime backend selection
- Unified API across all backends

✅ **Data Migration**
- Migrate data between backends
- Batch processing support
- Migration validation

✅ **Async/Await Support**
- Built for FastAPI async operations
- Non-blocking database operations
- Proper connection pooling

## Quick Start

### 1. Installation

```bash
# Install PocketDB support
pip install pocketbase-client

# Or install all database backends
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Choose your database backend
export AGRISENSE_DB_BACKEND=pocketdb

# PocketDB specific
export POCKETDB_URL=http://localhost:8090
export POCKETDB_DATA_DIR=/var/lib/agrisense/pb_data
export POCKETDB_ADMIN_EMAIL=admin@agrisense.local
export POCKETDB_ADMIN_PASSWORD=YourSecurePassword
```

### 3. Basic Usage

```python
from agrisense_app.backend.database import init_database

# Initialize database
db = await init_database("pocketdb")

# Insert sensor data
reading = {
    "zone_id": "field_1",
    "temperature_c": 25.5,
    "humidity": 60.0,
    "timestamp": "2024-01-04T10:00:00Z"
}
await db.insert_reading(reading)

# Get data
readings = await db.get_readings(zone_id="field_1", limit=100)

# Check health
is_healthy = await db.health_check()

# Get statistics
stats = await db.get_stats()

# Cleanup
await db.close()
```

## Architecture

### Module Structure

```
database/
├── __init__.py           # Package exports
├── config.py            # Configuration management
├── manager.py           # Unified database manager
├── pocketdb_adapter.py  # PocketDB implementation
├── migration.py         # Data migration utilities
└── POCKETDB_GUIDE.py    # Examples and documentation
```

### Supported Collections

All backends support these collections:

- **sensor_readings** - Raw sensor data (temperature, humidity, etc.)
- **recommendations** - AI/ML recommendations
- **recommendation_tips** - Actionable tips
- **tank_levels** - Water tank monitoring
- **rainwater_harvest** - Rainwater collection tracking
- **valve_events** - Irrigation control logs
- **alerts** - System alerts and notifications

## Database Backends

### SQLite (Default)

**Best for:** Development, testing, offline scenarios

```python
db = await init_database("sqlite")
```

**Configuration:**
```bash
AGRISENSE_DB_BACKEND=sqlite
AGRISENSE_DB_PATH=/path/to/sensors.db
```

**Pros:**
- No external dependencies
- Perfect for development
- Easy to backup/transport
- Included in Python stdlib

**Cons:**
- Limited concurrency
- Not suitable for production scale
- Single machine only

### PocketDB (Recommended for Edge)

**Best for:** Edge devices, IoT, offline-first applications

```python
db = await init_database("pocketdb")
```

**Configuration:**
```bash
AGRISENSE_DB_BACKEND=pocketdb
POCKETDB_URL=http://localhost:8090
POCKETDB_DATA_DIR=/var/lib/agrisense/pb_data
```

**Features:**
- Built on top of SQLite3 - same reliability
- Real-time API with built-in auth
- REST API out of the box
- Perfect for IoT/edge deployments
- Minimal dependencies

**Installation:**
```bash
# Download PocketBase binary from https://pocketbase.io/
# Or use Docker:
docker run -d -p 8090:8090 -v pocketdb_data:/pb_data ghcr.io/pocketbase/pocketbase:latest
```

### MongoDB (Production Scale)

**Best for:** Large-scale production deployments

```python
db = await init_database("mongodb")
```

**Configuration:**
```bash
AGRISENSE_DB_BACKEND=mongodb
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/agrisense
```

**Pros:**
- Horizontal scaling
- Complex queries
- Flexible schema
- Cloud-hosted options

**Cons:**
- External dependency
- Higher operational complexity
- Cloud costs

## API Reference

### DatabaseManager

Main interface for database operations.

```python
from agrisense_app.backend.database import get_database_manager

db = get_database_manager()
await db.init()

# Insert operations
await db.insert_reading(data)
await db.insert_recommendation(data)
await db.insert_alert(data)

# Query operations
readings = await db.get_readings(zone_id="field_1", limit=100)
recommendations = await db.get_recommendations()
alerts = await db.get_alerts()

# Management
stats = await db.get_stats()
is_healthy = await db.health_check()
await db.close()
```

### PocketDBAdapter

Direct PocketDB interface (if using PocketDB backend).

```python
from agrisense_app.backend.database import init_pocketdb, PocketDBConfig

config = PocketDBConfig(
    base_url="http://localhost:8090",
    auto_init=True
)

adapter = await init_pocketdb(config)

# Insert
reading = await adapter.insert_sensor_reading(data)

# Query
readings = await adapter.get_sensor_readings(zone_id="field_1")

# Management
stats = await adapter.get_stats()
await adapter.disconnect()
```

### MigrationManager

Migrate data between backends.

```python
from agrisense_app.backend.database import migrate_database

result = await migrate_database(
    from_backend="sqlite",
    to_backend="pocketdb"
)

print(result["migration_stats"])  # Records migrated
print(result["validation"])        # Validation results
```

## FastAPI Integration

### Complete Example

```python
from fastapi import FastAPI, HTTPException
from datetime import datetime
from agrisense_app.backend.database import init_database

app = FastAPI(title="AgriSense API")

# Startup: Initialize database
@app.on_event("startup")
async def startup_event():
    app.state.db = await init_database("pocketdb")

# Shutdown: Cleanup
@app.on_event("shutdown")
async def shutdown_event():
    await app.state.db.close()

# Sensor readings endpoint
@app.post("/api/v1/sensor-readings")
async def create_sensor_reading(
    zone_id: str,
    device_id: str,
    temperature_c: float,
    humidity: float,
    soil_moisture_pct: float
):
    try:
        reading = {
            "zone_id": zone_id,
            "device_id": device_id,
            "temperature_c": temperature_c,
            "humidity": humidity,
            "soil_moisture_pct": soil_moisture_pct,
            "timestamp": datetime.utcnow().isoformat()
        }
        result = await app.state.db.insert_reading(reading)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get readings endpoint
@app.get("/api/v1/readings")
async def get_readings(zone_id: str = None, limit: int = 100):
    readings = await app.state.db.get_readings(zone_id, limit)
    return {"data": readings, "count": len(readings)}

# Health check
@app.get("/api/v1/health/database")
async def database_health():
    is_healthy = await app.state.db.health_check()
    stats = await app.state.db.get_stats()
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "backend": app.state.db.backend.value,
        "statistics": stats
    }
```

## Docker Deployment

### Docker Compose

```yaml
version: "3.9"

services:
  # PocketDB instance
  pocketdb:
    image: ghcr.io/pocketbase/pocketbase:latest
    ports:
      - "8090:8090"
    volumes:
      - pocketdb_data:/pb_data
    networks:
      - agrisense

  # AgriSense Backend
  agrisense:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      AGRISENSE_DB_BACKEND: pocketdb
      POCKETDB_URL: http://pocketdb:8090
      POCKETDB_DATA_DIR: /app/pb_data
    depends_on:
      - pocketdb
    volumes:
      - agrisense_data:/app/pb_data
    networks:
      - agrisense

volumes:
  pocketdb_data:
  agrisense_data:

networks:
  agrisense:
    driver: bridge
```

### Run with Docker

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f agrisense

# Stop services
docker-compose down
```

## Data Migration

### Migrate from SQLite to PocketDB

```python
import asyncio
from agrisense_app.backend.database import migrate_database

async def main():
    result = await migrate_database(
        from_backend="sqlite",
        to_backend="pocketdb"
    )
    
    if result["status"] == "success":
        print(f"Migrated {result['migration_stats']['readings']} readings")
        print(f"Migrated {result['migration_stats']['recommendations']} recommendations")
    else:
        print(f"Migration failed: {result['error']}")

asyncio.run(main())
```

### Migration Script

```bash
# Via CLI
python -m agrisense_app.backend.database.migration sqlite pocketdb

# Verify migration
python -c "
import asyncio
from agrisense_app.backend.database import migrate_database
result = asyncio.run(migrate_database('sqlite', 'pocketdb'))
print(f'Status: {result[\"status\"]}')
"
```

## Configuration

### Environment Variables

```bash
# Backend selection
AGRISENSE_DB_BACKEND=pocketdb              # sqlite, pocketdb, mongodb

# PocketDB
POCKETDB_URL=http://localhost:8090
POCKETDB_DATA_DIR=/var/lib/agrisense
POCKETDB_ADMIN_EMAIL=admin@agrisense.local
POCKETDB_ADMIN_PASSWORD=SecurePassword

# SQLite
AGRISENSE_DB_PATH=/path/to/sensors.db

# MongoDB
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/agrisense

# Database behavior
AGRISENSE_DB_AUTO_INIT=1                   # Auto-create tables
AGRISENSE_DB_MIGRATIONS=1                  # Enable migrations
```

## Troubleshooting

### PocketDB Connection Issues

```python
# Test connection
from agrisense_app.backend.database import init_database

try:
    db = await init_database("pocketdb")
    health = await db.health_check()
    print(f"PocketDB Health: {health}")
except Exception as e:
    print(f"Connection error: {e}")
```

### Check Database Statistics

```python
stats = await db.get_stats()
for collection, info in stats["collections"].items():
    print(f"{collection}: {info['record_count']} records")
```

### Migration Debugging

```python
from agrisense_app.backend.database import migrate_database

result = await migrate_database("sqlite", "pocketdb")
print(f"Errors: {result['migration_stats']['errors']}")
```

## Performance Tips

### 1. Batch Operations

```python
# Insert multiple readings at once
readings = [
    {"zone_id": "field_1", "temperature_c": 25.5, ...},
    {"zone_id": "field_1", "temperature_c": 25.6, ...},
    # ... more readings
]

for reading in readings:
    await db.insert_reading(reading)
```

### 2. Indexed Queries

```python
# Efficient queries with zone_id filter
readings = await db.get_readings(zone_id="field_1", limit=100)
```

### 3. Cleanup Old Data

```python
from agrisense_app.backend.database import get_database_manager

db = get_database_manager()
await db.init()

# Keep only 90 days of data
await db._adapter.clear_old_records("sensor_readings", days_to_keep=90)
```

## Security Considerations

### Database Access

```python
# Use authentication for PocketDB in production
config = PocketDBConfig(
    base_url="https://pocketdb.agrisense.cloud",  # Use HTTPS
    admin_email="admin@agrisense.local",
    admin_password=os.getenv("POCKETDB_ADMIN_PASSWORD")
)
```

### Data Encryption

- PocketDB: Uses SQLite3 with SQLCipher extension
- MongoDB: Enable at-rest encryption
- All: Use TLS for network communication

## Monitoring and Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agrisense.database")

# Monitor database operations
db = get_database_manager()
await db.init()

# Get stats
stats = await db.get_stats()
logger.info(f"Database stats: {stats}")
```

## Contributing

To add support for a new database backend:

1. Create a new adapter file: `database/yourdb_adapter.py`
2. Implement the adapter interface
3. Update `config.py` with new `DatabaseBackend`
4. Update `manager.py` with initialization logic
5. Add tests in `tests/database/`

## References

- [PocketBase Documentation](https://pocketbase.io/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [FastAPI Databases](https://fastapi.tiangolo.com/advanced/databases/)
- [SQLite Documentation](https://www.sqlite.org/)

## License

Part of AgriSense project. See LICENSE file for details.
