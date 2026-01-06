# PocketDB Integration Guide - AgriSense

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Add to your Python environment
pip install pocketbase-client

# Or install from requirements
pip install -r src/backend/requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file in your project root:

```bash
# Database Configuration
AGRISENSE_DB_BACKEND=pocketdb
POCKETDB_URL=http://localhost:8090
POCKETDB_DATA_DIR=./pb_data
POCKETDB_ADMIN_EMAIL=admin@agrisense.local
POCKETDB_ADMIN_PASSWORD=YourSecurePassword123!
```

### 3. Start PocketDB (Choose One Method)

**Option A: Docker (Recommended)**
```bash
docker run -d \
  -p 8090:8090 \
  -v pocketdb_data:/pb_data \
  --name agrisense-pocketdb \
  ghcr.io/pocketbase/pocketbase:latest
```

**Option B: Docker Compose**
```bash
docker-compose -f docker-compose.pocketdb.yml up -d
```

**Option C: Download Binary**
```bash
# Download from https://pocketbase.io/
cd pb
./pocketbase serve
# Access at http://localhost:8090
```

### 4. Initialize AgriSense with PocketDB

```bash
# Using Python script
python setup_pocketdb.py --mode init

# Or in Python code
import asyncio
from agrisense_app.backend.database import init_database

async def setup():
    db = await init_database("pocketdb")
    await db.close()

asyncio.run(setup())
```

### 5. Verify Setup

```bash
# Check health
python setup_pocketdb.py --mode health-check

# View statistics
python setup_pocketdb.py --mode stats

# Access PocketDB Admin UI
# Open browser: http://localhost:8090/_/
```

## Integration with FastAPI

### Complete Example Application

```python
# main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
from agrisense_app.backend.database import init_database

# Initialize database
db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global db
    db = await init_database("pocketdb")
    print("✓ PocketDB initialized")
    yield
    # Shutdown
    await db.close()
    print("✓ Database closed")

# Create app with lifespan
app = FastAPI(
    title="AgriSense API",
    description="Agricultural IoT Platform with PocketDB",
    lifespan=lifespan
)

# API Endpoints
@app.get("/api/health/database")
async def database_health():
    health = await db.health_check()
    stats = await db.get_stats()
    return {
        "status": "healthy" if health else "unhealthy",
        "statistics": stats
    }

@app.post("/api/sensor-readings")
async def create_reading(zone_id: str, temperature: float, humidity: float):
    reading = {
        "zone_id": zone_id,
        "temperature_c": temperature,
        "humidity": humidity,
        "timestamp": datetime.utcnow().isoformat()
    }
    return await db.insert_reading(reading)

@app.get("/api/readings")
async def get_readings(zone_id: str = None):
    readings = await db.get_readings(zone_id, limit=100)
    return {"data": readings}

# Run with:
# uvicorn main:app --reload
```

## Migration from SQLite to PocketDB

### Automated Migration

```bash
# One-command migration
python setup_pocketdb.py --mode migrate --from sqlite --to pocketdb

# With verbose output
python setup_pocketdb.py --mode migrate --from sqlite --to pocketdb -v
```

### Manual Migration in Code

```python
import asyncio
from agrisense_app.backend.database import migrate_database

async def migrate():
    result = await migrate_database(
        from_backend="sqlite",
        to_backend="pocketdb"
    )
    
    print(f"Status: {result['status']}")
    print(f"Migrated readings: {result['migration_stats']['readings']}")
    print(f"Migrated recommendations: {result['migration_stats']['recommendations']}")
    print(f"Errors: {result['migration_stats']['errors']}")

asyncio.run(migrate())
```

## PocketDB Admin UI

### Access Admin Panel

1. Open http://localhost:8090/_/ in your browser
2. Login with configured credentials
3. View and manage collections
4. Create API keys and tokens
5. Configure rules and webhooks

### Common Admin Tasks

**Create a new collection:**
1. Click "Create New"
2. Define schema (fields with types)
3. Set access rules
4. Create indexes for performance

**View records:**
1. Select collection from sidebar
2. Use filters to narrow results
3. Export data as JSON/CSV

**Backup data:**
1. Export from admin UI, or
2. Copy `/pb_data` directory

## Configuration Reference

### Environment Variables

```bash
# Backend Selection
AGRISENSE_DB_BACKEND=pocketdb              # sqlite, pocketdb, mongodb

# PocketDB Connection
POCKETDB_URL=http://localhost:8090         # Server URL
POCKETDB_DATA_DIR=/path/to/pb_data         # Data storage directory
POCKETDB_ADMIN_EMAIL=admin@example.com     # Admin user email
POCKETDB_ADMIN_PASSWORD=SecurePassword     # Admin password

# Database Behavior
AGRISENSE_DB_AUTO_INIT=1                   # Auto-create tables/collections
AGRISENSE_DB_MIGRATIONS=1                  # Enable migration tools
```

### Python Configuration

```python
from agrisense_app.backend.database import (
    DatabaseConfig,
    DatabaseBackend,
    DatabaseManager
)

# Create custom config
config = DatabaseConfig(
    backend=DatabaseBackend.POCKETDB,
    pocketdb_url="http://prod-pocketdb:8090",
    pocketdb_data_dir="/mnt/storage/pb_data",
    auto_init=True
)

# Use config
db = DatabaseManager(config)
await db.init()
```

## Docker Deployment

### Development with Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.pocketdb.yml up -d

# View logs
docker-compose -f docker-compose.pocketdb.yml logs -f backend

# Access services
# - API: http://localhost:8000
# - Frontend: http://localhost:5173
# - PocketDB Admin: http://localhost:8090/_/

# Stop all
docker-compose -f docker-compose.pocketdb.yml down
```

### Production Deployment

**Using Azure Container Apps:**

```bash
# Build image
docker build -t agrisense:pocketdb .

# Push to registry
docker tag agrisense:pocketdb myregistry.azurecr.io/agrisense:pocketdb
docker push myregistry.azurecr.io/agrisense:pocketdb

# Deploy with environment variables
az containerapp create \
  --name agrisense \
  --environment agrisense-env \
  --image myregistry.azurecr.io/agrisense:pocketdb \
  --environment-variables \
    AGRISENSE_DB_BACKEND=pocketdb \
    POCKETDB_URL=https://pocketdb.agrisense.cloud \
  --memory 2.0Gi \
  --cpu 1.0
```

## Performance Optimization

### 1. Database Indexing

PocketDB supports index creation for faster queries:

```python
# Indexes on frequently filtered columns
# - zone_id (for zone-based queries)
# - timestamp (for time-range queries)
# - device_id (for device-specific queries)

# Created automatically or via admin UI
```

### 2. Batch Operations

```python
# Insert multiple readings efficiently
readings = [
    {"zone_id": "field_1", "temperature_c": 25.5, ...},
    {"zone_id": "field_1", "temperature_c": 25.6, ...},
    # ... more readings
]

for reading in readings:
    await db.insert_reading(reading)
```

### 3. Data Cleanup

```bash
# Keep only 90 days of data
python setup_pocketdb.py --mode cleanup --days-to-keep 90
```

### 4. Connection Management

```python
# Reuse database connection
db = get_database_manager()
# Use db for multiple operations
# Don't repeatedly open/close connections
```

## Backup and Restore

### Backup Data

**Docker volumes:**
```bash
# Backup PocketDB volume
docker run --rm -v pocketdb_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/pocketdb-backup.tar.gz -C /data .
```

**Direct filesystem:**
```bash
# Backup data directory
tar czf pocketdb-backup.tar.gz /path/to/pb_data/
```

### Restore Data

```bash
# From backup
tar xzf pocketdb-backup.tar.gz -C /path/to/pb_data/

# Restart PocketDB
docker restart agrisense-pocketdb
```

## Monitoring and Logging

### Health Checks

```python
db = get_database_manager()

# Check connectivity
health = await db.health_check()

# Get statistics
stats = await db.get_stats()
print(f"Sensor readings: {stats['collections']['sensor_readings']['record_count']}")
```

### Logging

```python
import logging

# Enable database logging
logging.getLogger("agrisense.database").setLevel(logging.DEBUG)

# Monitor operations
logger.debug("Database operation: insert_reading")
logger.info("Database stats: sensor_readings=1000, recommendations=500")
```

## Troubleshooting

### Connection Issues

```bash
# Test connectivity
curl http://localhost:8090/api/health

# Check logs
docker logs agrisense-pocketdb

# Verify URL and port
echo $POCKETDB_URL
```

### Data Migration Issues

```bash
# Retry migration with verbose output
python setup_pocketdb.py --mode migrate -v

# Check migration errors
# Review logs for specific failed records
```

### Performance Issues

```bash
# Check database statistics
python setup_pocketdb.py --mode stats

# Clean up old data
python setup_pocketdb.py --mode cleanup --days-to-keep 60

# Monitor collection sizes
# Use PocketDB Admin UI → Collections → Size info
```

### Authentication Issues

```bash
# Reset admin password (in PocketDB)
# Stop PocketDB and delete pb_data/
# Restart - will prompt for new admin credentials

# Or update via admin UI:
# Settings → User Management → Edit Admin
```

## API Examples

### Insert Sensor Reading

```python
reading = {
    "zone_id": "field_1",
    "device_id": "sensor_001",
    "plant": "rice",
    "temperature_c": 25.5,
    "humidity": 60.0,
    "soil_moisture_pct": 45.0,
    "timestamp": "2024-01-04T10:00:00Z"
}
result = await db.insert_reading(reading)
```

### Query Readings

```python
# Get all readings for a zone
readings = await db.get_readings("field_1", limit=100)

# Process results
for reading in readings:
    print(f"Temp: {reading['temperature_c']}°C")
```

### Create Recommendations

```python
rec = {
    "zone_id": "field_1",
    "plant": "rice",
    "water_liters": 50.0,
    "fert_n_g": 100.0,
    "timestamp": datetime.utcnow().isoformat()
}
result = await db.insert_recommendation(rec)
```

## Security Considerations

### In Development

```bash
# Default credentials
POCKETDB_ADMIN_EMAIL=admin@agrisense.local
POCKETDB_ADMIN_PASSWORD=AgriSense@2024!
```

### In Production

```bash
# Use strong passwords
POCKETDB_ADMIN_PASSWORD=$(openssl rand -base64 32)

# Use HTTPS/TLS
POCKETDB_URL=https://pocketdb.agrisense.production

# Enable API key authentication
# Configure in PocketDB admin UI

# Set collection-level access rules
# Restrict who can read/write
```

## Next Steps

1. ✅ Install PocketDB
2. ✅ Configure AgriSense
3. ✅ Migrate existing data
4. ✅ Test API endpoints
5. ✅ Set up monitoring
6. ✅ Deploy to production

## Support & Resources

- **PocketBase Docs**: https://pocketbase.io/docs/
- **AgriSense Documentation**: See `/documentation/` folder
- **GitHub Issues**: Report problems at project repository

## FAQ

**Q: Should I use PocketDB or SQLite?**
A: PocketDB for production/edge, SQLite for development/testing

**Q: Can I switch backends later?**
A: Yes, use migration tools to move data between backends

**Q: What about MongoDB?**
A: Also supported for large-scale deployments with `AGRISENSE_DB_BACKEND=mongodb`

**Q: How do I backup data?**
A: Copy `/pb_data` directory or export from admin UI

**Q: Is PocketDB suitable for production?**
A: Yes, it's based on SQLite3 - battle-tested, reliable, and lightweight

---

**Last Updated**: January 4, 2026
**AgriSense Version**: 2024.01
