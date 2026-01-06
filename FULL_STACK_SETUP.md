# AgriSense Full-Stack Setup with PocketDB

Complete guide to set up and run AgriSense with PocketDB backend integrated with React frontend.

## Quick Start (5 Minutes)

### Step 1: Install Dependencies

```bash
cd AGRISENSEFULL-STACK

# Install Python dependencies
pip install -r src/backend/requirements.txt

# Install frontend dependencies  
cd src/frontend
npm install
cd ../..
```

### Step 2: Set Up Environment

Copy the PocketDB environment file to your project:

```bash
cp .env.pocketdb AGRISENSEFULL-STACK/
```

This file contains all configuration for PocketDB, backend, and frontend.

### Step 3: Start PocketDB

**Option A: Docker (Recommended)**

```bash
# Start PocketDB in Docker
docker run -d \
  -p 8090:8090 \
  -v pocketdb_data:/pb_data \
  --name agrisense-pocketdb \
  ghcr.io/pocketbase/pocketbase:latest

# Verify it's running
curl http://localhost:8090/api/health
```

**Option B: Download Binary**

1. Download from https://pocketbase.io/
2. Run: `./pocketbase serve`
3. Access admin UI: http://localhost:8090/_/

### Step 4: Start Backend

```bash
# From project root directory
cd AGRISENSEFULL-STACK

# Option A: PowerShell
.\start_pocketdb.ps1

# Option B: Python
python startup_with_pocketdb.py

# Option C: Direct uvicorn
cd src/backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Backend will be available at: **http://localhost:8000**
API docs at: **http://localhost:8000/docs**

### Step 5: Start Frontend

In a new terminal:

```bash
cd AGRISENSEFULL-STACK/src/frontend
npm run dev
```

Frontend will be available at: **http://localhost:5173** (or next available port)

### Step 6: Verify Everything Works

Open in browser:
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs
- **PocketDB Admin**: http://localhost:8090/_/
- **Database Health**: http://localhost:8000/health/database

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AGRISENSE STACK                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           React Frontend (Vite)                      │   │
│  │         http://localhost:5173                        │   │
│  └────────────────┬─────────────────────────────────────┘   │
│                   │ API Calls & WebSocket                    │
│                   ▼                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        FastAPI Backend (Python)                      │   │
│  │         http://localhost:8000                        │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Database Module (Multi-backend support)        │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └────────┬────────────────────────────┬────────────────┘   │
│           │                            │                     │
│           ▼                            ▼                     │
│  ┌────────────────────┐      ┌────────────────────┐         │
│  │   PocketDB API     │      │  Legacy SQLite DB  │         │
│  │  (Recommended)     │      │  (Optional)        │         │
│  │ http:8090          │      │                    │         │
│  └────────────────────┘      └────────────────────┘         │
│           │                                                  │
│           ▼                                                  │
│  ┌────────────────────┐                                     │
│  │  SQLite3 Storage   │                                     │
│  │  (Embedded DB)     │                                     │
│  │  ./pb_data/        │                                     │
│  └────────────────────┘                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
AGRISENSEFULL-STACK/
├── .env.pocketdb                    # PocketDB configuration
├── start_pocketdb.ps1               # PowerShell startup script
├── startup_with_pocketdb.py         # Python startup script
├── docker-compose.pocketdb.yml      # Docker Compose configuration
├── setup_pocketdb.py                # PocketDB setup tool
│
├── src/
│   ├── backend/
│   │   ├── main.py                  # FastAPI application (updated)
│   │   ├── requirements.txt          # Python dependencies (updated)
│   │   │
│   │   └── database/                 # NEW: Database module
│   │       ├── __init__.py
│   │       ├── config.py            # Database configuration
│   │       ├── manager.py           # Database manager
│   │       ├── pocketdb_adapter.py  # PocketDB integration
│   │       ├── migration.py         # Data migration tools
│   │       ├── example_routes.py    # Example API routes
│   │       ├── README.md            # Full documentation
│   │       └── POCKETDB_GUIDE.py    # Code examples
│   │
│   └── frontend/
│       ├── .env.development         # Frontend config (updated)
│       ├── vite.config.ts
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── lib/
│       └── package.json
│
└── documentation/
    └── api/                         # API documentation
```

## Database Integration

### How It Works

1. **FastAPI Lifespan Management**
   - On startup: Initializes PocketDB connection
   - During runtime: Provides database access via `app.state.db`
   - On shutdown: Cleanly closes database connection

2. **Database Manager**
   - Unified interface for all database operations
   - Supports SQLite, PocketDB, and MongoDB
   - Easy backend switching via environment variable

3. **Frontend Communication**
   - API calls to FastAPI endpoints
   - Endpoints use database manager to persist data
   - Real-time updates via WebSocket

### Using the Database in Endpoints

```python
from fastapi import FastAPI, Depends

@app.post("/api/sensor-readings")
async def create_reading(data: dict):
    # Access database from app state
    result = await app.state.db.insert_reading(data)
    return result

@app.get("/api/readings")
async def get_readings(zone_id: str):
    readings = await app.state.db.get_readings(zone_id)
    return {"data": readings}
```

## API Endpoints

### Health & Status

```
GET /health                     # Basic health check
GET /health/database           # Database health status
GET /health/enhanced           # Full system health
```

### Sensor Readings (via database module)

See [example_routes.py](src/backend/database/example_routes.py) for full examples:

```
POST /api/v1/sensor-readings        # Create sensor reading
GET  /api/v1/sensor-readings        # Get readings
GET  /api/v1/sensor-readings/{zone} # Get readings for zone
```

### Recommendations

```
POST /api/v1/recommendations        # Create recommendation
GET  /api/v1/recommendations        # Get recommendations
```

### Alerts

```
POST /api/v1/alerts                 # Create alert
GET  /api/v1/alerts                 # Get alerts
```

## Frontend Integration

### Environment Configuration

Frontend loads environment from `.env.development`:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws
VITE_POCKETDB_URL=http://localhost:8090
```

### API Client Setup

The frontend automatically connects to the backend through:

1. **API Base URL**: `http://localhost:8000`
2. **WebSocket**: `ws://localhost:8000/ws`
3. **Proxy in Vite**: Configured in `vite.config.ts`

### Making API Calls

```typescript
// React component example
import { useEffect, useState } from 'react';

export function SensorDashboard() {
  const [readings, setReadings] = useState([]);

  useEffect(() => {
    // Fetch from backend API
    fetch('/api/v1/sensor-readings?zone_id=field_1')
      .then(res => res.json())
      .then(data => setReadings(data.data));
  }, []);

  return (
    <div>
      {readings.map(r => (
        <div key={r.id}>
          Temp: {r.temperature_c}°C
        </div>
      ))}
    </div>
  );
}
```

## Configuration

### Environment Variables

Create `.env.pocketdb` or `.env.local` in project root:

```bash
# Database
AGRISENSE_DB_BACKEND=pocketdb
POCKETDB_URL=http://localhost:8090
POCKETDB_DATA_DIR=./pb_data
POCKETDB_ADMIN_EMAIL=admin@agrisense.local
POCKETDB_ADMIN_PASSWORD=YourPassword123!

# Backend
FASTAPI_ENV=development
LOG_LEVEL=INFO
WORKERS=1

# Frontend
VITE_API_BASE_URL=http://localhost:8000
VITE_ENVIRONMENT=development
```

### Database Selection

Switch database backends by changing one variable:

```bash
# PocketDB (Recommended)
AGRISENSE_DB_BACKEND=pocketdb

# SQLite (Legacy)
AGRISENSE_DB_BACKEND=sqlite

# MongoDB (Production)
AGRISENSE_DB_BACKEND=mongodb
```

## Data Migration

If you have existing SQLite data, migrate to PocketDB:

```bash
python setup_pocketdb.py --mode migrate --from sqlite --to pocketdb
```

Or in Python code:

```python
from agrisense_app.backend.database import migrate_database

result = await migrate_database("sqlite", "pocketdb")
print(f"Migrated {result['migration_stats']['readings']} readings")
```

## Troubleshooting

### Backend Won't Start

**Check Python environment:**
```bash
# Verify virtual environment
python --version  # Should be 3.12+

# Check dependencies
pip list | grep -i fastapi
```

**Check port conflicts:**
```powershell
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

### PocketDB Connection Issues

**Check PocketDB is running:**
```bash
# Test connection
curl http://localhost:8090/api/health

# Check Docker container
docker ps | grep pocketbase

# View logs
docker logs agrisense-pocketdb
```

**Check environment variables:**
```bash
# View current settings
echo $env:POCKETDB_URL
echo $env:POCKETDB_DATA_DIR
```

### Frontend API Connection Issues

**Check browser console:**
- Open DevTools (F12)
- Network tab should show API requests
- Check for CORS errors

**Verify API is accessible:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

**Check Vite proxy configuration:**
- Vite config in `src/frontend/vite.config.ts` has proxy to `/api`
- Should forward requests to `http://localhost:8000`

## Development Workflow

### 1. Make Backend Changes

```bash
cd AGRISENSEFULL-STACK/src/backend
# Edit files - uvicorn --reload will auto-restart
```

### 2. Make Frontend Changes

```bash
cd AGRISENSEFULL-STACK/src/frontend
# Edit files - Vite dev server has hot reload
```

### 3. View Database

**PocketDB Admin UI:**
- Open http://localhost:8090/_/
- Browse collections
- View records
- Create API keys

**Programmatically:**
```python
db = get_database_manager()
await db.init()
stats = await db.get_stats()
print(stats)
```

### 4. Check API Documentation

Open http://localhost:8000/docs in browser
- Interactive API explorer
- Try out endpoints
- See request/response schemas

## Production Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose -f docker-compose.pocketdb.yml up -d

# View logs
docker-compose -f docker-compose.pocketdb.yml logs -f

# Stop services
docker-compose -f docker-compose.pocketdb.yml down
```

### Using Azure

See [AZURE_DEPLOYMENT_QUICKSTART.md](../AZURE_DEPLOYMENT_QUICKSTART.md)

## Performance Optimization

### 1. Database Indexing
PocketDB automatically indexes frequently accessed fields

### 2. API Response Caching
```python
from fastapi_cache2 import cache

@app.get("/api/readings")
@cache(expire=300)
async def get_readings():
    ...
```

### 3. Frontend Code Splitting
Already configured in `vite.config.ts`

### 4. Data Cleanup
Keep only 90 days of data:
```bash
python setup_pocketdb.py --mode cleanup --days-to-keep 90
```

## Monitoring

### Database Health

```bash
curl http://localhost:8000/health/database
```

Response:
```json
{
  "status": "healthy",
  "backend": "pocketdb",
  "connected": true,
  "collections": {
    "sensor_readings": { "record_count": 1000 },
    "recommendations": { "record_count": 250 }
  }
}
```

### System Metrics

```bash
curl http://localhost:8000/health/enhanced
```

### Logs

```bash
# Backend logs (in terminal where it's running)
# Look for INFO/ERROR/WARNING level messages

# Docker logs
docker logs agrisense-pocketdb
docker logs agrisense-backend
```

## Resources

- **PocketBase Docs**: https://pocketbase.io/docs/
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **React Docs**: https://react.dev/
- **AgriSense Database Module**: [README.md](src/backend/database/README.md)
- **Integration Guide**: [POCKETDB_INTEGRATION.md](../POCKETDB_INTEGRATION.md)

## Support

For issues or questions:

1. Check the relevant README in `src/backend/database/`
2. Review [POCKETDB_INTEGRATION.md](../POCKETDB_INTEGRATION.md)
3. Check browser console and network tab
4. Review backend logs
5. Check PocketDB admin UI for data issues

---

**Last Updated**: January 4, 2026  
**AgriSense Version**: 2024.01  
**Database**: PocketDB (SQLite3-based)
