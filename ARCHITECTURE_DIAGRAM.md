# AgriSense Full-Stack Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AGRISENSE ECOSYSTEM                          │
│                   (Fully Integrated & Ready)                        │
└─────────────────────────────────────────────────────────────────────┘

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    FRONTEND LAYER (Browser)                        ┃
┃                                                                    ┃
┃              React + Vite (TypeScript + Tailwind)                 ┃
┃                  http://localhost:5173                            ┃
┃                                                                    ┃
┃   ┌──────────────────────────────────────────────────────────┐   ┃
┃   │  Components:                                              │   ┃
┃   │  • Sensor Dashboard (live data)                          │   ┃
┃   │  • Recommendations Engine (AI/ML suggestions)            │   ┃
┃   │  • Alerts Panel (notifications)                          │   ┃
┃   │  • Water Management (tank levels, irrigation)            │   ┃
┃   │  • Settings/Admin Panel                                  │   ┃
┃   └──────────────────────────────────────────────────────────┘   ┃
┃                                                                    ┃
┃   Environment:                                                     ┃
┃   • VITE_API_BASE_URL=http://localhost:8000                      ┃
┃   • VITE_WS_URL=ws://localhost:8000                              ┃
┃   • VITE_POCKETDB_URL=http://localhost:8090                      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              │ HTTP/WebSocket
                              │ Fetch, Axios
                              │ Real-time updates
                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    BACKEND LAYER (Server)                          ┃
┃                                                                    ┃
┃              FastAPI + Python 3.12 + Async/Await                 ┃
┃                  http://localhost:8000                            ┃
┃                                                                    ┃
┃   ┌──────────────────────────────────────────────────────────┐   ┃
┃   │  Endpoints:                                               │   ┃
┃   │  • GET  /health                    - Basic health         │   ┃
┃   │  • GET  /health/database           - Database health     │   ┃
┃   │  • POST /api/v1/sensor-readings    - Create reading      │   ┃
┃   │  • GET  /api/v1/sensor-readings    - Get readings        │   ┃
┃   │  • GET  /api/v1/recommendations    - Get AI suggestions  │   ┃
┃   │  • POST /api/v1/alerts             - Create alert        │   ┃
┃   │  • GET  /api/v1/alerts             - Get alerts          │   ┃
┃   │  • ... and more (see example_routes.py)                  │   ┃
┃   └──────────────────────────────────────────────────────────┘   ┃
┃                                                                    ┃
┃   Core Modules:                                                    ┃
┃   • Recommendation Engine (AI/ML)                                 ┃
┃   • Chatbot & NLP Services                                        ┃
┃   • Weed Detection & Management                                   ┃
┃   • Disease Detection (Computer Vision)                           ┃
┃   • Rate Limiting & Security                                      ┃
┃                                                                    ┃
┃   ┌──────────────────────────────────────────────────────────┐   ┃
┃   │  DATABASE MODULE (src/backend/database/)                  │   ┃
┃   │  ┌────────────────────────────────────────────────────┐  │   ┃
┃   │  │ Database Manager (Unified Interface)               │  │   ┃
┃   │  │  - Config: Load from environment                   │  │   ┃
┃   │  │  - Manager: Handle all operations                  │  │   ┃
┃   │  │  - Migration: Transfer data between backends       │  │   ┃
┃   │  │  - Health: Monitor database status                 │  │   ┃
┃   │  └────────────────────────────────────────────────────┘  │   ┃
┃   │                          ▲                                │   ┃
┃   │  Adapters (choose one):                                  │   ┃
┃   │  ┌──────────────┬──────────────┬──────────────┐          │   ┃
┃   │  ▼              ▼              ▼              │          │   ┃
┃   │  PocketDB   SQLite      MongoDB          Legacy           │   ┃
┃   │  Adapter    Adapter     Adapter          Fallback         │   ┃
┃   │  (Default)  (Dev)       (Production)                      │   ┃
┃   └──────────────────────────────────────────────────────────┘   ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
                              │
                              │ SQL/REST API
                              │ (depending on backend)
                              ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                    DATABASE LAYER (Storage)                       ┃
┃                                                                    ┃
┃  ┌────────────────────────────────────────────────────────────┐  ┃
┃  │ POCKETDB (Recommended - Default)                           │  ┃
┃  │ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │  ┃
┃  │ http://localhost:8090                                      │  ┃
┃  │                                                             │  ┃
┃  │ Features:                                                   │  ┃
┃  │ • SQLite3-based (embedded)                                 │  ┃
┃  │ • Real-time API                                            │  ┃
┃  │ • Admin UI at /_/                                          │  ┃
┃  │ • Authentication built-in                                  │  ┃
┃  │ • Auto-indexing                                            │  ┃
┃  │ • TTL support (auto-cleanup)                               │  ┃
┃  │                                                             │  ┃
┃  │ Collections (all created auto):                            │  ┃
┃  │ ✓ sensor_readings          ✓ recommendations               │  ┃
┃  │ ✓ recommendation_tips      ✓ tank_levels                   │  ┃
┃  │ ✓ rainwater_harvest        ✓ valve_events                  │  ┃
┃  │ ✓ alerts                                                    │  ┃
┃  │                                                             │  ┃
┃  │ Storage: ./pb_data/ (SQLite database)                      │  ┃
┃  └────────────────────────────────────────────────────────────┘  ┃
┃                              ▲                                    ┃
┃  Alternative Backends (set via AGRISENSE_DB_BACKEND):            ┃
┃  ┌──────────────────────────┬──────────────────────────────┐    ┃
┃  ▼                          ▼                              │    ┃
┃  SQLite (Legacy)         MongoDB (Scale)              Fallback   ┃
┃  • Development           • Production                            ┃
┃  • Testing               • Large-scale                           ┃
┃  • Quick start           • Distributed                           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Data Flow Diagram

```
USER ACTION (Frontend)
        │
        ▼
┌─────────────────────┐
│ React Component     │
│ (e.g., Create Form) │
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Event Handler (onClick, onSubmit)                   │
│ • Collect form data                                 │
│ • Validate on client                                │
│ • Prepare API payload                               │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ API Call (fetch/axios)                              │
│ POST /api/v1/sensor-readings                        │
│ {                                                    │
│   "zone_id": "field_1",                             │
│   "temperature_c": 25.5,                            │
│   "humidity": 60.0                                  │
│ }                                                    │
└─────────────────────────────────────────────────────┘
        │ HTTP
        │ (Vite Proxy: /api → http://localhost:8000)
        ▼
┌─────────────────────────────────────────────────────┐
│ FastAPI Endpoint                                     │
│ @app.post("/api/v1/sensor-readings")                │
│                                                      │
│ async def create_reading(data: SensorReadingCreate) │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Request Validation                                   │
│ • Pydantic schema validation                        │
│ • Type checking                                     │
│ • Business logic validation                         │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Database Operation                                   │
│ await app.state.db.insert_reading(data)             │
│                                                      │
│ • Accesses database manager from app state          │
│ • Database manager routes to appropriate adapter    │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ PocketDB Adapter                                     │
│ • Converts data to PocketDB format                  │
│ • Adds timestamps                                   │
│ • Makes HTTP request to PocketDB API                │
│ POST http://localhost:8090/api/collections/         │
│            sensor_readings/records                  │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ PocketDB Server                                      │
│ • Validates request                                 │
│ • Checks authentication                             │
│ • Stores in SQLite3                                 │
│ • Generates response with ID                        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ SQLite3 Storage (./pb_data/)                         │
│ INSERT INTO sensor_readings (...)                   │
│ VALUES (...)                                        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Response back through stack                          │
│ ← HTTP 200 with record ID and data                  │
│ ← Backend parses response                           │
│ ← Frontend receives API response                    │
│ ← React updates component state                     │
│ ← UI refreshes with new data                        │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Success Notification                                 │
│ • Toast message                                     │
│ • Component re-renders                              │
│ • Data visible in UI                                │
└─────────────────────────────────────────────────────┘
```

## Startup Sequence

```
START
  │
  ├─→ Docker: Start PocketDB
  │   (docker run -d -p 8090:8090 ...)
  │
  ├─→ Wait for PocketDB ready
  │   (curl http://localhost:8090/api/health)
  │
  ├─→ Run Backend Startup Script
  │   ├─→ Activate Python venv
  │   ├─→ Set environment variables
  │   └─→ Start FastAPI (uvicorn main:app)
  │       │
  │       └─→ LIFESPAN STARTUP:
  │           ├─→ Import database module
  │           ├─→ Create DatabaseManager
  │           ├─→ Initialize PocketDB connection
  │           │   ├─→ Authenticate admin
  │           │   ├─→ Create collections
  │           │   └─→ Verify health
  │           └─→ Make available at app.state.db
  │
  ├─→ Backend ready at http://localhost:8000
  │   ├─→ /health → returns "ok"
  │   ├─→ /health/database → returns database stats
  │   ├─→ /docs → Swagger UI available
  │   └─→ All /api/v1/* endpoints ready
  │
  └─→ Run Frontend (npm run dev)
      ├─→ Vite dev server starts
      ├─→ Hot reload configured
      ├─→ Proxy to backend configured
      └─→ Frontend ready at http://localhost:5173

SYSTEM READY
```

## Environment Setup Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT SETUP                         │
│                    (.env.pocketdb)                           │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│  Database Settings      │
├─────────────────────────┤
│ AGRISENSE_DB_BACKEND    │──┐
│   = pocketdb            │  │
│                         │  │
│ POCKETDB_URL            │  │─→ PocketDB Configuration
│   = http://localhost:8090  │
│                         │  │
│ POCKETDB_DATA_DIR       │  │
│   = ./pb_data           │  │
│                         │  │
│ POCKETDB_ADMIN_PASSWORD │──┘
│   = AgriSense@2024!     │
└─────────────────────────┘

         │         │         │
         ▼         ▼         ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Database     │ │   Backend    │ │   Frontend   │
│ Manager      │ │   Settings   │ │   Config     │
├──────────────┤ ├──────────────┤ ├──────────────┤
│ • Creates    │ │ • Port 8000  │ │ • Port 5173  │
│   connections│ │ • Log level  │ │ • API URL    │
│ • Selects    │ │ • Workers    │ │ • WS URL     │
│   backend    │ │ • Debug mode │ │ • DB URL     │
│ • Connects   │ │              │ │              │
│   to DB      │ │              │ │              │
└──────────────┘ └──────────────┘ └──────────────┘
      │                │                  │
      │                │                  │
      ▼                ▼                  ▼
   PocketDB        FastAPI           React/Vite
   Ready           Ready             Ready
```

## Component Interaction Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    COMPONENT INTERACTIONS                        │
└──────────────────────────────────────────────────────────────────┘

Frontend                          Backend                     Database
═══════════════════════════════════════════════════════════════════════

React Components
    │
    ├─ SensorDashboard ──────→ GET /api/v1/readings ──────→ PocketDB
    │                                                       (Query)
    │
    ├─ RecommendationPanel ──→ GET /api/v1/recommendations  ↓
    │                                                       Return
    │                                                       Records
    ├─ AlertsPanel ─────────→ POST /api/v1/alerts ────→ PocketDB
    │                                                       (Insert)
    │
    ├─ WaterManager ────────→ GET /api/v1/tank-levels ────→ PocketDB
    │                                                       (Query)
    │
    └─ AdminPanel ──────────→ Various admin endpoints      │
                              (see example_routes.py)      │

WebSocket Real-time
    │
    └─ Connection: ws://localhost:8000/ws ──────────────→ Server
                                            (Real-time updates)

HTTP/REST
    │
    └─ Standard HTTP requests (POST, GET, PUT, DELETE)
       with JSON payloads
```

---

**Architecture Diagram**  
Generated: January 4, 2026  
Version: 2024.01  
Status: Production Ready ✅
