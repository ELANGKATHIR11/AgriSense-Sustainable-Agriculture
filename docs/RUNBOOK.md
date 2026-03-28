# AgriSense Operational Runbook

This runbook covers day-to-day operations: starting/stopping services, checking health, and resolving common problems.

---

## Startup Order

Always start services in this order to avoid dependency failures:

1. Database (MongoDB or PostgreSQL)
2. Python ML Service — port 8000
3. Node.js Backend API — port 5000
4. React Frontend — port 3001
5. Edge AI Service (optional) — port 5002

### Starting each service

**Database (MongoDB)**
```bash
mongod --dbpath /var/lib/mongodb   # Linux/macOS
# or use the system service: sudo systemctl start mongod
```

**Database (PostgreSQL — if DB_TYPE=postgres)**
```bash
sudo systemctl start postgresql
# Create database if needed:
createdb agrisense
```

**Python ML Service**
```bash
cd backend/ml
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uvicorn fastapi_service:app --host 0.0.0.0 --port 8000 --reload
```

**Node.js Backend**
```bash
cd backend
npm run dev       # development (nodemon, auto-reload)
# or
npm start         # production
```

**React Frontend**
```bash
cd frontend
npm run dev       # development (Vite, hot-reload on :3001)
# or
npm run build && npm run preview   # production preview
```

**Edge AI Service (optional)**
```bash
cd backend/ml
source .venv/bin/activate
python edge_ai_service.py   # Flask on :5002
```

### One-command dev startup (frontend + backend Node)

```bash
# From repo root
npm run dev:all
```

This runs the Node backend and React frontend concurrently via `concurrently`.  
The Python ML service must still be started separately.

---

## Stopping Services

| Service | How to stop |
|---|---|
| Node.js backend | `Ctrl+C` in terminal, or `kill $(lsof -ti:5000)` |
| React frontend | `Ctrl+C` in terminal, or `kill $(lsof -ti:3001)` |
| Python ML service | `Ctrl+C` in terminal, or `kill $(lsof -ti:8000)` |
| Edge AI service | `Ctrl+C` in terminal, or `kill $(lsof -ti:5002)` |
| MongoDB | `sudo systemctl stop mongod` |

---

## Healthcheck Endpoints

All the following are available once the Node backend is running.

| Endpoint | Method | Purpose | Expected response |
|---|---|---|---|
| `http://localhost:5000/api/health` | GET | Basic liveness | `{"message":"OK","uptime":...}` |
| `http://localhost:5000/api/health/live` | GET | Kubernetes liveness probe | `{"alive":true}` |
| `http://localhost:5000/api/health/ready` | GET | Readiness (DB + ML service) | `{"ready":true,"checks":{...}}` |
| `http://localhost:5000/api/health/detailed` | GET | Full dependency status | JSON with all services |
| `http://localhost:8000/docs` | GET | FastAPI Swagger UI | HTML page |
| `http://localhost:8000/health` | GET | ML service liveness | `{"status":"ok"}` |

**Quick check script:**
```bash
curl -s http://localhost:5000/api/health | python3 -m json.tool
curl -s http://localhost:5000/api/health/ready | python3 -m json.tool
```

---

## Common Failure Modes and Quick Fixes

### 1. Node backend won't start — `EADDRINUSE :5000`
Another process is using port 5000.
```bash
lsof -ti:5000 | xargs kill   # macOS/Linux
# or change PORT in backend/.env
```

### 2. `MongoServerSelectionError` / Cannot connect to MongoDB
MongoDB is not running or the URI is wrong.
```bash
# Check if mongod is running
ps aux | grep mongod
# Start it
mongod --dbpath ./data/db &
# Verify MONGODB_URI in backend/.env
```

### 3. ML service returns 503 or "service unavailable"
The Python FastAPI service is not running.
```bash
cd backend/ml && source .venv/bin/activate
uvicorn fastapi_service:app --port 8000 --reload
```
Also verify `ML_SERVICE_URL=http://localhost:8000` and `USE_ML_SERVICE=true` in `backend/.env`.

### 4. Frontend shows blank page or CORS error
- Vite dev server may not be running: `cd frontend && npm run dev`
- CORS mismatch: ensure `FRONTEND_URL=http://localhost:3001` is set in `backend/.env`
- API URL mismatch: ensure `VITE_API_BASE_URL=http://localhost:5000` in `frontend/.env`

### 5. `JWT malformed` / `invalid signature`
- Missing or wrong `JWT_SECRET` in `backend/.env`
- Clear localStorage in the browser and log in again

### 6. Python `ModuleNotFoundError`
The virtual environment is not activated or dependencies are not installed.
```bash
cd backend/ml
source .venv/bin/activate
pip install -r requirements.txt
```

### 7. Frontend build fails — TypeScript errors
```bash
cd frontend && npx tsc --noEmit
```
Fix any type errors shown. Common cause: outdated type definitions or mismatched versions.

### 8. `npm ci` fails — `package-lock.json` out of sync
```bash
cd frontend   # or backend
rm package-lock.json
npm install
```
Commit the updated `package-lock.json`.

### 9. Port 3001 in use
Change the Vite port:
```bash
# In frontend/.env or pass via CLI:
VITE_PORT=3002 npm run dev
# or edit vite.config.ts → server.port
```

### 10. Edge AI service not responding (:5002)
- Ensure `USE_EDGE_AI=true` in `backend/.env` and `VITE_USE_EDGE_AI=true` in `frontend/.env`
- Start the Flask service: `cd backend/ml && python edge_ai_service.py`
- Check that required model files exist in `backend/ml/models/edge/`

---

## Log Locations

| Service | Log path |
|---|---|
| Node.js combined | `backend/logs/combined.log` |
| Node.js errors | `backend/logs/error.log` |
| ML predictions | `backend/logs/ml-predictions.log` |
| Unhandled exceptions | `backend/logs/exceptions.log` |

**Tail logs:**
```bash
tail -f backend/logs/combined.log
tail -f backend/logs/error.log
```

---

## Running Tests

```bash
# Backend Node.js tests
cd backend && npm test

# Frontend Vitest tests
cd frontend && npx vitest run

# All tests from root
npm run test:all

# Frontend TypeScript type-check
npm run lint:frontend
```

---

## CI Pipeline

The GitHub Actions CI workflow (`.github/workflows/ci.yml`) runs automatically on every push and pull request to `main` / `develop`. It validates:

1. **Frontend** — TypeScript type-check, build, and Vitest tests
2. **Backend Node** — `npm test` with environment stubs
3. **Python ML service** — dependency install, smoke imports, syntax checks

If CI fails, check the **Actions** tab in GitHub for detailed logs.

---

## Related Documentation

| Document | Purpose |
|---|---|
| [`README.md`](../README.md) | Project overview and quick-start |
| [`SETUP_CHECKLIST.md`](../SETUP_CHECKLIST.md) | Full environment setup checklist |
| [`TROUBLESHOOTING.md`](../TROUBLESHOOTING.md) | Extended troubleshooting guide |
| [`RUN.md`](../RUN.md) | Detailed run instructions |
| [`QUICK_REFERENCE.md`](../QUICK_REFERENCE.md) | Command cheat-sheet |
