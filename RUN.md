# Running AgriSense Full-Stack

## Quick start

### Option 1: PowerShell script (two windows)

```powershell
.\run-dev.ps1
```

This opens two windows: **backend** (port 5000) and **frontend** (port 3001). Close those windows to stop.

### Option 2: Two terminals

**Terminal 1 – Backend**
```bash
cd backend && npm run dev
```

**Terminal 2 – Frontend**
```bash
cd frontend && npm run dev
```

### Option 3: Single command (concurrently)

```bash
npm run dev
```

Runs both backend and frontend. Stop with `Ctrl+C`.

---

## URLs

| Service   | URL                          |
|-----------|------------------------------|
| Frontend  | http://localhost:3001        |
| Backend   | http://localhost:5000        |
| API Docs  | http://localhost:5000/api/docs |
| Health    | http://localhost:5000/health |

---

## Environment

- **Backend:** `backend/.env` (copy from `backend/.env.example`). Uses `PORT=5000`, `FRONTEND_URL=http://localhost:3001`.
- **Frontend:** `frontend/.env` with `VITE_API_BASE_URL=http://localhost:5000/api`, `VITE_WS_URL=ws://localhost:5000`.

MongoDB is optional; the backend continues without it and uses in-memory fallbacks for IoT data.
