# AgriSense Firebase Deployment & Local-First Backend Quick Start

## 🚀 Quick Start (5 Minutes)

### Windows Users

**Step 1: Run Setup Script**
```powershell
# Open PowerShell and run:
.\setup-local-backend.ps1

# OR if using Command Prompt:
setup-local-backend.bat
```

**Step 2: Open 3 New Terminal Windows**

Terminal 1 - PouchDB Backend:
```powershell
node pouchdb-server.js
# Expected output: PouchDB server running on port 5984
```

Terminal 2 - FastAPI Backend:
```powershell
cd src\backend
python -m uvicorn main:app --reload --port 8004
# Expected output: Uvicorn running on http://127.0.0.1:8004
```

Terminal 3 - Frontend:
```powershell
cd src\frontend
npm run dev
# Expected output: VITE v7.2.6 ready in XX ms → Local: http://localhost:5173/
```

**Step 3: Access Application**
- Open browser: http://localhost:5173
- Application auto-syncs with local PouchDB backend
- Works offline - data syncs when connection returns

---

## 📱 Local Services Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Vue.js Frontend                      │
│              (http://localhost:5173)                    │
└────────────┬────────────────────────────────────────────┘
             │
             ├── Real-Time Sync ──────────┐
             │                             │
             ↓                             ↓
   ┌─────────────────────┐      ┌──────────────────────┐
   │   PouchDB IndexedDB  │      │  PouchDB REST API    │
   │   (Browser Storage)  │◄────►│ (http://5984)        │
   │ (Works Offline!)     │      │ Node.js + Express    │
   └─────────────────────┘      └──────────────────────┘
             │                             │
             │                             ↓
             │                   ┌──────────────────────┐
             │                   │  FastAPI Backend     │
             │                   │ (http://8004)        │
             │                   │ Business Logic, ML   │
             │                   │ SQLite/Cosmos DB     │
             └───────────────────┴──────────────────────┘
```

---

## 🔄 Real-Time Sync Flow

1. **User Action** → Save crop data in browser
2. **Local Storage** → Immediately saved to PouchDB IndexedDB (instant feedback)
3. **Background Sync** → Data syncs to local PouchDB server (CouchDB protocol)
4. **Server Processing** → Optional: PouchDB server pushes to FastAPI backend
5. **Offline Ready** → If connection drops, app still works with local data
6. **Auto-Reconnect** → When online, syncs all changes automatically

---

## 🌍 Firebase Deployment (When Ready)

### Prerequisites
```powershell
# Install Firebase CLI globally
npm install -g firebase-tools

# Login to Firebase
firebase login
```

### Deploy to Firebase

```powershell
# Step 1: Build frontend
cd src\frontend
npm run build

# Step 2: Deploy to Firebase
cd ..\..\
firebase deploy --only hosting

# Output will show:
# ✔ Deploy complete!
# Project Console: https://console.firebase.google.com/project/YOUR_PROJECT
# Hosting URL: https://YOUR_PROJECT.firebaseapp.com
```

### Access Deployed App
```
https://YOUR_PROJECT.firebaseapp.com
```

**Note**: Frontend will still sync with local PouchDB server on your network. For cloud sync, update `VITE_POUCHDB_SERVER_URL` to your cloud server.

---

## 📋 What Each Service Does

### 🖥️ PouchDB Backend (port 5984)
- **Role**: Local database with sync capability
- **Storage**: CouchDB-compatible document storage
- **Sync Protocol**: Real-time replication to other instances
- **Runs On**: Node.js + Express.js
- **Start Command**: `node pouchdb-server.js`
- **Features**:
  - Health check: GET `http://localhost:5984/health`
  - Database info: GET `http://localhost:5984/db/info`
  - Full REST API for CRUD operations

### 🐍 FastAPI Backend (port 8004)
- **Role**: Business logic, ML models, data validation
- **Responsibilities**: 
  - ML inference (crop disease detection, yield prediction)
  - Complex business logic
  - Authentication/authorization
  - Integration with Azure services
- **Runs On**: Python + uvicorn
- **Start Command**: `python -m uvicorn main:app --reload --port 8004`
- **Database**: SQLite (dev) or Azure Cosmos DB (prod)

### 🎨 Vue.js Frontend (port 5173)
- **Role**: User interface and local-first data management
- **Framework**: Vue 3 + Vite + TypeScript
- **Storage**: PouchDB with IndexedDB backend
- **Features**:
  - Offline-first operation
  - Real-time sync with PouchDB server
  - Responsive design
  - Multi-language support
- **Runs On**: Node.js + npm
- **Start Command**: `npm run dev`
- **Dev Port**: http://localhost:5173
- **Prod Port**: Deployed to Firebase Hosting

---

## ✨ Key Features

### ✅ Offline-First
- Works completely offline
- Data saved to browser's IndexedDB
- Syncs automatically when online

### ✅ Real-Time Sync
- Changes sync instantly across browser tabs
- Background replication to server
- Conflict resolution built-in

### ✅ Responsive Design
- Mobile-optimized UI
- Touch-friendly on tablets
- Works on all modern browsers

### ✅ Cloud Ready
- Firebase Hosting deployment
- Can connect to cloud database
- Scales to multiple users

---

## 🧪 Testing Offline Mode

### Simulate Offline in Browser:

1. Open DevTools (F12)
2. Go to "Network" tab
3. Check "Offline" checkbox
4. Continue using the app - it should still work!
5. Uncheck "Offline" - changes sync automatically

---

## 🐛 Troubleshooting

### Port Already in Use
```powershell
# Find process on port 5984
netstat -ano | findstr :5984

# Kill the process
taskkill /PID <PID> /F

# Then restart
node pouchdb-server.js
```

### Frontend Can't Connect to Backend
```powershell
# Check if backend is running
curl http://localhost:8004/docs

# Check if PouchDB is running
curl http://localhost:5984/health

# Verify .env.local in src/frontend/
type src\frontend\.env.local
```

### Sync Not Working
```powershell
# Check PouchDB logs
# Should see messages like:
# ✓ Sync started with http://localhost:5984/agrisense_db
# ✓ Received X changes

# Verify in browser console (F12)
# Look for CustomEvent messages: 'pouchdb-sync-change'
```

---

## 📊 Environment Variables

**Frontend** (`src/frontend/.env.local`):
```env
# Local PouchDB Server
VITE_POUCHDB_SERVER_URL=http://localhost:5984

# FastAPI Backend
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_BACKEND_WS_URL=ws://localhost:8004

# Features
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_POUCHDB_SYNC=true
VITE_LOG_LEVEL=info
```

---

## 🔐 Security Notes

### Development (Local)
- Run everything on localhost
- No authentication required for local testing
- CORS enabled for local dev

### Production (Firebase)
- Use HTTPS only
- Enable authentication
- Configure CORS properly
- Use environment variables for secrets
- See `FIREBASE_POUCHDB_DEPLOYMENT.md` section "Security Considerations"

---

## 📈 Performance Tips

### Browser Side:
- IndexedDB automatically limits storage (~50MB typical)
- Periodic cleanup: `await pouchdbSync.compactDB()`
- Monitor network tab in DevTools

### Server Side:
- PouchDB handles up to 10K docs efficiently
- For larger datasets, consider pagination
- Monitor memory usage: `node pouchdb-server.js`

### Deploy to Cloud:
- Use CDN for frontend assets (Firebase auto-handles)
- Scale PouchDB with replication
- Consider Azure Cosmos DB for production

---

## 📚 Complete Documentation

For detailed information, see:
- **`FIREBASE_POUCHDB_DEPLOYMENT.md`** - Full deployment guide with troubleshooting
- **`src/frontend/src/lib/pouchdb-sync.ts`** - Sync service documentation
- **`pouchdb-server.js`** - Backend API endpoints
- **`DOCUMENTATION_INDEX.md`** - Project-wide docs index

---

## 🎯 Next Steps

1. ✅ Run setup script
2. ✅ Start all 3 services
3. ✅ Test app at http://localhost:5173
4. ✅ Test offline mode (DevTools → Network → Offline)
5. ✅ Create some crops/readings
6. ✅ Verify data persists after refresh
7. ✅ Deploy to Firebase when ready

---

## 💡 Architecture Highlights

### Why This Setup?
- **Local-First**: Works offline, essential for agriculture (unreliable connectivity)
- **Real-Time Sync**: Changes sync instantly across devices
- **Cloud Ready**: Easy transition to cloud with Firebase
- **Scalable**: Can handle multiple users, multiple farms
- **Secure**: Local encryption + HTTPS on cloud

### Technology Choices:
- **PouchDB**: CouchDB replication, proven in production
- **IndexedDB**: Browser storage, 50MB+ capacity
- **FastAPI**: Fast, modern Python framework
- **Vue.js**: Lightweight, responsive UI
- **Firebase**: Zero-config hosting, auto-scaling

---

## 🚀 Deployment Summary

| Component | Development | Production |
|-----------|-------------|-----------|
| Frontend | http://localhost:5173 | https://your-project.firebaseapp.com |
| PouchDB | http://localhost:5984 | Cloud instance (optional) |
| FastAPI | http://localhost:8004 | Azure Container Apps (optional) |
| Database | SQLite local | Azure Cosmos DB (optional) |
| Storage | Browser IndexedDB | Cloud Storage (optional) |

---

**Happy farming with AgriSense! 🌾**
