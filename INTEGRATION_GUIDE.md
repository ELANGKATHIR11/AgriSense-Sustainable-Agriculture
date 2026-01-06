# 🔗 AgriSense Integration & Usage Guide

## Quick Access Commands

### Access Your Live Application
```bash
# Cloud (Global)
https://agrisense-fe79c.web.app

# Local Development
http://localhost:8080          # Frontend
http://localhost:8004/docs     # API Documentation
http://localhost:5984/health   # PouchDB Status
```

---

## 🚀 Starting All Services (3 Steps)

### Step 1: Open Terminal 1 - PouchDB Backend
```powershell
cd F:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK
node pouchdb-server.js
```
**Expected Output:**
```
🚀 PouchDB Sync Server running on http://localhost:5984
📊 Database: agrisense
🔄 Sync endpoint: http://localhost:5984/agrisense
```

### Step 2: Open Terminal 2 - FastAPI Backend
```powershell
cd F:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\src\backend
& F:/AGRISENSEFULL-STACK/.venv/Scripts/Activate.ps1
python -m uvicorn main:app --reload --port 8004
```
**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8004
INFO:     Application startup complete
```

### Step 3: Open Terminal 3 - Frontend
```powershell
cd F:\AGRISENSEFULL-STACK\AGRISENSEFULL-STACK\src\frontend
npm run dev
```
**Expected Output:**
```
VITE v7.2.6  ready in XXX ms
➜  Local:   http://127.0.0.1:8080/
```

---

## 📚 Data Models

### Crop Document
```json
{
  "_id": "crop_tomato_001",
  "type": "crop",
  "name": "Tomato",
  "category": "Vegetable",
  "growing_season": "Summer",
  "ph_min": 6.0,
  "ph_max": 6.8,
  "temperature_min": 20,
  "temperature_max": 30,
  "growth_duration_days": 60,
  "water_requirement_mm": 400
}
```

### Sensor Reading Document
```json
{
  "_id": "reading_device001_2026010401",
  "type": "sensor_reading",
  "deviceId": "DEVICE_001",
  "timestamp": "2026-01-04T10:30:00Z",
  "measurements": {
    "temperature": 28.5,
    "humidity": 65.2,
    "soilMoisture": 42.1,
    "phLevel": 6.8,
    "nitrogen": 25.3,
    "phosphorus": 18.7,
    "potassium": 32.4
  },
  "location": {
    "fieldId": "FIELD_001",
    "coordinates": {
      "lat": 12.34,
      "lon": 56.78
    }
  }
}
```

### Recommendation Document
```json
{
  "_id": "recommendation_field001_2026010401",
  "type": "recommendation",
  "fieldId": "FIELD_001",
  "cropId": "crop_tomato_001",
  "timestamp": "2026-01-04T10:30:00Z",
  "recommendations": [
    {
      "type": "irrigation",
      "action": "Increase watering",
      "reason": "Soil moisture below optimal",
      "urgency": "high"
    }
  ]
}
```

---

## 🔄 API Endpoints (FastAPI Backend)

### Health & Status
```
GET /health                          - Health check
GET /docs                            - Swagger documentation
GET /openapi.json                    - OpenAPI schema
```

### Sensor Management
```
POST   /api/v1/sensors              - Create sensor
GET    /api/v1/sensors              - List all sensors
GET    /api/v1/sensors/{id}         - Get sensor details
PUT    /api/v1/sensors/{id}         - Update sensor
DELETE /api/v1/sensors/{id}         - Delete sensor
```

### Sensor Readings
```
POST   /api/v1/sensors/readings     - Add reading
GET    /api/v1/sensors/readings     - List readings
GET    /api/v1/sensors/{id}/latest  - Latest reading for device
```

### Crops
```
GET    /api/v1/crops                - List all crops
GET    /api/v1/crops/{id}           - Get crop details
POST   /api/v1/crops                - Add crop
```

### ML Predictions
```
POST   /api/v1/ml/predict           - General prediction
POST   /api/v1/ml/disease           - Disease detection
POST   /api/v1/ml/weed              - Weed detection
POST   /api/v1/ml/yield             - Yield prediction
```

---

## 💾 PouchDB REST API

### Database Operations
```
GET    http://localhost:5984/health              - Server health
GET    http://localhost:5984/agrisense           - DB info
GET    http://localhost:5984/agrisense/_all_docs - List all docs
GET    http://localhost:5984/agrisense/_changes  - Change feed
POST   http://localhost:5984/agrisense           - Create DB
```

### Document Operations
```
GET    /agrisense/{id}                           - Get document
POST   /agrisense                                - Create document
PUT    /agrisense/{id}                           - Update document
DELETE /agrisense/{id}                           - Delete document
POST   /agrisense/_bulk_docs                     - Bulk operations
```

### Example: Create Crop via PouchDB
```bash
curl -X POST http://localhost:5984/agrisense \
  -H "Content-Type: application/json" \
  -d '{
    "_id": "crop_corn_001",
    "type": "crop",
    "name": "Corn",
    "category": "Grain",
    "growing_season": "Summer",
    "ph_min": 6.0,
    "ph_max": 7.5,
    "temperature_min": 15,
    "temperature_max": 35
  }'
```

---

## 🧪 Testing the Sync

### In Browser Console (F12)
```javascript
// Test 1: Listen for sync events
document.addEventListener('pouchdb-sync-change', (e) => {
  console.log('Sync change:', e.detail);
});

document.addEventListener('pouchdb-sync-active', (e) => {
  console.log('Sync active');
});

document.addEventListener('pouchdb-sync-paused', (e) => {
  console.log('Sync paused');
});

// Test 2: Save a document
await pouchdbSync.saveDoc({
  type: 'test',
  name: 'Test Document',
  timestamp: new Date().toISOString()
});

// Test 3: Query documents
const crops = await pouchdbSync.queryDocs('crops/by_name');
console.log(crops);

// Test 4: Check sync status
const info = await pouchdbSync.getDBInfo();
console.log(info);
```

---

## 📱 Feature Walkthroughs

### Add Sensor Reading
```
1. Open http://localhost:8080
2. Click "Dashboard" → "Add Sensor Reading"
3. Fill in temperature, humidity, soil moisture
4. Click "Save"
5. Verify: Data appears immediately (offline OK)
6. Verify: Syncs to PouchDB server in background
```

### Test Offline Mode
```
1. Open http://localhost:8080
2. Open DevTools (F12)
3. Go to Network tab
4. Check "Offline" checkbox
5. Try adding a new crop
6. Verify: Works without internet
7. Uncheck "Offline"
8. Verify: Changes sync automatically
```

### View API Documentation
```
1. Open http://localhost:8004/docs
2. Try endpoints:
   - GET /health
   - GET /api/v1/crops
   - POST /api/v1/sensors/readings
3. See request/response format
4. Test with sample data
```

### Monitor Real-Time Sync
```
1. Open http://localhost:8080 in 2 browser tabs
2. In Tab 1: Add a new crop
3. In Tab 2: Verify it appears instantly
4. Check browser console for sync events
5. Monitor PouchDB replication
```

---

## 🔧 Common Operations

### Rebuild Frontend
```bash
cd src/frontend
npm run build
```

### Redeploy to Firebase
```bash
firebase deploy --only hosting
```

### Reset Local Database
```bash
# Stop all services first
# Then delete:
rm -r .pouchdb-data/
rm -r src/frontend/node_modules/.vite

# Restart all services
```

### View Environment Config
```bash
# Frontend
cat src/frontend/.env.local

# Backend
cat src/backend/.env
```

### Monitor Logs
```bash
# PouchDB logs (check terminal 1)
# FastAPI logs (check terminal 2)
# Frontend logs (check DevTools Console)
```

---

## 📊 Database Inspection

### Via Browser DevTools
```
1. Open DevTools (F12)
2. Go to Application tab
3. Find IndexedDB → agrisense_db
4. View all documents
5. Check storage quota
```

### Via curl
```bash
# List all documents
curl http://localhost:5984/agrisense/_all_docs

# Get specific document
curl http://localhost:5984/agrisense/crop_tomato_001

# Check database info
curl http://localhost:5984/agrisense
```

### Via Firebase Console
```
1. Go to https://console.firebase.google.com
2. Select agrisense-fe79c project
3. Navigate to Hosting
4. View deployment history
5. Check analytics/performance
```

---

## 🚨 Troubleshooting Commands

### Check if Services are Running
```bash
# PouchDB
Invoke-WebRequest http://localhost:5984/health

# FastAPI
Invoke-WebRequest http://localhost:8004/health

# Frontend
Invoke-WebRequest http://localhost:8080
```

### Restart Services
```bash
# Kill process on port
netstat -ano | findstr :5984
taskkill /PID <PID> /F

# Then restart
node pouchdb-server.js
```

### Check Logs
```bash
# View recent git commits
git log --oneline -5

# Check firebase status
firebase status

# Verify project
firebase projects:list
```

### Clear Browser Cache
```javascript
// In DevTools Console:
// Clear IndexedDB
indexedDB.databases().then(dbs => {
  dbs.forEach(db => indexedDB.deleteDatabase(db.name));
});

// Clear localStorage
localStorage.clear();

// Clear sessionStorage
sessionStorage.clear();
```

---

## 🌐 URL Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│ QUICK URL REFERENCE                                         │
├─────────────────────────────────────────────────────────────┤
│ Production (Cloud)                                          │
│ ├─ App: https://agrisense-fe79c.web.app                    │
│ ├─ Console: https://console.firebase.google.com            │
│ └─ Hosting: https://agrisense-fe79c.web.app/               │
│                                                             │
│ Development (Local)                                         │
│ ├─ Frontend: http://localhost:8080                         │
│ ├─ API Docs: http://localhost:8004/docs                    │
│ ├─ PouchDB: http://localhost:5984                          │
│ └─ Health: http://localhost:5984/health                    │
│                                                             │
│ Project IDs                                                 │
│ ├─ Firebase Project: agrisense-fe79c                       │
│ ├─ Project Number: 711158080268                            │
│ └─ Account: elangkathir11@gmail.com                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 💡 Tips & Tricks

### Speed Up Development
```bash
# Watch and rebuild frontend automatically
npm run dev -- --watch

# Hot reload FastAPI
python -m uvicorn main:app --reload

# Monitor PouchDB in real-time
tail -f pouchdb-server.js output
```

### Debug Sync Issues
```javascript
// Enable detailed logging
localStorage.setItem('pouchdb_debug', 'true');

// Monitor sync handler
window.addEventListener('pouchdb-sync-*', (e) => {
  console.log('Sync event:', e.type, e.detail);
});

// Check local database size
navigator.storage.estimate().then(estimate => {
  console.log(`Using ${estimate.usage} of ${estimate.quota} bytes`);
});
```

### Performance Monitoring
```javascript
// Measure sync time
const start = performance.now();
await pouchdbSync.startSync('http://localhost:5984');
const elapsed = performance.now() - start;
console.log(`Sync took ${elapsed}ms`);

// Monitor replication
db.changes({live: true, include_docs: true})
  .on('change', (change) => {
    console.log('Change:', change);
  });
```

---

## 📋 Deployment Checklist

- [x] Frontend built and deployed
- [x] PouchDB server running
- [x] FastAPI backend operational
- [x] Real-time sync working
- [x] Offline mode enabled
- [x] Database populated with crops
- [x] Environment configured
- [x] Firebase project linked
- [x] All 3 services running
- [x] Health checks passing

---

## 🎓 Next Learning Steps

1. **Explore the Frontend Code**
   - `src/frontend/src/lib/pouchdb-sync.ts` - Sync logic
   - `src/frontend/src/components/` - React components
   - `src/frontend/src/pages/` - Page components

2. **Understand the Backend**
   - `src/backend/main.py` - FastAPI app
   - `src/backend/routes/` - API routes
   - `src/backend/services/` - Business logic

3. **Database Design**
   - CouchDB/PouchDB concepts
   - Document-oriented databases
   - Replication and sync

4. **Cloud Deployment**
   - Firebase Hosting features
   - Global CDN usage
   - Monitoring and analytics

---

## 🆘 Support & Help

### Documentation Files
- `QUICK_START.md` - 5-minute quick start
- `STARTUP_GUIDE.md` - Complete setup guide
- `FIREBASE_POUCHDB_DEPLOYMENT.md` - Architecture details
- `DEPLOYMENT_COMPLETE.md` - Deployment info
- `STATUS_DASHBOARD.md` - System status

### External Resources
- Firebase: https://firebase.google.com/docs
- PouchDB: https://pouchdb.com/guides/
- FastAPI: https://fastapi.tiangolo.com/
- Vue.js: https://vuejs.org/guide/

---

## ✅ You're All Set!

Your AgriSense application is:
- ✅ Deployed globally
- ✅ Running locally
- ✅ Syncing in real-time
- ✅ Working offline
- ✅ Ready for production

**Access your app now:**
## 🌐 https://agrisense-fe79c.web.app

Happy Farming! 🌾
