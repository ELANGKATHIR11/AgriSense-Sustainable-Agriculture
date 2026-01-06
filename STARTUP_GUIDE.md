# 🚀 AgriSense Complete Startup Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Automated Setup (Recommended)](#automated-setup)
3. [Manual Setup](#manual-setup)
4. [Running the Application](#running-the-application)
5. [Firebase Deployment](#firebase-deployment)
6. [Verification Checklist](#verification-checklist)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software
- **Node.js** 16+ (https://nodejs.org/)
- **Python** 3.10+ (https://www.python.org/)
- **Git** (https://git-scm.com/) - for version control
- **Visual Studio Code** (https://code.visualstudio.com/) - recommended

### Verify Installation
```powershell
# Check Node.js
node --version
npm --version

# Check Python
python --version

# Check Git
git --version
```

All commands in this guide should show version numbers.

---

## Automated Setup (Recommended)

### Windows PowerShell
```powershell
# Navigate to project root
cd F:\AGRISENSEFULL-STACK

# Run setup script
.\setup-local-backend.ps1
```

### Windows Command Prompt
```cmd
cd F:\AGRISENSEFULL-STACK
setup-local-backend.bat
```

### macOS/Linux
```bash
cd /path/to/AGRISENSEFULL-STACK
bash setup-local-backend.sh
```

The script will:
- ✅ Verify Node.js and npm
- ✅ Install PouchDB server dependencies
- ✅ Install frontend dependencies
- ✅ Create `.env.local` with correct defaults
- ✅ Check Firebase CLI installation

---

## Manual Setup

### Step 1: Install PouchDB Server Dependencies

```powershell
# From project root
npm install express cors pouchdb body-parser
```

### Step 2: Install Frontend Dependencies

```powershell
cd src\frontend
npm install pouchdb pouchdb-adapter-idb firebase
```

### Step 3: Create Frontend Environment File

Create `src/frontend/.env.local`:
```env
VITE_POUCHDB_SERVER_URL=http://localhost:5984
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_BACKEND_WS_URL=ws://localhost:8004
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_POUCHDB_SYNC=true
VITE_LOG_LEVEL=info
```

### Step 4: Install Python Backend Dependencies

```powershell
cd src\backend
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

---

## Running the Application

### Architecture Overview

```
User Browser (http://localhost:5173)
    ↓
Vue.js Frontend + PouchDB
    ↓
PouchDB Local Server (port 5984)
    ↓
FastAPI Backend (port 8004)
    ↓
SQLite/Azure Cosmos DB
```

### Launch All Services

**IMPORTANT: Use 3 separate terminal windows**

---

### Terminal 1: PouchDB Backend

```powershell
# From project root
node pouchdb-server.js
```

**Expected Output:**
```
✓ PouchDB server running on http://localhost:5984
✓ Database: agrisense_db
✓ Ready for sync connections
```

**Verify it's working:**
- Open http://localhost:5984/health in browser
- Should show: `{"status":"ok"}`

---

### Terminal 2: FastAPI Backend

```powershell
# From project root
cd src\backend

# Activate virtual environment (if not already active)
.\venv\Scripts\activate

# Start FastAPI
python -m uvicorn main:app --reload --port 8004
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8004 (Press CTRL+C to quit)
INFO:     Application startup complete
```

**Verify it's working:**
- Open http://localhost:8004/docs in browser
- Should show FastAPI Swagger documentation

---

### Terminal 3: Vue.js Frontend

```powershell
# From project root
cd src\frontend

# Start development server
npm run dev
```

**Expected Output:**
```
  VITE v7.2.6  ready in XX ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

**Verify it's working:**
- Open http://localhost:5173 in browser
- Application should load without errors

---

## ✨ Testing the Application

### Basic Functionality Test

1. **Load the App**
   - Navigate to http://localhost:5173
   - Should see AgriSense dashboard

2. **Test Data Entry**
   - Go to "Add Sensor" or similar feature
   - Create a new sensor reading
   - Verify it saves locally

3. **Test Offline Mode**
   - Open DevTools (F12)
   - Go to Network tab
   - Check "Offline" checkbox
   - Try creating another reading
   - Should work even without connection!

4. **Test Sync Recovery**
   - Uncheck "Offline" in DevTools
   - Wait 5-10 seconds
   - Check browser console (F12 → Console)
   - Should see sync messages like: `pouchdb-sync-change`

5. **Test Data Persistence**
   - Refresh the page (F5)
   - All data should still be there
   - Open browser DevTools → Application → IndexedDB
   - Should see `agrisense_db` database

---

## Firebase Deployment

### Prerequisites
1. Google Account (https://accounts.google.com)
2. Firebase Project (https://console.firebase.google.com)

### Step 1: Install Firebase CLI

```powershell
npm install -g firebase-tools
```

### Step 2: Create Firebase Project

1. Go to https://console.firebase.google.com
2. Click "Create a new project"
3. Name it "AgriSense"
4. Accept the defaults and create

### Step 3: Initialize Firebase in Project

```powershell
# From project root
firebase init hosting
```

**During initialization:**
- Use existing project: Select "AgriSense"
- Public directory: Type `src/frontend/dist`
- Configure as single-page app: Type `y` (yes)
- Overwrite index.html: Type `n` (no)

### Step 4: Build Frontend

```powershell
cd src\frontend
npm run build
```

This creates the `dist` folder with optimized production build.

### Step 5: Login to Firebase

```powershell
firebase login
```

This opens browser to authenticate. Allow access.

### Step 6: Deploy to Firebase

```powershell
# From project root
firebase deploy --only hosting
```

**Output will show:**
```
✔ Deploy complete!

Project Console: https://console.firebase.google.com/project/YOUR-PROJECT
Hosting URL: https://your-project.firebaseapp.com
```

### Step 7: Access Deployed App

```
https://your-project.firebaseapp.com
```

---

## Verification Checklist

### Before Running
- [ ] Node.js 16+ installed (`node --version`)
- [ ] Python 3.10+ installed (`python --version`)
- [ ] npm installed (`npm --version`)
- [ ] Project cloned/downloaded
- [ ] Opened project in IDE (VS Code)

### After Setup
- [ ] Dependencies installed (`npm list pouchdb` shows module)
- [ ] `.env.local` created in `src/frontend/`
- [ ] All scripts executable

### During Runtime
- [ ] Terminal 1: PouchDB running on port 5984
- [ ] Terminal 2: FastAPI running on port 8004
- [ ] Terminal 3: Frontend running on port 5173
- [ ] Browser shows http://localhost:5173 without errors
- [ ] Can create/view data in UI
- [ ] DevTools console shows no errors

### After Deployment
- [ ] Firebase Hosting shows green checkmark
- [ ] Deployed URL is accessible
- [ ] Frontend loads on deployed URL
- [ ] Mobile/responsive design works

---

## Troubleshooting

### Port Conflicts

**Problem:** "Address already in use"

**Solution:**
```powershell
# Find process using port
netstat -ano | findstr :5984

# Kill the process (replace XXXX with PID)
taskkill /PID XXXX /F

# Restart service
node pouchdb-server.js
```

### PouchDB Not Connecting

**Problem:** "Cannot connect to PouchDB server"

**Solution:**
```powershell
# Check if server is running
curl http://localhost:5984/health

# If not running, start it
node pouchdb-server.js

# Check firewall
netsh advfirewall firewall add rule name="PouchDB" dir=in action=allow protocol=tcp localport=5984
```

### FastAPI Not Starting

**Problem:** Python ModuleNotFoundError

**Solution:**
```powershell
cd src\backend

# Create fresh virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Try starting again
python -m uvicorn main:app --reload --port 8004
```

### Frontend Won't Load

**Problem:** CORS errors, dependencies missing

**Solution:**
```powershell
cd src\frontend

# Clear node_modules and reinstall
rmdir /S /Q node_modules
npm cache clean --force
npm install

# Rebuild
npm run dev
```

### Offline Mode Not Working

**Problem:** "Data not persisting offline"

**Solution:**
1. Check browser storage
   - DevTools → Application → Storage → IndexedDB
   - Should show `agrisense_db` database
   
2. Check permissions
   - Browser may block IndexedDB for localhost:port
   - Try in Chrome/Firefox instead of Edge
   
3. Check `.env.local`
   - Verify `VITE_ENABLE_OFFLINE_MODE=true`
   - Verify `VITE_ENABLE_POUCHDB_SYNC=true`

### Firebase Deployment Failed

**Problem:** Authentication error

**Solution:**
```powershell
# Clear Firebase cache
firebase logout
firebase login

# Verify project
firebase projects:list

# Try deploying again
firebase deploy --only hosting --debug
```

---

## Performance Tips

### Development Mode
- Use `npm run dev` for hot-reloading
- Keep DevTools closed for better performance
- Monitor Network tab for slow requests

### Production Mode
- Use `npm run build` before deploying
- Check bundle size: `npm run build -- --report`
- Use Firebase CDN for asset delivery (automatic)

### Database Optimization
```javascript
// In browser console or code
await pouchdbSync.compactDB();
// Reduces storage size by removing deleted docs
```

---

## Security Best Practices

### Development (Local)
- ✅ Run on localhost only
- ✅ No authentication needed
- ✅ CORS enabled for local testing
- ✅ SQLite database is local only

### Production (Firebase)
- ✅ Use HTTPS (Firebase handles automatically)
- ✅ Set strong Firebase rules
- ✅ Enable authentication in app
- ✅ Set environment variables for secrets
- ✅ Use Azure Cosmos DB for sensitive data
- ✅ Enable firewall rules

See `FIREBASE_POUCHDB_DEPLOYMENT.md` for detailed security guide.

---

## Getting Help

### Resources
- **Project Docs**: `DOCUMENTATION_INDEX.md`
- **Deployment Guide**: `FIREBASE_POUCHDB_DEPLOYMENT.md`
- **Code Examples**: `QUICK_START.md`
- **Backend API**: http://localhost:8004/docs (when running)

### Common Issues
- Check `FIREBASE_POUCHDB_DEPLOYMENT.md` → Troubleshooting section
- Check browser console (F12) for error messages
- Check terminal output for stack traces

### Support
1. Read documentation thoroughly
2. Check troubleshooting section
3. Review error messages carefully
4. Check port availability
5. Verify all services are running

---

## Next Steps

1. ✅ Run setup script
2. ✅ Start all 3 services in separate terminals
3. ✅ Access http://localhost:5173
4. ✅ Test offline mode
5. ✅ Test data creation/persistence
6. ✅ Deploy to Firebase (when ready)

---

**You're all set! AgriSense is ready to help farmers make better decisions. 🌾**
