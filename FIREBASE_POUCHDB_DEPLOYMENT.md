# AgriSense Local-First Backend Deployment Guide

## Overview

This guide covers:
1. **Local PouchDB Backend** - Native database running on your device
2. **Real-time Sync** - Frontend syncs with local backend in real-time
3. **Firebase Hosting** - Deploy frontend to Firebase
4. **Offline-First** - Works with or without internet connection

---

## Prerequisites

- Node.js 16+ installed
- Firebase CLI installed (`npm install -g firebase-tools`)
- Firebase project created (https://console.firebase.google.com)
- Git configured

---

## Step 1: Set Up Local PouchDB Backend

### 1.1 Install PouchDB Server Dependencies

```bash
cd F:\AGRISENSEFULL-STACK
npm install express cors pouchdb pouchdb-express-router body-parser
```

### 1.2 Start Local PouchDB Server

```bash
# In a new terminal
node pouchdb-server.js
```

Expected output:
```
🚀 PouchDB Sync Server running on http://localhost:5984
📊 Database: agrisense
🔄 Sync endpoint: http://localhost:5984/agrisense
💾 Data stored in: ./pouchdb-data
```

---

## Step 2: Configure Frontend for Local Sync

### 2.1 Install Frontend Dependencies

```bash
cd src/frontend
npm install pouchdb pouchdb-adapter-idb firebase
```

### 2.2 Update Environment Variables

Create `.env.local` in `src/frontend/`:

```env
VITE_POUCHDB_SERVER_URL=http://localhost:5984
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_POUCHDB_SYNC=true
```

### 2.3 Start Frontend with Sync

```bash
# In src/frontend directory
npm run dev
```

The frontend will now:
- Store data locally using PouchDB IndexedDB adapter
- Sync with your local PouchDB server
- Work offline without internet
- Auto-sync when connection is restored

---

## Step 3: Firebase Deployment Setup

### 3.1 Initialize Firebase Project

```bash
firebase login
firebase init hosting --project your-project-id
```

When prompted:
- Select "Existing project"
- Choose your Firebase project
- Public directory: `src/frontend/dist`
- Configure SPA routing: Yes
- Don't overwrite `firebase.json`

### 3.2 Build Frontend

```bash
cd src/frontend
npm run build
```

### 3.3 Deploy to Firebase Hosting

```bash
firebase deploy --only hosting
```

You'll get a URL like: `https://your-project.firebaseapp.com`

---

## Step 4: Configure Firebase Environment Variables

### 4.1 Get Firebase Credentials

1. Go to Firebase Console
2. Project Settings → General
3. Copy your project config

### 4.2 Create `.env.production`

In `src/frontend/`:

```env
VITE_FIREBASE_API_KEY=your_key_from_firebase
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=your-sender-id
VITE_FIREBASE_APP_ID=your-app-id
VITE_POUCHDB_SERVER_URL=https://your-pouchdb-server.com
VITE_BACKEND_API_URL=https://your-backend-api.com/api/v1
```

### 4.3 Rebuild and Redeploy

```bash
npm run build
firebase deploy --only hosting
```

---

## Step 5: Run Backend Natively

### 5.1 Backend (FastAPI) - Local Device

```bash
cd src/backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8004
```

Access: `http://localhost:8004`

API Docs: `http://localhost:8004/docs`

### 5.2 Ensure CORS is Configured

In `main.py`, verify CORS middleware:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8080",
        "https://your-project.firebaseapp.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│         Browser (Frontend Application)                   │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  React Components + Business Logic            │       │
│  └──────────────────────────────────────────────┘       │
│                       ↓↑                                 │
│  ┌──────────────────────────────────────────────┐       │
│  │  PouchDB (IndexedDB Adapter)                 │       │
│  │  - Offline storage                           │       │
│  │  - Local syncing                             │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
                       ↓↑
         ┌─────────────────────────────┐
         │  Local PouchDB Server       │
         │  (http://localhost:5984)    │
         │  - CouchDB compatible       │
         │  - Replication protocol     │
         └─────────────────────────────┘
                       ↓↑
┌─────────────────────────────────────────────────────────┐
│         FastAPI Backend (Your Device)                   │
│         (http://localhost:8004)                         │
│                                                          │
│  - REST APIs                                            │
│  - Business logic                                       │
│  - ML/AI models                                         │
│  - Database operations                                  │
└─────────────────────────────────────────────────────────┘
```

---

## Real-Time Sync Features

### Automatic Sync Events

The frontend listens to these events:

```typescript
// Sync started
window.addEventListener('pouchdb-sync-active', () => {
  console.log('Syncing...');
});

// Sync paused (offline)
window.addEventListener('pouchdb-sync-paused', () => {
  console.log('Offline mode');
});

// Sync error
window.addEventListener('pouchdb-sync-error', (e) => {
  console.error('Sync error:', e.detail);
});

// Data changed
window.addEventListener('pouchdb-sync-change', (e) => {
  console.log('Data updated:', e.detail);
});
```

### Manual Sync Control

```typescript
import { startSync, stopSync, getSyncStatus } from '@/lib/pouchdb-sync';

// Start syncing
await startSync('http://localhost:5984/agrisense');

// Check status
const { isSyncing, isConnected } = getSyncStatus();

// Stop syncing
stopSync();
```

---

## Database Operations

### Save Document

```typescript
import { saveDoc } from '@/lib/pouchdb-sync';

await saveDoc({
  _id: 'crop-1',
  type: 'crop',
  name: 'Tomato',
  category: 'Vegetable',
  timestamp: new Date().toISOString(),
});
```

### Query Documents

```typescript
import { queryDocs, getDocsByType } from '@/lib/pouchdb-sync';

// Query by view
const crops = await queryDocs('crops/by_category', {
  key: 'Vegetable',
});

// Get all documents of type
const readings = await getDocsByType('reading');
```

### Delete Document

```typescript
import { deleteDoc } from '@/lib/pouchdb-sync';

await deleteDoc('crop-1');
```

---

## Troubleshooting

### PouchDB Server Won't Start

```bash
# Check if port 5984 is in use
netstat -ano | findstr :5984

# Kill process (Windows)
taskkill /PID <PID> /F

# Or use a different port
PORT=5985 node pouchdb-server.js
```

### Sync Not Working

1. Verify PouchDB server is running
2. Check browser console for errors
3. Verify CORS is enabled on backend
4. Check firewall settings

### Firebase Deployment Issues

```bash
# Clear cache and rebuild
cd src/frontend
rm -rf dist node_modules
npm install
npm run build

# Test locally first
firebase hosting:channel:deploy preview-1

# Deploy to production
firebase deploy --only hosting
```

### Offline Mode Not Working

1. Enable offline mode in `.env.local`
2. Check IndexedDB in DevTools → Application → IndexedDB
3. Verify PouchDB adapter is installed

---

## Performance Optimization

### 1. Enable Database Compression

```typescript
await compactDB();
```

### 2. Monitor Sync Performance

```typescript
const info = await getDBInfo();
console.log(`Database has ${info.doc_count} documents`);
console.log(`Update sequence: ${info.update_seq}`);
```

### 3. Implement Selective Sync

```typescript
// Only sync specific document types
await startSync('http://localhost:5984/agrisense?filter=type/crop');
```

---

## Security Considerations

1. **Local Storage**: PouchDB stores unencrypted data locally
2. **HTTPS**: Use HTTPS in production
3. **Authentication**: Implement user authentication in backend
4. **CORS**: Restrict CORS origins to your domains
5. **Firewall**: Protect PouchDB server from public access

---

## Next Steps

1. ✅ Local PouchDB backend running
2. ✅ Frontend syncing with local database
3. ✅ Real-time offline-first functionality
4. ✅ Firebase hosting deployed
5. Deploy PouchDB server (external server like AWS/Azure)
6. Implement authentication layer
7. Add encryption for sensitive data
8. Set up CI/CD pipeline

---

## Support & Documentation

- PouchDB Docs: https://pouchdb.com/
- Firebase Docs: https://firebase.google.com/docs
- Express.js: https://expressjs.com/
- CouchDB Replication: https://docs.couchdb.org/en/stable/replication/

