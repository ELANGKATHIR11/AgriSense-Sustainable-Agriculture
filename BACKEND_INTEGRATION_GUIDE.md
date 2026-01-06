# Backend Integration Complete ✅

## Status
- ✅ **Frontend**: Deployed to https://agrisense-fe79c.web.app
- ✅ **Backend**: Running on localhost:8004
- ⏳ **Tunneling**: Ready to set up ngrok for public access

## What's Running Now
1. **FastAPI Backend** on `http://localhost:8004`
   - API endpoints at `/api/v1/*`
   - OpenAPI docs at `http://localhost:8004/docs`

2. **Frontend** deployed to Firebase
   - Accessible at https://agrisense-fe79c.web.app
   - Currently configured for localhost backend (dev mode)

## Setting Up ngrok for Public Access

### Option A: Using PowerShell Script (Recommended)
```powershell
.\setup-ngrok-tunnel.ps1
```

### Option B: Using Batch Script
```cmd
setup-ngrok-tunnel.bat
```

### Option C: Manual Setup
1. Create free ngrok account: https://dashboard.ngrok.com/signup
2. Get authtoken: https://dashboard.ngrok.com/get-started/your-authtoken
3. Install token:
   ```
   ngrok config add-authtoken YOUR_TOKEN_HERE
   ```
4. Start tunnel:
   ```
   ngrok http 8004
   ```

## After Starting ngrok Tunnel

Once ngrok shows your public URL (like `https://1234-56-789-012-34.ngrok.io`):

### 1. Update Frontend Configuration
Edit `src/frontend/.env.ngrok`:
```env
VITE_BACKEND_API_URL=https://YOUR_NGROK_URL/api/v1
VITE_BACKEND_WS_URL=wss://YOUR_NGROK_URL
```

### 2. Rebuild Frontend
```powershell
cd src/frontend
npm run build
firebase deploy --only hosting
```

### 3. Configure Backend CORS
The backend needs to accept requests from the frontend. Set environment variable:
```powershell
$env:CORS_ORIGINS="https://agrisense-fe79c.web.app,https://YOUR_NGROK_URL"
```

## Testing the Connection

After ngrok is running and frontend is rebuilt:

1. Open https://agrisense-fe79c.web.app
2. Open browser DevTools (F12 → Network tab)
3. Try an API call - you should see requests to your ngrok URL
4. Check no CORS errors in Console tab

## Quick Reference

| Component | URL | Status |
|-----------|-----|--------|
| Frontend | https://agrisense-fe79c.web.app | ✅ Deployed |
| Backend (local) | http://localhost:8004/api/v1 | ✅ Running |
| Backend (ngrok) | https://YOUR_NGROK_URL/api/v1 | ⏳ Waiting for token |
| OpenAPI Docs | http://localhost:8004/docs | ✅ Available |

## Environment Variables

### Development (localhost)
```env
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_BACKEND_WS_URL=ws://localhost:8004
```

### Production (ngrok)
```env
VITE_BACKEND_API_URL=https://NGROK_URL/api/v1
VITE_BACKEND_WS_URL=wss://NGROK_URL
```

## Troubleshooting

### ngrok not found
- Scripts will auto-download ngrok to `%TEMP%` if missing
- Ensure internet connection and Windows Defender doesn't block it

### CORS errors
- Check `CORS_ORIGINS` environment variable
- Include both Firebase domain and ngrok URL

### Backend not responding
- Verify running: `curl http://localhost:8004/health`
- Check terminal for error messages

### Frontend shows blank page
- Clear browser cache (Ctrl+Shift+Del)
- Check DevTools Console for errors
- Verify VITE_BACKEND_API_URL is set correctly

## Next Steps

1. Get ngrok authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
2. Run one of the setup scripts with your token
3. Copy the ngrok public URL when tunnel starts
4. Update `.env.ngrok` with the URL
5. Rebuild and redeploy frontend
6. Test the integration

---
**Setup Date**: January 4, 2026
**Frontend**: https://agrisense-fe79c.web.app
**Backend**: Ready on localhost:8004
