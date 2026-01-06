# AgriSense PocketDB Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Start PocketDB

```bash
# Using Docker (easiest)
docker run -d -p 8090:8090 -v pocketdb_data:/pb_data --name agrisense-pocketdb ghcr.io/pocketbase/pocketbase:latest
```

### Step 2: Start Backend

**Windows (PowerShell):**
```powershell
cd AGRISENSEFULL-STACK
.\start_pocketdb.ps1
```

**Linux/Mac:**
```bash
cd AGRISENSEFULL-STACK
chmod +x start_pocketdb.sh
./start_pocketdb.sh
```

**Any System:**
```bash
cd AGRISENSEFULL-STACK
python startup_with_pocketdb.py
```

### Step 3: Start Frontend

In a new terminal:
```bash
cd AGRISENSEFULL-STACK/src/frontend
npm install  # If not done yet
npm run dev
```

## 📍 Access Your Application

| Service | URL |
|---------|-----|
| **Frontend** | http://localhost:5173 |
| **Backend API** | http://localhost:8000 |
| **API Documentation** | http://localhost:8000/docs |
| **Database Admin** | http://localhost:8090/_/ |
| **Database Health** | http://localhost:8000/health/database |

## 🗄️ Database

### What is PocketDB?
- SQLite3-based embedded database
- Built-in REST API
- Admin UI included
- Perfect for IoT/Edge deployments
- No external DB required

### Default Credentials
- **Email**: admin@agrisense.local
- **Password**: AgriSense@2024!

### Change Password
1. Open http://localhost:8090/_/
2. Click Settings (top right)
3. Click Admin password
4. Enter new password

## 🔧 Configuration

### Environment File (.env.pocketdb)
Located in project root with all settings:

```bash
# Database
AGRISENSE_DB_BACKEND=pocketdb
POCKETDB_URL=http://localhost:8090
POCKETDB_DATA_DIR=./pb_data

# Backend  
FASTAPI_ENV=development
LOG_LEVEL=INFO

# Frontend
VITE_API_BASE_URL=http://localhost:8000
```

## 📝 Common Tasks

### View Database Statistics
```bash
curl http://localhost:8000/health/database
```

### Migrate from SQLite
```bash
python setup_pocketdb.py --mode migrate --from sqlite --to pocketdb
```

### Clean Up Old Data (Keep 90 days)
```bash
python setup_pocketdb.py --mode cleanup --days-to-keep 90
```

### Backup Database
```bash
# Docker backup
docker exec agrisense-pocketdb tar czf - /pb_data > backup.tar.gz

# Direct backup
cp -r pb_data/ pb_data.backup/
```

## 🐛 Troubleshooting

### Backend won't start?
```bash
# Check port is free
netstat -ano | findstr :8000  # Windows
lsof -i :8000                 # Linux/Mac

# Kill process using port 8000
taskkill /PID <PID> /F        # Windows
kill -9 <PID>                 # Linux/Mac
```

### PocketDB not running?
```bash
# Check container
docker ps | grep pocketbase

# View logs
docker logs agrisense-pocketdb

# Restart
docker restart agrisense-pocketdb
```

### Frontend can't connect to API?
1. Open DevTools (F12)
2. Check Network tab
3. Look for failed API requests
4. Check browser console for errors
5. Verify http://localhost:8000/health returns 200

### Database connection fails?
```bash
# Test connection
curl http://localhost:8000/health/database

# Check environment variables
echo $env:POCKETDB_URL
echo $env:POCKETDB_DATA_DIR
```

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [FULL_STACK_SETUP.md](FULL_STACK_SETUP.md) | Complete setup guide |
| [POCKETDB_INTEGRATION.md](../POCKETDB_INTEGRATION.md) | Integration details |
| [src/backend/database/README.md](src/backend/database/README.md) | Database module docs |
| [src/backend/database/POCKETDB_GUIDE.py](src/backend/database/POCKETDB_GUIDE.py) | Code examples |

## 💻 Development

### Hot Reload
- **Backend**: Automatically restarts on code changes
- **Frontend**: Hot module replacement (HMR)

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Then start backend
```

### Test API Endpoints
```bash
# Create sensor reading
curl -X POST http://localhost:8000/api/v1/sensor-readings \
  -H "Content-Type: application/json" \
  -d '{
    "zone_id": "field_1",
    "temperature_c": 25.5,
    "humidity": 60.0
  }'

# Get readings
curl http://localhost:8000/api/v1/readings?zone_id=field_1

# Check health
curl http://localhost:8000/health
```

## 🚀 Deployment

### Docker Compose (All Services)
```bash
docker-compose -f docker-compose.pocketdb.yml up -d
```

### Azure Container Apps
See [AZURE_DEPLOYMENT_QUICKSTART.md](../AZURE_DEPLOYMENT_QUICKSTART.md)

## 📊 Monitoring

### Check System Health
```bash
curl http://localhost:8000/health/enhanced
```

### View Database Stats
```bash
python setup_pocketdb.py --mode stats
```

### Monitor Logs
```bash
# Backend (in terminal)
# Shows all INFO/ERROR/WARNING

# PocketDB Docker
docker logs -f agrisense-pocketdb
```

## 🔗 Architecture

```
Browser (http://localhost:5173)
    ↓ HTTP/WebSocket
React Frontend
    ↓ API Calls (/api/v1/...)
FastAPI Backend (http://localhost:8000)
    ↓ Database Operations
Database Manager (Multi-backend)
    ↓ SQL Queries
PocketDB (http://localhost:8090)
    ↓ Storage
SQLite3 Database (./pb_data/)
```

## ⚡ Performance Tips

1. **Use zone_id filters** - Reduces data transfer
2. **Set appropriate limits** - Don't fetch all records
3. **Clean up old data** - Keep database responsive
4. **Use indexing** - PocketDB auto-indexes common fields
5. **Monitor health** - Check `/health/database` regularly

## 🔐 Security

### Development
- Default credentials are fine
- HTTPS not required
- CORS enabled for localhost

### Production
- Change admin password
- Use HTTPS/TLS
- Configure API keys
- Set environment variables
- Use secure secrets vault (Azure Key Vault)

## 📞 Need Help?

1. **Check documentation**
   - [FULL_STACK_SETUP.md](FULL_STACK_SETUP.md)
   - [POCKETDB_INTEGRATION.md](../POCKETDB_INTEGRATION.md)

2. **Review example code**
   - [src/backend/database/example_routes.py](src/backend/database/example_routes.py)
   - [src/backend/database/POCKETDB_GUIDE.py](src/backend/database/POCKETDB_GUIDE.py)

3. **Check logs**
   - Backend terminal output
   - Browser DevTools console
   - Docker logs for PocketDB

4. **Test endpoints**
   - http://localhost:8000/docs (Swagger UI)
   - Use curl commands above

## 🎯 Next Steps

1. ✅ Set up PocketDB and backend
2. ✅ Run frontend
3. 📖 Read FULL_STACK_SETUP.md for detailed info
4. 🏗️ Build your features
5. 🧪 Test with API docs
6. 📦 Deploy to production

---

**Version**: 2024.01  
**Last Updated**: January 4, 2026  
**Status**: Ready for Development & Deployment
