# 🚀 Hugging Face Spaces Deployment Guide - AgriSense

**Complete guide to deploy AgriSense full-stack AI/ML application on Hugging Face Spaces with FREE 16GB RAM using the "Golden Combo" architecture.**

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Setup External Services (Free Tier)](#setup-external-services)
4. [Deployment Process](#deployment-process)
5. [Environment Variables](#environment-variables)
6. [Testing & Verification](#testing-verification)
7. [Troubleshooting](#troubleshooting)
8. [Cost Breakdown](#cost-breakdown)
9. [Performance Optimization](#performance-optimization)

---

## 🏗️ Architecture Overview

### The "Golden Combo" (100% Free)

```
┌─────────────────────────────────────────────────────────────┐
│  Hugging Face Spaces (Docker SDK - 16GB RAM)               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Single Container (UID 1000, Port 7860)               │ │
│  │  ┌────────────────┐  ┌────────────────┐              │ │
│  │  │  Uvicorn       │  │  Celery Worker │              │ │
│  │  │  (FastAPI)     │  │  (Background)  │              │ │
│  │  │  - API         │  │  - ML Tasks    │              │ │
│  │  │  - Static UI   │  │  - Reports     │              │ │
│  │  └────────────────┘  └────────────────┘              │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
           │                            │
           │                            │
           ▼                            ▼
    ┌──────────────┐          ┌──────────────────┐
    │  MongoDB     │          │  Upstash Redis   │
    │  Atlas M0    │          │  Free Tier       │
    │  (512MB)     │          │  (10K commands)  │
    └──────────────┘          └──────────────────┘
```

**Components:**
- **Compute:** Hugging Face Spaces (Docker SDK) - 16GB RAM, 8 vCPU
- **Database:** MongoDB Atlas M0 Sandbox - 512MB storage
- **Broker:** Upstash Redis - 10,000 commands/day
- **Architecture:** Monolithic container (React + FastAPI + Celery)

---

## ✅ Prerequisites

### Local Requirements
- Git installed
- Docker installed (for local testing)
- Node.js 18+ (for frontend development)
- Python 3.12+ (for backend development)

### Accounts Needed (All Free)
1. **Hugging Face Account** - [Sign up](https://huggingface.co/join)
2. **MongoDB Atlas** - [Sign up](https://www.mongodb.com/cloud/atlas/register)
3. **Upstash Account** - [Sign up](https://console.upstash.com/)

---

## 🔧 Setup External Services

### 1️⃣ MongoDB Atlas Setup (5 minutes)

1. **Create Account & Cluster**
   - Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
   - Create a free account
   - Create new project: "AgriSense"
   - Click **"Build a Database"** → Choose **M0 FREE** tier
   - Select region closest to your users
   - Cluster name: `agrisense-cluster`

2. **Configure Security**
   - **Network Access:**
     - Click "Network Access" in left sidebar
     - Click "Add IP Address"
     - Select **"Allow Access from Anywhere"** (0.0.0.0/0)
     - This is required for Hugging Face Spaces
   
   - **Database Access:**
     - Click "Database Access" in left sidebar
     - Click "Add New Database User"
     - Username: `agrisense_user`
     - Password: Generate strong password (save it!)
     - Database User Privileges: **Read and write to any database**

3. **Get Connection String**
   - Go to "Database" → Click "Connect" on your cluster
   - Choose **"Connect your application"**
   - Driver: Python, Version: 3.12 or later
   - Copy connection string:
     ```
     mongodb+srv://agrisense_user:<password>@agrisense-cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
     ```
   - Replace `<password>` with your actual password
   - Save this as `MONGO_URI`

### 2️⃣ Upstash Redis Setup (3 minutes)

1. **Create Account**
   - Go to [Upstash Console](https://console.upstash.com/)
   - Sign up with GitHub or email

2. **Create Redis Database**
   - Click **"Create Database"**
   - Name: `agrisense-redis`
   - Type: **Regional**
   - Region: Choose closest to your users
   - TLS: **Enabled** (default)
   - Click "Create"

3. **Get Connection URL**
   - Click on your database name
   - Scroll to **"REST API"** section
   - Copy the **"Redis URL"** (starts with `redis://`)
   - Format: `redis://default:<password>@<host>:6379`
   - Save this as `REDIS_URL`

---

## 🚀 Deployment Process

### Step 1: Create Hugging Face Space

1. Go to https://huggingface.co/new-space
2. Fill in details:
   - **Owner:** Your username or organization
   - **Space name:** `agrisense-app` (or your choice)
   - **License:** Apache-2.0
   - **Select SDK:** **Docker**
   - **Space hardware:** **CPU basic (FREE)** - 16GB RAM
   - **Visibility:** Public or Private

3. Click **"Create Space"**

### Step 2: Clone and Setup Repository

```bash
# Clone your Hugging Face Space repo
git clone https://huggingface.co/spaces/<your-username>/agrisense-app
cd agrisense-app

# Copy AgriSense files
# Option A: If you have AgriSense locally
cp /path/to/AgriSense/Dockerfile.huggingface ./Dockerfile
cp /path/to/AgriSense/start.sh .
cp -r /path/to/AgriSense/agrisense_app .
cp -r /path/to/AgriSense/ml_models .

# Option B: Clone from your GitHub repo
git clone https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK.git temp_clone
cp temp_clone/Dockerfile.huggingface ./Dockerfile
cp temp_clone/start.sh .
cp -r temp_clone/agrisense_app .
cp -r temp_clone/ml_models .
rm -rf temp_clone
```

### Step 3: Create README.md for Hugging Face

Create `README.md` in the Space root:

```markdown
---
title: AgriSense AI Platform
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
---

# 🌾 AgriSense - AI-Powered Agricultural Platform

Full-stack AI/ML application for smart farming with:
- Real-time sensor monitoring
- Crop disease detection
- Intelligent recommendations
- Chatbot assistance

**Tech Stack:** FastAPI + React + PyTorch + TensorFlow + Celery

[Documentation](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK)
```

### Step 4: Configure Space Secrets

1. Go to your Space settings: `https://huggingface.co/spaces/<username>/agrisense-app/settings`

2. Click **"Variables and secrets"** → **"New secret"**

3. Add the following secrets:

| Secret Name | Value | Example |
|------------|-------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb+srv://user:pass@cluster.mongodb.net/agrisense` |
| `REDIS_URL` | Upstash Redis URL | `redis://default:pass@host:6379` |
| `AGRISENSE_ADMIN_TOKEN` | Generate random token | `sk-agrisense-abc123xyz789` |

**Generate Admin Token:**
```bash
# Linux/Mac
openssl rand -hex 32

# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

4. **Optional Secrets:**

| Secret Name | Value | Purpose |
|------------|-------|---------|
| `OPENAI_API_KEY` | Your OpenAI key | For GPT-4 chatbot |
| `GEMINI_API_KEY` | Your Gemini key | For Google AI features |
| `SENTRY_DSN` | Sentry project DSN | Error tracking |

5. **Optional Variables (Public):**

| Variable Name | Default | Description |
|--------------|---------|-------------|
| `AGRISENSE_DISABLE_ML` | `0` | Set to `1` to disable ML models (save RAM) |
| `WORKERS` | `2` | Number of Uvicorn workers |
| `CELERY_WORKERS` | `2` | Number of Celery workers |
| `LOG_LEVEL` | `info` | Logging level: debug, info, warning, error |

### Step 5: Deploy to Hugging Face

```bash
# Ensure all files are in the Space repo
ls -la
# Should see: Dockerfile, start.sh, agrisense_app/, ml_models/, README.md

# Add all files
git add .

# Commit
git commit -m "Initial deployment of AgriSense"

# Push to Hugging Face (this triggers build)
git push origin main
```

### Step 6: Monitor Build

1. Go to your Space URL: `https://huggingface.co/spaces/<username>/agrisense-app`
2. Watch the **"Building..."** logs
3. Build takes **10-15 minutes** first time
4. Look for:
   ```
   ✅ Frontend static files found at static/ui/
   ✅ Celery worker started
   🚀 Starting FastAPI server on port 7860...
   ```

5. Once you see **"Running"**, your app is live! 🎉

---

## 🔐 Environment Variables

### Required Secrets

#### `MONGO_URI` (Required)
MongoDB Atlas connection string for persistent data storage.

**Format:**
```
mongodb+srv://<username>:<password>@<cluster>.mongodb.net/<database>?retryWrites=true&w=majority
```

**Example:**
```
mongodb+srv://agrisense_user:SecurePass123@agrisense-cluster.abc123.mongodb.net/agrisense?retryWrites=true&w=majority
```

#### `REDIS_URL` (Required)
Upstash Redis connection URL for Celery broker and result backend.

**Format:**
```
redis://default:<password>@<host>:<port>
```

**Example:**
```
redis://default:AXylZDEyMzQ1Njc4OTBhYmNkZWY@us1-modern-firefly-12345.upstash.io:6379
```

#### `AGRISENSE_ADMIN_TOKEN` (Required)
Secure token for admin API endpoints and system operations.

**Generate:**
```bash
python -c "import secrets; print('sk-agrisense-' + secrets.token_urlsafe(32))"
```

### Optional Configuration

#### `AGRISENSE_DISABLE_ML` (Default: `0`)
Disable ML model loading to save memory.
- `0`: ML models enabled (requires ~4-6GB RAM)
- `1`: ML models disabled (chatbot and inference APIs will return errors)

**When to disable:**
- Testing deployment without ML
- Memory constraints
- API-only functionality needed

#### `WORKERS` (Default: `2`)
Number of Uvicorn worker processes for handling HTTP requests.
- Min: `1` (low traffic)
- Recommended: `2` (balanced)
- Max: `4` (high traffic, more RAM usage)

#### `CELERY_WORKERS` (Default: `2`)
Number of Celery worker threads for background tasks.
- Min: `1` (minimal background processing)
- Recommended: `2` (balanced)
- Max: `4` (intensive ML inference)

#### `LOG_LEVEL` (Default: `info`)
Application logging verbosity.
- `debug`: Detailed logs (development)
- `info`: Standard logs (production)
- `warning`: Warnings and errors only
- `error`: Errors only

---

## 🧪 Testing & Verification

### 1. Health Check

```bash
curl https://<your-username>-agrisense-app.hf.space/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-28T12:00:00Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "celery": "running"
  }
}
```

### 2. API Documentation

Open in browser:
```
https://<your-username>-agrisense-app.hf.space/docs
```

You should see FastAPI's interactive API documentation (Swagger UI).

### 3. Frontend UI

Open in browser:
```
https://<your-username>-agrisense-app.hf.space/ui/
```

You should see the AgriSense React dashboard.

### 4. Test API Endpoint

```bash
# Test sensor reading endpoint
curl -X POST "https://<your-username>-agrisense-app.hf.space/api/sensors/readings" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AGRISENSE_ADMIN_TOKEN" \
  -d '{
    "device_id": "TEST_SENSOR_001",
    "temperature": 25.5,
    "humidity": 65.0,
    "soil_moisture": 42.0
  }'
```

### 5. Check Celery Worker

```bash
# If you have Flower monitoring enabled
https://<your-username>-agrisense-app.hf.space/flower/
```

---

## 🔍 Troubleshooting

### Issue 1: "Application Error" on Space

**Symptoms:**
- Space shows "Application Error"
- Build logs show errors

**Solutions:**

1. **Check Build Logs:**
   - Click "Logs" tab in Space
   - Look for Python/Node errors

2. **Verify Secrets:**
   - Ensure `MONGO_URI` and `REDIS_URL` are correct
   - Test MongoDB connection:
     ```bash
     mongosh "mongodb+srv://<your-connection-string>"
     ```

3. **Check Docker Build:**
   Test locally:
   ```bash
   docker build -f Dockerfile.huggingface -t agrisense-test .
   docker run -p 7860:7860 \
     -e MONGO_URI="your-uri" \
     -e REDIS_URL="your-url" \
     -e AGRISENSE_ADMIN_TOKEN="test-token" \
     agrisense-test
   ```

### Issue 2: Frontend Not Loading (404)

**Symptoms:**
- `/ui/` returns 404
- Static files not found

**Solutions:**

1. **Verify Build Output:**
   Check build logs for:
   ```
   ✅ Frontend static files found at static/ui/
   ```

2. **Check Vite Build:**
   ```bash
   cd agrisense_app/frontend/farm-fortune-frontend-main
   npm run build
   ls -la dist/
   # Should show index.html and assets/
   ```

3. **Rebuild Space:**
   - Go to Space settings
   - Click "Factory reboot"
   - This forces a fresh build

### Issue 3: Out of Memory (OOM)

**Symptoms:**
- Space crashes
- "Container killed" in logs

**Solutions:**

1. **Disable ML Models:**
   ```bash
   # In Space secrets
   AGRISENSE_DISABLE_ML=1
   ```

2. **Reduce Workers:**
   ```bash
   WORKERS=1
   CELERY_WORKERS=1
   ```

3. **Optimize ML Models:**
   - Use TensorFlow Lite (4x smaller)
   - Lazy load models
   - Implement model caching

### Issue 4: Celery Not Starting

**Symptoms:**
- "Celery worker may not have started" in logs
- Background tasks not processing

**Solutions:**

1. **Check Redis Connection:**
   ```bash
   # Test Redis URL
   redis-cli -u "your-redis-url" ping
   # Should return: PONG
   ```

2. **Verify celery_config.py:**
   Ensure file exists at `agrisense_app/backend/celery_config.py`

3. **Check Celery Logs:**
   ```bash
   # Inside Space container (via SSH or logs)
   cat /home/agrisense/app/celery_logs/worker.log
   ```

### Issue 5: CORS Errors

**Symptoms:**
- Frontend can't call backend APIs
- Browser console shows CORS errors

**Solutions:**

1. **Add CORS Origins:**
   Update `main.py`:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Or specify Space URL
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Check Space URL:**
   Ensure frontend API calls use relative paths (`/api/...`) or correct Space URL.

---

## 💰 Cost Breakdown

| Service | Tier | RAM/Storage | Requests/Month | Cost |
|---------|------|-------------|----------------|------|
| **Hugging Face Spaces** | CPU Basic | 16GB | Unlimited | **$0** |
| **MongoDB Atlas** | M0 Sandbox | 512MB | Unlimited | **$0** |
| **Upstash Redis** | Free | 10K commands/day | ~300K/month | **$0** |
| **Total** | | | | **$0/month** |

### Cost Scaling (If Needed)

| Component | Free Limit | Upgrade Option | Upgrade Cost |
|-----------|-----------|----------------|--------------|
| HF Spaces | 16GB RAM | CPU Upgrade (32GB) | $0.60/hour |
| MongoDB | 512MB | M2 (2GB) | $9/month |
| Redis | 10K/day | 100K/day | $10/month |

**Estimated Costs at Scale:**
- **Low Traffic (<10K users/month):** FREE
- **Medium Traffic (10-50K users):** $10-20/month
- **High Traffic (>50K users):** $50-100/month

---

## ⚡ Performance Optimization

### 1. Lazy Load ML Models

Update `agrisense_app/backend/ml/model_loader.py`:

```python
from functools import lru_cache
import tensorflow as tf

@lru_cache(maxsize=3)
def load_model(model_name: str):
    """Load models on-demand, cache up to 3 models"""
    model_path = f"../../ml_models/{model_name}"
    return tf.keras.models.load_model(model_path)

# Usage in API endpoints
@app.post("/api/predict/disease")
async def predict_disease(image: UploadFile):
    model = load_model("disease_detection")
    # Run inference...
```

### 2. Use TensorFlow Lite

Convert models to TFLite for 4x size reduction:

```python
# Convert Keras model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

# Load and use TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
```

### 3. Enable Response Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

# Cache expensive computations
@lru_cache(maxsize=128)
def get_weather_forecast(location: str, date: str):
    # Expensive API call
    return fetch_weather(location, date)

# Cache with TTL using Redis
async def cached_prediction(sensor_id: str):
    cache_key = f"prediction:{sensor_id}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = await run_ml_inference(sensor_id)
    await redis_client.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
    return result
```

### 4. Database Indexing

Add indexes to MongoDB collections:

```python
# In startup event
@app.on_event("startup")
async def create_indexes():
    db = motor_client.agrisense
    
    # Sensor readings
    await db.sensor_readings.create_index([("device_id", 1), ("timestamp", -1)])
    
    # User authentication
    await db.users.create_index([("email", 1)], unique=True)
    
    # Recommendations
    await db.recommendations.create_index([("field_id", 1), ("created_at", -1)])
```

### 5. Frontend Optimization

**Vite Build Optimization:**

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'ui-vendor': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
          'chart-vendor': ['recharts', 'three'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
      },
    },
  },
});
```

---

## 📊 Monitoring

### Application Metrics

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Middleware to track metrics
@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    http_requests_total.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    http_request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### Health Dashboard

Create a simple health dashboard:

```python
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": await check_mongodb(),
            "redis": await check_redis(),
            "celery": await check_celery(),
        },
        "metrics": {
            "uptime": get_uptime(),
            "memory_usage": psutil.virtual_memory().percent,
            "cpu_usage": psutil.cpu_percent(),
        },
    }
```

---

## 🎯 Next Steps

### After Successful Deployment

1. **Custom Domain (Optional)**
   - Go to Space settings → "Domains"
   - Add your custom domain (requires Pro account)

2. **Enable Analytics**
   - Install Google Analytics or Plausible
   - Track user engagement

3. **Set Up Monitoring**
   - Integrate with Sentry for error tracking
   - Set up uptime monitoring (UptimeRobot)

4. **Continuous Deployment**
   - Set up GitHub Actions to auto-deploy on push
   - Example workflow in `.github/workflows/deploy.yml`

5. **Backup Strategy**
   - Schedule MongoDB backups (use Atlas built-in)
   - Export Redis data periodically

---

## 📚 Additional Resources

- **Hugging Face Spaces Docs:** https://huggingface.co/docs/hub/spaces-overview
- **MongoDB Atlas Docs:** https://www.mongodb.com/docs/atlas/
- **Upstash Redis Docs:** https://docs.upstash.com/redis
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Celery Docs:** https://docs.celeryq.dev/

---

## 🆘 Support

If you encounter issues:

1. **Check Logs:** Space logs tab
2. **GitHub Issues:** [Create an issue](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/issues)
3. **Hugging Face Forum:** [Community support](https://discuss.huggingface.co/)

---

## 📝 License

AgriSense is licensed under the Apache-2.0 License.

---

**🎉 Congratulations! Your AgriSense application is now running on Hugging Face Spaces for FREE!**

**Live URL:** `https://<your-username>-agrisense-app.hf.space`

**Estimated Deployment Time:** 30-45 minutes (including service setup)
