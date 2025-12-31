# 🚀 Hugging Face Spaces Deployment Checklist

**Quick reference for deploying AgriSense to Hugging Face Spaces**

---

## ✅ Pre-Deployment Checklist

### 1. External Services Setup

- [ ] **MongoDB Atlas Account Created**
  - URL: https://www.mongodb.com/cloud/atlas/register
  - Cluster created (M0 free tier)
  - Database user created with read/write permissions
  - Network access: 0.0.0.0/0 (allow from anywhere)
  - Connection string obtained and tested

- [ ] **Upstash Redis Account Created**
  - URL: https://console.upstash.com/
  - Database created (free tier)
  - Redis URL obtained (redis://default:password@host:6379)
  - Connection tested with redis-cli

- [ ] **Hugging Face Account Created**
  - URL: https://huggingface.co/join
  - Profile complete
  - Ready to create Space

### 2. Environment Variables Prepared

- [ ] `MONGO_URI` - Format: `mongodb+srv://user:pass@cluster.mongodb.net/agrisense`
- [ ] `REDIS_URL` - Format: `redis://default:pass@host:6379`
- [ ] `AGRISENSE_ADMIN_TOKEN` - Generated secure token (32+ characters)
- [ ] Optional: `OPENAI_API_KEY` (for GPT chatbot)
- [ ] Optional: `GEMINI_API_KEY` (for Gemini features)

**Generate Admin Token:**
```bash
python -c "import secrets; print('sk-agrisense-' + secrets.token_urlsafe(32))"
```

### 3. Local Testing (Optional but Recommended)

- [ ] Docker installed and running
- [ ] Build test passed:
  ```bash
  docker build -f Dockerfile.huggingface -t agrisense-test .
  ```
- [ ] Local run test passed:
  ```bash
  docker run -p 7860:7860 \
    -e MONGO_URI="your-uri" \
    -e REDIS_URL="your-redis-url" \
    -e AGRISENSE_ADMIN_TOKEN="test-token" \
    agrisense-test
  ```
- [ ] Health endpoint responds: `curl http://localhost:7860/health`
- [ ] Frontend loads: http://localhost:7860/ui/
- [ ] API docs accessible: http://localhost:7860/docs

---

## 📦 Deployment Steps

### Step 1: Create Hugging Face Space

- [ ] Go to https://huggingface.co/new-space
- [ ] Space name: `agrisense-app` (or your choice)
- [ ] SDK: **Docker**
- [ ] Hardware: **CPU basic (FREE)** - 16GB RAM
- [ ] Visibility: Public or Private
- [ ] Space created successfully

### Step 2: Prepare Files

**Manual Method:**
- [ ] Clone Space repo: `git clone https://huggingface.co/spaces/<username>/agrisense-app`
- [ ] Copy `Dockerfile.huggingface` as `Dockerfile`
- [ ] Copy `start.sh`
- [ ] Copy `agrisense_app/` directory
- [ ] Copy `ml_models/` directory (if exists)
- [ ] Create `README.md` for Space

**Automated Method:**
- [ ] Run script: `bash deploy_to_huggingface.sh agrisense-app <username>`
- [ ] Review generated files in `hf_space_temp/`

### Step 3: Configure Space Secrets

- [ ] Go to Space Settings → Variables and secrets
- [ ] Add **Secret**: `MONGO_URI` = `mongodb+srv://...`
- [ ] Add **Secret**: `REDIS_URL` = `redis://...`
- [ ] Add **Secret**: `AGRISENSE_ADMIN_TOKEN` = `sk-agrisense-...`
- [ ] Add **Variable** (optional): `AGRISENSE_DISABLE_ML` = `0`
- [ ] Add **Variable** (optional): `WORKERS` = `2`
- [ ] Add **Variable** (optional): `LOG_LEVEL` = `info`

### Step 4: Push to Hugging Face

```bash
cd hf_space_temp  # or your Space repo directory
git add .
git commit -m "Deploy AgriSense to Hugging Face Spaces"
git push origin main
```

- [ ] Files pushed successfully
- [ ] Build started (visible in Space UI)

### Step 5: Monitor Build

- [ ] Go to Space URL: `https://huggingface.co/spaces/<username>/agrisense-app`
- [ ] Click "Logs" tab
- [ ] Watch for successful stages:
  - [ ] Frontend build (Node 18)
  - [ ] Backend dependencies install (Python 3.12)
  - [ ] Container assembly
  - [ ] Health check passed
  - [ ] Celery worker started
  - [ ] Uvicorn started on port 7860
- [ ] Status changes to "Running" (build time: ~10-15 minutes)

---

## 🧪 Post-Deployment Verification

### Basic Health Checks

- [ ] **Health Endpoint**
  ```bash
  curl https://<username>-agrisense-app.hf.space/health
  ```
  Expected: `{"status": "healthy", ...}`

- [ ] **API Documentation**
  - URL: `https://<username>-agrisense-app.hf.space/docs`
  - Swagger UI loads correctly
  - All endpoints visible

- [ ] **Frontend UI**
  - URL: `https://<username>-agrisense-app.hf.space/ui/`
  - Dashboard loads
  - No console errors
  - Assets load correctly

### Functional Tests

- [ ] **Database Connection**
  - Submit test sensor reading via API
  - Verify data saved in MongoDB Atlas
  - Check database collections

- [ ] **Redis Connection**
  - Submit background task (if applicable)
  - Verify Celery processes task
  - Check Redis commands count in Upstash

- [ ] **Authentication**
  - Test user registration endpoint
  - Test user login endpoint
  - Verify JWT token generation

- [ ] **ML Models (if enabled)**
  - Test disease detection endpoint
  - Test recommendation endpoint
  - Check inference time (<5 seconds)

### Performance Checks

- [ ] **Response Times**
  - Health endpoint: <200ms
  - API endpoints: <1s
  - Frontend load: <3s
  - ML inference: <5s

- [ ] **Memory Usage**
  - Check Space metrics in HF UI
  - Idle: ~2-4GB
  - Under load: <12GB
  - No OOM errors

- [ ] **Logs Quality**
  - No critical errors in logs
  - Celery worker logs present
  - Request logs readable
  - Error messages informative

---

## 🔧 Troubleshooting Checklist

### Build Failed

- [ ] Check Dockerfile syntax
- [ ] Verify all COPY paths exist
- [ ] Check requirements.txt for conflicts
- [ ] Review build logs for specific error
- [ ] Test build locally with Docker

### Container Won't Start

- [ ] Verify secrets are set correctly
- [ ] Check MongoDB URI format and credentials
- [ ] Check Redis URL format and credentials
- [ ] Look for Python import errors in logs
- [ ] Check file permissions (UID 1000)

### Frontend 404 Error

- [ ] Verify frontend built successfully
- [ ] Check `static/ui/` directory exists in container
- [ ] Verify vite.config.ts has `base: '/ui/'`
- [ ] Check main.py mounts static files correctly
- [ ] Review Dockerfile frontend COPY command

### Out of Memory

- [ ] Set `AGRISENSE_DISABLE_ML=1`
- [ ] Reduce `WORKERS=1`
- [ ] Reduce `CELERY_WORKERS=1`
- [ ] Optimize model loading (lazy load)
- [ ] Convert models to TensorFlow Lite
- [ ] Check for memory leaks

### Celery Not Working

- [ ] Verify Redis URL is correct
- [ ] Check Celery logs in `/home/agrisense/app/celery_logs/`
- [ ] Verify `celery_config.py` exists
- [ ] Test Redis connection manually
- [ ] Check Celery worker process in logs

### Slow Performance

- [ ] Enable Redis caching
- [ ] Add database indexes
- [ ] Optimize expensive queries
- [ ] Implement lazy model loading
- [ ] Use CDN for static assets
- [ ] Enable response compression

---

## 📊 Monitoring & Maintenance

### Regular Checks (Weekly)

- [ ] Check Space uptime status
- [ ] Review error logs for patterns
- [ ] Monitor MongoDB storage usage (512MB limit)
- [ ] Check Redis command count (10K/day limit)
- [ ] Review response times and performance

### Updates & Upgrades

- [ ] Update Python dependencies monthly
- [ ] Update Node dependencies monthly
- [ ] Test updates locally before deploying
- [ ] Monitor for security vulnerabilities
- [ ] Review and apply Hugging Face updates

### Backup Strategy

- [ ] Export MongoDB data weekly
- [ ] Backup environment variables
- [ ] Keep git history clean
- [ ] Document configuration changes

---

## 🎯 Success Criteria

Your deployment is successful when:

- [x] Health endpoint returns `{"status": "healthy"}`
- [x] Frontend UI loads without errors
- [x] API documentation is accessible
- [x] Database operations work correctly
- [x] Background tasks process via Celery
- [x] ML inference completes successfully
- [x] No critical errors in logs
- [x] Response times within acceptable range
- [x] Memory usage under 12GB

---

## 📝 Deployment Record

**Fill this out after successful deployment:**

- **Deployment Date:** _____________________
- **Space URL:** https://huggingface.co/spaces/_______/________
- **MongoDB Cluster:** _____________________
- **Redis Instance:** _____________________
- **Initial Build Time:** _____ minutes
- **First Response Time:** _____ ms
- **ML Models Enabled:** Yes / No
- **Issues Encountered:** _____________________
- **Resolution Notes:** _____________________

---

## 🆘 Quick Reference URLs

- **Space Dashboard:** `https://huggingface.co/spaces/<username>/<space-name>`
- **Space Settings:** `https://huggingface.co/spaces/<username>/<space-name>/settings`
- **Space Logs:** `https://huggingface.co/spaces/<username>/<space-name>/logs`
- **MongoDB Atlas:** https://cloud.mongodb.com/
- **Upstash Console:** https://console.upstash.com/
- **AgriSense GitHub:** https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK

---

## 📞 Support Resources

If you need help:

1. **Review Full Guide:** [HF_DEPLOYMENT_GUIDE.md](HF_DEPLOYMENT_GUIDE.md)
2. **Check Logs:** Space logs tab
3. **Search Issues:** [GitHub Issues](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/issues)
4. **Ask Community:** [Hugging Face Forum](https://discuss.huggingface.co/)

---

**🎉 Good luck with your deployment!**

_Estimated total setup time: 30-45 minutes (including service signups)_
