# 🔐 Environment Variables Reference Card

**Quick reference for AgriSense Hugging Face Spaces deployment**

---

## Required Secrets (MUST SET)

### `MONGO_URI`
**MongoDB Atlas connection string**

**Format:**
```
mongodb+srv://<username>:<password>@<cluster>.<id>.mongodb.net/<database>?retryWrites=true&w=majority
```

**Example:**
```
mongodb+srv://agrisense_user:SecurePass123@agrisense-cluster.abc123.mongodb.net/agrisense?retryWrites=true&w=majority
```

**How to Get:**
1. Go to MongoDB Atlas → Database → Connect
2. Choose "Connect your application"
3. Copy connection string
4. Replace `<password>` with your actual password

**Common Issues:**
- ❌ Forgot to replace `<password>`
- ❌ Special characters not URL-encoded
- ❌ Network access not set to 0.0.0.0/0
- ❌ Database user doesn't have read/write permissions

---

### `REDIS_URL`
**Upstash Redis connection URL**

**Format:**
```
redis://default:<password>@<host>:<port>
```

**Example:**
```
redis://default:AXylZDEyMzQ1Njc4OTBhYmNkZWY@us1-modern-firefly-12345.upstash.io:6379
```

**How to Get:**
1. Go to Upstash Console → Your Database
2. Scroll to "REST API" section
3. Copy "Redis URL"

**Common Issues:**
- ❌ Using REST URL instead of Redis URL
- ❌ Missing `default:` username prefix
- ❌ Wrong port number (should be 6379)

---

### `AGRISENSE_ADMIN_TOKEN`
**Admin API authentication token**

**Format:**
```
sk-agrisense-<random-string>
```

**Example:**
```
sk-agrisense-K7jP9mN2vL4xQ8wY5tR3sE1fG6hU0dI
```

**How to Generate:**

**Option 1: Python**
```bash
python -c "import secrets; print('sk-agrisense-' + secrets.token_urlsafe(32))"
```

**Option 2: OpenSSL**
```bash
echo "sk-agrisense-$(openssl rand -hex 32)"
```

**Option 3: Node.js**
```bash
node -e "console.log('sk-agrisense-' + require('crypto').randomBytes(32).toString('hex'))"
```

**Security Notes:**
- ✅ Minimum 32 characters
- ✅ Use letters, numbers, and symbols
- ✅ Generate new token for each environment
- ❌ Don't commit to git
- ❌ Don't share publicly

---

## Optional Variables (Recommended Defaults)

### `AGRISENSE_DISABLE_ML`
**Control ML model loading**

| Value | Behavior | Memory Usage | Use Case |
|-------|----------|--------------|----------|
| `0` (default) | ML models enabled | 6-8GB | Production with AI features |
| `1` | ML models disabled | 2-3GB | Testing, API-only, low memory |

**When to Disable:**
- Testing deployment without ML inference
- Memory constraints (<8GB)
- API-only functionality needed
- Faster startup time required

**Example:**
```bash
AGRISENSE_DISABLE_ML=1
```

---

### `WORKERS`
**Number of Uvicorn worker processes**

| Value | Concurrency | Memory | Use Case |
|-------|------------|--------|----------|
| `1` | Low | 1-2GB | Development, testing |
| `2` (default) | Medium | 2-4GB | Production (recommended) |
| `4` | High | 4-8GB | High traffic |

**Formula:** `Workers = (CPU cores / 2) + 1`

**Example:**
```bash
WORKERS=2
```

**Recommendations:**
- 16GB RAM → 2 workers (balanced)
- High traffic → 4 workers (more RAM)
- Memory constrained → 1 worker

---

### `CELERY_WORKERS`
**Number of Celery worker threads**

| Value | Background Tasks | Memory | Use Case |
|-------|-----------------|--------|----------|
| `1` | Minimal processing | 500MB | Low task volume |
| `2` (default) | Balanced | 1GB | Standard workload |
| `4` | High throughput | 2GB | Heavy ML inference |

**Example:**
```bash
CELERY_WORKERS=2
```

**Recommendations:**
- Few background tasks → 1 worker
- ML inference tasks → 2-4 workers
- High task volume → Scale workers and Redis

---

### `LOG_LEVEL`
**Application logging verbosity**

| Value | Output | Use Case |
|-------|--------|----------|
| `debug` | Very detailed | Development, troubleshooting |
| `info` (default) | Standard logs | Production (recommended) |
| `warning` | Warnings + errors | Production (quiet) |
| `error` | Errors only | Production (minimal) |

**Example:**
```bash
LOG_LEVEL=info
```

**Log Hierarchy:**
```
DEBUG → INFO → WARNING → ERROR → CRITICAL
(most)                            (least)
```

---

## Optional Secrets (Third-Party APIs)

### `OPENAI_API_KEY`
**OpenAI GPT API key for chatbot**

**Format:**
```
sk-proj-<rest-of-key>
```

**Example:**
```
sk-proj-abc123XYZ789defGHI456jklMNO
```

**How to Get:**
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Copy immediately (won't show again)

**Required For:**
- GPT-4 chatbot features
- Advanced NLP capabilities
- Text generation endpoints

**Cost:** Pay-as-you-go (~$0.03/1K tokens)

---

### `GEMINI_API_KEY`
**Google Gemini API key**

**Format:**
```
AIza<rest-of-key>
```

**Example:**
```
AIzaSyBCDEF1234567890abcdefGHIJKL
```

**How to Get:**
1. Go to https://makersuite.google.com/app/apikey
2. Create API key
3. Copy key

**Required For:**
- Gemini AI features
- Multi-modal AI capabilities
- Alternative to OpenAI

**Cost:** Free tier available

---

### `SENTRY_DSN`
**Sentry error tracking DSN**

**Format:**
```
https://<key>@<host>.ingest.sentry.io/<project>
```

**Example:**
```
https://abc123def456@o123456.ingest.sentry.io/7654321
```

**How to Get:**
1. Go to https://sentry.io/
2. Create project
3. Copy DSN from Settings → Client Keys

**Required For:**
- Error tracking and monitoring
- Performance monitoring
- Real-time alerts

**Cost:** Free tier (5K events/month)

---

## Advanced Configuration

### `ALLOWED_ORIGINS`
**CORS allowed origins (comma-separated)**

**Default:** `*` (allow all)

**Example:**
```bash
ALLOWED_ORIGINS=https://your-domain.com,https://app.your-domain.com
```

**Security:** Restrict in production to known domains

---

### `MAX_TASKS_PER_CHILD`
**Celery: Max tasks before worker restart**

**Default:** `50`

**Example:**
```bash
MAX_TASKS_PER_CHILD=100
```

**Purpose:** Prevent memory leaks in long-running workers

---

### `MONGO_DB_NAME`
**MongoDB database name**

**Default:** `agrisense`

**Example:**
```bash
MONGO_DB_NAME=agrisense_production
```

---

### `PORT`
**Application port (DO NOT CHANGE for HF Spaces)**

**Required:** `7860` (Hugging Face requirement)

**Example:**
```bash
PORT=7860
```

**Note:** Hugging Face Spaces MUST use port 7860

---

## Configuration Templates

### Minimal (Testing)
```bash
# Required
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/agrisense
REDIS_URL=redis://default:pass@host:6379
AGRISENSE_ADMIN_TOKEN=sk-agrisense-abc123

# Recommended for testing
AGRISENSE_DISABLE_ML=1
WORKERS=1
LOG_LEVEL=debug
```

### Standard (Production)
```bash
# Required
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/agrisense
REDIS_URL=redis://default:pass@host:6379
AGRISENSE_ADMIN_TOKEN=sk-agrisense-xyz789

# Optional (defaults work well)
AGRISENSE_DISABLE_ML=0
WORKERS=2
CELERY_WORKERS=2
LOG_LEVEL=info
```

### High Performance
```bash
# Required
MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/agrisense
REDIS_URL=redis://default:pass@host:6379
AGRISENSE_ADMIN_TOKEN=sk-agrisense-prod123

# Optimized for high traffic
AGRISENSE_DISABLE_ML=0
WORKERS=4
CELERY_WORKERS=4
LOG_LEVEL=warning
MAX_TASKS_PER_CHILD=100

# Optional APIs
OPENAI_API_KEY=sk-proj-xyz789
SENTRY_DSN=https://abc@o123.ingest.sentry.io/456
```

---

## How to Set in Hugging Face Spaces

### Method 1: Web UI (Recommended)

1. Go to your Space: `https://huggingface.co/spaces/<username>/<space-name>`
2. Click **"Settings"** tab
3. Scroll to **"Variables and secrets"**
4. Click **"New secret"** for sensitive values (MONGO_URI, REDIS_URL, tokens)
5. Click **"New variable"** for public values (LOG_LEVEL, WORKERS)
6. Enter **Name** and **Value**
7. Click **"Save"**
8. Space will automatically restart

### Method 2: Space Configuration File

Create `.env.example` (for documentation):
```bash
# Copy this to Space settings, don't commit with real values
MONGO_URI=your-mongodb-uri-here
REDIS_URL=your-redis-url-here
AGRISENSE_ADMIN_TOKEN=your-admin-token-here
```

**Note:** Never commit `.env` with real secrets!

---

## Validation Checklist

Before deploying, verify:

- [ ] MongoDB URI format is correct
- [ ] MongoDB user has read/write permissions
- [ ] MongoDB network access allows 0.0.0.0/0
- [ ] Redis URL format is correct
- [ ] Redis connection works (test with redis-cli)
- [ ] Admin token is at least 32 characters
- [ ] Admin token is securely generated
- [ ] Optional variables use recommended values
- [ ] No secrets committed to git
- [ ] All secrets added to Space settings

---

## Testing Variables Locally

```bash
# Create .env.local (don't commit!)
cat > .env.local << 'EOF'
MONGO_URI=mongodb+srv://test:pass@cluster.mongodb.net/test
REDIS_URL=redis://default:pass@localhost:6379
AGRISENSE_ADMIN_TOKEN=test-token-123
AGRISENSE_DISABLE_ML=1
WORKERS=1
LOG_LEVEL=debug
EOF

# Load and test
export $(cat .env.local | xargs)
python -c "import os; print('MONGO_URI:', os.getenv('MONGO_URI'))"

# Test with Docker
docker run -p 7860:7860 --env-file .env.local agrisense-test
```

---

## Troubleshooting

### "Could not connect to MongoDB"
→ Check MONGO_URI format
→ Verify password has no special characters or URL-encode them
→ Check network access in MongoDB Atlas

### "Redis connection refused"
→ Check REDIS_URL format
→ Verify port is 6379
→ Check Redis database is active in Upstash

### "Invalid token"
→ Verify AGRISENSE_ADMIN_TOKEN is set
→ Check token format (starts with sk-agrisense-)
→ Ensure token has no extra spaces

### "Out of memory"
→ Set AGRISENSE_DISABLE_ML=1
→ Reduce WORKERS to 1
→ Reduce CELERY_WORKERS to 1

---

## Security Best Practices

✅ **DO:**
- Use strong, random admin tokens (32+ chars)
- Store secrets in HF Space settings
- Rotate tokens periodically
- Use separate tokens for dev/prod
- URL-encode special characters in URIs

❌ **DON'T:**
- Commit secrets to git
- Share tokens publicly
- Use simple/guessable tokens
- Reuse tokens across environments
- Leave default passwords

---

## Quick Reference URLs

- **MongoDB Atlas:** https://cloud.mongodb.com/
- **Upstash Console:** https://console.upstash.com/
- **OpenAI Keys:** https://platform.openai.com/api-keys
- **Gemini Keys:** https://makersuite.google.com/app/apikey
- **Sentry Projects:** https://sentry.io/
- **HF Space Settings:** `https://huggingface.co/spaces/<username>/<space>/settings`

---

**Need more help?** See [HF_DEPLOYMENT_GUIDE.md](HF_DEPLOYMENT_GUIDE.md) for detailed setup instructions.

**Generated:** December 28, 2025
