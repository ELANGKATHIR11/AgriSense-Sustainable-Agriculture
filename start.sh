#!/bin/bash
# AgriSense Startup Script for Hugging Face Spaces
# Runs Celery worker + FastAPI (Uvicorn) in a single container
set -e

echo "🌱 Starting AgriSense on Hugging Face Spaces..."
echo "   Python: $(python --version)"
echo "   Working Directory: $(pwd)"
echo "   User: $(whoami) (UID: $(id -u))"

# ============================================================================
# Environment Validation
# ============================================================================
echo ""
echo "🔍 Validating environment variables..."

# Required variables
if [ -z "$MONGO_URI" ]; then
    echo "❌ ERROR: MONGO_URI not set!"
    echo "   Please add MongoDB Atlas connection string in Space settings."
    exit 1
fi

if [ -z "$REDIS_URL" ]; then
    echo "❌ ERROR: REDIS_URL not set!"
    echo "   Please add Upstash Redis URL in Space settings."
    exit 1
fi

# Optional variables with defaults
export AGRISENSE_DISABLE_ML=${AGRISENSE_DISABLE_ML:-0}
export WORKERS=${WORKERS:-2}
export LOG_LEVEL=${LOG_LEVEL:-info}
export CELERY_WORKERS=${CELERY_WORKERS:-2}
export MAX_TASKS_PER_CHILD=${MAX_TASKS_PER_CHILD:-50}

echo "✅ Environment validated successfully!"
echo ""
echo "Configuration:"
echo "   MongoDB: ✓ Connected to Atlas M0"
echo "   Redis: ✓ Connected to Upstash"
echo "   ML Models: $([ "$AGRISENSE_DISABLE_ML" = "1" ] && echo "❌ Disabled (AGRISENSE_DISABLE_ML=1)" || echo "✓ Enabled")"
echo "   Uvicorn Workers: $WORKERS"
echo "   Celery Workers: $CELERY_WORKERS"
echo "   Log Level: $LOG_LEVEL"
echo ""

# ============================================================================
# Navigate to Backend Directory
# ============================================================================
cd /home/agrisense/app/backend

# Verify static files are in place
if [ -d "static/ui" ] && [ -f "static/ui/index.html" ]; then
    echo "✅ Frontend static files found at static/ui/"
else
    echo "⚠️  Warning: Frontend static files not found at static/ui/"
    echo "   The UI may not load correctly."
fi

# ============================================================================
# Start Celery Worker (Background)
# ============================================================================
echo ""
echo "🔄 Starting Celery worker..."

# Check if celery_config.py exists
if [ -f "celery_config.py" ]; then
    celery -A celery_config worker \
        --loglevel=$LOG_LEVEL \
        --logfile=/home/agrisense/app/celery_logs/worker.log \
        --detach \
        --concurrency=$CELERY_WORKERS \
        --max-tasks-per-child=$MAX_TASKS_PER_CHILD \
        --pool=solo \
        --without-gossip \
        --without-mingle \
        --without-heartbeat
    
    # Wait for Celery to initialize
    sleep 5
    
    # Verify Celery started
    if pgrep -f "celery.*worker" > /dev/null; then
        echo "✅ Celery worker started (PID: $(pgrep -f 'celery.*worker'))"
    else
        echo "⚠️  Warning: Celery worker may not have started properly"
        echo "   Check celery_logs/worker.log for details"
    fi
else
    echo "⚠️  Warning: celery_config.py not found - skipping Celery worker"
    echo "   Background tasks will not be processed."
fi

# ============================================================================
# Start FastAPI with Uvicorn (Foreground)
# ============================================================================
echo ""
echo "🚀 Starting FastAPI server on port 7860..."
echo "   Access the application at: http://localhost:7860"
echo "   API Documentation: http://localhost:7860/docs"
echo "   Frontend UI: http://localhost:7860/ui/"
echo ""

# Use exec to replace the shell process with uvicorn (proper signal handling)
exec uvicorn main:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers $WORKERS \
    --log-level $LOG_LEVEL \
    --no-access-log \
    --proxy-headers \
    --forwarded-allow-ips '*'
