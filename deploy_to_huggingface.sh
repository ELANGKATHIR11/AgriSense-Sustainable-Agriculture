#!/bin/bash
# Quick deployment script for Hugging Face Spaces
# Usage: ./deploy_to_huggingface.sh <space-name> <username>

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <space-name> <username>"
    echo "Example: $0 agrisense-app myusername"
    exit 1
fi

SPACE_NAME=$1
USERNAME=$2
SPACE_URL="https://huggingface.co/spaces/$USERNAME/$SPACE_NAME"

echo -e "${GREEN}🚀 AgriSense Hugging Face Deployment Script${NC}"
echo "========================================"
echo "Space Name: $SPACE_NAME"
echo "Username: $USERNAME"
echo "Space URL: $SPACE_URL"
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: git is not installed${NC}"
    exit 1
fi

# Step 1: Clone or use existing Space repo
echo -e "${YELLOW}Step 1: Setting up Space repository...${NC}"
if [ -d "hf_space_temp" ]; then
    echo "Removing existing hf_space_temp directory..."
    rm -rf hf_space_temp
fi

git clone "https://huggingface.co/spaces/$USERNAME/$SPACE_NAME" hf_space_temp
cd hf_space_temp

# Step 2: Copy files
echo -e "${YELLOW}Step 2: Copying AgriSense files...${NC}"

# Copy Dockerfile
echo "  - Copying Dockerfile.huggingface as Dockerfile..."
cp ../Dockerfile.huggingface ./Dockerfile

# Copy startup script
echo "  - Copying start.sh..."
cp ../start.sh .
chmod +x start.sh

# Copy backend
echo "  - Copying backend..."
rm -rf agrisense_app
mkdir -p agrisense_app
cp -r ../agrisense_app/backend agrisense_app/

# Copy frontend source (will be built in Docker)
echo "  - Copying frontend source..."
mkdir -p agrisense_app/frontend
cp -r ../agrisense_app/frontend/farm-fortune-frontend-main agrisense_app/frontend/

# Copy ML models (if they exist)
if [ -d "../ml_models" ]; then
    echo "  - Copying ML models..."
    cp -r ../ml_models .
else
    echo "  - Warning: ml_models directory not found, creating empty directory..."
    mkdir -p ml_models
fi

# Step 3: Create README
echo -e "${YELLOW}Step 3: Creating README.md...${NC}"
cat > README.md << 'EOF'
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
- Real-time sensor monitoring (IoT)
- Crop disease detection (Computer Vision)
- Intelligent recommendations (ML)
- Chatbot assistance (NLP)

**Tech Stack:** FastAPI + React + PyTorch + TensorFlow + Celery

## 🔧 Configuration Required

Before the app will work, you must add these **Secrets** in Space Settings:

### Required Secrets
1. `MONGO_URI` - MongoDB Atlas connection string
   ```
   mongodb+srv://user:pass@cluster.mongodb.net/agrisense
   ```

2. `REDIS_URL` - Upstash Redis connection URL
   ```
   redis://default:pass@host:6379
   ```

3. `AGRISENSE_ADMIN_TOKEN` - Admin API token (generate random string)
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

### Optional Variables
- `AGRISENSE_DISABLE_ML=1` - Disable ML models to save RAM
- `WORKERS=2` - Number of Uvicorn workers
- `LOG_LEVEL=info` - Logging level

## 📚 Documentation

See [Full Deployment Guide](https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK/blob/main/HF_DEPLOYMENT_GUIDE.md)

## 🔗 Links

- **GitHub:** https://github.com/ELANGKATHIR11/AGRISENSEFULL-STACK
- **API Docs:** Access `/docs` after deployment
- **UI:** Access `/ui/` after deployment
EOF

# Step 4: Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo -e "${YELLOW}Step 4: Creating .gitignore...${NC}"
    cat > .gitignore << 'EOF'
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.log
.env
.env.*
*.db
*.sqlite
node_modules/
EOF
fi

# Step 5: Git operations
echo -e "${YELLOW}Step 5: Committing changes...${NC}"
git add .
git status

echo ""
echo -e "${YELLOW}Ready to deploy!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the files in hf_space_temp/"
echo "2. Commit and push:"
echo "   cd hf_space_temp"
echo "   git commit -m 'Deploy AgriSense to Hugging Face Spaces'"
echo "   git push origin main"
echo ""
echo "3. Go to $SPACE_URL/settings"
echo "   - Add secrets: MONGO_URI, REDIS_URL, AGRISENSE_ADMIN_TOKEN"
echo ""
echo "4. Wait for build (~10-15 minutes)"
echo "5. Access your app at: $SPACE_URL"
echo ""
echo -e "${GREEN}✅ Files prepared successfully!${NC}"
echo ""
read -p "Do you want to commit and push now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Committing and pushing...${NC}"
    git commit -m "Deploy AgriSense to Hugging Face Spaces"
    git push origin main
    echo -e "${GREEN}✅ Deployment initiated!${NC}"
    echo "Monitor build at: $SPACE_URL"
else
    echo "Commit manually when ready:"
    echo "  cd hf_space_temp"
    echo "  git commit -m 'Deploy AgriSense'"
    echo "  git push origin main"
fi
