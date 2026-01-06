#!/bin/bash
# AgriSense Local-First Backend Quick Start
# This script sets up everything locally and prepares for Firebase deployment

set -e

echo "🚀 AgriSense Local-First Backend Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Node.js
echo "📋 Checking Node.js..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not found. Please install Node.js 16+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node -v)${NC}"

# Check npm
echo "📋 Checking npm..."
if ! command -v npm &> /dev/null; then
    echo -e "${RED}✗ npm not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ npm $(npm -v)${NC}"

# Install PouchDB server dependencies
echo ""
echo "📦 Installing PouchDB Server dependencies..."
npm install express cors pouchdb body-parser

# Install Frontend dependencies
echo ""
echo "📦 Installing Frontend dependencies..."
cd src/frontend
npm install pouchdb pouchdb-adapter-idb firebase

# Create .env.local for frontend
echo ""
echo "🔧 Creating frontend environment file..."
cat > .env.local << 'EOF'
VITE_POUCHDB_SERVER_URL=http://localhost:5984
VITE_BACKEND_API_URL=http://localhost:8004/api/v1
VITE_BACKEND_WS_URL=ws://localhost:8004
VITE_ENABLE_OFFLINE_MODE=true
VITE_ENABLE_POUCHDB_SYNC=true
VITE_LOG_LEVEL=info
EOF
echo -e "${GREEN}✓ Created .env.local${NC}"

# Go back to root
cd ../..

# Check if Firebase CLI is installed
echo ""
echo "📋 Checking Firebase CLI..."
if ! command -v firebase &> /dev/null; then
    echo -e "${YELLOW}⚠ Firebase CLI not found. Installing...${NC}"
    npm install -g firebase-tools
fi
echo -e "${GREEN}✓ Firebase CLI installed${NC}"

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "📚 Next steps:"
echo "=============="
echo ""
echo "1. Start PouchDB Backend (Terminal 1):"
echo -e "   ${YELLOW}node pouchdb-server.js${NC}"
echo ""
echo "2. Start FastAPI Backend (Terminal 2):"
echo -e "   ${YELLOW}cd src/backend && python -m uvicorn main:app --reload --port 8004${NC}"
echo ""
echo "3. Start Frontend Development Server (Terminal 3):"
echo -e "   ${YELLOW}cd src/frontend && npm run dev${NC}"
echo ""
echo "4. Access the application:"
echo -e "   ${YELLOW}http://localhost:5173${NC}"
echo ""
echo "5. For Firebase deployment:"
echo -e "   ${YELLOW}firebase login${NC}"
echo -e "   ${YELLOW}cd src/frontend && npm run build${NC}"
echo -e "   ${YELLOW}firebase deploy --only hosting${NC}"
echo ""
echo "📖 See FIREBASE_POUCHDB_DEPLOYMENT.md for detailed instructions"
