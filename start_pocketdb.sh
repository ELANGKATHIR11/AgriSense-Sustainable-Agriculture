#!/bin/bash
# AgriSense Full-Stack Startup with PocketDB
# Works on Linux and macOS

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_section() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Main startup
print_header "AgriSense Full-Stack with PocketDB"

# Check virtual environment
print_section "Checking Python Virtual Environment"
if [ ! -f ".venv/bin/activate" ]; then
    print_error "Virtual environment not found"
    echo "Please run: python3 -m venv .venv"
    exit 1
fi
print_success "Virtual environment found"

# Activate virtual environment
print_section "Activating Virtual Environment"
source .venv/bin/activate
print_success "Virtual environment activated"

# Set environment variables
print_section "Setting Environment Variables"
export AGRISENSE_DB_BACKEND=pocketdb
export POCKETDB_URL=http://localhost:8090
export POCKETDB_DATA_DIR=./pb_data
export VITE_API_URL=http://localhost:8000
export FASTAPI_ENV=development
export LOG_LEVEL=INFO

# Create pb_data directory if it doesn't exist
if [ ! -d "pb_data" ]; then
    mkdir -p pb_data
    print_success "Created pb_data directory"
fi
print_success "Environment variables set"

# Load .env.pocketdb if it exists
if [ -f ".env.pocketdb" ]; then
    print_section "Loading Configuration from .env.pocketdb"
    set -a
    source .env.pocketdb
    set +a
    print_success "Configuration loaded"
fi

# Print startup info
print_header "AgriSense Services"
echo ""
echo "Database Backend:   $AGRISENSE_DB_BACKEND"
echo "Backend Port:       8000"
echo "Frontend Port:      5173"
echo "PocketDB URL:       http://localhost:8090"
echo ""
echo -e "${YELLOW}Service URLs:${NC}"
echo -e "  • Backend API:     ${BLUE}http://localhost:8000${NC}"
echo -e "  • API Docs:        ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  • PocketDB Admin:  ${BLUE}http://localhost:8090/_/${NC}"
echo -e "  • Frontend:        ${BLUE}http://localhost:5173${NC}"
echo ""

# Initialize PocketDB collections
print_section "Initializing PocketDB Collections"
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'AGRISENSEFULL-STACK')

async def init():
    try:
        from agrisense_app.backend.database import init_database
        db = await init_database('pocketdb')
        stats = await db.get_stats()
        print('✓ Database initialized successfully')
        await db.close()
    except Exception as e:
        print(f'⚠ Warning: Could not verify PocketDB: {e}')
        print('Make sure PocketDB is running at http://localhost:8090')

asyncio.run(init())
" || true

# Start backend
print_header "Starting Backend Server"
print_section "FastAPI Backend"
print_info "Backend will start on port 8000"
print_info "Press Ctrl+C to stop the backend"
print_info "API Documentation: http://localhost:8000/docs"
echo ""

cd AGRISENSEFULL-STACK/src/backend || exit 1
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
