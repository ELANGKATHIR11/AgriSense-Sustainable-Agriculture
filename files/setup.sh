#!/bin/bash
# AgriSense Setup Script for Linux/Mac

echo "🌾 AgriSense Setup Starting..."
echo "================================"

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "❌ Python 3.12+ required"; exit 1; }

# Check Node.js version
echo "Checking Node.js version..."
node --version || { echo "❌ Node.js 20+ required"; exit 1; }

# Backend Setup
echo ""
echo "📦 Setting up Backend..."
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Initialize database
python3 -c "from core.data_store import init_sensor_db; init_sensor_db()"

echo "✅ Backend setup complete!"

# Frontend Setup
echo ""
echo "🎨 Setting up Frontend..."
cd ../frontend

# Install dependencies
npm install

echo "✅ Frontend setup complete!"

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo ""
echo "To start the application:"
echo "1. Backend:  cd backend && source .venv/bin/activate && uvicorn main:app --reload"
echo "2. Frontend: cd frontend && npm run dev"
echo ""
echo "Access the app at: http://localhost:5173"
echo "API documentation: http://localhost:8000/docs"
