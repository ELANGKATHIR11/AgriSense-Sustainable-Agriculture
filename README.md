# AgriSense — Smart Agriculture & IoT Platform

Production-grade smart agriculture, precision irrigation, plant disease diagnostics, digital twin analytics, and IoT monitoring platform with native 7-language Indian localization.

## Architecture

- **Frontend**: React 19 + TypeScript + Tailwind CSS + Lucide Icons + Three.js
- **Gateway**: Node.js / Express API Gateway (Port 3000)
- **AI / ML Backend**: Python 3.11/3.12 FastAPI + PyTorch + CUDA acceleration on NVIDIA RTX 5060 dGPU (Port 8004)
- **Local AI & ML**:
  - Plant Disease Vision CNN: PyTorch MobileNetV3 trained on RTX 5060 dGPU
  - Crop Yield Predictor: Gradient Boosted Regressor on Indian Crop Yields
  - Crop Recommendation: Random Forest NPK & Climate Classifier
  - Agricultural Advisory: Local Vectorized Knowledge Retrieval (FAISS) based on Indian Agriculture Expert Q&A

## Quick Start (Offline Mode)

### 1. Python ML & Agronomy Backend
```bash
cd backend/python
python -m uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

### 2. Node.js Gateway & Frontend
```bash
npm install
npm run dev
```

Visit `http://localhost:3000` to access the full AgriSense platform.
