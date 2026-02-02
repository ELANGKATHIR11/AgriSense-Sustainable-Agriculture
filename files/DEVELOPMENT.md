# AgriSense Development Guide

## 🛠️ Development Setup

### Prerequisites

Before starting development, ensure you have:

- **Python 3.12.10** or higher
- **Node.js 20.x LTS** or higher
- **Git** for version control
- **Visual Studio Code** (recommended) or your preferred IDE
- **Postman** or similar for API testing (optional)

### Initial Setup

1. **Clone the Repository**

```bash
git clone <repository-url>
cd AgriSense
```

2. **Backend Development Setup**

```bash
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black flake8

# Initialize database
python -c "from core.data_store import init_sensor_db; init_sensor_db()"
```

3. **Frontend Development Setup**

```bash
cd frontend

# Install dependencies
npm install

# Install development tools
npm install -D @types/node
```

### Running in Development Mode

**Backend (Terminal 1):**
```bash
cd backend
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uvicorn main:app --reload --port 8000
```

**Frontend (Terminal 2):**
```bash
cd frontend
npm run dev
```

Access:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## 📂 Project Structure Explained

### Backend Structure

```
backend/
├── main.py                 # FastAPI app entry point
├── api/                    # API route modules
│   ├── sensor_api.py       # Sensor data endpoints
│   └── ai_routes.py        # AI/ML endpoints
├── core/                   # Core business logic
│   ├── engine.py           # Recommendation engine
│   ├── data_store.py       # Database operations
│   └── config.yaml         # Crop configuration
├── routes/                 # Additional route modules
│   ├── ml_predictions.py   # ML prediction endpoints
│   ├── health_routes.py    # System health checks
│   └── admin_routes.py     # Admin endpoints
├── requirements.txt        # Python dependencies
└── sensors.db              # SQLite database (auto-generated)
```

### Frontend Structure

```
frontend/
├── src/
│   ├── pages/              # Page components
│   │   ├── Home.tsx        # Landing page
│   │   ├── Dashboard.tsx   # Real-time monitoring
│   │   ├── Chatbot.tsx     # AI assistant
│   │   ├── Crops.tsx       # Crop recommendations
│   │   ├── DiseaseManagement.tsx
│   │   ├── WeedManagement.tsx
│   │   ├── Irrigation.tsx
│   │   └── Admin.tsx       # Admin dashboard
│   ├── components/         # Reusable components
│   │   └── Layout.tsx      # App layout with navigation
│   ├── services/           # API communication
│   │   └── api.ts          # API service layer
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   ├── index.css           # Global styles
│   └── i18n.ts             # Internationalization
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

---

## 🔧 Development Workflow

### Adding a New API Endpoint

1. **Create endpoint in appropriate route file:**

```python
# backend/routes/ml_predictions.py

@router.post("/new-feature")
async def new_feature(input_data: InputModel):
    """New feature endpoint"""
    try:
        # Your logic here
        result = process_data(input_data)
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

2. **Add to main.py if new router:**

```python
# backend/main.py
from routes.new_router import router as new_router
app.include_router(new_router, prefix="/new", tags=["New Feature"])
```

3. **Update API service in frontend:**

```typescript
// frontend/src/services/api.ts
export const apiService = {
  // ... existing methods
  newFeature: (data: any) => api.post('/new-feature', data),
}
```

### Adding a New Frontend Page

1. **Create page component:**

```tsx
// frontend/src/pages/NewPage.tsx
import { useState } from 'react'

const NewPage = () => {
  return (
    <div className="max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">New Feature</h1>
      {/* Your content */}
    </div>
  )
}

export default NewPage
```

2. **Add route in App.tsx:**

```tsx
// frontend/src/App.tsx
import NewPage from './pages/NewPage'

<Route path="/new-page" element={<NewPage />} />
```

3. **Add navigation link in Layout.tsx:**

```tsx
{ path: '/new-page', label: 'New Feature', icon: Star }
```

---

## 🧪 Testing

### Backend Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html

# View coverage report
open htmlcov/index.html  # or start htmlcov/index.html on Windows
```

### Frontend Testing

```bash
# Run tests
npm run test

# Run with coverage
npm run test:coverage
```

### Manual API Testing

Use the interactive API documentation:
- Visit: http://localhost:8000/docs
- Try out endpoints directly in the browser
- View request/response schemas

---

## 🐛 Debugging

### Backend Debugging

1. **Enable Debug Logging:**

```python
# backend/main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Use Python debugger:**

```python
import pdb; pdb.set_trace()  # Add breakpoint
```

3. **Check logs:**

```bash
# Backend logs are printed to console
# Check for errors and warnings
```

### Frontend Debugging

1. **React DevTools:**
   - Install React DevTools browser extension
   - Inspect component props and state

2. **Console Logging:**

```tsx
console.log('Debug data:', data)
console.error('Error:', error)
```

3. **Network Tab:**
   - Open browser DevTools (F12)
   - Check Network tab for API calls
   - Verify request/response payloads

---

## 🔐 Environment Variables

### Backend `.env`

```env
# Development
AGRISENSE_DISABLE_ML=0
DEBUG=true
LOG_LEVEL=DEBUG

# Database
DATABASE_URL=sqlite:///./sensors.db

# Optional Services
REDIS_URL=redis://localhost:6379/0
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883

# API Keys (for external services)
OPENWEATHER_API_KEY=
OPENAI_API_KEY=
```

### Frontend `.env`

```env
# API Configuration
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws

# Feature Flags
VITE_ENABLE_3D=true
VITE_ENABLE_OFFLINE=false

# Localization
VITE_DEFAULT_LANGUAGE=en
```

---

## 📝 Code Style Guidelines

### Python (Backend)

- Follow **PEP 8** style guide
- Use **type hints** for function parameters
- Write **docstrings** for all functions
- Format code with **Black**:

```bash
black backend/ --line-length 100
```

- Lint with **Flake8**:

```bash
flake8 backend/ --max-line-length 100
```

### TypeScript (Frontend)

- Use **TypeScript** strict mode
- Follow **React best practices**
- Use **functional components** with hooks
- Format code with **Prettier** (if configured)

```bash
npm run lint
```

---

## 🚀 Deployment

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

### Production Build

**Backend:**
```bash
# Install production dependencies only
pip install -r requirements.txt --no-dev

# Run with Gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Frontend:**
```bash
# Build for production
npm run build

# Serve with static server
npm run preview
# or deploy dist/ folder to CDN/hosting
```

---

## 🤝 Contributing Guidelines

1. **Fork** the repository
2. Create a **feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**

### Commit Message Convention

```
feat: Add new crop recommendation algorithm
fix: Resolve sensor data parsing issue
docs: Update API documentation
style: Format code with Black
refactor: Optimize database queries
test: Add unit tests for water optimization
chore: Update dependencies
```

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/
- **TailwindCSS Docs**: https://tailwindcss.com/docs
- **TypeScript Handbook**: https://www.typescriptlang.org/docs/

---

## 🆘 Troubleshooting

### Common Issues

**1. Port Already in Use**

```bash
# Find process using port
# Linux/Mac:
lsof -i :8000
# Windows:
netstat -ano | findstr :8000

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

**2. Module Not Found**

```bash
# Backend
pip install -r requirements.txt

# Frontend
npm install
```

**3. Database Locked**

```bash
# Remove database file and reinitialize
rm backend/sensors.db
python -c "from core.data_store import init_sensor_db; init_sensor_db()"
```

**4. CORS Errors**

- Check that backend CORS settings allow frontend origin
- Verify `VITE_API_BASE_URL` in frontend `.env`

---

Happy coding! 🚀
