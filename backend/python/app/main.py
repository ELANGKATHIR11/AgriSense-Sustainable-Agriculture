import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.v1.router import api_router

# Create database tables
Base.metadata.create_all(bind=engine)

from contextlib import asynccontextmanager
from app.services.telemetry_service import telemetry_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    await telemetry_service.start()
    yield
    await telemetry_service.stop()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AgriSense Offline Backend & ML Inference Engine with NVIDIA RTX 5060 dGPU Acceleration",
    lifespan=lifespan
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time:.2f}"
    return response

# Mount routers at both /api and /api/v1 for 100% frontend compatibility
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "service": "AgriSense Offline Core Backend Engine",
        "version": settings.VERSION,
        "status": "online",
        "device": settings.DEVICE,
        "gpu": settings.GPU_NAME,
        "cuda_active": settings.CUDA_AVAILABLE,
        "docs_url": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": "connected",
        "hardware_acceleration": settings.GPU_NAME,
        "device": settings.DEVICE
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting AgriSense Backend on port {settings.FASTAPI_PORT} with {settings.GPU_NAME}...")
    uvicorn.run("app.main:app", host=settings.FASTAPI_HOST, port=settings.FASTAPI_PORT, reload=True)
