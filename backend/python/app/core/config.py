import os
from typing import List
import torch
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../.env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "../../../../.env"))
load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "AgriSense Backend Engine"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server ports
    FASTAPI_PORT: int = 8004
    FASTAPI_HOST: str = "0.0.0.0"
    
    # Hardware acceleration
    CUDA_AVAILABLE: bool = torch.cuda.is_available()
    DEVICE: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    GPU_NAME: str = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    
    # Database URL: uses PostgreSQL with user credentials, or fallback if configured
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:Akilaarasu1%21@localhost:5432/agrisense"
    )
    
    # Ollama Local LLM Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
    OLLAMA_ENABLED: bool = os.getenv("OLLAMA_ENABLED", "true").lower() in ("true", "1", "yes")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Dataset and model paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    DATASETS_DIR: str = os.path.join(BASE_DIR, "datasets")
    KNOWLEDGE_DIR: str = os.path.join(BASE_DIR, "knowledge")
    
    model_config = {"case_sensitive": True}

settings = Settings()
