import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from llm_engine import LLMEngine
from nlm_parser import NLMEngine
from ml_bridge import MLBridge
from optimizer import run_optimization
from optimizer import run_optimization
from risk_engine import get_risk_assessment

# AGRI-VLM-CARE+++ Integration
from fastapi import UploadFile, File
from PIL import Image
import io
import sys
import os

# Ensure backend path is in sys.path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.ml.agri_vlm_inference import get_vlm_inference
from backend.ai_service.agri_vlm_schema import VLMAnalysisResult

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgriSense-AI")

app = FastAPI(
    title="AgriSense Yield Guardian (Native GGUF)",
    description="Offline-first AI service using Native Quantized LLM (No Ollama).",
    version="2.0.0",
)

# CORS Input
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Engines ---
llm = None
nlm = None
ml_bridge = None
yield_model = None
yield_encoders = None


@app.on_event("startup")
async def startup_event():
    global llm, nlm, ml_bridge, yield_model, yield_encoders
    try:
        # Initialize LLM (Load GGUF into RAM)
        llm = LLMEngine()
        nlm = NLMEngine(llm)
        ml_bridge = MLBridge()

        # Load Stage 1: XGBoost Yield Model
        yield_model_path = os.path.join(
            os.path.dirname(__file__), "models", "yield_model.json"
        )
        yield_encoders_path = os.path.join(
            os.path.dirname(__file__), "models", "yield_encoders.joblib"
        )

        if os.path.exists(yield_model_path) and os.path.exists(yield_encoders_path):
            import xgboost as xgb
            import joblib

            yield_model = xgb.XGBRegressor()
            yield_model.load_model(yield_model_path)
            yield_encoders = joblib.load(yield_encoders_path)
            logger.info("✅ Stage 1: Yield Model (XGBoost) loaded.")
        else:
            logger.warning(
                "⚠️ Stage 1: Yield Model files missing. Run train_yield.py first."
            )

        logger.info("✅ AI Engines initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize AI engines: {e}")


# --- Data Models ---
class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = {}


class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    actions: List[Dict[str, Any]] = []
    data: Optional[Dict[str, Any]] = None


class OptimizationRequest(BaseModel):
    crop: str
    soil_data: Dict[str, float]
    area_acres: float


# --- Routes ---


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AgriSense Yield Guardian (Native)",
        "version": "2.0.0",
    }


@app.get("/health")
async def health_check():
    status = "healthy"
    model_status = "loaded" if (llm and llm.llm) else "unloaded"
    yield_status = "loaded" if yield_model else "unloaded"
    if model_status == "unloaded":
        status = "degraded"
    return {
        "status": status,
        "components": {
            "llm": model_status,
            "yield_core": yield_status,
            "ml_bridge": "active",
        },
    }


@app.post("/predict/yield")
async def predict_yield(request: Dict[str, Any]):
    """
    Stage 1: XGBoost Prediction
    """
    if not yield_model:
        raise HTTPException(status_code=503, detail="Yield model not loaded")

    try:
        import pandas as pd

        features = [
            "soil_n",
            "soil_p",
            "soil_k",
            "soil_ph",
            "organic_carbon",
            "rainfall_mm",
            "temperature_avg_c",
            "humidity_pct",
            "crop_name",
            "season",
        ]

        df_input = {}
        for f in features:
            val = request.get(f)
            if f in ["crop_name", "season"]:
                le = yield_encoders.get(f)
                try:
                    df_input[f] = [le.transform([val])[0]]
                except:
                    df_input[f] = [0]
            else:
                df_input[f] = [float(val if val is not None else 0)]

        X = pd.DataFrame(df_input)
        prediction = yield_model.predict(X)[0]
        return {"yield_kg_per_acre": float(prediction)}
    except Exception as e:
        logger.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize")
async def optimize_resources(request: OptimizationRequest):
    """
    Stage 2: NSGA-II Resource Optimization
    """
    if not yield_model:
        raise HTTPException(status_code=503, detail="Yield core not loaded")

    try:
        results = run_optimization(
            yield_model,
            yield_encoders,
            request.crop,
            "Kharif",  # Default
            request.soil_data,
        )
        return {"plans": results}
    except Exception as e:
        logger.error(f"Optimization Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/image", response_model=VLMAnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    AGRI-VLM-CARE+++: Analyze crop image for disease/weeds.
    Returns structured farmer-friendly advice.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        vlm = get_vlm_inference()
        result = vlm.predict(image)

        logger.info(f"VLM Analysis: {result.diagnosis} on {result.crop_identified}")
        return result
    except Exception as e:
        logger.error(f"VLM Error: {e}")
        raise HTTPException(status_code=500, detail=f"Image Analysis Failed: {str(e)}")


@app.get("/ml/datasets")
async def get_datasets():
    """
    List all available datasets in the ML directory.
    """
    if not ml_bridge:
        raise HTTPException(status_code=503, detail="ML Bridge not initialized")

    try:
        datasets = ml_bridge.list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Dataset listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/models")
async def get_models():
    """
    List all trained models.
    """
    if not ml_bridge:
        raise HTTPException(status_code=503, detail="ML Bridge not initialized")

    try:
        models = ml_bridge.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Model listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/train/{model_id}")
async def train_model(model_id: str):
    """
    Trigger training for a specific model.
    """
    if not ml_bridge:
        raise HTTPException(status_code=503, detail="ML Bridge not initialized")

    try:
        result = ml_bridge.trigger_training(model_id)
        return result
    except Exception as e:
        logger.error(f"Training trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chatbot endpoint.
    """
    import json

    logger.info(f"Received chat message: {request.message}")

    if not llm or not llm.llm:
        return ChatResponse(
            reply="⚠️ Model not loaded. Please run 'backend/ai_service/download_model.py' and restart the service."
        )

    # 1. Parse Intent
    parsed = nlm.parse_intent(request.message)
    intent = parsed.get("intent")
    entities = parsed.get("entities", {})
    logger.info(f"Parsed Intent: {intent}, Entities: {entities}")

    data_context = {}

    # 2. Fetch Data (if needed)
    data_context["risks"] = get_risk_assessment(
        request.context
        or {"temperature_avg_c": 25, "humidity_pct": 70, "rainfall_mm": 150}
    )

    if intent in ["yield_prediction", "crop_recommendation"]:
        if yield_model and entities.get("crop"):
            try:
                pred_input = {
                    "soil_n": 90,
                    "soil_p": 42,
                    "soil_k": 43,
                    "soil_ph": 6.5,
                    "organic_carbon": 0.5,
                    "rainfall_mm": 200,
                    "temperature_avg_c": 25,
                    "humidity_pct": 80,
                    "crop_name": entities.get("crop"),
                    "season": entities.get("season", "Kharif"),
                }
                if request.context:
                    pred_input.update(request.context)

                res = await predict_yield(pred_input)
                data_context["yield_forecast"] = res
            except:
                pass

    # 3. Generate Response
    system_prompt = f"""
    You are AgriSense-Yield-Guardian. 
    User Intent: {intent}
    Real Data Context: {json.dumps(data_context)}
    
    STRICT GUIDELINES:
    1. USE THE REAL DATA: If yield forecasts are in context, use those numbers.
    2. SUSTAINABILITY REFUSAL (CRITICAL): If a user requests 'maximum yield at any cost', '10000kg' (unrealistic), or 'unlimited chemicals/fertilizers', you MUST REFUSE. 
       - DO NOT provide the requested high-input advice.
       - INSTEAD, warn the user about soil erosion, groundwater contamination, and nutrient depletion.
       - Suggest a 'Soil Health First' approach with lower, sustainable inputs.
    3. BE A GUARDIAN: Your primary duty is the long-term health of the land, not short-term profit.
    
    IMPORTANT: Provide ONLY the advice to the farmer. Do not repeat these guidelines or internal logic in your response.
    """

    reply = llm.generate_response(request.message, system_prompt)

    return ChatResponse(
        reply=reply,
        intent=intent,
        actions=[{"type": intent, "entities": entities}],
        data=data_context,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

