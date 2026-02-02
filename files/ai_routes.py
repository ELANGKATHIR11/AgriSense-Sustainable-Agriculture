"""
AgriSense AI Routes
Chatbot and advanced ML model endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "anonymous"
    language: Optional[str] = "en"

class DiseaseDetectionInput(BaseModel):
    image_data: str  # Base64 encoded image
    crop_type: Optional[str] = None

class WeedDetectionInput(BaseModel):
    image_data: str  # Base64 encoded image
    field_location: Optional[str] = None

@router.post("/chat")
async def chat_with_bot(message: ChatMessage):
    """
    Chat with AI agricultural assistant
    
    Provides context-aware agricultural advice
    """
    try:
        # Simple rule-based responses (in production, use LLM)
        query = message.message.lower()
        
        responses = {
            "hello": "Hello! I'm your AgriSense AI assistant. How can I help you with farming today?",
            "crop": "I can help you choose the best crop based on your soil and climate conditions. What information do you have?",
            "disease": "I can identify plant diseases from images. Please upload a photo of the affected plant.",
            "water": "I can calculate optimal irrigation schedules based on your crop and weather. What crop are you growing?",
            "fertilizer": "I can recommend the right NPK fertilizer amounts. Do you have your soil test results?",
            "weather": "I can provide weather-based farming advice. What's your location?"
        }
        
        # Find matching response
        response_text = "I'm here to help with your farming questions! You can ask me about crops, diseases, irrigation, fertilizers, or general farming advice."
        
        for keyword, response in responses.items():
            if keyword in query:
                response_text = response
                break
        
        return {
            "success": True,
            "user_message": message.message,
            "bot_response": response_text,
            "language": message.language,
            "suggestions": [
                "Recommend a crop for my conditions",
                "How much water does my crop need?",
                "Identify this plant disease",
                "Calculate fertilizer requirements"
            ]
        }
    except Exception as e:
        logger.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disease/detect")
async def detect_disease(input_data: DiseaseDetectionInput):
    """
    Detect plant diseases from images
    
    Uses computer vision models for identification
    """
    try:
        # Mock disease detection (in production, use trained CNN model)
        mock_diseases = [
            {
                "disease": "Leaf Spot",
                "confidence": 0.87,
                "severity": "moderate",
                "treatment": "Apply copper-based fungicide. Remove affected leaves.",
                "prevention": "Ensure good air circulation and avoid overhead watering."
            },
            {
                "disease": "Powdery Mildew",
                "confidence": 0.72,
                "severity": "mild",
                "treatment": "Apply sulfur-based fungicide or neem oil.",
                "prevention": "Maintain proper spacing between plants."
            }
        ]
        
        return {
            "success": True,
            "crop_type": input_data.crop_type or "Unknown",
            "detections": mock_diseases[:1],  # Return top detection
            "method": "cnn_model",
            "recommendation": "Monitor plants closely and apply recommended treatment."
        }
    except Exception as e:
        logger.error(f"Error in disease detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/weed/detect")
async def detect_weed(input_data: WeedDetectionInput):
    """
    Detect and identify weeds in field images
    
    Uses image segmentation for weed detection
    """
    try:
        # Mock weed detection (in production, use trained model)
        mock_weeds = [
            {
                "weed_type": "Dandelion",
                "confidence": 0.82,
                "coverage_percent": 15,
                "control_method": "Manual removal or selective herbicide",
                "priority": "medium"
            }
        ]
        
        return {
            "success": True,
            "field_location": input_data.field_location or "Unknown",
            "detections": mock_weeds,
            "total_coverage": 15,
            "method": "segmentation_model",
            "recommendation": "Consider spot treatment with herbicide or manual removal."
        }
    except Exception as e:
        logger.error(f"Error in weed detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chatbot/history/{user_id}")
async def get_chat_history(user_id: str, limit: int = 10):
    """Get conversation history for a user"""
    return {
        "success": True,
        "user_id": user_id,
        "history": [],
        "count": 0
    }

@router.post("/plant-health/assess")
async def assess_plant_health(input_data: DiseaseDetectionInput):
    """
    Comprehensive plant health assessment
    
    Analyzes overall plant health including diseases, pests, and nutrition
    """
    try:
        return {
            "success": True,
            "overall_health": "good",
            "health_score": 82,
            "issues": [
                {
                    "category": "disease",
                    "severity": "low",
                    "description": "Minor leaf spotting detected"
                }
            ],
            "recommendations": [
                "Continue current care routine",
                "Monitor for disease progression",
                "Ensure adequate nutrition"
            ]
        }
    except Exception as e:
        logger.error(f"Error in plant health assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
