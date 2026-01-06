"""
Explainable AI (XAI) Module
SHAP and LIME explanations for farmer-friendly insights
"""
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from ..config.optimization import settings

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    lime_tabular = None  # type: ignore


class ExplainableAI:
    """
    Provides explanations for ML predictions using SHAP and LIME.
    Translates technical explanations into farmer-friendly language.
    """
    
    def __init__(self):
        self.shap_enabled = settings.enable_shap_explanations and SHAP_AVAILABLE
        self.lime_enabled = settings.enable_lime_explanations and LIME_AVAILABLE
        
        if not self.shap_enabled and not self.lime_enabled:
            logger.info("Explainable AI disabled or libraries not available")
    
    async def explain_crop_recommendation(
        self,
        model: Any,
        input_features: Dict[str, float],
        prediction: str,
        confidence: float
    ) -> Dict[str, Any]:
        """
        Explain why a specific crop was recommended.
        
        Args:
            model: Trained ML model
            input_features: Input feature dictionary
            prediction: Predicted crop
            confidence: Prediction confidence
            
        Returns:
            Dictionary with explanations
        """
        explanation = {
            "prediction": prediction,
            "confidence": confidence,
            "technical_explanation": {},
            "farmer_friendly_explanation": ""
        }
        
        if settings.farmer_friendly_explanations:
            explanation["farmer_friendly_explanation"] = self._generate_friendly_crop_explanation(
                input_features, prediction
            )
        
        # Generate SHAP explanation if enabled
        if self.shap_enabled:
            try:
                shap_values = self._compute_shap_values(model, input_features)
                explanation["technical_explanation"]["shap"] = shap_values
            except Exception as e:
                logger.error(f"SHAP explanation failed: {e}")
        
        return explanation
    
    def _generate_friendly_crop_explanation(
        self,
        features: Dict[str, float],
        crop: str
    ) -> str:
        """
        Generate farmer-friendly explanation in natural language.
        
        Example output:
        "Rice is recommended because your soil has high nitrogen (80mg/kg) and 
        the weather is humid (85%). Rice grows best in these wet conditions."
        """
        reasons = []
        
        # Temperature reasoning
        temp = features.get("temperature", 0)
        if temp > 30:
            reasons.append(f"warm temperature ({temp}°C)")
        elif temp < 15:
            reasons.append(f"cool temperature ({temp}°C)")
        else:
            reasons.append(f"moderate temperature ({temp}°C)")
        
        # Humidity reasoning
        humidity = features.get("humidity", 0)
        if humidity > 80:
            reasons.append(f"high humidity ({humidity}%)")
        elif humidity < 40:
            reasons.append(f"low humidity ({humidity}%)")
        
        # Rainfall reasoning
        rainfall = features.get("rainfall", 0)
        if rainfall > 200:
            reasons.append(f"heavy rainfall ({rainfall}mm)")
        elif rainfall < 50:
            reasons.append(f"low rainfall ({rainfall}mm)")
        
        # NPK reasoning
        N = features.get("N", 0)
        P = features.get("P", 0)
        K = features.get("K", 0)
        
        if N > 80:
            reasons.append(f"nitrogen-rich soil ({N}mg/kg)")
        if P > 60:
            reasons.append(f"high phosphorus ({P}mg/kg)")
        if K > 80:
            reasons.append(f"potassium-rich soil ({K}mg/kg)")
        
        # pH reasoning
        ph = features.get("ph", 7)
        if ph < 5.5:
            reasons.append(f"acidic soil (pH {ph})")
        elif ph > 7.5:
            reasons.append(f"alkaline soil (pH {ph})")
        
        # Combine into friendly explanation
        if len(reasons) > 0:
            reason_text = ", ".join(reasons[:3])  # Top 3 reasons
            explanation = (
                f"✅ **{crop.title()}** is recommended for your field because of {reason_text}. "
                f"This crop thrives in these conditions and will give you good yields."
            )
        else:
            explanation = f"✅ **{crop.title()}** is suitable for your field conditions."
        
        return explanation
    
    def _compute_shap_values(self, model: Any, features: Dict[str, float]) -> Dict[str, float]:
        """Compute SHAP values for feature importance"""
        if not self.shap_enabled:
            return {}
        
        # Convert features to array
        feature_array = np.array([list(features.values())])
        feature_names = list(features.keys())
        
        try:
            # Create SHAP explainer (use appropriate type based on model)
            explainer = shap.Explainer(model)
            shap_values = explainer(feature_array)
            
            # Extract feature importances
            importance_dict = {}
            for idx, name in enumerate(feature_names):
                importance_dict[name] = float(shap_values.values[0][idx])
            
            return importance_dict
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return {}
    
    async def explain_irrigation_decision(
        self,
        soil_moisture: float,
        temperature: float,
        humidity: float,
        recommendation: str
    ) -> Dict[str, Any]:
        """
        Explain irrigation recommendation.
        
        Returns farmer-friendly explanation of why irrigation is needed or not.
        """
        explanation = {
            "recommendation": recommendation,
            "reasons": [],
            "friendly_explanation": ""
        }
        
        # Build reasons list
        if soil_moisture < 30:
            explanation["reasons"].append({
                "factor": "Low Soil Moisture",
                "value": f"{soil_moisture}%",
                "impact": "High priority - soil is dry"
            })
        
        if temperature > 35:
            explanation["reasons"].append({
                "factor": "High Temperature",
                "value": f"{temperature}°C",
                "impact": "Plants need more water in hot weather"
            })
        
        if humidity < 40:
            explanation["reasons"].append({
                "factor": "Low Humidity",
                "value": f"{humidity}%",
                "impact": "Dry air increases water loss"
            })
        
        # Generate friendly explanation
        if recommendation == "irrigate_immediately":
            explanation["friendly_explanation"] = (
                f"🚨 **Urgent:** Your soil moisture is at {soil_moisture}%, which is too low. "
                f"Water your crops now to prevent stress and yield loss. "
                f"Recommended irrigation: 30 minutes."
            )
        elif recommendation == "irrigate_soon":
            explanation["friendly_explanation"] = (
                f"⚠️ **Action needed:** Soil moisture is {soil_moisture}%. "
                f"Plan to irrigate within the next few hours. "
                f"Recommended irrigation: 20 minutes."
            )
        elif recommendation == "schedule_irrigation":
            explanation["friendly_explanation"] = (
                f"📅 **Plan ahead:** Soil moisture is {soil_moisture}%. "
                f"Schedule irrigation for later today or tomorrow. "
                f"Recommended irrigation: 15 minutes."
            )
        else:
            explanation["friendly_explanation"] = (
                f"✅ **No action needed:** Soil moisture is good at {soil_moisture}%. "
                f"Your crops have enough water for now."
            )
        
        return explanation
    
    async def explain_pest_risk(
        self,
        temperature: float,
        humidity: float,
        rainfall: float,
        risk_level: str,
        risk_score: float
    ) -> Dict[str, Any]:
        """
        Explain pest/disease risk assessment.
        """
        explanation = {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "contributing_factors": [],
            "friendly_explanation": "",
            "prevention_tips": []
        }
        
        # Identify contributing factors
        if humidity > 80:
            explanation["contributing_factors"].append({
                "factor": "High Humidity",
                "value": f"{humidity}%",
                "impact": "Humid conditions favor fungal diseases"
            })
            explanation["prevention_tips"].append(
                "Improve air circulation by spacing plants properly"
            )
        
        if 25 < temperature < 35:
            explanation["contributing_factors"].append({
                "factor": "Warm Temperature",
                "value": f"{temperature}°C",
                "impact": "Optimal temperature for pest reproduction"
            })
            explanation["prevention_tips"].append(
                "Monitor plants daily for early signs of pests"
            )
        
        if rainfall > 50:
            explanation["contributing_factors"].append({
                "factor": "Recent Rainfall",
                "value": f"{rainfall}mm",
                "impact": "Wet conditions increase disease risk"
            })
            explanation["prevention_tips"].append(
                "Avoid overhead irrigation, water at soil level"
            )
        
        # Friendly explanation
        if risk_level == "high":
            explanation["friendly_explanation"] = (
                f"🔴 **High Risk:** Weather conditions (temp: {temperature}°C, humidity: {humidity}%) "
                f"are very favorable for pests and diseases. Inspect your crops carefully and "
                f"consider preventive measures."
            )
        elif risk_level == "medium":
            explanation["friendly_explanation"] = (
                f"🟡 **Medium Risk:** Current conditions may encourage pest activity. "
                f"Keep an eye on your plants and take preventive action if needed."
            )
        else:
            explanation["friendly_explanation"] = (
                f"🟢 **Low Risk:** Weather conditions are not favorable for pests right now. "
                f"Continue regular monitoring."
            )
        
        return explanation


# Singleton instance
explainable_ai = ExplainableAI()
