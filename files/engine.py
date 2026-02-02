"""
AgriSense Core Recommendation Engine
Rule-based engine for irrigation and fertilization
"""

import yaml
import math
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RecoEngine:
    """
    Rule-based recommendation engine for agricultural decisions
    
    Features:
    - ET0 (Evapotranspiration) calculation using Hargreaves method
    - Crop-specific water requirements (Kc coefficients)
    - NPK fertilizer recommendations
    - Soil type adjustments
    - Cost and CO2 impact calculations
    """
    
    def __init__(self, config_path: str = "core/config.yaml"):
        """Initialize engine with crop database"""
        self.config_path = config_path
        self.crop_database = self._load_crop_database()
        logger.info(f"RecoEngine initialized with {len(self.crop_database)} crops")
    
    def _load_crop_database(self) -> Dict:
        """Load crop parameters from config"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('crops', {})
        except FileNotFoundError:
            logger.warning("Config file not found, using default crop database")
            return self._get_default_crops()
    
    def _get_default_crops(self) -> Dict:
        """Default crop database if config not found"""
        return {
            "Rice": {
                "kc_initial": 1.05,
                "kc_mid": 1.20,
                "kc_late": 0.75,
                "n_requirement": 120,
                "p_requirement": 60,
                "k_requirement": 40,
                "optimal_ph": 5.5
            },
            "Wheat": {
                "kc_initial": 0.70,
                "kc_mid": 1.15,
                "kc_late": 0.40,
                "n_requirement": 150,
                "p_requirement": 60,
                "k_requirement": 40,
                "optimal_ph": 6.5
            },
            "Cotton": {
                "kc_initial": 0.45,
                "kc_mid": 1.15,
                "kc_late": 0.70,
                "n_requirement": 120,
                "p_requirement": 50,
                "k_requirement": 50,
                "optimal_ph": 7.0
            },
            "Maize": {
                "kc_initial": 0.70,
                "kc_mid": 1.20,
                "kc_late": 0.60,
                "n_requirement": 150,
                "p_requirement": 75,
                "k_requirement": 75,
                "optimal_ph": 6.0
            }
        }
    
    def calculate_et0(self, temp_min: float, temp_max: float, 
                     latitude: float = 13.0827) -> float:
        """
        Calculate reference evapotranspiration (ET0) using Hargreaves method
        
        Args:
            temp_min: Minimum temperature (°C)
            temp_max: Maximum temperature (°C)
            latitude: Latitude in degrees (default: Chennai)
        
        Returns:
            ET0 in mm/day
        """
        # Calculate average temperature
        temp_avg = (temp_min + temp_max) / 2
        
        # Simplified Hargreaves formula
        # ET0 = 0.0023 * Ra * (Tavg + 17.8) * sqrt(Tmax - Tmin)
        # Ra (extraterrestrial radiation) approximation for tropics: 15.3 MJ/m²/day
        ra = 15.3
        
        temp_range = abs(temp_max - temp_min)
        et0 = 0.0023 * ra * (temp_avg + 17.8) * math.sqrt(temp_range)
        
        return round(et0, 2)
    
    def get_water_requirement(self, crop: str, growth_stage: str,
                              temp_min: float, temp_max: float,
                              soil_type: str = "loam") -> Dict:
        """
        Calculate crop water requirement
        
        Args:
            crop: Crop name
            growth_stage: "initial", "mid", "late"
            temp_min: Min temperature
            temp_max: Max temperature
            soil_type: Soil type (sandy/loam/clay)
        
        Returns:
            Water requirement details
        """
        # Get crop parameters
        crop_data = self.crop_database.get(crop, self.crop_database["Maize"])
        
        # Get Kc coefficient based on growth stage
        kc_map = {
            "initial": crop_data.get("kc_initial", 0.7),
            "mid": crop_data.get("kc_mid", 1.15),
            "late": crop_data.get("kc_late", 0.6)
        }
        kc = kc_map.get(growth_stage, 1.0)
        
        # Calculate ET0
        et0 = self.calculate_et0(temp_min, temp_max)
        
        # Calculate crop water requirement (ETc = ET0 * Kc)
        etc = et0 * kc
        
        # Adjust for soil type
        soil_multiplier = {
            "sandy": 1.2,
            "loam": 1.0,
            "clay": 0.85
        }
        etc_adjusted = etc * soil_multiplier.get(soil_type, 1.0)
        
        # Calculate irrigation amount (accounting for 80% efficiency)
        irrigation_mm = etc_adjusted / 0.8
        
        # Convert to liters per hectare (1 mm = 10,000 L/ha)
        irrigation_liters = irrigation_mm * 10000
        
        return {
            "crop": crop,
            "growth_stage": growth_stage,
            "et0_mm_day": et0,
            "kc_coefficient": kc,
            "etc_mm_day": round(etc_adjusted, 2),
            "irrigation_mm_day": round(irrigation_mm, 2),
            "irrigation_liters_ha": round(irrigation_liters, 0),
            "soil_type": soil_type,
            "recommendation": self._get_irrigation_advice(irrigation_mm)
        }
    
    def _get_irrigation_advice(self, irrigation_mm: float) -> str:
        """Generate irrigation advice based on amount"""
        if irrigation_mm < 3:
            return "Low water requirement. Irrigate every 3-4 days."
        elif irrigation_mm < 6:
            return "Moderate water requirement. Irrigate every 2-3 days."
        elif irrigation_mm < 10:
            return "High water requirement. Irrigate daily or alternate days."
        else:
            return "Very high water requirement. Daily irrigation recommended."
    
    def get_fertilizer_recommendation(self, crop: str, soil_n: float,
                                     soil_p: float, soil_k: float,
                                     soil_ph: float) -> Dict:
        """
        Calculate NPK fertilizer requirements
        
        Args:
            crop: Crop name
            soil_n: Current soil nitrogen (kg/ha)
            soil_p: Current soil phosphorus (kg/ha)
            soil_k: Current soil potassium (kg/ha)
            soil_ph: Soil pH level
        
        Returns:
            Fertilizer recommendations
        """
        crop_data = self.crop_database.get(crop, self.crop_database["Maize"])
        
        # Calculate deficiencies
        n_deficit = max(0, crop_data.get("n_requirement", 120) - soil_n)
        p_deficit = max(0, crop_data.get("p_requirement", 60) - soil_p)
        k_deficit = max(0, crop_data.get("k_requirement", 40) - soil_k)
        
        # pH adjustment factor
        optimal_ph = crop_data.get("optimal_ph", 6.5)
        ph_diff = abs(soil_ph - optimal_ph)
        
        if ph_diff > 1.5:
            ph_advice = f"Critical: Adjust pH from {soil_ph} to {optimal_ph}"
        elif ph_diff > 0.5:
            ph_advice = f"Consider adjusting pH from {soil_ph} to {optimal_ph}"
        else:
            ph_advice = "pH is within optimal range"
        
        # Calculate fertilizer amounts
        # Assuming standard fertilizer compositions
        urea_kg = n_deficit * 2.17  # 46% N in urea
        dap_kg = p_deficit * 2.17   # 46% P2O5 in DAP
        mop_kg = k_deficit * 1.67   # 60% K2O in MOP
        
        # Cost estimation (INR per kg)
        urea_cost = urea_kg * 6
        dap_cost = dap_kg * 27
        mop_cost = mop_kg * 17
        total_cost = urea_cost + dap_cost + mop_cost
        
        return {
            "crop": crop,
            "deficiencies": {
                "nitrogen_kg_ha": round(n_deficit, 2),
                "phosphorus_kg_ha": round(p_deficit, 2),
                "potassium_kg_ha": round(k_deficit, 2)
            },
            "fertilizers": {
                "urea_kg_ha": round(urea_kg, 2),
                "dap_kg_ha": round(dap_kg, 2),
                "mop_kg_ha": round(mop_kg, 2)
            },
            "cost_inr_ha": round(total_cost, 2),
            "ph_status": ph_advice,
            "application_schedule": self._get_application_schedule(crop),
            "organic_alternatives": [
                "Compost (5-10 tonnes/ha)",
                "Vermicompost (3-5 tonnes/ha)",
                "Green manure (Dhaincha/Sunhemp)"
            ]
        }
    
    def _get_application_schedule(self, crop: str) -> List[str]:
        """Get fertilizer application schedule"""
        schedules = {
            "Rice": [
                "Basal: 50% N, 100% P, 50% K",
                "Tillering: 25% N",
                "Panicle initiation: 25% N, 50% K"
            ],
            "Wheat": [
                "Basal: 50% N, 100% P, 50% K",
                "Crown root initiation: 25% N, 25% K",
                "Flowering: 25% N, 25% K"
            ]
        }
        return schedules.get(crop, [
            "Basal: 50% of NPK",
            "Mid-season: 30% of NPK",
            "Late-season: 20% of NPK"
        ])
    
    def get_crop_recommendation(self, temperature: float, humidity: float,
                               ph: float, rainfall: float) -> List[Dict]:
        """
        Recommend suitable crops based on conditions
        
        Args:
            temperature: Average temperature (°C)
            humidity: Relative humidity (%)
            ph: Soil pH
            rainfall: Annual rainfall (mm)
        
        Returns:
            List of recommended crops with scores
        """
        recommendations = []
        
        for crop_name, crop_data in self.crop_database.items():
            score = 100
            
            # pH compatibility
            optimal_ph = crop_data.get("optimal_ph", 6.5)
            ph_diff = abs(ph - optimal_ph)
            score -= ph_diff * 15
            
            # Temperature compatibility (simplified)
            if temperature < 15 or temperature > 40:
                score -= 30
            
            # Humidity compatibility
            if humidity < 40:
                score -= 20
            elif humidity > 80:
                score -= 10
            
            # Ensure score is non-negative
            score = max(0, score)
            
            if score > 40:  # Only include viable crops
                recommendations.append({
                    "crop": crop_name,
                    "suitability_score": round(score, 2),
                    "confidence": "High" if score > 70 else "Medium" if score > 50 else "Low",
                    "reason": self._get_recommendation_reason(crop_name, score)
                })
        
        # Sort by score
        recommendations.sort(key=lambda x: x["suitability_score"], reverse=True)
        
        return recommendations[:5]  # Return top 5
    
    def _get_recommendation_reason(self, crop: str, score: float) -> str:
        """Generate reason for recommendation"""
        if score > 70:
            return f"{crop} is highly suitable for current conditions"
        elif score > 50:
            return f"{crop} is moderately suitable, minor adjustments may be needed"
        else:
            return f"{crop} is marginally suitable, significant management required"

# Initialize global engine instance
engine = RecoEngine()
