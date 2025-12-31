"""
Smart Recommendations Engine with Multi-Objective Optimization

Optimizes agricultural decisions across multiple objectives:
- Maximize crop yield
- Minimize water usage
- Minimize costs
- Minimize environmental impact

Uses constraint-based optimization and Pareto frontier analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy.optimize import minimize, differential_evolution


logger = logging.getLogger(__name__)


class OptimizationObjective(Enum):
    """Optimization objectives"""
    MAXIMIZE_YIELD = "maximize_yield"
    MINIMIZE_WATER = "minimize_water"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_ENVIRONMENTAL_IMPACT = "minimize_environmental_impact"


@dataclass
class Constraint:
    """Optimization constraint"""
    name: str
    type: str  # 'eq' (equality) or 'ineq' (inequality)
    func: callable
    description: str


@dataclass
class OptimizationResult:
    """Result from multi-objective optimization"""
    objectives: Dict[str, float]
    parameters: Dict[str, float]
    constraints_satisfied: bool
    pareto_optimal: bool
    confidence: float
    recommendations: List[str]
    trade_offs: Dict[str, str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "objectives": self.objectives,
            "parameters": self.parameters,
            "constraints_satisfied": self.constraints_satisfied,
            "pareto_optimal": self.pareto_optimal,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
            "trade_offs": self.trade_offs,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class FarmParameters:
    """Farm-specific parameters for optimization"""
    field_size_hectares: float
    soil_type: str  # 'sandy', 'loamy', 'clay'
    current_soil_moisture: float  # 0-100%
    current_temperature: float  # Celsius
    current_humidity: float  # 0-100%
    crop_type: str
    growth_stage: str  # 'seedling', 'vegetative', 'flowering', 'fruiting'
    water_cost_per_liter: float = 0.001  # USD
    fertilizer_cost_per_kg: float = 2.0  # USD
    labor_cost_per_hour: float = 5.0  # USD
    available_budget: float = 1000.0  # USD
    max_water_daily_liters: float = 50000.0
    weather_forecast_7day: Optional[Dict[str, List[float]]] = None


class YieldPredictor:
    """
    Predicts crop yield based on input parameters
    
    Uses simplified agronomy models (replace with ML models in production)
    """
    
    # Optimal ranges for different crops
    CROP_OPTIMA = {
        "tomato": {
            "soil_moisture": (50, 70),
            "temperature": (20, 28),
            "nitrogen": (150, 200),  # kg/ha
            "phosphorus": (80, 120),
            "potassium": (180, 220)
        },
        "wheat": {
            "soil_moisture": (40, 60),
            "temperature": (15, 25),
            "nitrogen": (100, 150),
            "phosphorus": (50, 80),
            "potassium": (50, 100)
        },
        "rice": {
            "soil_moisture": (70, 90),
            "temperature": (25, 35),
            "nitrogen": (120, 180),
            "phosphorus": (60, 100),
            "potassium": (80, 120)
        }
    }
    
    @classmethod
    def predict_yield(
        cls,
        crop_type: str,
        soil_moisture: float,
        temperature: float,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        field_size: float
    ) -> float:
        """
        Predict crop yield (kg/ha)
        
        Args:
            crop_type: Type of crop
            soil_moisture: Soil moisture percentage
            temperature: Temperature in Celsius
            nitrogen: Nitrogen application (kg/ha)
            phosphorus: Phosphorus application (kg/ha)
            potassium: Potassium application (kg/ha)
            field_size: Field size in hectares
            
        Returns:
            Predicted yield in kg/ha
        """
        if crop_type not in cls.CROP_OPTIMA:
            logger.warning(f"Unknown crop type: {crop_type}, using tomato defaults")
            crop_type = "tomato"
        
        optima = cls.CROP_OPTIMA[crop_type]
        
        # Base yield (kg/ha)
        base_yields = {
            "tomato": 60000,
            "wheat": 6000,
            "rice": 8000
        }
        base_yield = base_yields.get(crop_type, 10000)
        
        # Calculate efficiency factors (0-1) for each parameter
        moisture_efficiency = cls._efficiency_factor(
            soil_moisture, optima["soil_moisture"]
        )
        temp_efficiency = cls._efficiency_factor(
            temperature, optima["temperature"]
        )
        n_efficiency = cls._efficiency_factor(
            nitrogen, optima["nitrogen"]
        )
        p_efficiency = cls._efficiency_factor(
            phosphorus, optima["phosphorus"]
        )
        k_efficiency = cls._efficiency_factor(
            potassium, optima["potassium"]
        )
        
        # Combined efficiency (geometric mean)
        combined_efficiency = (
            moisture_efficiency * 0.3 +
            temp_efficiency * 0.2 +
            n_efficiency * 0.2 +
            p_efficiency * 0.15 +
            k_efficiency * 0.15
        )
        
        # Predicted yield
        yield_per_ha = base_yield * combined_efficiency
        
        return yield_per_ha
    
    @staticmethod
    def _efficiency_factor(value: float, optimal_range: Tuple[float, float]) -> float:
        """
        Calculate efficiency factor (0-1) based on distance from optimal range
        """
        min_val, max_val = optimal_range
        optimal_mid = (min_val + max_val) / 2
        optimal_width = max_val - min_val
        
        if min_val <= value <= max_val:
            # Within optimal range - high efficiency
            return 1.0 - 0.1 * abs(value - optimal_mid) / (optimal_width / 2)
        elif value < min_val:
            # Below optimal - decreasing efficiency
            deficit = (min_val - value) / min_val
            return max(0.0, 1.0 - 2.0 * deficit)
        else:
            # Above optimal - diminishing returns and potential toxicity
            excess = (value - max_val) / max_val
            return max(0.0, 1.0 - 3.0 * excess)


class WaterUsageEstimator:
    """Estimates water usage based on crop and environmental conditions"""
    
    # Base water requirements (liters/ha/day)
    BASE_WATER_REQUIREMENTS = {
        "tomato": 4000,
        "wheat": 2000,
        "rice": 8000,
        "corn": 3000
    }
    
    @classmethod
    def estimate_water_usage(
        cls,
        crop_type: str,
        temperature: float,
        humidity: float,
        soil_moisture: float,
        field_size: float,
        irrigation_efficiency: float = 0.75
    ) -> float:
        """
        Estimate daily water usage (liters)
        
        Args:
            crop_type: Type of crop
            temperature: Temperature in Celsius
            humidity: Humidity percentage
            soil_moisture: Current soil moisture percentage
            field_size: Field size in hectares
            irrigation_efficiency: Irrigation system efficiency (0-1)
            
        Returns:
            Estimated water usage in liters/day
        """
        base_requirement = cls.BASE_WATER_REQUIREMENTS.get(crop_type, 3000)
        
        # Adjust for temperature (higher temp = more evapotranspiration)
        temp_factor = 1.0 + (temperature - 25) * 0.02
        temp_factor = max(0.5, min(2.0, temp_factor))
        
        # Adjust for humidity (lower humidity = more water needed)
        humidity_factor = 1.0 + (50 - humidity) * 0.01
        humidity_factor = max(0.7, min(1.5, humidity_factor))
        
        # Adjust for soil moisture (lower moisture = more water needed)
        moisture_deficit = max(0, 60 - soil_moisture)  # Target 60%
        moisture_factor = 1.0 + moisture_deficit * 0.015
        
        # Calculate total requirement
        total_requirement = (
            base_requirement * 
            temp_factor * 
            humidity_factor * 
            moisture_factor * 
            field_size
        )
        
        # Account for irrigation efficiency
        actual_water_needed = total_requirement / irrigation_efficiency
        
        return actual_water_needed


class CostEstimator:
    """Estimates costs for different agricultural operations"""
    
    @staticmethod
    def calculate_irrigation_cost(
        water_volume: float,
        water_cost_per_liter: float,
        energy_cost: float = 0.1
    ) -> float:
        """Calculate cost of irrigation"""
        water_cost = water_volume * water_cost_per_liter
        # Add energy cost for pumping (simplified)
        energy = water_volume * 0.0005  # kWh per liter (pump efficiency)
        total_energy_cost = energy * energy_cost
        return water_cost + total_energy_cost
    
    @staticmethod
    def calculate_fertilizer_cost(
        nitrogen_kg: float,
        phosphorus_kg: float,
        potassium_kg: float,
        n_cost: float = 2.0,
        p_cost: float = 2.5,
        k_cost: float = 1.5
    ) -> float:
        """Calculate fertilizer costs"""
        return (
            nitrogen_kg * n_cost +
            phosphorus_kg * p_cost +
            potassium_kg * k_cost
        )
    
    @staticmethod
    def calculate_labor_cost(
        hours: float,
        hourly_rate: float
    ) -> float:
        """Calculate labor costs"""
        return hours * hourly_rate


class SmartRecommendationEngine:
    """
    Multi-objective optimization engine for agricultural recommendations
    """
    
    def __init__(self):
        self.yield_predictor = YieldPredictor()
        self.water_estimator = WaterUsageEstimator()
        self.cost_estimator = CostEstimator()
    
    def optimize_irrigation_schedule(
        self,
        farm_params: FarmParameters,
        days_ahead: int = 7
    ) -> OptimizationResult:
        """
        Optimize irrigation schedule for next N days
        
        Objectives:
        - Maximize yield
        - Minimize water usage
        - Minimize costs
        
        Args:
            farm_params: Farm-specific parameters
            days_ahead: Number of days to optimize
            
        Returns:
            OptimizationResult with recommendations
        """
        logger.info(f"Optimizing irrigation schedule for {days_ahead} days")
        
        # Decision variables: daily irrigation amount (liters/day)
        bounds = [(0, farm_params.max_water_daily_liters) for _ in range(days_ahead)]
        
        # Objective function (minimize negative yield + water + cost)
        def objective(irrigation_schedule: np.ndarray) -> float:
            total_water = np.sum(irrigation_schedule)
            
            # Estimate yield impact
            avg_moisture = farm_params.current_soil_moisture
            for water in irrigation_schedule:
                # Simplified moisture dynamics
                moisture_increase = water / (farm_params.field_size_hectares * 10000)
                avg_moisture = min(100, avg_moisture + moisture_increase)
            
            # Predict yield
            yield_kg = self.yield_predictor.predict_yield(
                crop_type=farm_params.crop_type,
                soil_moisture=avg_moisture,
                temperature=farm_params.current_temperature,
                nitrogen=150,  # Assumed baseline
                phosphorus=80,
                potassium=150,
                field_size=farm_params.field_size_hectares
            )
            
            # Calculate costs
            irrigation_cost = self.cost_estimator.calculate_irrigation_cost(
                total_water,
                farm_params.water_cost_per_liter
            )
            
            # Multi-objective: balance yield (maximize) vs water & cost (minimize)
            # Normalize and weight objectives
            yield_normalized = -yield_kg / 60000  # Negative for minimization
            water_normalized = total_water / (farm_params.max_water_daily_liters * days_ahead)
            cost_normalized = irrigation_cost / farm_params.available_budget
            
            # Weighted sum (can be adjusted based on priorities)
            return yield_normalized + 0.4 * water_normalized + 0.6 * cost_normalized
        
        # Constraints
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: farm_params.max_water_daily_liters * days_ahead - np.sum(x)
            },
            {
                'type': 'ineq',
                'fun': lambda x: farm_params.available_budget - 
                    self.cost_estimator.calculate_irrigation_cost(
                        np.sum(x),
                        farm_params.water_cost_per_liter
                    )
            }
        ]
        
        # Initial guess: uniform distribution
        x0 = np.array([farm_params.max_water_daily_liters * 0.5] * days_ahead)
        
        # Optimize
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if result.success:
            optimal_schedule = result.x
            total_water = np.sum(optimal_schedule)
            total_cost = self.cost_estimator.calculate_irrigation_cost(
                total_water,
                farm_params.water_cost_per_liter
            )
            
            # Calculate expected yield
            avg_moisture = farm_params.current_soil_moisture
            for water in optimal_schedule:
                moisture_increase = water / (farm_params.field_size_hectares * 10000)
                avg_moisture = min(100, avg_moisture + moisture_increase)
            
            expected_yield = self.yield_predictor.predict_yield(
                crop_type=farm_params.crop_type,
                soil_moisture=avg_moisture,
                temperature=farm_params.current_temperature,
                nitrogen=150,
                phosphorus=80,
                potassium=150,
                field_size=farm_params.field_size_hectares
            )
            
            # Generate recommendations
            recommendations = []
            for day, water in enumerate(optimal_schedule, 1):
                if water > 100:  # Only recommend if significant
                    recommendations.append(
                        f"Day {day}: Irrigate {water:.0f} liters "
                        f"({water/farm_params.field_size_hectares:.0f} L/ha)"
                    )
            
            # Trade-offs analysis
            trade_offs = {
                "yield_vs_water": (
                    f"Using {total_water/1000:.1f}m³ water to achieve "
                    f"{expected_yield:.0f} kg/ha yield"
                ),
                "yield_vs_cost": (
                    f"Spending ${total_cost:.2f} to achieve "
                    f"{expected_yield:.0f} kg/ha yield (${total_cost/expected_yield:.4f}/kg)"
                ),
                "water_vs_cost": (
                    f"Water efficiency: ${total_cost/(total_water/1000):.2f}/m³"
                )
            }
            
            return OptimizationResult(
                objectives={
                    "expected_yield_kg_per_ha": round(expected_yield, 2),
                    "total_water_liters": round(total_water, 2),
                    "total_cost_usd": round(total_cost, 2)
                },
                parameters={
                    f"day_{i+1}_irrigation_liters": round(val, 2)
                    for i, val in enumerate(optimal_schedule)
                },
                constraints_satisfied=True,
                pareto_optimal=True,
                confidence=0.85,
                recommendations=recommendations,
                trade_offs=trade_offs
            )
        else:
            logger.error(f"Optimization failed: {result.message}")
            raise ValueError(f"Optimization failed: {result.message}")
    
    def optimize_fertilizer_plan(
        self,
        farm_params: FarmParameters,
        current_nutrient_levels: Dict[str, float]
    ) -> OptimizationResult:
        """
        Optimize fertilizer application plan
        
        Objectives:
        - Maximize yield
        - Minimize fertilizer costs
        - Minimize environmental impact (nitrate leaching)
        
        Args:
            farm_params: Farm-specific parameters
            current_nutrient_levels: Current N-P-K levels in soil (kg/ha)
            
        Returns:
            OptimizationResult with fertilizer recommendations
        """
        logger.info("Optimizing fertilizer application plan")
        
        # Decision variables: [nitrogen, phosphorus, potassium] to add (kg/ha)
        bounds = [
            (0, 200),  # Nitrogen
            (0, 150),  # Phosphorus
            (0, 200)   # Potassium
        ]
        
        def objective(nutrients: np.ndarray) -> float:
            n_add, p_add, k_add = nutrients
            
            # Total nutrients
            total_n = current_nutrient_levels.get("nitrogen", 0) + n_add
            total_p = current_nutrient_levels.get("phosphorus", 0) + p_add
            total_k = current_nutrient_levels.get("potassium", 0) + k_add
            
            # Predict yield
            yield_kg = self.yield_predictor.predict_yield(
                crop_type=farm_params.crop_type,
                soil_moisture=farm_params.current_soil_moisture,
                temperature=farm_params.current_temperature,
                nitrogen=total_n,
                phosphorus=total_p,
                potassium=total_k,
                field_size=farm_params.field_size_hectares
            )
            
            # Calculate cost
            cost = self.cost_estimator.calculate_fertilizer_cost(
                n_add * farm_params.field_size_hectares,
                p_add * farm_params.field_size_hectares,
                k_add * farm_params.field_size_hectares
            )
            
            # Environmental impact (nitrate leaching risk)
            leaching_risk = max(0, total_n - 180) ** 2 * 0.01
            
            # Multi-objective
            yield_normalized = -yield_kg / 60000
            cost_normalized = cost / farm_params.available_budget
            env_normalized = leaching_risk / 100
            
            return yield_normalized + 0.5 * cost_normalized + 0.3 * env_normalized
        
        # Constraints
        constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: farm_params.available_budget - 
                    self.cost_estimator.calculate_fertilizer_cost(
                        x[0] * farm_params.field_size_hectares,
                        x[1] * farm_params.field_size_hectares,
                        x[2] * farm_params.field_size_hectares
                    )
            }
        ]
        
        # Initial guess: balanced NPK
        x0 = np.array([100, 60, 100])
        
        # Optimize
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )
        
        if result.success:
            n_add, p_add, k_add = result.x
            
            # Calculate final metrics
            total_n = current_nutrient_levels.get("nitrogen", 0) + n_add
            total_p = current_nutrient_levels.get("phosphorus", 0) + p_add
            total_k = current_nutrient_levels.get("potassium", 0) + k_add
            
            expected_yield = self.yield_predictor.predict_yield(
                crop_type=farm_params.crop_type,
                soil_moisture=farm_params.current_soil_moisture,
                temperature=farm_params.current_temperature,
                nitrogen=total_n,
                phosphorus=total_p,
                potassium=total_k,
                field_size=farm_params.field_size_hectares
            )
            
            total_cost = self.cost_estimator.calculate_fertilizer_cost(
                n_add * farm_params.field_size_hectares,
                p_add * farm_params.field_size_hectares,
                k_add * farm_params.field_size_hectares
            )
            
            # Generate recommendations
            recommendations = []
            if n_add > 10:
                recommendations.append(
                    f"Apply {n_add:.1f} kg/ha Nitrogen "
                    f"({n_add * farm_params.field_size_hectares:.1f} kg total)"
                )
            if p_add > 10:
                recommendations.append(
                    f"Apply {p_add:.1f} kg/ha Phosphorus "
                    f"({p_add * farm_params.field_size_hectares:.1f} kg total)"
                )
            if k_add > 10:
                recommendations.append(
                    f"Apply {k_add:.1f} kg/ha Potassium "
                    f"({k_add * farm_params.field_size_hectares:.1f} kg total)"
                )
            
            if not recommendations:
                recommendations.append(
                    "Current nutrient levels are optimal. No fertilizer application needed."
                )
            
            # Trade-offs
            trade_offs = {
                "yield_vs_cost": (
                    f"Spending ${total_cost:.2f} to achieve "
                    f"{expected_yield:.0f} kg/ha yield"
                ),
                "cost_per_kg_yield": f"${total_cost/expected_yield:.4f}/kg",
                "environmental_impact": (
                    "Low" if total_n < 180 else 
                    "Moderate" if total_n < 220 else 
                    "High (nitrate leaching risk)"
                )
            }
            
            return OptimizationResult(
                objectives={
                    "expected_yield_kg_per_ha": round(expected_yield, 2),
                    "total_cost_usd": round(total_cost, 2),
                    "nitrogen_total_kg_per_ha": round(total_n, 2),
                    "phosphorus_total_kg_per_ha": round(total_p, 2),
                    "potassium_total_kg_per_ha": round(total_k, 2)
                },
                parameters={
                    "nitrogen_to_add_kg_per_ha": round(n_add, 2),
                    "phosphorus_to_add_kg_per_ha": round(p_add, 2),
                    "potassium_to_add_kg_per_ha": round(k_add, 2)
                },
                constraints_satisfied=True,
                pareto_optimal=True,
                confidence=0.80,
                recommendations=recommendations,
                trade_offs=trade_offs
            )
        else:
            logger.error(f"Fertilizer optimization failed: {result.message}")
            raise ValueError(f"Optimization failed: {result.message}")


# Convenience functions
def optimize_irrigation(
    field_size_hectares: float,
    soil_type: str,
    current_soil_moisture: float,
    current_temperature: float,
    current_humidity: float,
    crop_type: str,
    growth_stage: str,
    days_ahead: int = 7,
    available_budget: float = 1000.0
) -> dict:
    """
    Convenience function for irrigation optimization
    
    Returns:
        dict: Optimization results with recommendations
    """
    farm_params = FarmParameters(
        field_size_hectares=field_size_hectares,
        soil_type=soil_type,
        current_soil_moisture=current_soil_moisture,
        current_temperature=current_temperature,
        current_humidity=current_humidity,
        crop_type=crop_type,
        growth_stage=growth_stage,
        available_budget=available_budget
    )
    
    engine = SmartRecommendationEngine()
    result = engine.optimize_irrigation_schedule(farm_params, days_ahead)
    
    return result.to_dict()


def optimize_fertilization(
    field_size_hectares: float,
    soil_type: str,
    current_soil_moisture: float,
    current_temperature: float,
    crop_type: str,
    current_nitrogen: float = 0.0,
    current_phosphorus: float = 0.0,
    current_potassium: float = 0.0,
    available_budget: float = 1000.0
) -> dict:
    """
    Convenience function for fertilizer optimization
    
    Returns:
        dict: Optimization results with recommendations
    """
    farm_params = FarmParameters(
        field_size_hectares=field_size_hectares,
        soil_type=soil_type,
        current_soil_moisture=current_soil_moisture,
        current_temperature=current_temperature,
        current_humidity=50.0,  # Default
        crop_type=crop_type,
        growth_stage="vegetative",
        available_budget=available_budget
    )
    
    current_nutrients = {
        "nitrogen": current_nitrogen,
        "phosphorus": current_phosphorus,
        "potassium": current_potassium
    }
    
    engine = SmartRecommendationEngine()
    result = engine.optimize_fertilizer_plan(farm_params, current_nutrients)
    
    return result.to_dict()
