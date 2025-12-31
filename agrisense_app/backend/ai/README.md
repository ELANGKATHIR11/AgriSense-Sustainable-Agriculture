# 🧠 AgriSense AI Modules

**Advanced AI capabilities for intelligent agricultural decision-making**

---

## 📚 Table of Contents

1. [Overview](#overview)
2. [Smart Recommendations](#smart-recommendations)
3. [Explainable AI](#explainable-ai)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [API Reference](#api-reference)
7. [Examples](#examples)
8. [Best Practices](#best-practices)

---

## 🎯 Overview

The AgriSense AI modules provide cutting-edge artificial intelligence capabilities for agricultural optimization:

### **Smart Recommendations** (`ai/smart_recommendations.py`)
Multi-objective optimization engine that balances:
- 📈 **Maximize Yield**: Optimize for maximum crop production
- 💧 **Minimize Water**: Reduce water consumption and costs
- 💰 **Minimize Costs**: Optimize fertilizer and operational expenses
- 🌍 **Minimize Environmental Impact**: Reduce nitrate leaching and carbon footprint

### **Explainable AI** (`ai/explainable_ai.py`)
Makes ML predictions interpretable using:
- 🔍 **SHAP**: SHapley Additive exPlanations for feature importance
- 🎯 **LIME**: Local Interpretable Model-agnostic Explanations
- 📊 **Feature Contributions**: Understand which factors drive predictions
- 💬 **Natural Language**: Farmer-friendly explanations in plain English

---

## 🚀 Smart Recommendations

### Features

1. **Irrigation Optimization**
   - Optimizes irrigation schedule for next 7-30 days
   - Balances yield, water usage, and costs
   - Accounts for weather forecasts and soil conditions
   - Generates day-by-day irrigation plans

2. **Fertilizer Optimization**
   - Optimizes N-P-K (Nitrogen-Phosphorus-Potassium) application
   - Minimizes costs while maximizing yield
   - Considers environmental impact (nitrate leaching)
   - Provides precise application rates

3. **Multi-Objective Optimization**
   - Uses constraint-based optimization (scipy.optimize)
   - Finds Pareto-optimal solutions
   - Trade-off analysis between objectives
   - Supports custom constraints (budget, water limits)

### Algorithm Details

**Optimization Method**: Sequential Least Squares Programming (SLSQP)
- Gradient-based optimization
- Handles constraints and bounds
- Fast convergence for agricultural problems
- Alternative: Differential Evolution for global optimization

**Yield Prediction Model**:
```python
yield = base_yield × efficiency_factor(moisture, temp, N, P, K)
efficiency_factor = weighted_average(
    moisture_efficiency × 0.3,
    temp_efficiency × 0.2,
    nitrogen_efficiency × 0.2,
    phosphorus_efficiency × 0.15,
    potassium_efficiency × 0.15
)
```

**Water Estimation Model**:
```python
water_needed = base_requirement × temp_factor × humidity_factor × moisture_deficit
actual_water = water_needed / irrigation_efficiency
```

### Usage Example

```python
from agrisense_app.backend.ai.smart_recommendations import (
    optimize_irrigation,
    optimize_fertilization
)

# Optimize irrigation schedule
result = optimize_irrigation(
    field_size_hectares=5.0,
    soil_type="loamy",
    current_soil_moisture=42.0,
    current_temperature=28.5,
    current_humidity=65.0,
    crop_type="tomato",
    growth_stage="flowering",
    days_ahead=7,
    available_budget=500.0
)

print(result['recommendations'])
# Output:
# [
#   "Day 1: Irrigate 12500 liters (2500 L/ha)",
#   "Day 2: Irrigate 10000 liters (2000 L/ha)",
#   ...
# ]

print(result['objectives'])
# Output:
# {
#   "expected_yield_kg_per_ha": 58500.0,
#   "total_water_liters": 65000.0,
#   "total_cost_usd": 97.50
# }
```

### API Endpoints

Add to your FastAPI app:

```python
from agrisense_app.backend.ai.smart_recommendations import SmartRecommendationEngine

@app.post("/api/v1/recommendations/irrigation")
async def recommend_irrigation(
    field_size: float,
    soil_type: str,
    current_moisture: float,
    temperature: float,
    humidity: float,
    crop_type: str
):
    """
    Get optimized irrigation schedule
    """
    result = optimize_irrigation(
        field_size_hectares=field_size,
        soil_type=soil_type,
        current_soil_moisture=current_moisture,
        current_temperature=temperature,
        current_humidity=humidity,
        crop_type=crop_type,
        growth_stage="vegetative",
        days_ahead=7
    )
    return result

@app.post("/api/v1/recommendations/fertilizer")
async def recommend_fertilizer(
    field_size: float,
    crop_type: str,
    current_n: float,
    current_p: float,
    current_k: float
):
    """
    Get optimized fertilizer application plan
    """
    result = optimize_fertilization(
        field_size_hectares=field_size,
        soil_type="loamy",
        current_soil_moisture=50.0,
        current_temperature=25.0,
        crop_type=crop_type,
        current_nitrogen=current_n,
        current_phosphorus=current_p,
        current_potassium=current_k
    )
    return result
```

---

## 🔍 Explainable AI

### Features

1. **Multiple Explanation Methods**
   - **SHAP**: Precise feature attributions using game theory
   - **LIME**: Local explanations for black-box models
   - **Rule-Based**: Domain knowledge-based explanations (no dependencies)

2. **Natural Language Generation**
   - Converts technical outputs to farmer-friendly language
   - Contextualizes values against optimal ranges
   - Explains positive/negative contributions

3. **Actionable Insights**
   - Specific recommendations to improve predictions
   - Quantified adjustments needed (e.g., "Increase moisture by 15%")
   - Prioritized by impact

4. **What-If Scenarios**
   - Shows alternative outcomes
   - Demonstrates impact of changes
   - Helps farmers understand trade-offs

### Installation

```bash
# Install SHAP (optional, for advanced explanations)
pip install shap

# Install LIME (optional, for local explanations)
pip install lime

# Both are optional - rule-based explanations work without them
```

### Usage Example

```python
from agrisense_app.backend.ai.explainable_ai import (
    ExplainableAI,
    explain_model_prediction
)
import joblib

# Load your model
model = joblib.load("ml_models/disease_detection.joblib")

# Define features
feature_names = [
    "temperature", "humidity", "soil_moisture",
    "ph_level", "nitrogen", "phosphorus", "potassium"
]

# Input data
input_features = {
    "temperature": 32.5,
    "humidity": 45.0,
    "soil_moisture": 35.0,
    "ph_level": 6.8,
    "nitrogen": 120.0,
    "phosphorus": 60.0,
    "potassium": 150.0
}

# Get explanation (no dependencies needed)
explanation = explain_model_prediction(
    model=model,
    feature_names=feature_names,
    input_features=input_features,
    model_type="classifier",
    method="rule_based"  # or "shap" if installed
)

print(explanation['natural_language_explanation'])
# Output:
# "The model predicts class 1 with 87.5% confidence.
#  Key factors influencing this prediction:
#  1. The soil moisture is 35.0 (below optimal range of 40-60),
#     which negatively impacts the prediction (contribution: 28.3%).
#  2. The temperature is 32.5 (above optimal range of 20-30),
#     which negatively impacts the prediction (contribution: 21.7%).
#  3. The humidity is 45.0 (below optimal range of 50-70),
#     which negatively impacts the prediction (contribution: 15.2%)."

print(explanation['actionable_insights'])
# Output:
# [
#   "⚠️ Increase soil moisture by 5.0 to reach optimal range (40-60)",
#   "⚠️ Decrease temperature by 2.5 to reach optimal range (20-30)",
#   "⚠️ Increase humidity by 5.0 to reach optimal range (50-70)"
# ]
```

### Advanced Usage with SHAP

```python
from agrisense_app.backend.ai.explainable_ai import ExplainableAI, ExplanationMethod

# Initialize explainer
explainer = ExplainableAI(
    model=disease_model,
    feature_names=feature_names,
    model_type="classifier"
)

# Setup SHAP with background data
import numpy as np
background_data = np.random.randn(100, len(feature_names))  # Or use real training data
explainer.setup_shap_explainer(background_data)

# Get SHAP-based explanation
explanation = explainer.explain_prediction(
    input_features=input_features,
    method=ExplanationMethod.SHAP,
    top_k=5
)

# Access detailed feature contributions
for fc in explanation.feature_contributions:
    print(f"{fc.feature_name}: {fc.contribution:.4f} ({fc.direction})")
```

### API Integration

```python
from agrisense_app.backend.ai.explainable_ai import explain_model_prediction

@app.post("/api/v1/predict/explain")
async def predict_with_explanation(
    temperature: float,
    humidity: float,
    soil_moisture: float,
    ph_level: float,
    nitrogen: float,
    phosphorus: float,
    potassium: float
):
    """
    Make prediction and provide explanation
    """
    # Prepare input
    input_features = {
        "temperature": temperature,
        "humidity": humidity,
        "soil_moisture": soil_moisture,
        "ph_level": ph_level,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium
    }
    
    # Get explanation
    explanation = explain_model_prediction(
        model=global_disease_model,
        feature_names=FEATURE_NAMES,
        input_features=input_features,
        model_type="classifier",
        method="shap"  # or "rule_based"
    )
    
    return {
        "prediction": explanation['prediction'],
        "confidence": explanation['prediction_confidence'],
        "explanation": explanation['natural_language_explanation'],
        "insights": explanation['actionable_insights'],
        "top_features": explanation['feature_contributions'][:3],
        "alternative_scenarios": explanation['alternative_scenarios']
    }
```

---

## 📦 Installation

### Basic Installation (No AI dependencies)

```bash
# Core requirements only
pip install numpy scipy pandas
```

### Full Installation (With SHAP and LIME)

```bash
# Install AI libraries
pip install numpy scipy pandas scikit-learn
pip install shap lime

# For visualization (optional)
pip install matplotlib seaborn
```

### Docker Installation

```dockerfile
# In your Dockerfile
RUN pip install numpy scipy pandas scikit-learn shap lime
```

---

## 🎯 Quick Start

### 1. Import Modules

```python
from agrisense_app.backend.ai.smart_recommendations import (
    SmartRecommendationEngine,
    FarmParameters,
    optimize_irrigation,
    optimize_fertilization
)

from agrisense_app.backend.ai.explainable_ai import (
    ExplainableAI,
    ExplanationMethod,
    explain_model_prediction
)
```

### 2. Optimize Irrigation

```python
# Get irrigation recommendations
irrigation_plan = optimize_irrigation(
    field_size_hectares=10.0,
    soil_type="loamy",
    current_soil_moisture=45.0,
    current_temperature=28.0,
    current_humidity=60.0,
    crop_type="wheat",
    growth_stage="vegetative",
    days_ahead=7,
    available_budget=1000.0
)

# Display results
print("Expected Yield:", irrigation_plan['objectives']['expected_yield_kg_per_ha'], "kg/ha")
print("Water Needed:", irrigation_plan['objectives']['total_water_liters'], "liters")
print("Total Cost:", irrigation_plan['objectives']['total_cost_usd'], "USD")
print("\nRecommendations:")
for rec in irrigation_plan['recommendations']:
    print(f"  • {rec}")
```

### 3. Optimize Fertilizer

```python
# Get fertilizer recommendations
fertilizer_plan = optimize_fertilization(
    field_size_hectares=10.0,
    soil_type="clay",
    current_soil_moisture=55.0,
    current_temperature=24.0,
    crop_type="rice",
    current_nitrogen=80.0,
    current_phosphorus=40.0,
    current_potassium=70.0,
    available_budget=800.0
)

# Display results
print("Expected Yield:", fertilizer_plan['objectives']['expected_yield_kg_per_ha'], "kg/ha")
print("Total Cost:", fertilizer_plan['objectives']['total_cost_usd'], "USD")
print("\nRecommendations:")
for rec in fertilizer_plan['recommendations']:
    print(f"  • {rec}")
```

### 4. Explain Predictions

```python
import joblib

# Load model
model = joblib.load("ml_models/crop_recommendation.joblib")

# Get explanation
explanation = explain_model_prediction(
    model=model,
    feature_names=["temperature", "humidity", "soil_moisture", "nitrogen", "phosphorus", "potassium"],
    input_features={
        "temperature": 25.0,
        "humidity": 65.0,
        "soil_moisture": 52.0,
        "nitrogen": 150.0,
        "phosphorus": 80.0,
        "potassium": 180.0
    },
    model_type="classifier",
    method="rule_based"
)

# Display explanation
print(explanation['natural_language_explanation'])
print("\nActionable Insights:")
for insight in explanation['actionable_insights']:
    print(f"  {insight}")
```

---

## 📖 API Reference

### Smart Recommendations

#### `optimize_irrigation()`

```python
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
) -> dict
```

**Returns:**
```python
{
    "objectives": {
        "expected_yield_kg_per_ha": 58500.0,
        "total_water_liters": 65000.0,
        "total_cost_usd": 97.50
    },
    "parameters": {
        "day_1_irrigation_liters": 12500.0,
        "day_2_irrigation_liters": 10000.0,
        ...
    },
    "constraints_satisfied": true,
    "pareto_optimal": true,
    "confidence": 0.85,
    "recommendations": [...],
    "trade_offs": {...}
}
```

#### `optimize_fertilization()`

```python
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
) -> dict
```

**Returns:**
```python
{
    "objectives": {
        "expected_yield_kg_per_ha": 7800.0,
        "total_cost_usd": 245.60,
        "nitrogen_total_kg_per_ha": 150.0,
        "phosphorus_total_kg_per_ha": 80.0,
        "potassium_total_kg_per_ha": 150.0
    },
    "parameters": {
        "nitrogen_to_add_kg_per_ha": 70.0,
        "phosphorus_to_add_kg_per_ha": 40.0,
        "potassium_to_add_kg_per_ha": 80.0
    },
    ...
}
```

### Explainable AI

#### `explain_model_prediction()`

```python
def explain_model_prediction(
    model,
    feature_names: List[str],
    input_features: Dict[str, float],
    model_type: str = "classifier",
    method: str = "rule_based"
) -> dict
```

**Returns:**
```python
{
    "prediction": 1,
    "prediction_confidence": 0.875,
    "method": "rule_based",
    "feature_contributions": [
        {
            "feature_name": "soil_moisture",
            "value": 35.0,
            "contribution": -0.2834,
            "contribution_percent": 28.34,
            "direction": "negative",
            "importance_rank": 1
        },
        ...
    ],
    "natural_language_explanation": "...",
    "actionable_insights": [...],
    "alternative_scenarios": [...]
}
```

---

## 💡 Best Practices

### Smart Recommendations

1. **Use Realistic Constraints**
   ```python
   # Set realistic budget and water limits
   available_budget=1000.0,  # USD
   max_water_daily_liters=50000.0  # Liters
   ```

2. **Consider Weather Forecasts**
   ```python
   # Include 7-day weather forecast if available
   farm_params.weather_forecast_7day = {
       "temperature": [28, 30, 29, 27, 26, 28, 29],
       "rainfall": [0, 5, 0, 0, 10, 0, 0]  # mm
   }
   ```

3. **Validate Crop Types**
   ```python
   SUPPORTED_CROPS = ["tomato", "wheat", "rice", "corn"]
   if crop_type not in SUPPORTED_CROPS:
       raise ValueError(f"Unsupported crop: {crop_type}")
   ```

### Explainable AI

1. **Choose Right Method**
   - Use **SHAP** for precise attributions (requires SHAP library)
   - Use **LIME** for model-agnostic explanations (requires LIME library)
   - Use **Rule-Based** for fast, dependency-free explanations

2. **Provide Background Data for SHAP**
   ```python
   # Use representative training data
   explainer.setup_shap_explainer(background_data=X_train[:100])
   ```

3. **Limit Feature Count**
   ```python
   # Show only top 5 features for clarity
   explanation = explainer.explain_prediction(
       input_features,
       top_k=5
   )
   ```

---

## 🔬 Examples

### Example 1: Complete Irrigation Workflow

```python
from agrisense_app.backend.ai.smart_recommendations import optimize_irrigation

# Get sensor data
sensor_data = {
    "soil_moisture": 38.0,
    "temperature": 32.0,
    "humidity": 48.0
}

# Optimize irrigation
plan = optimize_irrigation(
    field_size_hectares=15.0,
    soil_type="loamy",
    current_soil_moisture=sensor_data['soil_moisture'],
    current_temperature=sensor_data['temperature'],
    current_humidity=sensor_data['humidity'],
    crop_type="tomato",
    growth_stage="fruiting",
    days_ahead=7,
    available_budget=2000.0
)

# Display results
print(f"💧 Water Plan: {plan['objectives']['total_water_liters']/1000:.1f} m³")
print(f"💰 Estimated Cost: ${plan['objectives']['total_cost_usd']:.2f}")
print(f"📈 Expected Yield: {plan['objectives']['expected_yield_kg_per_ha']:.0f} kg/ha")
print(f"\n📋 Schedule:")
for rec in plan['recommendations']:
    print(f"   {rec}")
```

### Example 2: Explainable Disease Prediction

```python
from agrisense_app.backend.ai.explainable_ai import explain_model_prediction
import joblib

# Load disease detection model
model = joblib.load("ml_models/disease_model.joblib")

# Sensor readings
readings = {
    "temperature": 28.5,
    "humidity": 82.0,
    "soil_moisture": 68.0,
    "ph_level": 6.2,
    "nitrogen": 140.0,
    "phosphorus": 75.0,
    "potassium": 160.0
}

# Get prediction with explanation
result = explain_model_prediction(
    model=model,
    feature_names=list(readings.keys()),
    input_features=readings,
    model_type="classifier",
    method="rule_based"
)

# Display results
print(f"🔍 Prediction: Disease Class {result['prediction']}")
print(f"📊 Confidence: {result['prediction_confidence']*100:.1f}%")
print(f"\n💬 Explanation:")
print(result['natural_language_explanation'])
print(f"\n✅ Actions:")
for insight in result['actionable_insights']:
    print(f"   {insight}")
```

---

## 🎓 Further Reading

- **Optimization Theory**: [SciPy Optimization Guide](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
- **SHAP Documentation**: [SHAP GitHub](https://github.com/slundberg/shap)
- **LIME Documentation**: [LIME GitHub](https://github.com/marcotcr/lime)
- **Agricultural Optimization**: FAO Guidelines on Irrigation Scheduling
- **Precision Agriculture**: Journal of Precision Agriculture Research

---

## 🤝 Contributing

To extend the AI modules:

1. **Add New Crops**: Update `CROP_OPTIMA` in `smart_recommendations.py`
2. **Add Objectives**: Extend optimization objectives in `OptimizationObjective` enum
3. **Improve Models**: Replace rule-based yield predictor with trained ML models
4. **Custom Explanations**: Add domain-specific explanation templates

---

## 📄 License

Part of the AgriSense project. See main README for license details.

---

**For questions or support, refer to the main project documentation or create an issue on GitHub.**
