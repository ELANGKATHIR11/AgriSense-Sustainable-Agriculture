from backend import twin_engine


def adapt_disease_to_twin(disease_name: str, confidence: float, severity: str):
    """Adapts disease analysis findings directly into the Disease digital twin state."""
    state = twin_engine.get_state()

    # Map severity to risk values
    severity_map = {"low": 30, "medium": 65, "high": 90}
    risk_score = severity_map.get(severity.lower(), 45)

    state["diseaseTwin"]["riskScore"] = int(risk_score)
    state["diseaseTwin"]["outbreakProbability"] = int(risk_score * 0.85)

    if disease_name not in state["diseaseTwin"]["susceptibleCrops"]:
        state["diseaseTwin"]["susceptibleCrops"].append(disease_name)

    # Recalculate twin state
    twin_engine.update_state()


def adapt_nutrient_to_twin(deficiency: str, severity: str):
    """Adapts nutrient deficiency detections into the Soil digital twin state."""
    state = twin_engine.get_state()

    # Map depletion impact based on severity
    depletion = (
        5 if severity.lower() == "low" else (12 if severity.lower() == "medium" else 20)
    )

    def_lower = deficiency.lower()
    if "nitrogen" in def_lower:
        state["soilTwin"]["nitrogen"] = max(
            10, state["soilTwin"]["nitrogen"] - depletion
        )
    elif "phosphorus" in def_lower:
        state["soilTwin"]["phosphorus"] = max(
            10, state["soilTwin"]["phosphorus"] - depletion
        )
    elif "potassium" in def_lower:
        state["soilTwin"]["potassium"] = max(
            10, state["soilTwin"]["potassium"] - depletion
        )

    # Recalculate twin state
    twin_engine.update_state()


def adapt_health_to_twin(health_score: float, stress_level: str):
    """Adapts general plant health scores into the Crop digital twin state."""
    state = twin_engine.get_state()
    state["cropTwin"]["cropHealthScore"] = (
        int(health_score * 100) if health_score <= 1.0 else int(health_score)
    )

    # Adjust predicted yield multiplier based on plant health
    if stress_level.lower() == "high":
        state["cropTwin"]["predictedYieldMultiplier"] = max(
            0.5, state["cropTwin"]["predictedYieldMultiplier"] - 0.15
        )
    elif stress_level.lower() == "low":
        state["cropTwin"]["predictedYieldMultiplier"] = min(
            1.5, state["cropTwin"]["predictedYieldMultiplier"] + 0.05
        )

    # Recalculate twin state
    twin_engine.update_state()


def adapt_yield_to_twin(estimated_yield: float):
    """Adapts yield estimation predictions into the main Farm / Crop digital twin index."""
    state = twin_engine.get_state()

    # Adjust projected yield multiplier dynamically
    if estimated_yield > 0:
        multiplier = min(1.8, estimated_yield / 4.0)
        state["cropTwin"]["predictedYieldMultiplier"] = round(multiplier, 2)

    # Recalculate twin state
    twin_engine.update_state()
