import logging
from typing import Dict, Any, List

logger = logging.getLogger("AgriSense-AI")


class RiskEngine:
    def __init__(self):
        # Thresholds based on general agronomic studies
        self.risk_thresholds = {
            "heat_stress": {"temp": 35.0},
            "pest_blight": {"humidity": 85.0, "temp_range": (20.0, 30.0)},
            "drought_risk": {"rainfall": 50.0},
        }

    def analyze_risks(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyzes environmental data for potential risks.
        """
        risks = []
        temp = context.get("temperature_avg_c", 25.0)
        humidity = context.get("humidity_pct", 70.0)
        rainfall = context.get("rainfall_mm", 150.0)

        # 1. Heat Stress
        if temp > self.risk_thresholds["heat_stress"]["temp"]:
            risks.append(
                {
                    "type": "weather",
                    "risk": "High Heat Stress",
                    "severity": "High",
                    "advice": "Increase irrigation frequency to maintain soil moisture during the heatwave.",
                }
            )

        # 2. Pest/Disease Risk (e.g., Blight)
        t_min, t_max = self.risk_thresholds["pest_blight"]["temp_range"]
        if (
            humidity > self.risk_thresholds["pest_blight"]["humidity"]
            and t_min <= temp <= t_max
        ):
            risks.append(
                {
                    "type": "pest",
                    "risk": "Potential Fungal Blight",
                    "severity": "Medium",
                    "advice": "High humidity detected. Check for early signs of leaf spots or fungal growth.",
                }
            )

        # 3. Drought Risk
        if rainfall < self.risk_thresholds["drought_risk"]["rainfall"]:
            risks.append(
                {
                    "type": "weather",
                    "risk": "High Drought Risk",
                    "severity": "Critical",
                    "advice": "Low rainfall season projected. Consider drought-resistant crop varieties or water-saving mulch.",
                }
            )

        return risks


def get_risk_assessment(data: Dict[str, Any]):
    engine = RiskEngine()
    return engine.analyze_risks(data)
