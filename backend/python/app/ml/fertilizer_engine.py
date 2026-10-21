from datetime import datetime
from typing import Dict, Any

CROP_NPK_TARGETS = {
    "rice": {"N": 120, "P": 60, "K": 60},
    "paddy": {"N": 120, "P": 60, "K": 60},
    "wheat": {"N": 120, "P": 60, "K": 40},
    "maize": {"N": 120, "P": 60, "K": 50},
    "cotton": {"N": 100, "P": 50, "K": 50},
    "sugarcane": {"N": 250, "P": 100, "K": 120},
    "banana": {"N": 200, "P": 70, "K": 300},
    "tomato": {"N": 150, "P": 80, "K": 100},
    "potato": {"N": 160, "P": 100, "K": 120},
    "chickpea": {"N": 30, "P": 60, "K": 30},
}

class FertilizerEngine:
    def calculate(self, crop_type: str, n: float, p: float, k: float, soil_ph: float, moisture: float) -> Dict[str, Any]:
        key = crop_type.lower().split()[0]
        targets = CROP_NPK_TARGETS.get(key, {"N": 100, "P": 50, "K": 50})

        # Calculate nutrient deficits (cannot be negative)
        n_deficit = max(0.0, round(targets["N"] - n, 1))
        p_deficit = max(0.0, round(targets["P"] - p, 1))
        k_deficit = max(0.0, round(targets["K"] - k, 1))

        # Commercial fertilizer products (Urea: 46% N, DAP: 18% N + 46% P, MOP: 60% K)
        dap_dosage = round((p_deficit / 0.46), 1) if p_deficit > 0 else 0.0
        n_from_dap = dap_dosage * 0.18
        remaining_n = max(0.0, n_deficit - n_from_dap)
        urea_dosage = round((remaining_n / 0.46), 1) if remaining_n > 0 else 0.0
        mop_dosage = round((k_deficit / 0.60), 1) if k_deficit > 0 else 0.0

        schedule = []
        if dap_dosage > 0:
            schedule.append({
                "productKey": "dap",
                "productName": "Di-Ammonium Phosphate (18:46:0)",
                "dosageKgHa": dap_dosage,
                "timingKey": "basal_sowing",
                "applicationNote": "Incorporate 5-7 cm below seed furrow at sowing."
            })
        if urea_dosage > 0:
            schedule.append({
                "productKey": "urea_split1",
                "productName": "Neem-Coated Urea (46% N) - Split 1",
                "dosageKgHa": round(urea_dosage * 0.5, 1),
                "timingKey": "active_tillering",
                "applicationNote": "Broadcast during active tillering stage under moist soil."
            })
            schedule.append({
                "productKey": "urea_split2",
                "productName": "Neem-Coated Urea (46% N) - Split 2",
                "dosageKgHa": round(urea_dosage * 0.5, 1),
                "timingKey": "panicle_initiation",
                "applicationNote": "Top-dress before panicle initiation; avoid waterlogging."
            })
        if mop_dosage > 0:
            schedule.append({
                "productKey": "mop",
                "productName": "Muriate of Potash (60% K2O)",
                "dosageKgHa": mop_dosage,
                "timingKey": "basal_vegetative",
                "applicationNote": "Apply 50% basal and 50% at flowering to enhance grain filling."
            })

        # Organic amendment recommendation
        organic_amendment = (
            "Well-decomposed Farm Yard Manure (FYM) @ 10 tonnes/ha + Azospirillum @ 2 kg/ha bio-inoculant."
            if soil_ph >= 6.5 else
            "Agricultural Lime @ 500 kg/ha with FYM to correct soil acidity and enhance phosphorus availability."
        )

        # Moisture warning
        moisture_warning = None
        if moisture < 30.0:
            moisture_warning = "CRITICAL: Soil moisture is below 30%. Irrigate before applying urea to prevent ammonia volatilization."
        elif moisture > 85.0:
            moisture_warning = "WARNING: Saturated soil conditions detected. Delay nitrogen top-dressing to prevent denitrification and leaching."

        return {
            "cropType": crop_type,
            "deficitsKgHa": {
                "nitrogen": n_deficit,
                "phosphorus": p_deficit,
                "potassium": k_deficit
            },
            "schedule": schedule,
            "organicAmendment": organic_amendment,
            "moistureWarning": moisture_warning,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }

fertilizer_engine = FertilizerEngine()
