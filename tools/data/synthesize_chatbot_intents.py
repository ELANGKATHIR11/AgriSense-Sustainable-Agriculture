"""
Synthesize chatbot intent-label dataset for AgriSense
Produces CSV: agrisense_app/backend/data/chatbot_intents.csv
Columns: intent,utterance
"""
import csv
import random
from pathlib import Path

INPUT = Path("agrisense_app/backend/india_crop_dataset.csv")
OUTPUT_DIR = Path("agrisense_app/backend/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT = OUTPUT_DIR / "chatbot_intents.csv"

SAMPLES_PER_CROP = 5
random.seed(1234)

INTENTS = {
    "recommend_crop": [
        "What crop should I grow if my soil pH is {ph} and rainfall {rain}mm?",
        "Recommend a crop for pH {ph} and temperature {temp}C",
        "Which crop suits my soil with pH {ph} and rainfall {rain}mm?",
        "Suggest a crop for low nitrogen soils (N={n} kg/ha)",
        "Best crop for temperature {temp}C and humidity {humidity}%"
    ],
    "fertilizer_advice": [
        "How much nitrogen should I apply for {crop}?",
        "Fertilizer recommendation for {crop} given N={n}, P={p}, K={k}",
        "What fertilizer schedule is best for {crop}?",
        "Suggest fertilizer amounts for {crop}",
        "When to apply phosphorus for {crop}?"
    ],
    "irrigation_advice": [
        "How often should I irrigate {crop}?",
        "Water requirement for {crop} with rainfall {rain}mm",
        "Irrigation schedule for {crop} in {season} season",
        "How much water does {crop} need per week?",
        "Best irrigation practice for {crop}"
    ],
    "pest_disease_help": [
        "My {crop} leaves are yellow — what could be wrong?",
        "How to manage pests in {crop} fields?",
        "Symptoms of common diseases in {crop}",
        "How to treat fungal infection in {crop}?",
        "Best pesticide practice for {crop}"
    ],
    "planting_schedule": [
        "When should I plant {crop} in my region?",
        "What is the planting month for {crop}?",
        "Optimal planting time for {crop}",
        "When does {crop} mature after planting?",
        "Planting and harvesting months for {crop}"
    ]
}

def read_crops():
    crops = []
    with INPUT.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            crops.append(r)
    return crops


def synthesize():
    crops = read_crops()
    rows = []
    for crop in crops:
        name = crop.get("Crop", "Unknown")
        # numeric fallbacks
        ph_opt = crop.get("pH_Optimal") or crop.get("pH_Min") or "6.5"
        rain = crop.get("Rainfall_Requirement_mm") or "400"
        temp = crop.get("Temperature_Optimal_C") or "25"
        humidity = crop.get("Humidity_Optimal_percent") or "70"
        n = crop.get("Nitrogen_Optimal_kg_ha") or "100"
        p = crop.get("Phosphorus_Optimal_kg_ha") or "40"
        k = crop.get("Potassium_Optimal_kg_ha") or "50"
        season = crop.get("Growing_Season") or "Kharif"

        for intent, templates in INTENTS.items():
            for i in range(SAMPLES_PER_CROP):
                template = random.choice(templates)
                utterance = template.format(
                    crop=name,
                    ph=ph_opt,
                    rain=rain,
                    temp=temp,
                    humidity=humidity,
                    n=n,
                    p=p,
                    k=k,
                    season=season
                )
                rows.append({"intent": intent, "utterance": utterance})

    # Shuffle
    random.shuffle(rows)

    # Write CSV
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["intent", "utterance"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"✅ Generated {len(rows)} chatbot intent samples to {OUTPUT}")

if __name__ == "__main__":
    synthesize()
