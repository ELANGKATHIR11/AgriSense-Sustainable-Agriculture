import pandas as pd
import random
from pathlib import Path

# Intents and templates
INTENTS = {
    "crop_recommendation": [
        "What crop should I grow in {soil} soil?",
        "Best crops for {soil} soil type?",
        "I have {soil} soil, what can I cultivate?",
        "Suggest crops for high {nutrient} levels.",
        "My soil pH is {ph}, what grows best?",
    ],
    "disease_identification": [
        "My {crop} has {symptom}, what is it?",
        "Yellow leaves on {crop}, help!",
        "IDENTIFY: {crop} with {symptom}",
        "What disease causes {symptom} in {crop}?",
        "How to treat {symptom} on {crop}?",
    ],
    "fertilizer_advice": [
        "How much {fertilizer} for {crop}?",
        "When to apply {fertilizer} to {crop}?",
        "Fertilizer schedule for {crop}",
        "Best organic fertilizer for {crop}?",
        "Is {fertilizer} good for {crop}?",
    ],
    "market_price": [
        "Current price of {crop}?",
        "Market rate for {crop} today",
        "How much is {crop} selling for?",
        "Price trend for {crop}",
        "Is it a good time to sell {crop}?",
    ],
}

SLOTS = {
    "soil": ["black", "red", "alluvial", "clay", "loamy"],
    "nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
    "ph": ["6.5", "7.0", "5.5", "8.0"],
    "crop": ["Rice", "Wheat", "Tomato", "Cotton", "Potato", "Maize"],
    "symptom": ["yellow spots", "wilting", "brown patches", "curled leaves", "rot"],
    "fertilizer": ["Urea", "DAP", "Compost", "NPK", "Manure"],
}


def generate_sentence(template):
    sentence = template
    for slot, values in SLOTS.items():
        if "{" + slot + "}" in sentence:
            sentence = sentence.replace("{" + slot + "}", random.choice(values))
    return sentence


def main():
    print("🤖 Generating Synthetic Chatbot Intents...")

    data = []
    for intent, templates in INTENTS.items():
        # Generate 50 examples per intent
        for _ in range(50):
            tmpl = random.choice(templates)
            utterance = generate_sentence(tmpl)
            data.append({"intent": intent, "utterance": utterance})

    df = pd.DataFrame(data)

    # Save to standard datasets location
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(exist_ok=True)

    output_path = datasets_dir / "chatbot_intents.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Generated {len(df)} intents at: {output_path}")
    print(df["intent"].value_counts())


if __name__ == "__main__":
    main()
