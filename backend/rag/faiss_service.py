# -*- coding: utf-8 -*-
"""
Lightweight knowledge base lookup - pure NumPy, no FAISS, no heavy model loading.
Prevents segfaults on Windows caused by FAISS AVX2 / BGE-M3 loading.
"""
from typing import List, Dict, Any

# In-memory lightweight knowledge base - no model required
_KNOWLEDGE_BASE = [
    {"text": "Tomato Leaf Mold is caused by Passalora fulva. Symptoms include yellow spots on upper leaf surfaces and olive-green velvet mold underneath. Improve greenhouse ventilation and avoid overhead watering.", "disease": "Tomato Leaf Mold", "keywords": ["tomato", "leaf", "mold", "yellow", "spots", "velvet"]},
    {"text": "Late Blight on Squash is caused by Phytophthora. Dark water-soaked lesions appear on leaves. Apply copper-based biological fungicide immediately.", "disease": "Late Blight", "keywords": ["blight", "squash", "phytophthora", "lesion", "dark", "necrosis"]},
    {"text": "Nitrogen deficiency causes uniform yellowing of older leaves starting at the tips. Supplement with organic blood meal, urea, or legume compost.", "nutrient": "Nitrogen", "keywords": ["nitrogen", "npk", "yellowing", "deficiency", "fertilizer", "n"]},
    {"text": "Potassium deficiency leads to leaf margin curling and brown necrotic edges. Apply wood ash or kelp meal for correction.", "nutrient": "Potassium", "keywords": ["potassium", "k", "curling", "necrosis", "margin", "deficiency"]},
    {"text": "Phosphorus deficiency shows as purple/red discolouration on undersides of leaves. Apply bone meal or rock phosphate.", "nutrient": "Phosphorus", "keywords": ["phosphorus", "p", "purple", "red", "deficiency", "discolouration"]},
    {"text": "Powdery Mildew shows white talcum-like powdery spots on leaves. Apply neem oil extract or potassium bicarbonate. Ensure full sunlight and good spacing.", "disease": "Powdery Mildew", "keywords": ["mildew", "powdery", "white", "spots", "fungal", "squash"]},
    {"text": "Corn Common Rust shows reddish-brown powdery pustules on both leaf surfaces. Apply strobilurin or triazole fungicides. Plant rust-resistant hybrids.", "disease": "Corn Rust", "keywords": ["rust", "corn", "maize", "pustules", "orange", "fungal"]},
    {"text": "Weeds compete for soil nitrogen and moisture. Use mulching or selective organic pre-emergents to control weed growth.", "weed": "Weed competition", "keywords": ["weed", "mulch", "competition", "organic", "pre-emergent"]},
    {"text": "Sandy soil has low water retention and nutrients. Add organic matter (compost), use drip irrigation, and apply slow-release NPK fertilizers.", "soil": "Sandy Soil", "keywords": ["sandy", "soil", "drainage", "water", "retention", "compost"]},
    {"text": "For rice cultivation, optimal soil pH is 5.5-6.5. Maintain 80-90% soil moisture. Use nitrogen fertilizers in split doses.", "crop": "Rice", "keywords": ["rice", "paddy", "ph", "moisture", "nitrogen", "cultivation"]},
    {"text": "For tomato cultivation, optimal soil pH is 6.0-6.8. Water deeply but infrequently. Apply calcium to prevent blossom end rot.", "crop": "Tomato", "keywords": ["tomato", "ph", "calcium", "blossom", "water", "cultivation"]},
    {"text": "Irrigation optimization: water requirements depend on crop type, soil moisture, temperature and evapotranspiration. Use drip irrigation for 40% water savings.", "topic": "Irrigation", "keywords": ["irrigation", "water", "drip", "moisture", "evapotranspiration", "schedule"]},
]

def query_kb(query: str, k: int = 2) -> List[Dict[str, Any]]:
    """
    Lightweight keyword-based knowledge base lookup.
    No heavy models, no FAISS - just fast string matching.
    """
    query_lower = query.lower()
    scored = []
    for doc in _KNOWLEDGE_BASE:
        score = sum(1 for kw in doc.get("keywords", []) if kw in query_lower)
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "text": doc["text"],
            "score": float(s),
            "metadata": {k2: v for k2, v in doc.items() if k2 not in ("text", "keywords")}
        }
        for s, doc in scored[:k]
    ]

def init_kb():
    """No-op - knowledge base is initialized at module level."""
    pass
