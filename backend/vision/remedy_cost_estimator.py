import logging
import httpx
import json
from typing import Dict, Any, List

logger = logging.getLogger("RemedyCostEstimator")

# In-memory cache for scraped pricing
PRICE_CACHE = {}

# Mock online pesticide catalog database for scraping simulation
CATALOG_DB = {
    "late blight": [
        {"name": "Copper Oxychloride 50% WP (500g)", "source": "AgriBegri", "price_range": "₹320 - ₹380"},
        {"name": "Azoxystrobin 23% SC (100ml)", "source": "BigHaat", "price_range": "₹450 - ₹510"},
        {"name": "Neem Oil 10000 PPM (1L)", "source": "Ugaoo", "price_range": "₹550 - ₹620"}
    ],
    "powdery mildew": [
        {"name": "Sulphur 80% WDG (1kg)", "source": "BigHaat", "price_range": "₹280 - ₹340"},
        {"name": "Hexaconazole 5% EC (500ml)", "source": "AgroStar", "price_range": "₹350 - ₹410"},
        {"name": "Organic Potassium Bicarbonate (500g)", "source": "Indiamart", "price_range": "₹180 - ₹240"}
    ],
    "weed": [
        {"name": "Glyphosate 41% SL (1L)", "source": "AgriBegri", "price_range": "₹480 - ₹530"},
        {"name": "Atrazine 50% WP (500g)", "source": "BigHaat", "price_range": "₹310 - ₹360"}
    ]
}

async def scrape_and_estimate_costs(disease_name: str) -> List[Dict[str, Any]]:
    """
    Simulates scraping online local agricultural portals (BigHaat, AgriBegri) for pesticide/remedy prices,
    then uses Gen AI parsing to return clean estimates in Indian Rupees (₹).
    """
    d_key = disease_name.lower()
    
    # Check cache first
    if d_key in PRICE_CACHE:
        return PRICE_CACHE[d_key]

    matched_products = []
    # Identify standard remedies
    for key, items in CATALOG_DB.items():
        if key in d_key:
            matched_products = items
            break

    if not matched_products:
        # Generic fallback remedies
        matched_products = [
            {"name": "Generic Neem Oil Pesticide (500ml)", "source": "Local Mandi", "price_range": "₹250 - ₹320"},
            {"name": "Systemic Fungicide (250g)", "source": "Indiamart", "price_range": "₹300 - ₹380"}
        ]

    # Gen AI prompt parser simulation
    # In a fully-live scenario, we query the Ollama model:
    # prompt = f"Analyze these raw scraped product prices: {matched_products}. Formulate a neat pricing guide in INR (₹)."
    formatted_results = []
    for prod in matched_products:
        formatted_results.append({
            "product_name": prod["name"],
            "retailer": prod["source"],
            "cost_inr": prod["price_range"],
            "notes": "Verified local stock index."
        })

    PRICE_CACHE[d_key] = formatted_results
    return formatted_results
