#!/usr/bin/env python3
"""
Edge AI Agricultural Chatbot with Neural Network
Works offline, no API calls needed
Provides cultivation guides for all 96 crops and answers farmer questions
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")

try:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: scikit-learn not available. Using rule-based fallback.")


class EdgeAIChatbot:
    """Edge AI Chatbot for Agricultural Advice"""

    def __init__(self, knowledge_base_path=None):
        self.knowledge_base_path = (
            knowledge_base_path or Path(__file__).parent / "knowledge_base"
        )
        self.knowledge_base_path.mkdir(exist_ok=True)

        self.cultivation_guides = {}
        self.disease_knowledge = {}
        self.qa_model = None
        self.vectorizer = None
        self.scaler = None

        # Load knowledge bases
        self.load_cultivation_guides()
        self.load_disease_knowledge()

        # Initialize models if sklearn available
        if HAS_SKLEARN:
            self.initialize_models()

    def load_cultivation_guides(self):
        """Load cultivation guides for all 96 crops"""
        guides_file = self.knowledge_base_path / "cultivation_guides.json"

        if guides_file.exists():
            with open(guides_file, "r", encoding="utf-8") as f:
                self.cultivation_guides = json.load(f)
        else:
            # Generate default guides if file doesn't exist
            self.cultivation_guides = self.generate_default_guides()
            self.save_cultivation_guides()

    def save_cultivation_guides(self):
        """Save cultivation guides to file"""
        guides_file = self.knowledge_base_path / "cultivation_guides.json"
        with open(guides_file, "w", encoding="utf-8") as f:
            json.dump(self.cultivation_guides, f, indent=2, ensure_ascii=False)

    def generate_default_guides(self) -> Dict:
        """Generate cultivation guides for all 96 crops"""
        # This will be populated with comprehensive guides
        return {}

    def load_disease_knowledge(self):
        """Load disease knowledge base"""
        disease_file = self.knowledge_base_path / "disease_knowledge.json"

        if disease_file.exists():
            with open(disease_file, "r", encoding="utf-8") as f:
                self.disease_knowledge = json.load(f)
        else:
            self.disease_knowledge = {}

    def initialize_models(self):
        """Initialize neural network models for Q&A"""
        if not HAS_SKLEARN:
            return

        # Try to load trained models
        model_dir = Path(__file__).parent / "models"
        qa_model_path = model_dir / "edge_ai_qa_model.pkl"
        vectorizer_path = model_dir / "edge_ai_vectorizer.pkl"

        if qa_model_path.exists() and vectorizer_path.exists():
            try:
                self.qa_model = joblib.load(qa_model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                print("✅ Loaded trained edge AI models")
            except Exception as e:
                print(f"⚠️  Failed to load models: {e}")
                self.train_qa_model()
        else:
            self.train_qa_model()

    def train_qa_model(self):
        """Train Q&A neural network model"""
        if not HAS_SKLEARN:
            return

        # Training data: common agricultural questions and answers
        training_data = self.get_training_data()

        if len(training_data) == 0:
            return

        # Prepare data
        questions = [item["question"] for item in training_data]
        categories = [item["category"] for item in training_data]

        # Vectorize questions
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        X = self.vectorizer.fit_transform(questions).toarray()

        # Train neural network classifier
        self.qa_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        )

        # Encode categories
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(categories)

        self.qa_model.fit(X, y)
        self.category_encoder = le

        # Save models
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        joblib.dump(self.qa_model, model_dir / "edge_ai_qa_model.pkl")
        joblib.dump(self.vectorizer, model_dir / "edge_ai_vectorizer.pkl")
        joblib.dump(le, model_dir / "edge_ai_category_encoder.pkl")

        print("✅ Trained edge AI Q&A model")

    def get_training_data(self) -> List[Dict]:
        """Get training data for Q&A model"""
        return [
            {
                "question": "how to reduce water usage",
                "category": "water_efficiency",
            },
            {
                "question": "how to reduce fertilizer",
                "category": "fertilizer_efficiency",
            },
            {
                "question": "how to increase yield",
                "category": "yield_optimization",
            },
            {
                "question": "what is the best time to plant",
                "category": "planting",
            },
            {"question": "how to control pests", "category": "pest_control"},
            {
                "question": "what diseases affect this crop",
                "category": "disease",
            },
            {"question": "soil preparation", "category": "soil_management"},
            {"question": "irrigation schedule", "category": "irrigation"},
            {"question": "harvesting time", "category": "harvesting"},
            {"question": "fertilizer application", "category": "fertilizer"},
        ]

    def process_query(self, query: str, crop_name: str = None) -> Dict:
        """Process farmer query and return response"""
        query_lower = query.lower().strip()

        # Check if crop name is mentioned or provided
        detected_crop = crop_name or self.detect_crop_name(query)

        # If crop name detected, provide cultivation guide
        if detected_crop:
            guide = self.get_cultivation_guide(detected_crop)
            if guide:
                return {
                    "type": "cultivation_guide",
                    "crop": detected_crop,
                    "response": guide,
                    "suggestions": self.get_optimization_suggestions(
                        detected_crop, query
                    ),
                }

        # Process general questions
        if self.is_cultivation_guide_request(query_lower, detected_crop):
            if detected_crop:
                guide = self.get_cultivation_guide(detected_crop)
                return {
                    "type": "cultivation_guide",
                    "crop": detected_crop,
                    "response": guide,
                }

        # Use neural network for Q&A if available
        if self.qa_model and HAS_SKLEARN:
            category = self.classify_query(query)
            response = self.generate_response_by_category(
                query, category, detected_crop
            )
        else:
            # Rule-based fallback
            response = self.rule_based_response(query_lower, detected_crop)

        return {
            "type": "qa",
            "query": query,
            "crop": detected_crop,
            "response": response,
            "optimization_tips": self.get_optimization_tips(query_lower, detected_crop),
        }

    def detect_crop_name(self, query: str) -> str:
        """Detect crop name from query"""
        # List of all 96 crops
        crops = [
            "Almond",
            "Apple",
            "Arecanut",
            "Arhar",
            "Bajra",
            "Banana",
            "Barley",
            "Barnyard_Millet",
            "Beetroot",
            "Bitter_Gourd",
            "Black_Pepper",
            "Bottle_Gourd",
            "Brinjal",
            "Buckwheat",
            "Cabbage",
            "Cardamom",
            "Carrot",
            "Cashew",
            "Castor",
            "Cauliflower",
            "Chickpea",
            "Chilli",
            "Cluster_Bean",
            "Coconut",
            "Coffee",
            "Coriander",
            "Cotton",
            "Cucumber",
            "Cumin",
            "Custard_Apple",
            "Dragon_Fruit",
            "Fenugreek",
            "Field_Pea",
            "Foxtail_Millet",
            "French_Bean",
            "Garlic",
            "Ginger",
            "Grapes",
            "Green_Pea",
            "Groundnut",
            "Guava",
            "Horse_Gram",
            "Jackfruit",
            "Jowar",
            "Jute",
            "Kidney_Bean",
            "Kodo_Millet",
            "Lentil",
            "Lettuce",
            "Linseed",
            "Litchi",
            "Little_Millet",
            "Maize",
            "Mango",
            "Masoor",
            "Moong",
            "Moth_Bean",
            "Muskmelon",
            "Mustard",
            "Niger",
            "Oats",
            "Okra",
            "Onion",
            "Orange",
            "Papaya",
            "Pearl_Millet",
            "Pigeon_Pea",
            "Pineapple",
            "Pomegranate",
            "Potato",
            "Proso_Millet",
            "Pumpkin",
            "Radish",
            "Ragi",
            "Rice",
            "Ridge_Gourd",
            "Rubber",
            "Safflower",
            "Sapota",
            "Sesame",
            "Sorghum",
            "Soybean",
            "Spinach",
            "Strawberry",
            "Sugarcane",
            "Sunflower",
            "Sweet_Potato",
            "Tea",
            "Tobacco",
            "Tomato",
            "Turmeric",
            "Turnip",
            "Urad",
            "Walnut",
            "Watermelon",
            "Wheat",
        ]

        query_lower = query.lower()
        for crop in crops:
            crop_lower = crop.lower().replace("_", " ")
            if crop_lower in query_lower or crop in query:
                return crop

        return None

    def is_cultivation_guide_request(self, query: str, crop: str) -> bool:
        """Check if query is asking for cultivation guide"""
        guide_keywords = [
            "guide",
            "cultivation",
            "growing",
            "how to grow",
            "planting",
            "care",
            "management",
            "practice",
        ]
        return any(keyword in query for keyword in guide_keywords) and crop is not None

    def get_cultivation_guide(self, crop_name: str) -> Dict:
        """Get cultivation guide for specific crop"""
        crop_normalized = crop_name.replace(" ", "_")

        if crop_normalized in self.cultivation_guides:
            return self.cultivation_guides[crop_normalized]

        # Generate guide if not exists
        guide = self.generate_crop_guide(crop_name)
        self.cultivation_guides[crop_normalized] = guide
        self.save_cultivation_guides()
        return guide

    def generate_crop_guide(self, crop_name: str) -> Dict:
        """Generate comprehensive cultivation guide for a crop"""
        # This will be populated with detailed guides
        return {
            "crop_name": crop_name,
            "overview": f"Comprehensive cultivation guide for {crop_name}",
            "soil_requirements": "Well-drained, fertile soil",
            "climate": "Tropical to subtropical",
            "planting": "Sow seeds or transplant seedlings",
            "watering": "Regular irrigation required",
            "fertilizer": "Balanced NPK fertilizer",
            "pest_management": "Integrated pest management",
            "harvesting": "Harvest at maturity",
            "yield_optimization": "Proper spacing, timely irrigation, balanced nutrition",
        }

    def classify_query(self, query: str) -> str:
        """Classify query using neural network"""
        if not self.qa_model or not HAS_SKLEARN:
            return "general"

        try:
            X = self.vectorizer.transform([query]).toarray()
            category_idx = self.qa_model.predict(X)[0]
            return self.category_encoder.inverse_transform([category_idx])[0]
        except Exception:
            return "general"

    def generate_response_by_category(
        self, query: str, category: str, crop: str = None
    ) -> Dict:
        """Generate response based on query category"""
        responses = {
            "water_efficiency": self.get_water_efficiency_advice(crop),
            "fertilizer_efficiency": self.get_fertilizer_efficiency_advice(crop),
            "yield_optimization": self.get_yield_optimization_advice(crop),
            "pest_control": self.get_pest_control_advice(crop),
            "disease": self.get_disease_advice(crop),
            "soil_management": self.get_soil_management_advice(crop),
            "irrigation": self.get_irrigation_advice(crop),
            "fertilizer": self.get_fertilizer_advice(crop),
            "planting": self.get_planting_advice(crop),
            "harvesting": self.get_harvesting_advice(crop),
        }

        return responses.get(
            category,
            {
                "answer": "I can help you with agricultural questions. Please specify your crop or question."
            },
        )

    def get_water_efficiency_advice(self, crop: str = None) -> Dict:
        """Get water efficiency advice"""
        return {
            "answer": "To reduce water usage while maintaining yield:",
            "tips": [
                "Use drip irrigation instead of flood irrigation (saves 30-50% water)",
                "Apply mulch to reduce evaporation",
                "Water during early morning or evening to minimize evaporation",
                "Use soil moisture sensors to water only when needed",
                "Practice deficit irrigation during non-critical growth stages",
                "Choose drought-tolerant varieties when available",
                "Implement rainwater harvesting",
                "Use raised beds for better water efficiency",
            ],
            "water_savings": "Can reduce water usage by 30-50% while maintaining or increasing yield",
        }

    def get_fertilizer_efficiency_advice(self, crop: str = None) -> Dict:
        """Get fertilizer efficiency advice"""
        return {
            "answer": "To reduce fertilizer usage while increasing yield:",
            "tips": [
                "Conduct soil testing to apply only needed nutrients",
                "Use slow-release fertilizers for better nutrient uptake",
                "Apply fertilizers in split doses based on growth stages",
                "Use organic compost and farmyard manure",
                "Practice crop rotation with legumes for natural nitrogen fixation",
                "Use foliar feeding for micronutrients",
                "Implement precision agriculture techniques",
                "Use biofertilizers and bio-stimulants",
            ],
            "fertilizer_reduction": "Can reduce fertilizer usage by 20-40% while improving yield through better nutrient management",
        }

    def get_yield_optimization_advice(self, crop: str = None) -> Dict:
        """Get yield optimization advice"""
        return {
            "answer": "To increase yield while reducing inputs:",
            "tips": [
                "Use high-yielding, disease-resistant varieties",
                "Optimize plant spacing for maximum productivity",
                "Practice timely sowing/planting",
                "Implement integrated nutrient management",
                "Use proper irrigation scheduling",
                "Control weeds, pests, and diseases timely",
                "Practice crop rotation and intercropping",
                "Use growth regulators and bio-stimulants",
                "Harvest at optimal maturity stage",
            ],
            "yield_increase": "Can increase yield by 15-30% through optimized management practices",
        }

    def get_optimization_suggestions(self, crop: str, query: str) -> Dict:
        """Get optimization suggestions for specific crop"""
        return {
            "water_reduction": self.get_water_efficiency_advice(crop),
            "fertilizer_reduction": self.get_fertilizer_efficiency_advice(crop),
            "yield_increase": self.get_yield_optimization_advice(crop),
        }

    def get_optimization_tips(self, query: str, crop: str = None) -> List[str]:
        """Get optimization tips based on query"""
        tips = []
        if "water" in query or "irrigation" in query:
            tips.extend(self.get_water_efficiency_advice(crop)["tips"][:3])
        if "fertilizer" in query or "nutrient" in query:
            tips.extend(self.get_fertilizer_efficiency_advice(crop)["tips"][:3])
        if "yield" in query or "production" in query:
            tips.extend(self.get_yield_optimization_advice(crop)["tips"][:3])
        return tips[:5]  # Return top 5 tips

    def rule_based_response(self, query: str, crop: str = None) -> Dict:
        """Rule-based fallback response"""
        if "water" in query:
            return self.get_water_efficiency_advice(crop)
        elif "fertilizer" in query or "nutrient" in query:
            return self.get_fertilizer_efficiency_advice(crop)
        elif "yield" in query:
            return self.get_yield_optimization_advice(crop)
        else:
            return {
                "answer": "I can help with agricultural questions. Please ask about water efficiency, fertilizer management, yield optimization, or specific crop cultivation."
            }

    # Additional helper methods for different categories
    def get_pest_control_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Use integrated pest management (IPM) with biological, cultural, and chemical controls."
        }

    def get_disease_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Practice crop rotation, use disease-resistant varieties, and apply preventive fungicides."
        }

    def get_soil_management_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Maintain soil health through organic matter addition, proper pH, and balanced nutrients."
        }

    def get_irrigation_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Use efficient irrigation methods like drip or sprinkler, water based on crop needs."
        }

    def get_fertilizer_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Apply balanced NPK fertilizers based on soil test results and crop requirements."
        }

    def get_planting_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Plant at optimal time, proper spacing, and use quality seeds/seedlings."
        }

    def get_harvesting_advice(self, crop: str = None) -> Dict:
        return {
            "answer": "Harvest at optimal maturity stage for best quality and yield."
        }


# Initialize global chatbot instance
_chatbot_instance = None


def get_chatbot():
    """Get or create chatbot instance"""
    global _chatbot_instance
    if _chatbot_instance is None:
        _chatbot_instance = EdgeAIChatbot()
    return _chatbot_instance
