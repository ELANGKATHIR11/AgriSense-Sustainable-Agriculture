"""
Retrieval-Augmented Generation (RAG) Pipeline for AgriSense
Implements hybrid RAG with Intent Classification + Retrieval + Generation
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set up paths
MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class CropData:
    """Structured crop information"""
    crop_name: str
    scientific_name: str
    season: str
    crop_type: str
    min_temp: float
    max_temp: float
    min_ph: float
    max_ph: float
    soil_type: str
    water_req: float
    rainfall_min: float
    rainfall_max: float
    npk_ratio: str
    growth_days: int
    

class IntentClassifier:
    """Classify user intent from query using SVM"""
    
    INTENTS = {
        'weather': ['temperature', 'rainfall', 'weather', 'climate', 'season', 'monsoon', 'rain', 'cold', 'hot'],
        'disease': ['disease', 'pest', 'blight', 'fungal', 'bacterial', 'viral', 'infection', 'damage'],
        'soil': ['soil', 'pH', 'nutrients', 'NPK', 'fertilizer', 'amendments', 'nitrogen', 'phosphorus'],
        'crop_recommendation': ['recommend', 'suitable', 'best', 'grow', 'cultivate', 'select', 'suggest'],
        'pricing': ['price', 'market', 'cost', 'sell', 'buy', 'profit', 'rate', 'income']
    }
    
    def __init__(self):
        """Load intent classifier model"""
        model_path = MODELS_DIR / "intent_classifier_model.pkl"
        scaler_path = MODELS_DIR / "intent_classifier_scaler.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Intent classifier model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
    def classify(self, query: str) -> Tuple[str, float]:
        """Classify query intent with confidence score"""
        # Feature extraction
        features = np.array([[
            len(query),
            ord(query[0].lower()),
            query.lower().count('e')
        ]])
        
        # Scale and predict
        features_scaled = self.scaler.transform(features)
        intent = self.model.predict(features_scaled)[0]
        
        # Get confidence
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        
        return intent, confidence
    
    def get_keywords_for_intent(self, intent: str) -> List[str]:
        """Get keywords for a specific intent"""
        return self.INTENTS.get(intent, [])


class CropRetriever:
    """Retrieve relevant crops using semantic similarity"""
    
    def __init__(self):
        """Load crop dataset and prepare retrieval"""
        self.crop_df = pd.read_csv(DATA_DIR / "raw" / "india_crops_complete.csv")
        self.crop_embeddings = self._create_embeddings()
    
    def _create_embeddings(self) -> np.ndarray:
        """Create feature embeddings for crops"""
        # Use normalized environmental features
        features = self.crop_df[[
            'min_temp_C', 'max_temp_C', 'min_pH', 'max_pH',
            'water_req_mm_day', 'rainfall_min_mm', 'rainfall_max_mm',
            'N_kg_per_ha', 'P_kg_per_ha', 'K_kg_per_ha'
        ]].values
        
        # Normalize features
        scaler = StandardScaler()
        return scaler.fit_transform(features)
    
    def retrieve(self, query_features: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k most similar crops"""
        # Normalize query
        scaler = StandardScaler()
        scaler.fit(self.crop_embeddings)
        query_normalized = scaler.transform([query_features])[0].reshape(1, -1)
        
        # Compute similarity
        similarities = cosine_similarity(query_normalized, self.crop_embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return crop data
        results = []
        for idx in top_indices:
            row = self.crop_df.iloc[idx]
            results.append({
                'crop_name': row['crop_name'],
                'scientific_name': row['scientific_name'],
                'season': row['season'],
                'crop_type': row['crop_type'],
                'min_temp': float(row['min_temp_C']),
                'max_temp': float(row['max_temp_C']),
                'water_requirement': float(row['water_req_mm_day']),
                'npk': f"{int(row['N_kg_per_ha'])}-{int(row['P_kg_per_ha'])}-{int(row['K_kg_per_ha'])}",
                'growth_days': int(row['growth_duration_days']),
                'similarity_score': float(similarities[idx])
            })
        
        return results
    
    def search_by_criteria(self, season: Optional[str] = None, 
                          crop_type: Optional[str] = None,
                          temp_range: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """Search crops by specific criteria"""
        df = self.crop_df.copy()
        
        if season:
            df = df[df['season'] == season]
        if crop_type:
            df = df[df['crop_type'] == crop_type]
        if temp_range:
            min_t, max_t = temp_range
            df = df[(df['min_temp_C'] <= max_t) & (df['max_temp_C'] >= min_t)]
        
        results = []
        for _, row in df.iterrows():
            results.append({
                'crop_name': row['crop_name'],
                'scientific_name': row['scientific_name'],
                'season': row['season'],
                'crop_type': row['crop_type'],
                'min_temp': float(row['min_temp_C']),
                'max_temp': float(row['max_temp_C']),
                'water_requirement': float(row['water_req_mm_day']),
                'npk': f"{int(row['N_kg_per_ha'])}-{int(row['P_kg_per_ha'])}-{int(row['K_kg_per_ha'])}",
                'growth_days': int(row['growth_duration_days'])
            })
        
        return results


class RAGPipeline:
    """Complete Retrieval-Augmented Generation Pipeline"""
    
    def __init__(self):
        """Initialize RAG components"""
        self.intent_classifier = IntentClassifier()
        self.retriever = CropRetriever()
        self.crop_df = pd.read_csv(DATA_DIR / "raw" / "india_crops_complete.csv")
        
        # Load trained models
        self._load_models()
    
    def _load_models(self):
        """Load all trained ML models"""
        self.models = {}
        model_files = [
            'crop_recommendation_model.pkl',
            'crop_type_classification_model.pkl',
            'growth_duration_model.pkl',
            'water_requirement_model.pkl',
            'season_classification_model.pkl'
        ]
        
        for model_file in model_files:
            model_path = MODELS_DIR / model_file
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = pickle.load(f)
    
    def process_query(self, query: str, context: Dict = None) -> Dict:
        """
        Process a user query through the RAG pipeline
        
        Args:
            query: User question/request
            context: Optional user context (location, season, etc.)
        
        Returns:
            Complete RAG response with intent, retrieved data, and predictions
        """
        # Step 1: Intent Classification
        intent, confidence = self.intent_classifier.classify(query)
        
        response = {
            'query': query,
            'intent': intent,
            'confidence': confidence,
            'context': context or {},
        }
        
        # Step 2: Retrieval based on intent
        if intent == 'crop_recommendation':
            response['recommendations'] = self._handle_crop_recommendation(context)
        elif intent == 'weather':
            response['weather_info'] = self._handle_weather_query(context)
        elif intent == 'disease':
            response['disease_info'] = self._handle_disease_query(context)
        elif intent == 'soil':
            response['soil_info'] = self._handle_soil_query(context)
        elif intent == 'pricing':
            response['pricing_info'] = self._handle_pricing_query(context)
        
        # Step 3: Generation (formatted response)
        response['response_text'] = self._generate_response(intent, response)
        
        return response
    
    def _handle_crop_recommendation(self, context: Dict) -> List[Dict]:
        """Handle crop recommendation queries"""
        season = context.get('season') if context else None
        crops = self.retriever.search_by_criteria(season=season)
        
        # Limit to top 5
        return crops[:5]
    
    def _handle_weather_query(self, context: Dict) -> Dict:
        """Handle weather-related queries"""
        crops = self.retriever.search_by_criteria()
        
        return {
            'temperature_range': {
                'min': float(self.crop_df['min_temp_C'].min()),
                'max': float(self.crop_df['max_temp_C'].max()),
                'optimal': float(self.crop_df['min_temp_C'].mean())
            },
            'rainfall_range': {
                'min': float(self.crop_df['rainfall_min_mm'].min()),
                'max': float(self.crop_df['rainfall_max_mm'].max()),
                'optimal': float(self.crop_df['rainfall_min_mm'].mean())
            },
            'crops_in_season': len(self.crop_df[self.crop_df['season'] == context.get('season', 'Kharif')]) if context else len(crops)
        }
    
    def _handle_disease_query(self, context: Dict) -> Dict:
        """Handle disease-related queries"""
        return {
            'preventive_measures': [
                'Maintain proper field sanitation',
                'Use disease-resistant varieties',
                'Practice crop rotation',
                'Monitor regularly for early symptoms',
                'Apply fungicides when necessary'
            ],
            'common_diseases': ['Leaf blight', 'Root rot', 'Powdery mildew', 'Bacterial wilt'],
            'recommendation': 'Consult local agricultural extension officer for specific pest management'
        }
    
    def _handle_soil_query(self, context: Dict) -> Dict:
        """Handle soil-related queries"""
        return {
            'soil_types': list(self.crop_df['soil_type'].unique())[:5],
            'ph_range': {
                'min': float(self.crop_df['min_pH'].min()),
                'max': float(self.crop_df['max_pH'].max())
            },
            'npk_guidelines': {
                'nitrogen': f"{int(self.crop_df['N_kg_per_ha'].min())}-{int(self.crop_df['N_kg_per_ha'].max())} kg/ha",
                'phosphorus': f"{int(self.crop_df['P_kg_per_ha'].min())}-{int(self.crop_df['P_kg_per_ha'].max())} kg/ha",
                'potassium': f"{int(self.crop_df['K_kg_per_ha'].min())}-{int(self.crop_df['K_kg_per_ha'].max())} kg/ha"
            }
        }
    
    def _handle_pricing_query(self, context: Dict) -> Dict:
        """Handle pricing-related queries"""
        return {
            'note': 'Pricing varies by market, quality, and seasonal demand',
            'recommendation': 'Check local agricultural markets or APMC for current prices',
            'cash_crops': list(self.crop_df[self.crop_df['crop_type'] == 'Cash']['crop_name'].unique())[:5]
        }
    
    def _generate_response(self, intent: str, data: Dict) -> str:
        """Generate natural language response"""
        if intent == 'crop_recommendation':
            crops = data.get('recommendations', [])
            if crops:
                crop_list = ', '.join([c['crop_name'] for c in crops[:3]])
                return f"Based on your requirements, I recommend: {crop_list}. These crops are well-suited to your season and climate conditions."
            return "No crops found matching your criteria. Please provide more details."
        
        elif intent == 'weather':
            return "The crop data shows temperature ranges from 10-40°C and rainfall from 200-3500mm depending on the crop. Please specify your location for more precise recommendations."
        
        elif intent == 'disease':
            return "To prevent crop diseases: maintain field sanitation, use disease-resistant varieties, practice crop rotation, and monitor regularly. Consult your local agriculture officer for specific issues."
        
        elif intent == 'soil':
            return "Soil pH typically ranges from 5.0-8.5 depending on the crop. NPK requirements vary: Nitrogen 20-250 kg/ha, Phosphorus 30-100 kg/ha, Potassium 25-300 kg/ha. Get a soil test done for accurate recommendations."
        
        elif intent == 'pricing':
            return "Agricultural prices fluctuate based on market demand, quality, and season. Check your local APMC (Agricultural Produce Market Committee) for current market rates."
        
        else:
            return "I can help you with crop recommendations, weather guidance, disease prevention, soil management, and pricing information. What would you like to know?"


# Utility functions for backend integration
def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline"""
    return RAGPipeline()


def process_user_query(pipeline: RAGPipeline, query: str, context: Dict = None) -> Dict:
    """Process a user query through the RAG pipeline"""
    return pipeline.process_query(query, context)
