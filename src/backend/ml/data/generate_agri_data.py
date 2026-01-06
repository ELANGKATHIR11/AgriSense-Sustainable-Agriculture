#!/usr/bin/env python3
"""
AgriSense Tabular Data Generator
=================================
Generates realistic synthetic agricultural datasets for Indian crops.

Features:
- 22 Indian crop profiles with region-specific parameters
- Gaussian distribution for realistic value variation
- Soil type assignment based on crop requirements
- Historical yield data with temporal features

Outputs:
- india_crops_complete.csv (44,000 samples: 22 crops × 2000 each)
- historical_yields.csv (yield prediction dataset)

Usage:
    python generate_agri_data.py [--samples 2000] [--output-dir ./tabular]

Author: AgriSense ML Team
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import argparse
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# INDIAN CROP PROFILES
# =============================================================================
# Based on ICAR (Indian Council of Agricultural Research) recommendations
# and state agricultural department data

CROP_PROFILES: Dict[str, Dict] = {
    # -------------------------------------------------------------------------
    # CEREALS
    # -------------------------------------------------------------------------
    'Rice': {
        'N': (80, 20),      # Nitrogen kg/ha (mean, std)
        'P': (40, 10),      # Phosphorus kg/ha
        'K': (40, 10),      # Potassium kg/ha
        'temperature': (25, 3),  # Celsius
        'humidity': (80, 10),    # Percentage
        'ph': (6.0, 0.5),        # pH level
        'rainfall': (1200, 300), # mm/year
        'soil_types': ['Alluvial', 'Clayey', 'Loamy'],
        'states': ['Punjab', 'West Bengal', 'Tamil Nadu', 'Andhra Pradesh'],
        'season': ['Kharif'],
        'yield_range': (2.5, 5.5)  # tonnes/hectare
    },
    'Maize': {
        'N': (120, 25),
        'P': (60, 15),
        'K': (40, 10),
        'temperature': (25, 4),
        'humidity': (65, 12),
        'ph': (6.5, 0.6),
        'rainfall': (600, 150),
        'soil_types': ['Loamy', 'Sandy Loam', 'Alluvial'],
        'states': ['Karnataka', 'Madhya Pradesh', 'Bihar', 'Rajasthan'],
        'season': ['Kharif', 'Rabi'],
        'yield_range': (2.0, 4.5)
    },
    
    # -------------------------------------------------------------------------
    # PULSES
    # -------------------------------------------------------------------------
    'Chickpea': {
        'N': (20, 5),  # Legumes fix nitrogen
        'P': (50, 12),
        'K': (20, 5),
        'temperature': (22, 4),
        'humidity': (45, 10),
        'ph': (6.8, 0.4),
        'rainfall': (400, 100),
        'soil_types': ['Black Soil', 'Loamy', 'Sandy Loam'],
        'states': ['Madhya Pradesh', 'Maharashtra', 'Rajasthan', 'Karnataka'],
        'season': ['Rabi'],
        'yield_range': (0.8, 1.8)
    },
    'Kidneybeans': {
        'N': (20, 5),
        'P': (60, 15),
        'K': (30, 8),
        'temperature': (20, 3),
        'humidity': (55, 10),
        'ph': (6.2, 0.4),
        'rainfall': (600, 150),
        'soil_types': ['Loamy', 'Sandy Loam', 'Alluvial'],
        'states': ['Jammu Kashmir', 'Himachal Pradesh', 'Uttarakhand'],
        'season': ['Kharif'],
        'yield_range': (0.6, 1.5)
    },
    'Pigeonpeas': {
        'N': (25, 6),
        'P': (50, 12),
        'K': (25, 6),
        'temperature': (28, 4),
        'humidity': (60, 12),
        'ph': (6.5, 0.5),
        'rainfall': (600, 180),
        'soil_types': ['Black Soil', 'Red Soil', 'Loamy'],
        'states': ['Maharashtra', 'Karnataka', 'Madhya Pradesh', 'Gujarat'],
        'season': ['Kharif'],
        'yield_range': (0.6, 1.2)
    },
    'Mothbeans': {
        'N': (15, 4),
        'P': (40, 10),
        'K': (20, 5),
        'temperature': (32, 5),
        'humidity': (40, 10),
        'ph': (7.2, 0.5),
        'rainfall': (300, 100),
        'soil_types': ['Sandy', 'Sandy Loam', 'Arid'],
        'states': ['Rajasthan', 'Gujarat', 'Maharashtra'],
        'season': ['Kharif'],
        'yield_range': (0.3, 0.8)
    },
    'Mungbean': {
        'N': (20, 5),
        'P': (40, 10),
        'K': (20, 5),
        'temperature': (30, 4),
        'humidity': (65, 12),
        'ph': (6.5, 0.5),
        'rainfall': (500, 150),
        'soil_types': ['Loamy', 'Sandy Loam', 'Alluvial'],
        'states': ['Rajasthan', 'Maharashtra', 'Andhra Pradesh', 'Karnataka'],
        'season': ['Kharif', 'Summer'],
        'yield_range': (0.5, 1.2)
    },
    'Blackgram': {
        'N': (20, 5),
        'P': (50, 12),
        'K': (20, 5),
        'temperature': (28, 4),
        'humidity': (70, 12),
        'ph': (6.5, 0.5),
        'rainfall': (600, 180),
        'soil_types': ['Black Soil', 'Alluvial', 'Loamy'],
        'states': ['Uttar Pradesh', 'Madhya Pradesh', 'Maharashtra', 'Tamil Nadu'],
        'season': ['Kharif'],
        'yield_range': (0.4, 1.0)
    },
    'Lentil': {
        'N': (20, 5),
        'P': (45, 10),
        'K': (20, 5),
        'temperature': (20, 3),
        'humidity': (50, 10),
        'ph': (6.5, 0.4),
        'rainfall': (350, 100),
        'soil_types': ['Loamy', 'Sandy Loam', 'Alluvial'],
        'states': ['Uttar Pradesh', 'Madhya Pradesh', 'West Bengal', 'Bihar'],
        'season': ['Rabi'],
        'yield_range': (0.6, 1.4)
    },
    
    # -------------------------------------------------------------------------
    # FRUITS
    # -------------------------------------------------------------------------
    'Pomegranate': {
        'N': (60, 15),
        'P': (30, 8),
        'K': (50, 12),
        'temperature': (30, 5),
        'humidity': (45, 10),
        'ph': (6.8, 0.5),
        'rainfall': (500, 150),
        'soil_types': ['Black Soil', 'Sandy Loam', 'Red Soil'],
        'states': ['Maharashtra', 'Karnataka', 'Andhra Pradesh', 'Gujarat'],
        'season': ['Perennial'],
        'yield_range': (8, 15)
    },
    'Banana': {
        'N': (200, 40),
        'P': (60, 15),
        'K': (300, 60),
        'temperature': (28, 3),
        'humidity': (80, 10),
        'ph': (6.5, 0.5),
        'rainfall': (1800, 400),
        'soil_types': ['Loamy', 'Alluvial', 'Clayey'],
        'states': ['Tamil Nadu', 'Maharashtra', 'Gujarat', 'Andhra Pradesh'],
        'season': ['Perennial'],
        'yield_range': (25, 50)
    },
    'Mango': {
        'N': (100, 25),
        'P': (50, 12),
        'K': (100, 25),
        'temperature': (30, 4),
        'humidity': (60, 12),
        'ph': (6.0, 0.5),
        'rainfall': (1000, 300),
        'soil_types': ['Alluvial', 'Loamy', 'Laterite'],
        'states': ['Uttar Pradesh', 'Andhra Pradesh', 'Karnataka', 'Maharashtra'],
        'season': ['Perennial'],
        'yield_range': (5, 12)
    },
    'Grapes': {
        'N': (100, 25),
        'P': (60, 15),
        'K': (120, 30),
        'temperature': (25, 4),
        'humidity': (50, 10),
        'ph': (6.5, 0.4),
        'rainfall': (600, 150),
        'soil_types': ['Sandy Loam', 'Black Soil', 'Red Soil'],
        'states': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh'],
        'season': ['Perennial'],
        'yield_range': (15, 25)
    },
    'Watermelon': {
        'N': (80, 20),
        'P': (40, 10),
        'K': (60, 15),
        'temperature': (32, 4),
        'humidity': (60, 12),
        'ph': (6.5, 0.5),
        'rainfall': (500, 150),
        'soil_types': ['Sandy Loam', 'Loamy', 'Alluvial'],
        'states': ['Karnataka', 'Maharashtra', 'Andhra Pradesh', 'Tamil Nadu'],
        'season': ['Summer'],
        'yield_range': (20, 35)
    },
    'Muskmelon': {
        'N': (70, 18),
        'P': (35, 8),
        'K': (50, 12),
        'temperature': (30, 4),
        'humidity': (55, 10),
        'ph': (6.5, 0.5),
        'rainfall': (400, 120),
        'soil_types': ['Sandy Loam', 'Loamy', 'Alluvial'],
        'states': ['Punjab', 'Rajasthan', 'Uttar Pradesh', 'Maharashtra'],
        'season': ['Summer'],
        'yield_range': (12, 22)
    },
    'Apple': {
        'N': (80, 20),
        'P': (40, 10),
        'K': (80, 20),
        'temperature': (18, 3),
        'humidity': (70, 10),
        'ph': (6.0, 0.4),
        'rainfall': (1000, 250),
        'soil_types': ['Loamy', 'Mountain Soil', 'Alluvial'],
        'states': ['Jammu Kashmir', 'Himachal Pradesh', 'Uttarakhand'],
        'season': ['Perennial'],
        'yield_range': (8, 15)
    },
    'Orange': {
        'N': (100, 25),
        'P': (50, 12),
        'K': (100, 25),
        'temperature': (25, 4),
        'humidity': (65, 12),
        'ph': (6.0, 0.5),
        'rainfall': (1200, 300),
        'soil_types': ['Loamy', 'Sandy Loam', 'Alluvial'],
        'states': ['Maharashtra', 'Punjab', 'Madhya Pradesh', 'Rajasthan'],
        'season': ['Perennial'],
        'yield_range': (10, 18)
    },
    'Papaya': {
        'N': (200, 40),
        'P': (200, 40),
        'K': (400, 80),
        'temperature': (30, 3),
        'humidity': (80, 10),
        'ph': (6.5, 0.5),
        'rainfall': (1500, 350),
        'soil_types': ['Loamy', 'Alluvial', 'Sandy Loam'],
        'states': ['Gujarat', 'Maharashtra', 'Andhra Pradesh', 'Karnataka'],
        'season': ['Perennial'],
        'yield_range': (30, 60)
    },
    'Coconut': {
        'N': (100, 25),
        'P': (50, 12),
        'K': (200, 40),
        'temperature': (28, 3),
        'humidity': (85, 8),
        'ph': (6.0, 0.5),
        'rainfall': (2000, 500),
        'soil_types': ['Coastal Sandy', 'Laterite', 'Loamy'],
        'states': ['Kerala', 'Karnataka', 'Tamil Nadu', 'Andhra Pradesh'],
        'season': ['Perennial'],
        'yield_range': (50, 100)  # nuts per tree per year
    },
    
    # -------------------------------------------------------------------------
    # COMMERCIAL CROPS
    # -------------------------------------------------------------------------
    'Cotton': {
        'N': (100, 25),
        'P': (50, 12),
        'K': (50, 12),
        'temperature': (28, 4),
        'humidity': (60, 12),
        'ph': (7.0, 0.5),
        'rainfall': (700, 200),
        'soil_types': ['Black Soil', 'Alluvial', 'Red Soil'],
        'states': ['Gujarat', 'Maharashtra', 'Telangana', 'Andhra Pradesh'],
        'season': ['Kharif'],
        'yield_range': (1.5, 3.0)
    },
    'Jute': {
        'N': (60, 15),
        'P': (30, 8),
        'K': (30, 8),
        'temperature': (30, 3),
        'humidity': (85, 8),
        'ph': (6.5, 0.5),
        'rainfall': (1500, 350),
        'soil_types': ['Alluvial', 'Clayey', 'Loamy'],
        'states': ['West Bengal', 'Bihar', 'Assam', 'Odisha'],
        'season': ['Kharif'],
        'yield_range': (2.0, 3.5)
    },
    'Coffee': {
        'N': (150, 35),
        'P': (50, 12),
        'K': (150, 35),
        'temperature': (22, 3),
        'humidity': (75, 10),
        'ph': (5.5, 0.4),
        'rainfall': (1800, 400),
        'soil_types': ['Laterite', 'Forest Soil', 'Loamy'],
        'states': ['Karnataka', 'Kerala', 'Tamil Nadu'],
        'season': ['Perennial'],
        'yield_range': (0.8, 1.5)
    }
}

# Soil type characteristics for additional features
SOIL_CHARACTERISTICS = {
    'Alluvial': {'organic_matter': 0.5, 'drainage': 'good', 'fertility': 'high'},
    'Black Soil': {'organic_matter': 0.8, 'drainage': 'poor', 'fertility': 'high'},
    'Red Soil': {'organic_matter': 0.3, 'drainage': 'good', 'fertility': 'medium'},
    'Laterite': {'organic_matter': 0.4, 'drainage': 'good', 'fertility': 'low'},
    'Sandy': {'organic_matter': 0.2, 'drainage': 'excellent', 'fertility': 'low'},
    'Sandy Loam': {'organic_matter': 0.4, 'drainage': 'good', 'fertility': 'medium'},
    'Loamy': {'organic_matter': 0.6, 'drainage': 'good', 'fertility': 'high'},
    'Clayey': {'organic_matter': 0.5, 'drainage': 'poor', 'fertility': 'medium'},
    'Mountain Soil': {'organic_matter': 0.7, 'drainage': 'good', 'fertility': 'medium'},
    'Coastal Sandy': {'organic_matter': 0.2, 'drainage': 'excellent', 'fertility': 'low'},
    'Forest Soil': {'organic_matter': 0.9, 'drainage': 'good', 'fertility': 'high'},
    'Arid': {'organic_matter': 0.1, 'drainage': 'excellent', 'fertility': 'very_low'}
}


class AgriDataGenerator:
    """
    Generator for realistic Indian agricultural datasets.
    
    Attributes:
        samples_per_crop: Number of samples to generate per crop
        random_state: Random seed for reproducibility
    """
    
    def __init__(self, samples_per_crop: int = 2000, random_state: int = 42):
        self.samples_per_crop = samples_per_crop
        self.random_state = random_state
        np.random.seed(random_state)
        
    def _generate_gaussian_values(
        self, 
        mean: float, 
        std: float, 
        n_samples: int,
        min_val: float = 0,
        max_val: float = None
    ) -> np.ndarray:
        """Generate Gaussian distributed values with bounds."""
        values = np.random.normal(mean, std, n_samples)
        values = np.maximum(values, min_val)
        if max_val is not None:
            values = np.minimum(values, max_val)
        return values
    
    def _assign_soil_type(self, crop_profile: Dict, n_samples: int) -> List[str]:
        """Assign soil types based on crop-specific probabilities."""
        soil_types = crop_profile['soil_types']
        # Primary soil type gets higher probability
        probs = [0.5] + [0.5 / (len(soil_types) - 1)] * (len(soil_types) - 1)
        return np.random.choice(soil_types, n_samples, p=probs).tolist()
    
    def _assign_state(self, crop_profile: Dict, n_samples: int) -> List[str]:
        """Assign Indian state based on crop cultivation regions."""
        states = crop_profile['states']
        return np.random.choice(states, n_samples).tolist()
    
    def generate_crop_data(self) -> pd.DataFrame:
        """
        Generate complete crop recommendation dataset.
        
        Returns:
            DataFrame with columns: N, P, K, temperature, humidity, ph, rainfall, 
                                   soil_type, state, season, label
        """
        logger.info(f"Generating crop data: {len(CROP_PROFILES)} crops × {self.samples_per_crop} samples")
        
        all_data = []
        
        for crop_name, profile in CROP_PROFILES.items():
            logger.info(f"  Generating {crop_name}...")
            
            n = self.samples_per_crop
            
            # Generate soil features with Gaussian distribution
            data = {
                'N': self._generate_gaussian_values(*profile['N'], n, min_val=0),
                'P': self._generate_gaussian_values(*profile['P'], n, min_val=0),
                'K': self._generate_gaussian_values(*profile['K'], n, min_val=0),
                'temperature': self._generate_gaussian_values(*profile['temperature'], n, min_val=5, max_val=45),
                'humidity': self._generate_gaussian_values(*profile['humidity'], n, min_val=20, max_val=100),
                'ph': self._generate_gaussian_values(*profile['ph'], n, min_val=3.5, max_val=9.5),
                'rainfall': self._generate_gaussian_values(*profile['rainfall'], n, min_val=100),
                'soil_type': self._assign_soil_type(profile, n),
                'state': self._assign_state(profile, n),
                'season': np.random.choice(profile['season'], n),
                'label': [crop_name] * n
            }
            
            df = pd.DataFrame(data)
            all_data.append(df)
        
        result = pd.concat(all_data, ignore_index=True)
        
        # Shuffle the data
        result = result.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        # Round numerical columns
        for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            result[col] = result[col].round(2)
        
        logger.info(f"✓ Generated {len(result)} total samples")
        return result
    
    def generate_yield_data(self, crop_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate historical yield prediction dataset.
        
        Adds temporal and management features to base crop data.
        
        Args:
            crop_df: Base crop recommendation DataFrame
            
        Returns:
            DataFrame with additional columns: year, pest_incidence, 
                                              fertilizer_usage_kg, yield_t_ha
        """
        logger.info("Generating yield prediction data...")
        
        yield_df = crop_df.copy()
        n = len(yield_df)
        
        # Add temporal features
        yield_df['year'] = np.random.randint(2010, 2024, n)
        
        # Pest incidence (0-100%)
        yield_df['pest_incidence'] = np.random.beta(2, 5, n) * 100  # Skewed towards lower values
        yield_df['pest_incidence'] = yield_df['pest_incidence'].round(1)
        
        # Fertilizer usage (kg/ha) - correlated with NPK
        base_fertilizer = (yield_df['N'] + yield_df['P'] + yield_df['K']) * 1.2
        yield_df['fertilizer_usage_kg'] = (
            base_fertilizer + np.random.normal(0, 20, n)
        ).clip(lower=50).round(1)
        
        # Calculate yield based on multiple factors
        def calculate_yield(row):
            crop_profile = CROP_PROFILES.get(row['label'], {})
            yield_range = crop_profile.get('yield_range', (1, 3))
            base_yield = np.random.uniform(*yield_range)
            
            # Modifiers
            pest_modifier = 1 - (row['pest_incidence'] / 200)  # Max 50% reduction
            
            # Temperature deviation from optimal
            optimal_temp = crop_profile.get('temperature', (25, 5))[0]
            temp_diff = abs(row['temperature'] - optimal_temp)
            temp_modifier = 1 - (temp_diff / 50)  # Max 20% reduction for 10°C diff
            
            # pH deviation from optimal
            optimal_ph = crop_profile.get('ph', (6.5, 0.5))[0]
            ph_diff = abs(row['ph'] - optimal_ph)
            ph_modifier = 1 - (ph_diff / 5)  # Max 20% reduction for 1 pH unit diff
            
            # Climate improvement factor (newer years have slightly better yields)
            year_modifier = 1 + (row['year'] - 2010) * 0.005
            
            final_yield = base_yield * pest_modifier * temp_modifier * ph_modifier * year_modifier
            return max(0.1, final_yield)
        
        yield_df['yield_t_ha'] = yield_df.apply(calculate_yield, axis=1).round(2)
        
        logger.info(f"✓ Generated yield data with {n} samples")
        return yield_df
    
    def generate_intent_corpus(self) -> Dict:
        """
        Generate NLP training data for agricultural chatbot.
        
        Returns:
            Dictionary with intent classification training data
        """
        logger.info("Generating intent corpus...")
        
        intents = {
            "crop_recommendation": {
                "patterns": [
                    "What crop should I grow?",
                    "Which crop is suitable for my land?",
                    "Suggest a crop for {soil_type} soil",
                    "Best crop for {state} region",
                    "What to plant in {season} season?",
                    "Crop recommendation for pH {ph}",
                    "Which crop needs {rainfall}mm rainfall?",
                    "मेरी मिट्टी के लिए कौन सी फसल अच्छी है?",  # Hindi
                    "என் நிலத்திற்கு எந்த பயிர் சிறந்தது?"  # Tamil
                ],
                "responses": [
                    "Based on your soil conditions, I recommend {crop}.",
                    "For {soil_type} soil with {temperature}°C temperature, {crop} would be ideal.",
                    "Considering your {state} location, {crop} is a good choice."
                ]
            },
            "disease_detection": {
                "patterns": [
                    "My {crop} leaves have spots",
                    "Is this a disease on my plant?",
                    "What disease affects {crop}?",
                    "How to identify leaf blight?",
                    "My plant leaves are turning yellow",
                    "पत्तियों पर धब्बे क्या हैं?",  # Hindi
                    "இலைகளில் புள்ளிகள் என்ன?"  # Tamil
                ],
                "responses": [
                    "This appears to be {disease}. Treatment involves {treatment}.",
                    "Based on the symptoms, your {crop} may have {disease}."
                ]
            },
            "yield_prediction": {
                "patterns": [
                    "How much yield can I expect?",
                    "Predict my harvest",
                    "Expected {crop} production",
                    "What will be my crop output?",
                    "फसल की उपज कितनी होगी?",  # Hindi
                    "விளைச்சல் எவ்வளவு இருக்கும்?"  # Tamil
                ],
                "responses": [
                    "Based on current conditions, expected yield is {yield} tonnes/hectare.",
                    "With optimal management, you can expect {yield} t/ha of {crop}."
                ]
            },
            "weather_query": {
                "patterns": [
                    "What is the weather forecast?",
                    "Will it rain tomorrow?",
                    "Temperature prediction for farming",
                    "Best time to sow {crop}?",
                    "मौसम का पूर्वानुमान क्या है?"  # Hindi
                ],
                "responses": [
                    "The forecast shows {weather_condition} with {temperature}°C.",
                    "Expected rainfall: {rainfall}mm. Good conditions for {activity}."
                ]
            },
            "pest_management": {
                "patterns": [
                    "How to control pests in {crop}?",
                    "Organic pest control methods",
                    "Insects attacking my plants",
                    "Natural pesticide alternatives",
                    "कीट नियंत्रण कैसे करें?"  # Hindi
                ],
                "responses": [
                    "For {pest} in {crop}, use {treatment}.",
                    "Organic options include neem oil, pyrethrin, or companion planting."
                ]
            },
            "fertilizer_advice": {
                "patterns": [
                    "How much fertilizer for {crop}?",
                    "NPK ratio recommendation",
                    "Organic vs chemical fertilizer",
                    "When to apply fertilizer?",
                    "खाद कितनी देनी चाहिए?"  # Hindi
                ],
                "responses": [
                    "For {crop}, apply N:{n}, P:{p}, K:{k} kg/ha.",
                    "Based on soil test, apply {fertilizer_amount}kg/ha."
                ]
            },
            "market_price": {
                "patterns": [
                    "Current {crop} price",
                    "When to sell my produce?",
                    "Market rate for {crop}",
                    "MSP for {crop} this year",
                    "मंडी में भाव क्या है?"  # Hindi
                ],
                "responses": [
                    "Current {crop} price is ₹{price}/quintal at {market}.",
                    "MSP for {crop} is ₹{msp}/quintal this season."
                ]
            }
        }
        
        # Generate training samples
        training_data = []
        for intent_name, intent_data in intents.items():
            for pattern in intent_data["patterns"]:
                # Generate variations with actual crop/soil values
                for crop in list(CROP_PROFILES.keys())[:5]:  # Top 5 crops
                    sample = {
                        "text": pattern.format(
                            crop=crop,
                            soil_type=np.random.choice(CROP_PROFILES[crop]['soil_types']),
                            state=np.random.choice(CROP_PROFILES[crop]['states']),
                            season=np.random.choice(CROP_PROFILES[crop]['season']),
                            ph=round(np.random.uniform(5.5, 7.5), 1),
                            rainfall=np.random.randint(300, 1500),
                            temperature=np.random.randint(20, 35)
                        ),
                        "intent": intent_name
                    }
                    training_data.append(sample)
        
        result = {
            "intents": intents,
            "training_data": training_data,
            "metadata": {
                "num_intents": len(intents),
                "num_samples": len(training_data),
                "languages": ["English", "Hindi", "Tamil"],
                "generated_at": datetime.now().isoformat()
            }
        }
        
        logger.info(f"✓ Generated intent corpus with {len(training_data)} samples")
        return result


def main():
    """Main entry point for data generation."""
    parser = argparse.ArgumentParser(description='Generate AgriSense training data')
    parser.add_argument('--samples', type=int, default=2000, 
                        help='Samples per crop (default: 2000)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: ./tabular)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / 'tabular'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    intent_dir = Path(__file__).parent / 'intent_corpus'
    intent_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("🌾 AgriSense Data Generator")
    print("=" * 70)
    print(f"Samples per crop: {args.samples}")
    print(f"Output directory: {output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)
    
    # Initialize generator
    generator = AgriDataGenerator(
        samples_per_crop=args.samples,
        random_state=args.seed
    )
    
    # Generate crop recommendation data
    print("\n📊 Generating Crop Recommendation Dataset...")
    crop_df = generator.generate_crop_data()
    crop_path = output_dir / 'india_crops_complete.csv'
    crop_df.to_csv(crop_path, index=False)
    print(f"   Saved: {crop_path}")
    print(f"   Shape: {crop_df.shape}")
    print(f"   Crops: {crop_df['label'].nunique()}")
    
    # Generate yield prediction data
    print("\n📈 Generating Yield Prediction Dataset...")
    yield_df = generator.generate_yield_data(crop_df)
    yield_path = output_dir / 'historical_yields.csv'
    yield_df.to_csv(yield_path, index=False)
    print(f"   Saved: {yield_path}")
    print(f"   Shape: {yield_df.shape}")
    print(f"   Year range: {yield_df['year'].min()}-{yield_df['year'].max()}")
    
    # Generate intent corpus
    print("\n💬 Generating Intent Corpus...")
    intent_data = generator.generate_intent_corpus()
    intent_path = intent_dir / 'intents.json'
    with open(intent_path, 'w', encoding='utf-8') as f:
        json.dump(intent_data, f, indent=2, ensure_ascii=False)
    print(f"   Saved: {intent_path}")
    print(f"   Intents: {intent_data['metadata']['num_intents']}")
    print(f"   Samples: {intent_data['metadata']['num_samples']}")
    
    # Print data summary
    print("\n" + "=" * 70)
    print("📋 Data Summary")
    print("=" * 70)
    print("\nCrop Distribution:")
    print(crop_df['label'].value_counts().head(10).to_string())
    print("\nSoil Type Distribution:")
    print(crop_df['soil_type'].value_counts().to_string())
    print("\nNumerical Features Statistics:")
    print(crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].describe().round(2).to_string())
    
    print("\n" + "=" * 70)
    print("✅ Data Generation Complete!")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {crop_path}")
    print(f"  2. {yield_path}")
    print(f"  3. {intent_path}")
    
    return crop_df, yield_df


if __name__ == "__main__":
    main()
