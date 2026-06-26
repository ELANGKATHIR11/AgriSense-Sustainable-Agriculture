"""
CROP RECOMMENDATION SYSTEM - USER INTERFACE
============================================
Easy-to-use script for getting crop recommendations based on soil test data
"""

import pandas as pd
import joblib

class CropRecommender:
    def __init__(self, model_path='crop_recommendation_model.pkl'):
        """Load the trained model"""
        print("Loading Crop Recommendation Model...")
        model_data = joblib.load(model_path)
        self.model = model_data['models'][model_data['best_model_name']]['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"✓ Model loaded successfully (Using: {self.best_model_name})\n")
    
    def recommend(self, pH, N, P, K, Fe, Mn, Zn, Cu, B, Water, Moisture, Temperature, Rainfall, top_n=5):
        """
        Get crop recommendations based on soil parameters
        
        Parameters:
        -----------
        pH : float - Soil pH (5.0 - 8.5)
        N : float - Nitrogen in kg/ha (10 - 300)
        P : float - Phosphorus in kg/ha (10 - 150)
        K : float - Potassium in kg/ha (10 - 200)
        Fe : float - Iron in ppm (2.0 - 8.0)
        Mn : float - Manganese in ppm (0.8 - 5.0)
        Zn : float - Zinc in ppm (0.4 - 3.0)
        Cu : float - Copper in ppm (0.15 - 1.5)
        B : float - Boron in ppm (0.15 - 1.5)
        Water : float - Water requirement in mm/season (200 - 2500)
        Moisture : float - Soil moisture in % (35 - 90)
        Temperature : float - Temperature in Celsius (10 - 40)
        Rainfall : float - Rainfall in mm/season (200 - 2500)
        top_n : int - Number of recommendations (default: 5)
        
        Returns:
        --------
        DataFrame with crop recommendations
        """
        # Prepare input
        soil_data = {
            'pH': pH, 'N': N, 'P': P, 'K': K,
            'Fe': Fe, 'Mn': Mn, 'Zn': Zn, 'Cu': Cu, 'B': B,
            'Water': Water, 'Moisture': Moisture,
            'Temperature': Temperature, 'Rainfall': Rainfall
        }
        
        input_df = pd.DataFrame([soil_data])
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Get predictions
        probabilities = self.model.predict_proba(input_scaled)[0]
        top_indices = probabilities.argsort()[-top_n:][::-1]
        
        results = pd.DataFrame({
            'Rank': range(1, top_n + 1),
            'Crop': self.label_encoder.inverse_transform(top_indices),
            'Suitability (%)': (probabilities[top_indices] * 100).round(2)
        })
        
        return results
    
    def interactive_input(self):
        """Interactive mode to get soil parameters from user"""
        print("=" * 70)
        print("INTERACTIVE CROP RECOMMENDATION")
        print("=" * 70)
        print("\nPlease enter your soil test parameters:\n")
        
        try:
            pH = float(input("Soil pH (5.0-8.5): "))
            print()
            
            print("Macronutrients (kg/ha):")
            N = float(input("  Nitrogen (N): "))
            P = float(input("  Phosphorus (P): "))
            K = float(input("  Potassium (K): "))
            print()
            
            print("Micronutrients (ppm):")
            Fe = float(input("  Iron (Fe): "))
            Mn = float(input("  Manganese (Mn): "))
            Zn = float(input("  Zinc (Zn): "))
            Cu = float(input("  Copper (Cu): "))
            B = float(input("  Boron (B): "))
            print()
            
            print("Environmental parameters:")
            Water = float(input("  Water availability (mm/season): "))
            Moisture = float(input("  Soil moisture (%): "))
            Temperature = float(input("  Average temperature (°C): "))
            Rainfall = float(input("  Average rainfall (mm/season): "))
            
            print("\n" + "=" * 70)
            print("CROP RECOMMENDATIONS")
            print("=" * 70 + "\n")
            
            recommendations = self.recommend(
                pH, N, P, K, Fe, Mn, Zn, Cu, B,
                Water, Moisture, Temperature, Rainfall
            )
            
            print(recommendations.to_string(index=False))
            print("\n" + "=" * 70)
            
        except ValueError:
            print("\n❌ Invalid input! Please enter numeric values only.")
        except KeyboardInterrupt:
            print("\n\nExiting...")


def example_usage():
    """Show example usage of the recommender"""
    print("=" * 70)
    print("CROP RECOMMENDATION SYSTEM - EXAMPLE USAGE")
    print("=" * 70)
    
    # Load recommender
    recommender = CropRecommender('/home/claude/crop_recommendation_model.pkl')
    
    # Example 1: From soil health card
    print("\n📋 Example 1: Soil test from health card")
    print("-" * 70)
    print("Soil Parameters:")
    print("  pH: 7.0, N: 120 kg/ha, P: 54 kg/ha, K: 100 kg/ha")
    print("  Fe: 4.06 ppm, Mn: 1.68 ppm, Zn: 0.83 ppm, Cu: 0.46 ppm, B: 0.3 ppm")
    print("  Water: 500mm, Moisture: 60%, Temp: 28°C, Rainfall: 600mm\n")
    
    result1 = recommender.recommend(
        pH=7.0, N=120, P=54, K=100,
        Fe=4.06, Mn=1.68, Zn=0.83, Cu=0.46, B=0.3,
        Water=500, Moisture=60, Temperature=28, Rainfall=600
    )
    print(result1.to_string(index=False))
    
    # Example 2: High fertility
    print("\n\n📋 Example 2: High fertility soil")
    print("-" * 70)
    print("Soil Parameters:")
    print("  pH: 6.5, N: 180 kg/ha, P: 80 kg/ha, K: 150 kg/ha")
    print("  Fe: 5.5 ppm, Mn: 2.5 ppm, Zn: 1.5 ppm, Cu: 0.8 ppm, B: 0.6 ppm")
    print("  Water: 800mm, Moisture: 70%, Temp: 25°C, Rainfall: 800mm\n")
    
    result2 = recommender.recommend(
        pH=6.5, N=180, P=80, K=150,
        Fe=5.5, Mn=2.5, Zn=1.5, Cu=0.8, B=0.6,
        Water=800, Moisture=70, Temperature=25, Rainfall=800
    )
    print(result2.to_string(index=False))
    
    # Example 3: Arid region
    print("\n\n📋 Example 3: Arid region (low fertility)")
    print("-" * 70)
    print("Soil Parameters:")
    print("  pH: 7.8, N: 40 kg/ha, P: 25 kg/ha, K: 30 kg/ha")
    print("  Fe: 2.0 ppm, Mn: 0.8 ppm, Zn: 0.4 ppm, Cu: 0.2 ppm, B: 0.15 ppm")
    print("  Water: 350mm, Moisture: 45%, Temp: 35°C, Rainfall: 350mm\n")
    
    result3 = recommender.recommend(
        pH=7.8, N=40, P=25, K=30,
        Fe=2.0, Mn=0.8, Zn=0.4, Cu=0.2, B=0.15,
        Water=350, Moisture=45, Temperature=35, Rainfall=350
    )
    print(result3.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("\n💡 To use interactive mode, run:")
    print("   recommender = CropRecommender()")
    print("   recommender.interactive_input()")
    print("=" * 70)


if __name__ == "__main__":
    # Run example usage
    example_usage()
    
    # Uncomment below to use interactive mode
    # recommender = CropRecommender('/home/claude/crop_recommendation_model.pkl')
    # recommender.interactive_input()
