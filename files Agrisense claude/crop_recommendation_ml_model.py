import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

class CropRecommendationSystem:
    """
    Intelligent Crop Recommendation System
    Recommends best crops based on soil test parameters
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.best_model_name = None
        
    def generate_training_data(self, df, samples_per_crop=50):
        """
        Generate synthetic training data based on crop requirements
        Creates samples within the min-max ranges for each crop
        """
        training_data = []
        
        for idx, row in df.iterrows():
            crop_name = row['Crop_Name']
            
            # Generate samples within the acceptable range for each crop
            for _ in range(samples_per_crop):
                sample = {
                    'pH': np.random.uniform(row['pH_Min'], row['pH_Max']),
                    'N': np.random.uniform(row['N_Min'], row['N_Max']),
                    'P': np.random.uniform(row['P_Min'], row['P_Max']),
                    'K': np.random.uniform(row['K_Min'], row['K_Max']),
                    'Fe': np.random.uniform(row['Fe_Min'], row['Fe_Max']),
                    'Mn': np.random.uniform(row['Mn_Min'], row['Mn_Max']),
                    'Zn': np.random.uniform(row['Zn_Min'], row['Zn_Max']),
                    'Cu': np.random.uniform(row['Cu_Min'], row['Cu_Max']),
                    'B': np.random.uniform(row['B_Min'], row['B_Max']),
                    'Water': np.random.uniform(row['Water_Requirement_Min'], row['Water_Requirement_Max']),
                    'Moisture': np.random.uniform(row['Moisture_Min'], row['Moisture_Max']),
                    'Temperature': np.random.uniform(row['Temp_Min'], row['Temp_Max']),
                    'Rainfall': np.random.uniform(row['Rainfall_Min'], row['Rainfall_Max']),
                    'Crop': crop_name
                }
                training_data.append(sample)
        
        return pd.DataFrame(training_data)
    
    def train(self, crop_requirements_df):
        """
        Train multiple ML models and select the best one
        """
        print("Generating training data...")
        training_df = self.generate_training_data(crop_requirements_df)
        
        print(f"Training data generated: {len(training_df)} samples for {crop_requirements_df['Crop_Name'].nunique()} crops")
        
        # Separate features and target
        X = training_df.drop('Crop', axis=1)
        y = training_df['Crop']
        
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        print("\nTraining multiple models...")
        
        # 1. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        rf_pred = rf.predict(X_test_scaled)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        self.models['Random Forest'] = {'model': rf, 'accuracy': rf_accuracy}
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        # 2. Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_pred = gb.predict(X_test_scaled)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        self.models['Gradient Boosting'] = {'model': gb, 'accuracy': gb_accuracy}
        print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
        
        # 3. SVM
        print("Training SVM...")
        svm = SVC(kernel='rbf', random_state=42, probability=True)
        svm.fit(X_train_scaled, y_train)
        svm_pred = svm.predict(X_test_scaled)
        svm_accuracy = accuracy_score(y_test, svm_pred)
        self.models['SVM'] = {'model': svm, 'accuracy': svm_accuracy}
        print(f"SVM Accuracy: {svm_accuracy:.4f}")
        
        # 4. Neural Network
        print("Training Neural Network...")
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
        mlp.fit(X_train_scaled, y_train)
        mlp_pred = mlp.predict(X_test_scaled)
        mlp_accuracy = accuracy_score(y_test, mlp_pred)
        self.models['Neural Network'] = {'model': mlp, 'accuracy': mlp_accuracy}
        print(f"Neural Network Accuracy: {mlp_accuracy:.4f}")
        
        # Select best model
        self.best_model_name = max(self.models.items(), key=lambda x: x[1]['accuracy'])[0]
        print(f"\n✓ Best Model: {self.best_model_name} with accuracy: {self.models[self.best_model_name]['accuracy']:.4f}")
        
        # Feature importance (for Random Forest)
        if self.best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.models['Random Forest']['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            print("\nTop 5 Most Important Features:")
            print(feature_importance.head())
        
        return self.models[self.best_model_name]['accuracy']
    
    def predict_crop(self, soil_data, top_n=5):
        """
        Predict the best crops for given soil parameters
        
        Parameters:
        -----------
        soil_data : dict
            Dictionary containing soil parameters:
            {'pH': value, 'N': value, 'P': value, 'K': value, 
             'Fe': value, 'Mn': value, 'Zn': value, 'Cu': value, 'B': value,
             'Water': value, 'Moisture': value, 'Temperature': value, 'Rainfall': value}
        
        top_n : int
            Number of top crop recommendations to return
            
        Returns:
        --------
        DataFrame with top N crop recommendations and their probabilities
        """
        # Prepare input data
        input_df = pd.DataFrame([soil_data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in input_df.columns:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Scale input
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Get prediction probabilities from best model
        best_model = self.models[self.best_model_name]['model']
        probabilities = best_model.predict_proba(input_scaled)[0]
        
        # Get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Rank': range(1, top_n + 1),
            'Recommended_Crop': top_crops,
            'Suitability_Score': (top_probs * 100).round(2),
            'Confidence': ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                          for p in top_probs]
        })
        
        return results
    
    def batch_predict(self, soil_samples_df):
        """
        Predict crops for multiple soil samples
        
        Parameters:
        -----------
        soil_samples_df : DataFrame
            DataFrame containing multiple soil samples
            
        Returns:
        --------
        DataFrame with predictions for each sample
        """
        predictions = []
        
        for idx, row in soil_samples_df.iterrows():
            soil_data = row.to_dict()
            top_crop = self.predict_crop(soil_data, top_n=1).iloc[0]
            predictions.append({
                'Sample_ID': idx,
                'Recommended_Crop': top_crop['Recommended_Crop'],
                'Suitability_Score': top_crop['Suitability_Score']
            })
        
        return pd.DataFrame(predictions)
    
    def save_model(self, filepath='crop_recommendation_model.pkl'):
        """Save the trained model"""
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'best_model_name': self.best_model_name
        }
        joblib.dump(model_data, filepath)
        print(f"\n✓ Model saved to {filepath}")
    
    def load_model(self, filepath='crop_recommendation_model.pkl'):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"✓ Model loaded from {filepath}")


# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("CROP RECOMMENDATION SYSTEM - TRAINING")
    print("=" * 70)
    
    # Load crop requirements dataset
    print("\nLoading crop requirements dataset...")
    crop_df = pd.read_csv('/home/claude/india_crops_dataset_complete.csv')
    print(f"Loaded {len(crop_df)} crops")
    
    # Initialize and train the system
    crs = CropRecommendationSystem()
    
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    accuracy = crs.train(crop_df)
    
    # Save the model
    crs.save_model('/home/claude/crop_recommendation_model.pkl')
    
    print("\n" + "=" * 70)
    print("TESTING THE MODEL")
    print("=" * 70)
    
    # Example prediction - Soil test from the uploaded image (approximate values)
    print("\nExample 1: Soil test similar to uploaded image")
    soil_test_1 = {
        'pH': 7.0,
        'N': 120,      # Medium level
        'P': 54,       # Medium level  
        'K': 100,      # Medium level
        'Fe': 4.06,
        'Mn': 1.68,
        'Zn': 0.83,
        'Cu': 0.46,
        'B': 0.3,
        'Water': 500,
        'Moisture': 60,
        'Temperature': 28,
        'Rainfall': 600
    }
    
    recommendations_1 = crs.predict_crop(soil_test_1, top_n=5)
    print("\nTop 5 Crop Recommendations:")
    print(recommendations_1.to_string(index=False))
    
    # Example 2: High fertility soil
    print("\n\nExample 2: High fertility soil")
    soil_test_2 = {
        'pH': 6.5,
        'N': 180,
        'P': 80,
        'K': 150,
        'Fe': 5.5,
        'Mn': 2.5,
        'Zn': 1.5,
        'Cu': 0.8,
        'B': 0.6,
        'Water': 800,
        'Moisture': 70,
        'Temperature': 25,
        'Rainfall': 800
    }
    
    recommendations_2 = crs.predict_crop(soil_test_2, top_n=5)
    print("\nTop 5 Crop Recommendations:")
    print(recommendations_2.to_string(index=False))
    
    # Example 3: Low fertility soil
    print("\n\nExample 3: Low fertility soil (arid region)")
    soil_test_3 = {
        'pH': 7.5,
        'N': 40,
        'P': 25,
        'K': 30,
        'Fe': 2.0,
        'Mn': 0.8,
        'Zn': 0.4,
        'Cu': 0.2,
        'B': 0.15,
        'Water': 350,
        'Moisture': 45,
        'Temperature': 35,
        'Rainfall': 350
    }
    
    recommendations_3 = crs.predict_crop(soil_test_3, top_n=5)
    print("\nTop 5 Crop Recommendations:")
    print(recommendations_3.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\nYou can now use this model to predict suitable crops")
    print("based on soil test parameters.")
