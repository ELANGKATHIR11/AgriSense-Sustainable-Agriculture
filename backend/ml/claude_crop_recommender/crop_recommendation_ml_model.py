import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class CropRecommendationSystem:
    """
    Intelligent Crop Recommendation System
    Recommends best crops based on soil test parameters for 100+ Indian crops
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.best_model_name = None

    def generate_training_data(self, df, samples_per_crop=50):
        """
        Generate synthetic training data based on crop requirements.

        Creates samples within the min-max ranges for each crop.

        Args:
            df: DataFrame with crop requirements
            samples_per_crop: Number of samples to generate per crop

        Returns:
            DataFrame with synthetic training data
        """
        training_data = []

        for idx, row in df.iterrows():
            crop_name = row['Crop_Name']

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
                    'Water': np.random.uniform(
                        row['Water_Requirement_Min'],
                        row['Water_Requirement_Max']
                    ),
                    'Moisture': np.random.uniform(
                        row['Moisture_Min'],
                        row['Moisture_Max']
                    ),
                    'Temperature': np.random.uniform(
                        row['Temp_Min'],
                        row['Temp_Max']
                    ),
                    'Rainfall': np.random.uniform(
                        row['Rainfall_Min'],
                        row['Rainfall_Max']
                    ),
                    'Crop': crop_name
                }
                training_data.append(sample)

        return pd.DataFrame(training_data)
    
    def train(self, crop_requirements_df):
        """Train multiple ML models and select the best one"""
        print("🌾 Generating training data...")
        training_df = self.generate_training_data(crop_requirements_df)

        num_crops = crop_requirements_df['Crop_Name'].nunique()
        num_samples = len(training_df)
        print(f"✓ Training data generated: {num_samples} samples "
              f"for {num_crops} crops")
        X = training_df.drop('Crop', axis=1)
        y = training_df['Crop']

        self.feature_names = X.columns.tolist()
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42,
            stratify=y_encoded
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("\n📚 Training multiple models...")
        
        # Random Forest
        print("  • Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        rf_accuracy = accuracy_score(y_test, rf.predict(X_test_scaled))
        self.models['Random Forest'] = {'model': rf, 'accuracy': rf_accuracy}
        print(f"    ✓ Accuracy: {rf_accuracy:.4f}")

        # Gradient Boosting
        print("  • Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb.fit(X_train_scaled, y_train)
        gb_accuracy = accuracy_score(y_test, gb.predict(X_test_scaled))
        self.models['Gradient Boosting'] = {
            'model': gb,
            'accuracy': gb_accuracy
        }
        print(f"    ✓ Accuracy: {gb_accuracy:.4f}")

        # SVM
        print("  • Support Vector Machine...")
        svm = SVC(kernel='rbf', random_state=42, probability=True)
        svm.fit(X_train_scaled, y_train)
        svm_accuracy = accuracy_score(y_test, svm.predict(X_test_scaled))
        self.models['SVM'] = {'model': svm, 'accuracy': svm_accuracy}
        print(f"    ✓ Accuracy: {svm_accuracy:.4f}")

        # Neural Network
        print("  • Neural Network...")
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=500,
            random_state=42
        )
        mlp.fit(X_train_scaled, y_train)
        mlp_accuracy = accuracy_score(y_test, mlp.predict(X_test_scaled))
        self.models['Neural Network'] = {
            'model': mlp,
            'accuracy': mlp_accuracy
        }
        print(f"    ✓ Accuracy: {mlp_accuracy:.4f}")

        best_model_tuple = max(
            self.models.items(),
            key=lambda x: x[1]['accuracy']
        )
        self.best_model_name = best_model_tuple[0]
        best_accuracy = self.models[self.best_model_name]['accuracy']
        print(f"\n✅ Best Model: {self.best_model_name} "
              f"(Accuracy: {best_accuracy:.4f})")
        return self.models[self.best_model_name]['accuracy']

    def predict_crop(self, soil_data, top_n=5):
        """
        Predict the best crops for given soil parameters

        Args:
            soil_data (dict): Dictionary with keys: pH, N, P, K, Fe, Mn, Zn,
                             Cu, B, Water, Moisture, Temperature, Rainfall
            top_n (int): Number of top recommendations to return

        Returns:
            dict: Recommendations with crop names and suitability scores
        """
        input_df = pd.DataFrame([soil_data])

        for feature in self.feature_names:
            if feature not in input_df.columns:
                msg = f"Missing required feature: {feature}"
                raise ValueError(msg)

        input_scaled = self.scaler.transform(input_df[self.feature_names])
        best_model = self.models[self.best_model_name]['model']
        probabilities = best_model.predict_proba(input_scaled)[0]

        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_crops = self.label_encoder.inverse_transform(top_indices)
        top_probs = probabilities[top_indices]

        recommendations = []
        for rank, (crop, prob) in enumerate(zip(top_crops, top_probs), 1):
            if prob > 0.7:
                confidence = 'High'
            elif prob > 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'

            recommendations.append({
                'rank': rank,
                'crop_name': crop,
                'suitability_score': round(float(prob) * 100, 2),
                'confidence': confidence
            })

        return recommendations

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
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath='crop_recommendation_model.pkl'):
        """Load a previously trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"✓ Model loaded from {filepath}")
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.models = model_data['models']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.best_model_name = model_data['best_model_name']
        print(f"✓ Model loaded from {filepath}")
