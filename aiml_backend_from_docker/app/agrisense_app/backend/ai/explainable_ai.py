"""
Explainable AI Module for AgriSense

Provides human-readable explanations for ML model predictions using:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Feature importance analysis
- Natural language explanation generation

Helps farmers understand WHY the model made a specific recommendation.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

# Optional: Install with `pip install shap lime`
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logging.warning("LIME not installed. Install with: pip install lime")


logger = logging.getLogger(__name__)


class ExplanationMethod(Enum):
    """Available explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    FEATURE_IMPORTANCE = "feature_importance"
    RULE_BASED = "rule_based"


@dataclass
class FeatureContribution:
    """Individual feature's contribution to prediction"""
    feature_name: str
    value: float
    contribution: float
    contribution_percent: float
    direction: str  # 'positive' or 'negative'
    importance_rank: int
    
    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "value": self.value,
            "contribution": round(self.contribution, 4),
            "contribution_percent": round(self.contribution_percent, 2),
            "direction": self.direction,
            "importance_rank": self.importance_rank
        }


@dataclass
class Explanation:
    """Complete explanation for a model prediction"""
    prediction: Union[float, str, int]
    prediction_confidence: float
    method: str
    feature_contributions: List[FeatureContribution]
    natural_language_explanation: str
    actionable_insights: List[str]
    alternative_scenarios: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict:
        return {
            "prediction": self.prediction,
            "prediction_confidence": self.prediction_confidence,
            "method": self.method,
            "feature_contributions": [fc.to_dict() for fc in self.feature_contributions],
            "natural_language_explanation": self.natural_language_explanation,
            "actionable_insights": self.actionable_insights,
            "alternative_scenarios": self.alternative_scenarios,
            "timestamp": self.timestamp.isoformat()
        }


class ExplainableAI:
    """
    Provides explanations for ML model predictions
    """
    
    # Feature descriptions for natural language generation
    FEATURE_DESCRIPTIONS = {
        "temperature": "temperature",
        "humidity": "humidity",
        "soil_moisture": "soil moisture",
        "ph_level": "soil pH",
        "nitrogen": "nitrogen level",
        "phosphorus": "phosphorus level",
        "potassium": "potassium level",
        "rainfall": "rainfall",
        "season": "season",
        "crop_type": "crop type"
    }
    
    # Optimal ranges for context
    OPTIMAL_RANGES = {
        "temperature": (20, 30),
        "humidity": (50, 70),
        "soil_moisture": (40, 60),
        "ph_level": (6.0, 7.5),
        "nitrogen": (100, 200),
        "phosphorus": (50, 100),
        "potassium": (100, 200)
    }
    
    def __init__(self, model, feature_names: List[str], model_type: str = "classifier"):
        """
        Initialize ExplainableAI
        
        Args:
            model: Trained ML model (sklearn, xgboost, etc.)
            feature_names: List of feature names used by the model
            model_type: 'classifier' or 'regressor'
        """
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None
        self.lime_explainer = None
        
        logger.info(f"Initialized ExplainableAI for {model_type} with {len(feature_names)} features")
    
    def setup_shap_explainer(self, background_data: Optional[np.ndarray] = None):
        """
        Setup SHAP explainer
        
        Args:
            background_data: Background dataset for SHAP (optional, uses summary if None)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Install with: pip install shap")
            return
        
        try:
            # Use TreeExplainer for tree-based models, KernelExplainer for others
            if hasattr(self.model, 'predict_proba'):
                # Classifier
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        background_data
                    )
                else:
                    # Try TreeExplainer for tree-based models
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                    except:
                        # Fallback to basic explainer with mock data
                        mock_data = np.zeros((10, len(self.feature_names)))
                        self.explainer = shap.KernelExplainer(
                            self.model.predict_proba,
                            mock_data
                        )
            else:
                # Regressor
                if background_data is not None:
                    self.explainer = shap.KernelExplainer(
                        self.model.predict,
                        background_data
                    )
                else:
                    try:
                        self.explainer = shap.TreeExplainer(self.model)
                    except:
                        mock_data = np.zeros((10, len(self.feature_names)))
                        self.explainer = shap.KernelExplainer(
                            self.model.predict,
                            mock_data
                        )
            
            logger.info("SHAP explainer setup complete")
        except Exception as e:
            logger.error(f"Failed to setup SHAP explainer: {e}")
            self.explainer = None
    
    def setup_lime_explainer(
        self,
        training_data: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """
        Setup LIME explainer
        
        Args:
            training_data: Training dataset for LIME
            class_names: Class names for classification (optional)
        """
        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Install with: pip install lime")
            return
        
        try:
            mode = "classification" if self.model_type == "classifier" else "regression"
            
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=self.feature_names,
                class_names=class_names,
                mode=mode,
                discretize_continuous=True
            )
            
            logger.info("LIME explainer setup complete")
        except Exception as e:
            logger.error(f"Failed to setup LIME explainer: {e}")
            self.lime_explainer = None
    
    def explain_prediction(
        self,
        input_features: Union[np.ndarray, Dict[str, float]],
        method: ExplanationMethod = ExplanationMethod.SHAP,
        top_k: int = 5
    ) -> Explanation:
        """
        Generate explanation for a single prediction
        
        Args:
            input_features: Input features (numpy array or dict)
            method: Explanation method to use
            top_k: Number of top features to highlight
            
        Returns:
            Explanation object with detailed insights
        """
        # Convert input to numpy array if dict
        if isinstance(input_features, dict):
            features_array = np.array([
                input_features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
            feature_dict = input_features
        else:
            features_array = input_features.reshape(1, -1)
            feature_dict = dict(zip(self.feature_names, input_features))
        
        # Make prediction
        if self.model_type == "classifier":
            if hasattr(self.model, 'predict_proba'):
                prediction_proba = self.model.predict_proba(features_array)[0]
                prediction = int(np.argmax(prediction_proba))
                confidence = float(np.max(prediction_proba))
            else:
                prediction = int(self.model.predict(features_array)[0])
                confidence = 0.85  # Default confidence if no probability available
        else:
            prediction = float(self.model.predict(features_array)[0])
            confidence = 0.80  # Default confidence for regression
        
        # Generate explanation based on method
        if method == ExplanationMethod.SHAP and self.explainer is not None:
            feature_contributions = self._explain_with_shap(features_array, feature_dict)
        elif method == ExplanationMethod.LIME and self.lime_explainer is not None:
            feature_contributions = self._explain_with_lime(features_array, feature_dict)
        else:
            # Fallback to rule-based explanation
            feature_contributions = self._explain_rule_based(feature_dict, prediction)
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Assign importance ranks
        for i, fc in enumerate(feature_contributions, 1):
            fc.importance_rank = i
        
        # Generate natural language explanation
        nl_explanation = self._generate_natural_language_explanation(
            prediction,
            confidence,
            feature_contributions[:top_k],
            feature_dict
        )
        
        # Generate actionable insights
        actionable_insights = self._generate_actionable_insights(
            prediction,
            feature_contributions[:top_k],
            feature_dict
        )
        
        # Generate alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(
            feature_dict,
            feature_contributions[:top_k]
        )
        
        return Explanation(
            prediction=prediction,
            prediction_confidence=confidence,
            method=method.value,
            feature_contributions=feature_contributions[:top_k],
            natural_language_explanation=nl_explanation,
            actionable_insights=actionable_insights,
            alternative_scenarios=alternative_scenarios
        )
    
    def _explain_with_shap(
        self,
        features_array: np.ndarray,
        feature_dict: Dict[str, float]
    ) -> List[FeatureContribution]:
        """Generate explanation using SHAP"""
        try:
            shap_values = self.explainer.shap_values(features_array)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # Use first class
            
            # Flatten if needed
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            # Calculate total contribution
            total_contribution = np.sum(np.abs(shap_values))
            
            # Create feature contributions
            contributions = []
            for i, feature_name in enumerate(self.feature_names):
                contribution = float(shap_values[i])
                contribution_percent = (abs(contribution) / total_contribution * 100) if total_contribution > 0 else 0
                
                contributions.append(FeatureContribution(
                    feature_name=feature_name,
                    value=feature_dict.get(feature_name, 0.0),
                    contribution=contribution,
                    contribution_percent=contribution_percent,
                    direction="positive" if contribution >= 0 else "negative",
                    importance_rank=0  # Will be set later
                ))
            
            return contributions
        
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._explain_rule_based(feature_dict, None)
    
    def _explain_with_lime(
        self,
        features_array: np.ndarray,
        feature_dict: Dict[str, float]
    ) -> List[FeatureContribution]:
        """Generate explanation using LIME"""
        try:
            if self.model_type == "classifier":
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict
            
            explanation = self.lime_explainer.explain_instance(
                features_array[0],
                predict_fn,
                num_features=len(self.feature_names)
            )
            
            # Extract feature contributions
            lime_contributions = explanation.as_list()
            
            # Parse LIME output
            contributions = []
            total_contribution = sum(abs(contrib) for _, contrib in lime_contributions)
            
            for feature_desc, contribution in lime_contributions:
                # Parse feature name from description (e.g., "temperature > 25.0")
                feature_name = feature_desc.split()[0]
                
                if feature_name in self.feature_names:
                    contribution_percent = (abs(contribution) / total_contribution * 100) if total_contribution > 0 else 0
                    
                    contributions.append(FeatureContribution(
                        feature_name=feature_name,
                        value=feature_dict.get(feature_name, 0.0),
                        contribution=float(contribution),
                        contribution_percent=contribution_percent,
                        direction="positive" if contribution >= 0 else "negative",
                        importance_rank=0
                    ))
            
            return contributions
        
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return self._explain_rule_based(feature_dict, None)
    
    def _explain_rule_based(
        self,
        feature_dict: Dict[str, float],
        prediction: Optional[Any] = None
    ) -> List[FeatureContribution]:
        """
        Generate rule-based explanation (fallback method)
        
        Uses domain knowledge and optimal ranges
        """
        contributions = []
        
        for feature_name in self.feature_names:
            value = feature_dict.get(feature_name, 0.0)
            
            # Calculate contribution based on deviation from optimal range
            if feature_name in self.OPTIMAL_RANGES:
                optimal_min, optimal_max = self.OPTIMAL_RANGES[feature_name]
                optimal_mid = (optimal_min + optimal_max) / 2
                
                # Contribution: negative if outside optimal, positive if inside
                if optimal_min <= value <= optimal_max:
                    deviation = abs(value - optimal_mid) / (optimal_max - optimal_min)
                    contribution = 1.0 - deviation
                    direction = "positive"
                else:
                    if value < optimal_min:
                        deviation = (optimal_min - value) / optimal_min
                    else:
                        deviation = (value - optimal_max) / optimal_max
                    contribution = -min(1.0, deviation)
                    direction = "negative"
            else:
                # Unknown optimal range: assume value itself is contribution
                contribution = value / 100.0  # Normalize
                direction = "positive" if contribution >= 0 else "negative"
            
            contributions.append(FeatureContribution(
                feature_name=feature_name,
                value=value,
                contribution=contribution,
                contribution_percent=0.0,  # Will calculate after
                direction=direction,
                importance_rank=0
            ))
        
        # Calculate percentages
        total_contribution = sum(abs(c.contribution) for c in contributions)
        if total_contribution > 0:
            for c in contributions:
                c.contribution_percent = abs(c.contribution) / total_contribution * 100
        
        return contributions
    
    def _generate_natural_language_explanation(
        self,
        prediction: Any,
        confidence: float,
        top_features: List[FeatureContribution],
        feature_dict: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation
        """
        explanation_parts = []
        
        # Start with prediction
        if self.model_type == "classifier":
            explanation_parts.append(
                f"The model predicts class {prediction} with {confidence*100:.1f}% confidence."
            )
        else:
            explanation_parts.append(
                f"The model predicts a value of {prediction:.2f} with {confidence*100:.1f}% confidence."
            )
        
        # Explain top contributing features
        if top_features:
            explanation_parts.append("\nKey factors influencing this prediction:")
            
            for i, fc in enumerate(top_features[:3], 1):
                feature_desc = self.FEATURE_DESCRIPTIONS.get(fc.feature_name, fc.feature_name)
                value_str = f"{fc.value:.1f}"
                
                # Add context about optimal range
                if fc.feature_name in self.OPTIMAL_RANGES:
                    optimal_min, optimal_max = self.OPTIMAL_RANGES[fc.feature_name]
                    
                    if fc.value < optimal_min:
                        context = f" (below optimal range of {optimal_min}-{optimal_max})"
                    elif fc.value > optimal_max:
                        context = f" (above optimal range of {optimal_min}-{optimal_max})"
                    else:
                        context = f" (within optimal range)"
                else:
                    context = ""
                
                impact = "positively" if fc.direction == "positive" else "negatively"
                
                explanation_parts.append(
                    f"{i}. The {feature_desc} is {value_str}{context}, "
                    f"which {impact} impacts the prediction "
                    f"(contribution: {fc.contribution_percent:.1f}%)."
                )
        
        return " ".join(explanation_parts)
    
    def _generate_actionable_insights(
        self,
        prediction: Any,
        top_features: List[FeatureContribution],
        feature_dict: Dict[str, float]
    ) -> List[str]:
        """
        Generate actionable recommendations
        """
        insights = []
        
        for fc in top_features:
            if fc.feature_name in self.OPTIMAL_RANGES:
                optimal_min, optimal_max = self.OPTIMAL_RANGES[fc.feature_name]
                value = fc.value
                feature_desc = self.FEATURE_DESCRIPTIONS.get(fc.feature_name, fc.feature_name)
                
                if value < optimal_min:
                    deficit = optimal_min - value
                    insights.append(
                        f"⚠️ Increase {feature_desc} by {deficit:.1f} to reach optimal range ({optimal_min}-{optimal_max})"
                    )
                elif value > optimal_max:
                    excess = value - optimal_max
                    insights.append(
                        f"⚠️ Decrease {feature_desc} by {excess:.1f} to reach optimal range ({optimal_min}-{optimal_max})"
                    )
                else:
                    insights.append(
                        f"✅ {feature_desc.capitalize()} is within optimal range - maintain current levels"
                    )
        
        return insights[:5]  # Limit to top 5 insights
    
    def _generate_alternative_scenarios(
        self,
        feature_dict: Dict[str, float],
        top_features: List[FeatureContribution]
    ) -> List[Dict[str, Any]]:
        """
        Generate what-if scenarios showing how changes affect prediction
        """
        scenarios = []
        
        for fc in top_features[:3]:  # Top 3 features
            if fc.feature_name not in self.OPTIMAL_RANGES:
                continue
            
            optimal_min, optimal_max = self.OPTIMAL_RANGES[fc.feature_name]
            optimal_mid = (optimal_min + optimal_max) / 2
            
            # Create scenario where this feature is optimal
            alt_features = feature_dict.copy()
            alt_features[fc.feature_name] = optimal_mid
            
            # Convert to array and predict
            alt_array = np.array([
                alt_features.get(name, 0.0) for name in self.feature_names
            ]).reshape(1, -1)
            
            try:
                if self.model_type == "classifier" and hasattr(self.model, 'predict_proba'):
                    alt_prediction = int(np.argmax(self.model.predict_proba(alt_array)))
                    alt_confidence = float(np.max(self.model.predict_proba(alt_array)))
                else:
                    alt_prediction = float(self.model.predict(alt_array)[0])
                    alt_confidence = 0.80
                
                feature_desc = self.FEATURE_DESCRIPTIONS.get(fc.feature_name, fc.feature_name)
                
                scenarios.append({
                    "scenario": f"If {feature_desc} was optimal ({optimal_mid:.1f})",
                    "prediction": alt_prediction,
                    "confidence": round(alt_confidence, 3),
                    "change_from_current": round(abs(alt_prediction - feature_dict.get(fc.feature_name, 0)), 2)
                })
            except Exception as e:
                logger.warning(f"Failed to generate scenario for {fc.feature_name}: {e}")
        
        return scenarios


# Convenience function
def explain_model_prediction(
    model,
    feature_names: List[str],
    input_features: Dict[str, float],
    model_type: str = "classifier",
    method: str = "rule_based"
) -> dict:
    """
    Convenience function to explain a model prediction
    
    Args:
        model: Trained ML model
        feature_names: List of feature names
        input_features: Input features as dictionary
        model_type: 'classifier' or 'regressor'
        method: 'shap', 'lime', or 'rule_based'
        
    Returns:
        dict: Explanation in dictionary format
    """
    explainer = ExplainableAI(model, feature_names, model_type)
    
    method_enum = ExplanationMethod.RULE_BASED
    if method == "shap" and SHAP_AVAILABLE:
        explainer.setup_shap_explainer()
        method_enum = ExplanationMethod.SHAP
    elif method == "lime" and LIME_AVAILABLE:
        # Note: LIME requires training data, not provided here
        method_enum = ExplanationMethod.RULE_BASED
    
    explanation = explainer.explain_prediction(
        input_features,
        method=method_enum,
        top_k=5
    )
    
    return explanation.to_dict()
