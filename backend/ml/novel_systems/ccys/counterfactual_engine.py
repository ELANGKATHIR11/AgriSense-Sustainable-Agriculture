import numpy as np


class TLearnerSimulator:
    """
    Counterfactual Simulation Engine using Meta-Learners.

    Patent Claim Element:
    "A simulation engine that utilizes a 'T-Learner' architecture to separate
    the treated distribution from the control distribution, enabling
    estimation of Conditional Average Treatment Effects (CATE)."
    """

    def __init__(self, causal_graph):
        self.graph = causal_graph
        # In a real system, these would be trained XGBoost/LightGBM models
        self.model_control = None  # M0(X) given T=0
        self.model_treated = None  # M1(X) given T=1

    def fit_mock(self):
        """
        Simulate training of the T-Learner for prototype purposes.
        M0(X): Yield function without Nitrogen boost.
        M1(X): Yield function WITH Nitrogen boost.
        """
        # Mock functions for prototype demo
        # Base Yield = 2000 + 50 * Soil_Carbon
        self.model_control = (
            lambda x: 2000 + 50 * x["Soil_Organic_Carbon"] + 10 * x["Rainfall_mm"]
        )

        # Treated Yield = Base Yield + 15 * N - 0.1 * N^2 (Diminishing returns)
        # Note: 'x' typically contains confounders.
        self.model_treated = lambda x, n_amount: self.model_control(x) + (
            15 * n_amount - 0.1 * n_amount**2
        )

    def predict_ate(self, n_amount):
        """
        Predict Average Treatment Effect (ATE) of applying 'n_amount' Nitrogen.
        """
        # ATE is average over a population.
        # Here we mock it for a standard field.
        standard_field = {"Soil_Organic_Carbon": 1.5, "Rainfall_mm": 500}
        y0 = self.model_control(standard_field)
        y1 = self.model_treated(standard_field, n_amount)
        return y1 - y0

    def simulate_counterfactual(self, field_context, intervention_dict):
        """
        Simulate "What If" scenario.

        Args:
            field_context: Dict of covariates (Soil, Rain, etc.)
            intervention_dict: {'Nitrogen_Applied': 50}

        Returns:
            Dict containing Observational vs Counterfactual outcomes.
        """
        if self.model_control is None:
            self.fit_mock()

        # 1. Estimate Factual (Observational) - "What happens if I do nothing (or standard)?"
        # Assuming Control T=0
        y_factual = self.model_control(field_context)

        # 2. Estimate Counterfactual - "What if I do(N=50)?"
        n_amount = intervention_dict.get("Nitrogen_Applied", 0)
        y_counterfactual = self.model_treated(field_context, n_amount)

        cate = y_counterfactual - y_factual

        return {
            "factual_yield": y_factual,
            "counterfactual_yield": y_counterfactual,
            "treatment_effect": cate,
            "roi_recommendation": "Positive" if cate > 100 else "Negative",
        }
