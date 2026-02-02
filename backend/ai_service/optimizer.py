import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import logging

logger = logging.getLogger("AgriSense-AI")


class AgriOptimizationProblem(ElementwiseProblem):
    def __init__(self, yield_model, encoders, crop_name, season, soil_data):
        # We optimize for 3 inputs: Nitrogen (N), Phosphorus (P), Potassium (K)
        # Bounds are based on typical agricultural ranges
        super().__init__(
            n_var=3,
            n_obj=2,
            n_constr=1,
            xl=np.array([0, 0, 0]),
            xu=np.array([140, 140, 140]),
        )
        self.yield_model = yield_model
        self.encoders = encoders
        self.crop_name = crop_name
        self.season = season
        self.soil_data = soil_data

    def _evaluate(self, x, out, *args, **kwargs):
        # x[0]=N, x[1]=P, x[2]=K

        # 1. Predict Yield using XGBoost Core
        try:
            import pandas as pd

            # Map categorical values
            crop_idx = self.encoders["crop_name"].transform([self.crop_name])[0]
            season_idx = self.encoders["season"].transform([self.season])[0]

            input_df = pd.DataFrame(
                [
                    {
                        "soil_n": x[0],
                        "soil_p": x[1],
                        "soil_k": x[2],
                        "soil_ph": self.soil_data.get("ph", 6.5),
                        "organic_carbon": self.soil_data.get("oc", 0.5),
                        "rainfall_mm": self.soil_data.get("rainfall", 200),
                        "temperature_avg_c": self.soil_data.get("temp", 25),
                        "humidity_pct": self.soil_data.get("humidity", 80),
                        "crop_name": crop_idx,
                        "season": season_idx,
                    }
                ]
            )

            # Objective 1: Maximize Yield (Pymoo minimizes, so use negative)
            yield_pred = self.yield_model.predict(input_df)[0]
            f1 = -yield_pred

            # Objective 2: Minimize Resource Cost (Proxied by sum of NPK)
            f2 = x[0] + x[1] + x[2]

            # Constraint 1: Sustainability safeguard (e.g., pH balance or excess N)
            # Example: Total NPK shouldn't exceed a 'safe' threshold for soil health
            # g1 <= 0 is satisfied. Let's say max safe NPK is 300
            g1 = (x[0] + x[1] + x[2]) - 300

            out["F"] = [f1, f2]
            out["G"] = [g1]

        except Exception as e:
            logger.error(f"Optimization Eval Error: {e}")
            out["F"] = [0, 1000]
            out["G"] = [1]


def run_optimization(yield_model, encoders, crop_name, season, soil_data):
    """
    Runs NSGA-II to find the Pareto-optimal trade-offs between Yield and Cost.
    """
    problem = AgriOptimizationProblem(
        yield_model, encoders, crop_name, season, soil_data
    )

    algorithm = NSGA2(pop_size=40)
    termination = get_termination("n_gen", 50)

    res = minimize(
        problem, algorithm, termination, seed=1, save_history=True, verbose=False
    )

    if res.X is not None:
        # Sort results by yield descending
        # res.F contains [-yield, cost]
        sorted_indices = np.argsort(res.F[:, 0])
        trade_offs = []
        for idx in sorted_indices:
            trade_offs.append(
                {
                    "n": float(res.X[idx][0]),
                    "p": float(res.X[idx][1]),
                    "k": float(res.X[idx][2]),
                    "expected_yield": float(-res.F[idx][0]),
                    "resource_cost_index": float(res.F[idx][1]),
                }
            )
        return trade_offs
    return []
