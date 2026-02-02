import torch
import torch.nn as nn


class WaterBalanceLoss(nn.Module):
    """
    Physics-Informed Loss Function for Water Balance Network.

    Enforces the Hydrological Mass Balance Equation:
    ΔS = P + I - ET - R - D

    Where:
    ΔS = Change in Soil Moisture Storage
    P  = Precipitation (Input)
    I  = Irrigation (Control Variable/Output)
    ET = Evapotranspiration (Predicted)
    R  = Runoff (Modeled function of S)
    D  = Drainage (Modeled function of S)

    Patent Claim Element:
    "A loss function comprising a data-fidelity term and a physics-residual term,
    wherein the physics-residual term minimizes the violation of the mass balance equation."
    """

    def __init__(self, weight_physics=1.0, soil_saturation_point=0.45):
        super(WaterBalanceLoss, self).__init__()
        self.lambda_phys = weight_physics
        self.n_sat = soil_saturation_point  # Porosity
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted_sm_t1, true_sm_t1, sm_t0, precip, irrigation, et_pred):
        """
        Calculate total loss = MSE(Data) + λ * MSE(Physics_Residual)

        Args:
            predicted_sm_t1: Model predicted soil moisture at t+1
            true_sm_t1     : Observed soil moisture at t+1 (from data)
            sm_t0          : Soil moisture at t (input state)
            precip         : Rainfall amount (input)
            irrigation     : Applied water (input)
            et_pred        : Predicted Evapotranspiration
        """

        # 1. Data Fidelity Loss (Standard ML)
        loss_data = self.mse_loss(predicted_sm_t1, true_sm_t1)

        # 2. Physics Residual Calculation
        # Approximate Runoff (SCS Curve Number method simplified)
        # R = (P - 0.2S)^2 / (P + 0.8S) if P > 0.2S else 0
        # For differentiable approx, we use Relu:
        potential_retention = (1 - sm_t0) * 100  # simplified
        runoff = torch.relu(precip - 0.2 * potential_retention) ** 2 / (
            precip + 0.8 * potential_retention + 1e-6
        )

        # Approximate Drainage (Darcy's Law exponential decay)
        # D = K_sat * (S/n_sat)^(3+2/lambda)
        # Simplified: D = 0.1 * exp(10 * (sm - saturation)) if sm is high
        drainage = torch.relu(sm_t0 - self.n_sat) * 1.0  # Simple overflow drainage

        # The Physical Law: S_{t+1} should be S_t + Inflows - Outflows
        phys_sm_t1 = sm_t0 + precip + irrigation - et_pred - runoff - drainage

        # Physics Residual (Difference between NN prediction and Physical Law)
        # We want the NN to learn dynamics that adhere to this balance.
        loss_physics = self.mse_loss(predicted_sm_t1, phys_sm_t1)

        # Total Loss
        return loss_data + self.lambda_phys * loss_physics
