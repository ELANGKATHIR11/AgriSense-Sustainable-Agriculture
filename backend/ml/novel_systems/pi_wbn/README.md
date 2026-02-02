# Physics-Informed Water Balance Network (PI-WBN)

## System Overview
The **PI-WBN** is a novel deep learning architecture designed for sustainable irrigation optimization. It differentiates itself from standard ML models by enforcing the **Hydrological Mass Balance Equation** ($ \Delta S = P + I - ET - R - D $) directly within the training loss function.

## Components
1.  **`physics_loss.py`**:
    *   **Class**: `WaterBalanceLoss`
    *   **Function**: Calculates the residual error between the neural network's state prediction and the state mandated by physical laws.
    *   **Patent Novelty**: "A training method dependent on a physics-residual loss term."

2.  **`pinn_model.py`**:
    *   **Class**: `PIWBNModel`
    *   **Function**: An LSTM-based encoder that predicts Soil Moisture ($S$) and Evapotranspiration ($ET$) fluxes.
    *   **Constraints**: Uses `Sigmoid` and `ReLU` activations to enforce hard physical constraints (non-negativity, saturation limits).

## Training Procedure (Claimable Method)
1.  **Input**: Time-series of weather ($P, T$) and Earth Observation data ($NDVI$).
2.  **Forward Pass**: Predict next state $S_{t+1}$ and flux $ET_t$.
3.  **Physics Check**: Calculate `Loss_Physics` using the customized `WaterBalanceLoss`.
4.  **Optimization**: Backpropagate `Loss_Total = Loss_Data + λ * Loss_Physics`.

## Usage
```python
import torch
from pinn_model import PIWBNModel
from physics_loss import WaterBalanceLoss

model = PIWBNModel()
loss_fn = WaterBalanceLoss(weight_physics=0.5)
# ... training loop ...
```
