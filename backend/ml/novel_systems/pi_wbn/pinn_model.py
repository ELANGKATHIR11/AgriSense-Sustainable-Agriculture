import torch
import torch.nn as nn


class PIWBNModel(nn.Module):
    """
    Physics-Informed Water Balance Network (PI-WBN).

    A hybrid neural architecture that predicts future soil moisture states
    based on historical weather and earth observation data.

    Architecture:
    1. LSTM Encoder: Processes time-series of [Precip, Temp, NDVI]
    2. State Predictor: MLP predicting S_{t+1} and ET_{t}

    Novelty:
    Designed to be trained with 'physics_loss.WaterBalanceLoss' to ensure
    outputs are hydrologically consistent.
    """

    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2):
        super(PIWBNModel, self).__init__()

        # Encoder for temporal context (Past 7 days weather + NDVI)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Readout layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # Outputs: [SoilMoisture_t1, Evapotranspiration_t]
        )

        # Hard constraint layer (Sigmoid to keep SM between 0 and 1)
        self.saturation_activation = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, features)
               features: [Precip, Temp, Humidity, NDVI, SoilMoisture_t0]

        Returns:
            soil_moisture_next: Predicted SM state
            et_pred: Predicted ET flux
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        # Prediction
        raw_out = self.fc(last_step)

        # Extract components
        # 1. Soil Moisture (Bounded 0-1)
        sm_pred = self.saturation_activation(raw_out[:, 0])

        # 2. ET (Non-negative)
        et_pred = torch.relu(raw_out[:, 1])

        return sm_pred, et_pred


# Example training loop snippet for Patent Disclosure
def train_step_example(model, optimizer, loss_fn, batch_data):
    """
    Demonstrates the Physics-Informed Training Step.
    """
    x_sequence, y_true_sm = batch_data

    # Inputs for Physics Calculation
    prev_sm = x_sequence[:, -1, 4]  # Assuming 5th feature is SM
    precip = x_sequence[:, -1, 0]  # Assuming 1st feature is Precip
    irrigation = torch.zeros_like(precip)  # Zero during training (observational)

    # Forward Pass
    optimizer.zero_grad()
    pred_sm, pred_et = model(x_sequence)

    # Calculate Physics-Informed Loss
    loss = loss_fn(
        predicted_sm_t1=pred_sm,
        true_sm_t1=y_true_sm,
        sm_t0=prev_sm,
        precip=precip,
        irrigation=irrigation,
        et_pred=pred_et,
    )

    loss.backward()
    optimizer.step()
    return loss.item()
