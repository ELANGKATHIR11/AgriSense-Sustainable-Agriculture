import torch
import torch.nn as nn
import copy


class SoilLSTM(nn.Module):
    """
    Local Soil Trajectory Model.
    Predicts Soil Health (OC, pH) 5 years into future.
    """

    def __init__(self, input_dim=5):
        super(SoilLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 32, batch_first=True)
        self.fc = nn.Linear(32, 2)  # Outcome: [OC, pH]

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class FederatedClient:
    """
    Edge Node simulation for Federated Learning.

    Patent Claim Element:
    "A distributed client node configured to compute local model updates
    based on private soil data and apply differential privacy noise
    before transmission to the aggregator."
    """

    def __init__(self, client_id, private_data):
        self.client_id = client_id
        self.data = private_data
        self.model = SoilLSTM()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def set_weights(self, global_weights):
        """Load global model parameters."""
        self.model.load_state_dict(global_weights)

    def train_locally(self, epochs=5, dp_sigma=0.01):
        """
        Train on private data and return 'Noisy Updates'.
        """
        self.model.train()
        for _ in range(epochs):
            x, y = self.data
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

        # Compute Gradients / Weight Differences
        # For simplicity in this demo, we return the new weights directly
        new_weights = copy.deepcopy(self.model.state_dict())

        # Apply Differential Privacy (Gaussian Noise)
        # Weight Perturbation mechanism
        noisy_weights = {}
        for key, val in new_weights.items():
            noise = torch.randn_like(val) * dp_sigma
            noisy_weights[key] = val + noise

        return noisy_weights
