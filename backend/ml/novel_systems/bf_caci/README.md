# Bayesian Fusion Engine (BF-CACI)

## System Overview
The **BF-CACI System** provides confidence-aware crop classification. Unlike standard models that force a decision even on cloudy days, BF-CACI quantifies "Epistemic Uncertainty" (I don't know) vs "Aleatoric Uncertainty" (Data noise). It achieves this by fusing Optical and Radar data using a **Dirichlet Distribution Output Layer**.

## Components
1.  **`evidential_loss.py`**:
    *   **Class**: `EvidentialLoss`
    *   **Function**: Minimizes the Bayes Risk of the predicted Dirichlet distribution and penalizes high-confidence errors using KL-Divergence.
    *   **Patent Novelty**: "Deep Evidential Learning loss function applied to multi-modal agricultural sensor fusion."

2.  **`fusion_net.py`**:
    *   **Class**: `BFCACINetwork`
    *   **Architecture**: Parallel LSTMs for Optical/Radar -> Gated Attention Fusion -> ReLU Output.
    *   **Output**: Dirichlet parameters $\alpha$, where $\alpha_k = evidence_k + 1$.

## Usage
```python
import torch
from fusion_net import BFCACINetwork
from evidential_loss import EvidentialLoss

# Inputs
opt_data = torch.randn(16, 12, 10) # Batch, Time, Bands
rad_data = torch.randn(16, 12, 2)

model = BFCACINetwork(num_classes=5)
# Forward pass returns 'alpha' parameters, not probabilities
alpha = model(opt_data, rad_data)

# Compute Uncertainty
S = torch.sum(alpha, dim=1, keepdim=True)
uncertainty = 5 / S # K / S
```
