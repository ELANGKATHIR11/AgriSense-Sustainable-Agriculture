# Federated Soil-Health Trajectory Forecaster (F-SHTF)

## System Overview
**F-SHTF** is a privacy-preserving AI system for forecasting long-term soil health. It allows training a Global Soil Model on private farmer data without that data ever leaving the farmer's device. It solves the "Data Privacy vs Model Accuracy" dilemma in Agri-Tech.

## Components
1.  **`client_node.py`**:
    *   **Class**: `FederatedClient`
    *   **Function**: Simulates an Edge Device (e.g., Mobile Phone). Trains a local LSTM on private soil logs.
    *   **Privacy**: Adds Gaussian Noise (Differential Privacy) to gradients before export.

2.  **`aggregator.py`**:
    *   **Class**: `FederatedAggregator`
    *   **Function**: Implements the **FedAvg** algorithm. Receives privacy-noised weights from multiple clients and averages them to update the Global Model.

## Usage
```python
from aggregator import FederatedAggregator
# Run the simulation script
# python aggregator.py
```

## Patent Claim Novelty
"A distributed soil-modeling system utilizing federated averaging algorithms to aggregate soil-chemistry degradation patterns across disparate, privacy-shielded nodes without centralizing raw soil test reports."
