import torch
import torch.nn as nn


class PhenologyAnchorDetector(nn.Module):
    """
    Stage I of PATA Network: Phenological Anchor Detection.

    Novelty:
    Instead of predicting yield directly, this sub-network predicts the
    'Biological Clock' of the crop from a shifted time-series.

    Inputs:
    - Raw NDVI Time Series (52 weeks, potentially shifted/misaligned)

    Outputs:
    - Sowing Date Offset (scalar regression)
    - Phenological State Vector (classification logits)
    """

    def __init__(self, seq_len=52, input_channels=1):
        super(PhenologyAnchorDetector, self).__init__()

        # 1D CNN Feature Extractor
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 52 -> 26
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 26 -> 13
        )

        self.flatten_dim = 128 * 13

        # Head 1: Anchor Point Regressor (Detects 'Time Zero' / Sowing Week)
        self.anchor_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Output: Scalar 'delta_t' (Sowing Shift)
        )

        # Head 2: Current Stage Classifier (Vegetative, Reproductive, Ripening)
        self.stage_head = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # Classes: [Pre-Sow, Veg, Repro, Mature]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor (Batch, Channels, Time) -> e.g., (32, 1, 52)
        """
        features = self.encoder(x)
        flat = features.view(features.size(0), -1)

        sowing_offset = self.anchor_head(flat)
        stage_logits = self.stage_head(flat)

        return sowing_offset, stage_logits
