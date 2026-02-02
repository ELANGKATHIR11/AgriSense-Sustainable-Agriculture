import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalWarpingLayer(nn.Module):
    """
    Stage II of PATA Network: Differentiable Temporal Warping.

    Patent Claim Element:
    "A temporal alignment layer configured to dynamically resample the input
    time-series vector based on a learned phenological inflection point."

    Mechanism:
    Uses a 1D version of Spatial Transformer Networks (STN).
    Constructs a sampling grid based on the predicted 'sowing_offset' and
    interpolates the input features onto this canonical grid.
    """

    def __init__(self, seq_len=52):
        super(TemporalWarpingLayer, self).__init__()
        self.seq_len = seq_len
        # Canonical 'Biological Time' grid (0.0 to 1.0)
        self.register_buffer("base_grid", torch.linspace(-1, 1, seq_len))

    def forward(self, x, sowing_offset_norm):
        """
        Args:
            x: Input Feature Map (Batch, Channels, Time)
            sowing_offset_norm: Predicted shift [-1, 1] relative to window center.
                                Output from PhenologyAnchorDetector.

        Returns:
            aligned_x: Time-series warped to start at Sowing Date.
        """
        batch_size = x.size(0)

        # 1. Create Sampling Grid
        # The grid defines *where* in the input 'x' we should sample from
        # to populate the output canonical slots.
        # If sowing was late (positive offset), we need to look 'later' in input.

        # Expand base grid to batch size
        grid = self.base_grid.unsqueeze(0).repeat(batch_size, 1)  # (B, T)

        # Add the offset (broadcasting)
        # grid[b, t] = base_time[t] + shift[b]
        sampling_locations = grid + sowing_offset_norm

        # Reshape for grid_sample: (Batch, 1, Time, 1) - treating 1D as thin 2D
        # grid_sample expects coordinates in range [-1, 1]
        sampling_locations = sampling_locations.view(batch_size, 1, -1, 1)
        # We need (x, y) coordinates. Since it's 1D, y is dummy (0).
        dummy_y = torch.zeros_like(sampling_locations)
        grid_2d = torch.cat([sampling_locations, dummy_y], dim=-1)  # (B, 1, T, 2)

        # 2. Resample
        # Input needs to be 4D (Batch, C, Height=1, Width=Time)
        x_4d = x.unsqueeze(2)

        # Bilinear interpolation extracts values at the calculated time points
        aligned_x = F.grid_sample(x_4d, grid_2d, align_corners=True)

        # Return to 3D (Batch, C, Time)
        return aligned_x.squeeze(2)


class PATANetwork(nn.Module):
    """
    Full End-to-End PATA System.
    Combines Anchor Detection + Warping + Prediction.
    """

    def __init__(self):
        super(PATANetwork, self).__init__()
        import phenology_anchor  # lazy import to avoid circular dependency if any

        self.anchor_detector = phenology_anchor.PhenologyAnchorDetector()
        self.warping_layer = TemporalWarpingLayer()
        self.yield_predictor = nn.LSTM(1, 32, batch_first=True)  # Simple regressor
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # 1. Detect Sowing Shift
        shift, _ = self.anchor_detector(x)

        # 2. Align Time Series
        # Extract last layer features or raw input? Using raw input for demo.
        aligned_input = self.warping_layer(x, shift)

        # 3. Predict Yield from ALIGNED data
        # Transpose for LSTM (Batch, Channel, Time) -> (Batch, Time, Channel)
        aligned_seq = aligned_input.permute(0, 2, 1)
        lstm_out, _ = self.yield_predictor(aligned_seq)
        prediction = self.fc(lstm_out[:, -1, :])

        return prediction, shift, aligned_input
