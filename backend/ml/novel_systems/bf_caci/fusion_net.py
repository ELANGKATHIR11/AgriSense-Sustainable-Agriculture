import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Gated Attention Fusion Layer.

    Dynamically weights Optical vs Radar features based on data quality (clouds).
    If Optical is noisy, Radar weight increases.
    """

    def __init__(self, feature_dim=64):
        super(GatedFusion, self).__init__()
        # Attention Gate
        self.gate_fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),  # Output: 0 to 1 weighting
        )

    def forward(self, optical_feat, radar_feat):
        combined = torch.cat([optical_feat, radar_feat], dim=1)
        z = self.gate_fc(combined)

        # Weighted mixture
        # If z is 1, fully Optical. If 0, fully Radar.
        # Ideally, we want the network to learn this.
        fused = z * optical_feat + (1 - z) * radar_feat
        return fused


class BFCACINetwork(nn.Module):
    """
    Bayesian Fusion Engine for Confidence-Aware Crop Identification (BF-CACI).

    Inputs:
    - Optical Series (B, T, C_opt)
    - Radar Series (B, T, C_rad)

    Outputs:
    - Alpha Parameters (B, Classes) -> Defines Dirichlet Density
    """

    def __init__(self, num_classes=5):
        super(BFCACINetwork, self).__init__()

        # Encoders (LSTM/CNN)
        self.optical_encoder = nn.LSTM(10, 64, batch_first=True)  # 10 bands
        self.radar_encoder = nn.LSTM(2, 64, batch_first=True)  # 2 bands (VV, VH)

        # Fusion
        self.fusion = GatedFusion(feature_dim=64)

        # Evidential Output Head
        # Note: Activation is NOT Softmax. It is RELU (to ensure alpha > 0) + 1
        self.output_fc = nn.Linear(64, num_classes)

    def forward(self, opt_input, rad_input):
        # 1. Encode Streams
        _, (h_opt, _) = self.optical_encoder(opt_input)  # Get last hidden state
        _, (h_rad, _) = self.radar_encoder(rad_input)

        h_opt = h_opt.squeeze(0)  # (Batch, 64)
        h_rad = h_rad.squeeze(0)

        # 2. Fuse
        fused_feat = self.fusion(h_opt, h_rad)

        # 3. Generate Evidence
        logits = self.output_fc(fused_feat)

        # 4. Activation for Dirichlet Parameters
        # evidence >= 0
        evidence = F.relu(logits)
        # alpha = evidence + 1
        alpha = evidence + 1

        return alpha

    def predict(self, opt, rad):
        """
        Inference Method.
        Returns: Class Probabilities, Epistemic Uncertainty
        """
        alpha = self.forward(opt, rad)
        S = torch.sum(alpha, dim=1, keepdim=True)

        probs = alpha / S
        uncertainty = self.output_fc.out_features / S

        return probs, uncertainty
