import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLoss(nn.Module):
    """
    Loss Function for Deep Evidential Learning (Classification).

    Patent Novelty:
    "A loss function minimizing the Bayes Risk of a Dirichlet distribution
    combined with a complexity-penalizing Kullback-Leibler divergence term
    to quantify epistemic uncertainty."

    References:
    Sensoy et al. "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(self, num_classes=10, annealing_step=10):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step

    def forward(self, alpha, y_true, epoch_num=1):
        """
        Args:
            alpha: Predicted Dirichlet parameters (Batch, Classes). Must be > 1.
            y_true: One-hot encoded ground truth labels (Batch, Classes).
            epoch_num: Current training epoch (for annealing KL weight).
        """
        # 1. Calc S (Total Strength) and Probabilities
        S = torch.sum(alpha, dim=1, keepdim=True)
        # Expected Probability: E[p] = alpha / S

        # 2. Equation 4 from Paper: Bayes Risk (Sum of Squares Error adaptation)
        # L_ace = sum( (y_ij - E[p_ij])^2 + Var(p_ij) )
        m = alpha / S

        A = torch.sum((y_true - m) ** 2, dim=1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

        loss_ace = A + B

        # 3. KL Divergence Regularizer (Eq 5)
        # Forces model to output uniform distribution (high uncertainty)
        # when evidence is zero (alphas close to 1).

        # Target uniform Dirichlet parameters: alpha_tilde = [1, 1, ... 1]
        # We want to minimize KL(Dir(alpha) || Dir(1))
        # BUT only for non-target classes.

        alpha_tilde = y_true + (1 - y_true) * alpha

        # Log Gamma function approximation/calls
        lgamma_alpha_tilde = torch.lgamma(alpha_tilde)
        lgamma_1 = torch.lgamma(torch.tensor(1.0))  # 0
        lgamma_S = torch.lgamma(S)
        lgamma_K = torch.lgamma(torch.tensor(float(self.num_classes)))

        # KL term
        kl = (
            lgamma_S
            - lgamma_K
            - torch.sum(lgamma_alpha_tilde, dim=1, keepdim=True)
            + torch.sum(
                (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S)),
                dim=1,
                keepdim=True,
            )
        )

        # Annealing coefficient (lambda)
        # Don't penalize uncertainty early in training.
        annealing_coef = min(1, epoch_num / self.annealing_step)

        total_loss = loss_ace + annealing_coef * kl
        return torch.mean(total_loss)

    def compute_uncertainty(self, alpha):
        """
        Returns metric for Epistemic Uncertainty.
        u = K / S
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        uncertainty = self.num_classes / S
        return uncertainty
