from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def covariance_decorrelation_loss(x: torch.Tensor) -> torch.Tensor:
    """
    Computes Covariance Decorrelation Loss from VICReg (Bardes et al., ICLR 2022).
    Forces off-diagonal terms of the covariance matrix towards zero to prevent
    dimensional collapse and maximize representation capacity / Effective Rank.

    Args:
        x: Tensor of shape (N, D), where N is number of samples/tokens, D is feature dimension.

    Returns:
        Scalar loss = (1 / D) * sum_{i != j} C_{i, j}^2
    """
    n, d = x.shape
    if n < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    x_centered = x.float() - x.float().mean(dim=0, keepdim=True)
    cov = (x_centered.T @ x_centered) / (n - 1)
    
    # Sum of off-diagonal elements squared: ||C||_F^2 - ||diag(C)||_2^2
    frobenius_sq = torch.linalg.matrix_norm(cov, ord="fro").square()
    diag_sq = cov.diag().square().sum()
    off_diag_sq = (frobenius_sq - diag_sq).clamp_min(0.0)
    
    return off_diag_sq / d


def variance_regularization_loss(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Computes Variance Regularization Loss from VICReg (Bardes et al., ICLR 2022).
    Prevents complete representation collapse by penalizing standard deviations below target_std.

    Args:
        x: Tensor of shape (N, D).
        target_std: Desired minimum standard deviation floor (default: 1.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar loss = (1 / D) * sum_j max(0, target_std - std(x_j))
    """
    n, d = x.shape
    if n < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    std = torch.sqrt(x.float().var(dim=0, unbiased=True) + eps)
    return F.relu(target_std - std).mean()


class RobustJEPALoss(nn.Module):
    """
    Enhanced JEPA Loss combining Smooth L1 latent prediction with VICReg variance
    and covariance decorrelation to eliminate dimensional collapse while maintaining
    strong invariance to natural corruptions.
    """
    def __init__(
        self,
        variance_weight: float = 0.1,
        variance_floor: float = 0.5,
        covariance_weight: float = 0.05,
    ):
        super().__init__()
        self.variance_weight = variance_weight
        self.variance_floor = variance_floor
        self.covariance_weight = covariance_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            prediction: (B, N_masked, D) predicted latent tokens.
            target: (B, N_masked, D) clean target latent tokens.
            context: (B, N_visible, D) online context representations.
        """
        # 1. Primary latent prediction loss
        pred_loss = F.smooth_l1_loss(prediction.float(), target.float())

        # Flatten context tokens across batch: (B * N_vis, D)
        context_flat = context.float().flatten(0, 1)

        # 2. Variance loss: ensures each dimension has sufficient spread
        var_loss = variance_regularization_loss(context_flat, target_std=self.variance_floor)

        # 3. Covariance decorrelation loss: forces dimensions to be orthogonal
        cov_loss = covariance_decorrelation_loss(context_flat)

        total_loss = pred_loss + (self.variance_weight * var_loss) + (self.covariance_weight * cov_loss)

        metrics = {
            "prediction_loss": pred_loss.detach(),
            "variance_loss": var_loss.detach(),
            "covariance_loss": cov_loss.detach(),
            "context_std_mean": context_flat.std(dim=0).mean().detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss, metrics
