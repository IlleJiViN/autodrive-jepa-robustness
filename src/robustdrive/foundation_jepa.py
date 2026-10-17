from __future__ import annotations

import copy
import math
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import covariance_decorrelation_loss


def select_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gathers tokens at specified indices: tokens (B, N, D), indices (B, K) -> (B, K, D)."""
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


def ema_momentum(start: float = 0.996, step: int = 0, total: int = 1000) -> float:
    """Cosine momentum schedule from start (e.g. 0.996) to 1.0."""
    return 1.0 - (1.0 - start) * (1.0 + math.cos(math.pi * min(step / max(total, 1), 1.0))) / 2.0


class MaskedPredictor(nn.Module):
    """
    Lightweight Transformer Predictor that maps visible context tokens + learned mask tokens
    to the target representations at masked patch positions.
    """
    def __init__(
        self,
        count: int = 960,
        encoder_dim: int = 384,
        dim: int = 192,
        depth: int = 4,
        heads: int = 6,
    ):
        super().__init__()
        self.count = count
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.projection = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, count, dim))

        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer,
            depth,
            norm=nn.LayerNorm(dim),
            enable_nested_tensor=False,
        )
        self.output = nn.Linear(dim, encoder_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        for block in self.blocks.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(
        self,
        context: torch.Tensor,
        visible: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context: (B, N_visible, encoder_dim) context tokens from online encoder.
            visible: (B, N_visible) indices of visible tokens.
            target_indices: (B, N_masked) indices of masked target tokens.
        Returns:
            (B, N_masked, encoder_dim) predicted representations at target positions.
        """
        encoded = self.projection(context)
        # Expand mask tokens across all positions: (B, count, dim)
        tokens = self.mask_token.to(dtype=encoded.dtype).expand(context.shape[0], self.position.shape[1], -1).clone()
        tokens.scatter_(1, visible.unsqueeze(-1).expand_as(encoded), encoded)
        prediction = self.output(self.blocks(tokens + self.position.to(dtype=encoded.dtype)))
        return select_tokens(prediction, target_indices)


class FoundationCorruptionJEPA(nn.Module):
    """
    Foundation Natural-Noise Image-based Joint-Embedding Predictive Architecture (I-JEPA).
    
    Mathematical Formulation:
    - Online Student Context Encoder (self.online): Pretrained ViT-S/16 backbone.
      Extracts patch + pos embeddings and strictly selects ONLY visible tokens BEFORE
      passing into Transformer blocks. Target pixels NEVER enter the student context encoder!
    - Target Teacher Encoder (self.target): EMA copy of self.online (requires_grad=False).
      Encodes clean image under torch.no_grad() and extracts normalized masked patch representations.
    - Masked Predictor (self.predictor): Predicts clean target representations from corrupted context.
    - VICReg Variance-Covariance Regularization: Guarantees non-collapsing, high Effective Rank representations.
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_384",
        img_size: tuple[int, int] = (384, 640),
        pretrained: bool = True,
        mask_ratio: float = 0.50,
        predictor_dim: int = 192,
        predictor_depth: int = 4,
        predictor_heads: int = 6,
        variance_floor: float = 1.0,
        variance_weight: float = 0.1,
        covariance_weight: float = 0.05,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.variance_floor = variance_floor
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight

        self.grid = (img_size[0] // 16, img_size[1] // 16)  # (24, 40)
        self.token_count = self.grid[0] * self.grid[1]       # 960 tokens
        self.visible_count = max(1, min(self.token_count - 1, int(self.token_count * (1.0 - mask_ratio))))
        self.masked_count = self.token_count - self.visible_count

        # Student Online Context Encoder
        self.online = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.embed_dim = self.online.embed_dim

        # Teacher Target Encoder (EMA)
        self.target = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()

        # Masked Latent Predictor
        self.predictor = MaskedPredictor(
            count=self.token_count,
            encoder_dim=self.embed_dim,
            dim=predictor_dim,
            depth=predictor_depth,
            heads=predictor_heads,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        # Teacher target encoder always stays in eval mode
        self.target.eval()
        return self

    def forward_encoder(
        self,
        backbone: nn.Module,
        images: torch.Tensor,
        visible: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass through ViT backbone with strict pre-attention token selection.
        If visible is provided, ONLY visible tokens are processed by backbone.blocks!
        """
        x = backbone.patch_embed(images)
        x = backbone._pos_embed(x)
        if x.shape[1] == 1 + self.token_count:
            x = x[:, 1:, :]  # Strip class token
        if hasattr(backbone, "norm_pre") and backbone.norm_pre is not None:
            x = backbone.norm_pre(x)

        # STRICT CONTEXT MASKING: select visible patches BEFORE attention blocks
        if visible is not None:
            x = select_tokens(x, visible)

        for blk in backbone.blocks:
            x = blk(x)
        x = backbone.norm(x)
        return x

    def forward(
        self,
        clean: torch.Tensor,
        view: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            clean: (B, 3, H, W) clean ground-truth images for teacher target.
            view: (B, 3, H, W) naturally corrupted images for student context.
        Returns:
            loss: scalar total JEPA loss.
            metrics: dict of detached scalar metrics.
        """
        batch_size = clean.shape[0]
        permutation = torch.rand(batch_size, self.token_count, device=clean.device, generator=generator).argsort(dim=1)
        visible, masked = permutation[:, :self.visible_count], permutation[:, self.visible_count:]

        # 1. Student Context Encoder: processes ONLY visible tokens
        context = self.forward_encoder(self.online, view, visible=visible)  # (B, N_visible, D)

        # 2. Masked Predictor: predicts latent representations at masked target positions
        prediction = self.predictor(context, visible, masked)  # (B, N_masked, D)

        # 3. Teacher Target Encoder: processes full clean image under no_grad
        with torch.no_grad():
            target_tokens = self.forward_encoder(self.target, clean, visible=None)  # (B, 960, D)
            target = select_tokens(target_tokens, masked).float()
            target = F.layer_norm(target, (target.shape[-1],))

        # 4. Losses: Prediction + VICReg Variance Regularization + Covariance Decorrelation
        prediction_loss = F.smooth_l1_loss(prediction.float(), target)

        context_flat = context.float().flatten(0, 1)  # (B * N_visible, D)
        context_std = context_flat.std(dim=0, unbiased=False)
        variance_loss = F.relu(self.variance_floor - context_std).mean()

        cov_loss = torch.tensor(0.0, device=clean.device)
        if self.covariance_weight > 0.0:
            cov_loss = covariance_decorrelation_loss(context_flat)

        total_loss = prediction_loss + (self.variance_weight * variance_loss) + (self.covariance_weight * cov_loss)

        metrics = {
            "prediction_loss": prediction_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "covariance_loss": cov_loss.detach(),
            "total_loss": total_loss.detach(),
            "context_std": context_std.mean().detach(),
            "target_std": target.flatten(0, 1).std(dim=0, unbiased=False).mean().detach(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def update_target(self, momentum: float):
        """EMA target encoder update."""
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(self.target.buffers(), self.online.buffers()):
            target.copy_(online)

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extracts full unmasked token representations (B, 960, 384) from online encoder."""
        return self.forward_encoder(self.online, images, visible=None)

    def get_student_encoder(self) -> nn.Module:
        """Returns the trained student ViT backbone for detector transfer."""
        return self.online
