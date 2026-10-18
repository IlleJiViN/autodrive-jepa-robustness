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


def sample_2d_block_mask(
    batch_size: int,
    grid: tuple[int, int] = (24, 40),
    visible_count: int = 480,
    num_blocks: int = 4,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples 2D contiguous rectangular block masks instead of point-wise random masking.
    Generates contiguous target regions so the network cannot rely on trivial local smoothing.
    Returns (visible_indices, masked_indices) with exact shapes (B, visible_count) and (B, masked_count).
    """
    H, W = grid
    total = H * W
    masked_count = total - visible_count

    y = torch.arange(H, device=device).float().unsqueeze(1).expand(H, W)
    x = torch.arange(W, device=device).float().unsqueeze(0).expand(H, W)

    heatmaps = torch.zeros(batch_size, H, W, device=device)
    for b in range(batch_size):
        for _ in range(num_blocks):
            cy = torch.randint(0, H, (1,), device=device, generator=generator).float()
            cx = torch.randint(0, W, (1,), device=device, generator=generator).float()
            sigma_y = torch.randint(2, 6, (1,), device=device, generator=generator).float()
            sigma_x = torch.randint(3, 9, (1,), device=device, generator=generator).float()
            dist_sq = ((y - cy) / sigma_y).pow(2) + ((x - cx) / sigma_x).pow(2)
            heatmaps[b] += torch.exp(-0.5 * dist_sq)

    flat_heat = heatmaps.view(batch_size, total)
    flat_heat = flat_heat + torch.rand(batch_size, total, device=device, generator=generator) * 1e-4
    perm = flat_heat.argsort(dim=1, descending=True)
    masked = perm[:, :masked_count]
    visible = perm[:, masked_count:]
    return visible, masked


def uniformity_loss(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Wang & Isola (ICML 2020) Uniformity Loss for batch instance representations.
    Forces representations to spread uniformly on the hypersphere, preventing collapse.
    """
    if len(z) < 2:
        return torch.tensor(0.0, device=z.device)
    z_norm = F.normalize(z.float(), dim=-1)
    sq_pdist = torch.pdist(z_norm, p=2).pow(2)
    val = sq_pdist.mul(-t).exp().mean().clamp_min(1e-12)
    return val.log()


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
        variance_weight: float = 1.0,
        covariance_weight: float = 0.5,
        uniformity_weight: float = 0.5,
        masking_strategy: str = "block",
        num_mask_blocks: int = 4,
        freeze_lower_blocks: int = 4,
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.mask_ratio = mask_ratio
        self.variance_floor = variance_floor
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.uniformity_weight = uniformity_weight
        self.masking_strategy = masking_strategy
        self.num_mask_blocks = num_mask_blocks
        self.freeze_lower_blocks = freeze_lower_blocks

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

        # Freeze lower blocks to preserve low-level edge/corner detectors
        if freeze_lower_blocks > 0:
            self.online.patch_embed.requires_grad_(False)
            for blk in self.online.blocks[:freeze_lower_blocks]:
                blk.requires_grad_(False)

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
        if self.masking_strategy == "block":
            visible, masked = sample_2d_block_mask(
                batch_size=batch_size,
                grid=self.grid,
                visible_count=self.visible_count,
                num_blocks=self.num_mask_blocks,
                device=clean.device,
                generator=generator,
            )
        else:
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

        # 4. Losses: Prediction + Instance/Token VICReg Regularization + Wang & Isola Uniformity
        pred_norm = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        prediction_loss = F.smooth_l1_loss(pred_norm, target)

        # Instance-level pooled representation across batch (B, D) to prevent cross-image collapse
        z_instance = context.float().mean(dim=1)  # (B, D)
        y_instance = F.layer_norm(z_instance, (z_instance.shape[-1],))
        instance_std = y_instance.std(dim=0, unbiased=False)
        instance_variance_loss = F.relu(self.variance_floor - instance_std).mean()

        # Token-level representation across patches (B * N_visible, D)
        context_flat = context.float().flatten(0, 1)  # (B * N_visible, D)
        token_std = context_flat.std(dim=0, unbiased=False)
        token_variance_loss = F.relu(self.variance_floor - token_std).mean()

        variance_loss = 0.6 * instance_variance_loss + 0.4 * token_variance_loss

        cov_loss = torch.tensor(0.0, device=clean.device)
        if self.covariance_weight > 0.0 and batch_size > 1:
            cov_loss = covariance_decorrelation_loss(y_instance)

        unif_loss = torch.tensor(0.0, device=clean.device)
        if self.uniformity_weight > 0.0 and batch_size > 1:
            unif_loss = uniformity_loss(z_instance)

        total_loss = (
            prediction_loss
            + (self.variance_weight * variance_loss)
            + (self.covariance_weight * cov_loss)
            + (self.uniformity_weight * unif_loss)
        )

        # Inter-image cosine similarity across batch (diagnostic for instance collapse)
        inter_cos = torch.tensor(0.0, device=clean.device)
        if batch_size > 1:
            norm_inst = F.normalize(z_instance, dim=-1)
            sim_mat = norm_inst @ norm_inst.T
            mask_offdiag = ~torch.eye(batch_size, dtype=torch.bool, device=clean.device)
            inter_cos = sim_mat[mask_offdiag].mean()

        metrics = {
            "prediction_loss": prediction_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "instance_var_loss": instance_variance_loss.detach(),
            "token_var_loss": token_variance_loss.detach(),
            "covariance_loss": cov_loss.detach(),
            "uniformity_loss": unif_loss.detach(),
            "total_loss": total_loss.detach(),
            "instance_std": instance_std.mean().detach(),
            "token_std": token_std.mean().detach(),
            "inter_image_cosine": inter_cos.detach(),
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
