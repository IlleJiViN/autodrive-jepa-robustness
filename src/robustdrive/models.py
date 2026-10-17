from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F


def select_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tokens.gather(1, indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))


class VisionEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads):
        super().__init__()
        self.image_size = tuple(image_size)
        self.grid = (image_size[0] // patch_size, image_size[1] // patch_size)
        self.patch = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.zeros(1, self.grid[0] * self.grid[1], dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        nn.init.trunc_normal_(self.position, std=0.02)
        # TransformerEncoder clones one layer: initialize each block independently.
        for block in self.blocks.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, images: torch.Tensor, visible: torch.Tensor | None = None):
        if tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(f"Expected image size {self.image_size}, got {tuple(images.shape[-2:])}")
        tokens = self.patch((images - self.mean) / self.std).flatten(2).transpose(1, 2) + self.position
        # Select patches BEFORE attention: target pixels cannot enter the context encoder.
        if visible is not None:
            tokens = select_tokens(tokens, visible)
        return self.blocks(tokens)


class MaskedPredictor(nn.Module):
    def __init__(self, count, encoder_dim, dim, depth, heads):
        super().__init__()
        self.projection = nn.Linear(encoder_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, count, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.0, activation="gelu",
                                          batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, norm=nn.LayerNorm(dim), enable_nested_tensor=False)
        self.output = nn.Linear(dim, encoder_dim)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)
        for block in self.blocks.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, context, visible, target_indices):
        encoded = self.projection(context)
        # Autocast makes projection BF16 while learned parameters remain FP32.
        tokens = self.mask_token.to(dtype=encoded.dtype).expand(context.shape[0], self.position.shape[1], -1).clone()
        tokens.scatter_(1, visible.unsqueeze(-1).expand_as(encoded), encoded)
        prediction = self.output(self.blocks(tokens + self.position.to(dtype=encoded.dtype)))
        return select_tokens(prediction, target_indices)


class CorruptionJEPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        model = config["model"]
        self.online = VisionEncoder(config["data"]["image_size"], model["patch_size"], model["dim"],
                                    model["depth"], model["heads"])
        self.target = copy.deepcopy(self.online).requires_grad_(False)
        self.token_count = self.online.grid[0] * self.online.grid[1]
        self.visible_count = max(1, min(self.token_count - 1, int(self.token_count * (1 - model["mask_ratio"]))))
        self.predictor = MaskedPredictor(self.token_count, model["dim"], model["predictor_dim"],
                                         model["predictor_depth"], model["predictor_heads"])
        self.variance_floor = config["training"]["variance_floor"]
        self.variance_weight = config["training"]["variance_weight"]
        self.covariance_weight = config["training"].get("covariance_weight", 0.0)
        self.target.eval()

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    def forward(self, clean, view, generator=None):
        permutation = torch.rand(clean.shape[0], self.token_count, device=clean.device, generator=generator).argsort(dim=1)
        visible, masked = permutation[:, :self.visible_count], permutation[:, self.visible_count:]
        context = self.online(view, visible)
        prediction = self.predictor(context, visible, masked)
        with torch.no_grad():
            target = select_tokens(self.target(clean), masked).float()
            target = F.layer_norm(target, (target.shape[-1],))
        prediction_loss = F.smooth_l1_loss(prediction.float(), target)
        context_flat = context.float().flatten(0, 1)
        context_std = context_flat.std(dim=0, unbiased=False)
        variance_loss = F.relu(self.variance_floor - context_std).mean()
        
        loss = prediction_loss + self.variance_weight * variance_loss
        cov_loss = torch.tensor(0.0, device=clean.device)
        if self.covariance_weight > 0.0:
            from .losses import covariance_decorrelation_loss
            cov_loss = covariance_decorrelation_loss(context_flat)
            loss = loss + self.covariance_weight * cov_loss

        metrics = {"prediction_loss": prediction_loss.detach(), "variance_loss": variance_loss.detach(),
                   "covariance_loss": cov_loss.detach(),
                   "context_std": context_std.mean().detach(),
                   "target_std": target.flatten(0, 1).std(dim=0, unbiased=False).mean().detach()}
        return loss, metrics

    @torch.no_grad()
    def update_target(self, momentum):
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)
        for target, online in zip(self.target.buffers(), self.online.buffers()):
            target.copy_(online)


def ema_momentum(start, step, total):
    return 1 - (1 - start) * (1 + math.cos(math.pi * min(step / max(total, 1), 1))) / 2


class YOLOFeatureAdapter(nn.Module):
    """P3/P4/P5 feature contract only. A trained YOLO neck/head is Phase 2 work."""
    def __init__(self, dim=384, channels=(128, 256, 512)):
        super().__init__()
        self.projections = nn.ModuleList([nn.Conv2d(dim, c, 1) for c in channels])

    def forward(self, tokens, grid):
        batch, count, dim = tokens.shape
        if count != grid[0] * grid[1]:
            raise ValueError("Token count does not match grid")
        feature = tokens.transpose(1, 2).reshape(batch, dim, *grid)
        maps = [F.interpolate(feature, scale_factor=2, mode="bilinear", align_corners=False),
                feature, F.avg_pool2d(feature, 2)]
        return [projection(value) for projection, value in zip(self.projections, maps)]
