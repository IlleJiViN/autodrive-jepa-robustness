"""
Foundation ViT-based Robust Object Detector.
Leverages pretrained Vision Transformers (e.g. timm vit_small_patch16_384)
with multi-scale Feature Pyramid Adapter (P3, P4, P5) and Anchor-Free Detection Heads.
Eliminates dimensional collapse and enables practical mAP on driving & CCTV domains.
"""
from __future__ import annotations

import math
from pathlib import Path
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import CLASS_NAMES
from .models import YOLOFeatureAdapter

SLOTS = 3
VALUES = 5 + len(CLASS_NAMES)  # objectness, class logits (9), centre offsets (2), log width/height (2)
STRIDES = (8, 16, 32)
LEVEL_NAMES = ("P3", "P4", "P5")


class ScaleHead(nn.Module):
    """Detection prediction head for one scale level."""
    def __init__(self, in_channels: int, num_classes: int = len(CLASS_NAMES)):
        super().__init__()
        self.num_classes = num_classes
        self.values = 5 + num_classes
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, SLOTS * self.values, kernel_size=1),
        )
        with torch.no_grad():
            bias = self.layers[-1].bias.view(SLOTS, self.values)
            bias[:, 0].fill_(-5.0)  # Prior bias for objectness

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = feature.shape
        out = self.layers(feature)
        # Reshape to (B, SLOTS, H, W, VALUES)
        return out.view(batch, SLOTS, self.values, height, width).permute(0, 1, 3, 4, 2).contiguous()


class FoundationViTDetector(nn.Module):
    """
    End-to-End Object Detector with Foundation ViT Backbone.
    Input image size: (384, 640). Grid: 24 x 40 = 960 patches of size 16x16.
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_384",
        pretrained: bool = False,
        freeze_backbone_blocks: int = 0,
        num_classes: int = len(CLASS_NAMES),
        img_size: tuple[int, int] = (384, 640),
    ):
        super().__init__()
        self.backbone_name = model_name
        self.img_size = img_size
        self.grid = (img_size[0] // 16, img_size[1] // 16)  # 24 x 40
        # Load timm ViT backbone with custom rectangular resolution
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.embed_dim = self.backbone.embed_dim  # 384 for ViT-S

        # Feature Adapter converting ViT tokens into P3 (1/8), P4 (1/16), P5 (1/32)
        self.adapter = YOLOFeatureAdapter(dim=self.embed_dim, channels=(128, 256, 512))
        
        # Scale-specific detection prediction heads
        self.heads = nn.ModuleList([
            ScaleHead(128, num_classes),
            ScaleHead(256, num_classes),
            ScaleHead(512, num_classes),
        ])

        if freeze_backbone_blocks > 0:
            self._freeze_early_blocks(freeze_backbone_blocks)

    def _freeze_early_blocks(self, num_blocks: int):
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, "blocks"):
            for i in range(min(num_blocks, len(self.backbone.blocks))):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = False

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial patch tokens from ViT backbone (excludes CLS token)."""
        feats = self.backbone.forward_features(x)
        # timm ViT forward_features returns either (B, 1 + N, D) or dict with tokens
        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens", feats.get("x"))
        elif feats.ndim == 3:
            # Check if CLS token is at index 0
            if feats.shape[1] == 1 + self.grid[0] * self.grid[1]:
                tokens = feats[:, 1:, :]
            else:
                tokens = feats
        else:
            raise ValueError(f"Unexpected feature shape from backbone: {feats.shape}")
        return tokens

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass producing multi-scale detection outputs:
        Returns:
            [P3_out (B, 3, 48, 80, V), P4_out (B, 3, 24, 40, V), P5_out (B, 3, 12, 20, V)]
        """
        tokens = self.forward_backbone(x)
        features = self.adapter(tokens, grid=self.grid)
        return [head(f) for head, f in zip(self.heads, features)]


def compute_detection_loss(
    predictions: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    negative_weight: float = 0.025,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes Anchor-Free detection loss across P3, P4, P5 scales:
    - Positive-aware Objectness BCE loss (balanced negative sampling)
    - Category CrossEntropy loss on positive slots
    - Bounding Box regression (Centre Sigmoid Offset + Log-size Smooth L1)
    """
    device = predictions[0].device
    total_obj = torch.zeros((), device=device)
    total_cls = torch.zeros((), device=device)
    total_box = torch.zeros((), device=device)
    total_pos = 0

    for pred, target in zip(predictions, targets):
        obj_pred = pred[..., 0].float()
        cls_pred = pred[..., 1:1 + len(CLASS_NAMES)].float()
        box_pred = pred[..., 1 + len(CLASS_NAMES):].float()

        present = target["present"].to(device)  # (B, SLOTS, H, W) bool
        num_pos = int(present.sum().item())
        total_pos += num_pos

        # Positive-aware Objectness Loss
        obj_target = present.float()
        bce = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction="none")
        pos_bce = bce[present]
        neg_bce = bce[~present]
        
        scale_obj = torch.zeros((), device=device)
        if pos_bce.numel() > 0:
            scale_obj = scale_obj + pos_bce.mean()
        if neg_bce.numel() > 0:
            scale_obj = scale_obj + negative_weight * neg_bce.mean()
        total_obj = total_obj + scale_obj

        if num_pos > 0:
            # Classification loss on positive slots
            pos_cls_pred = cls_pred[present]
            pos_cls_target = target["classes"].to(device)[present]
            cls_loss = F.cross_entropy(pos_cls_pred, pos_cls_target, reduction="mean")
            total_cls = total_cls + cls_loss

            # Box regression loss (centre offsets & log width/height)
            pos_box_pred = box_pred[present]
            pos_box_target = target["boxes"].to(device)[present]
            
            center_pred = torch.sigmoid(pos_box_pred[:, :2])
            center_loss = F.smooth_l1_loss(center_pred, pos_box_target[:, :2], reduction="mean")
            size_loss = F.smooth_l1_loss(pos_box_pred[:, 2:], pos_box_target[:, 2:], reduction="mean")
            total_box = total_box + center_loss + size_loss

    total_loss = (1.0 * total_obj) + (0.5 * total_cls) + (1.5 * total_box)
    metrics = {
        "obj_loss": float(total_obj.item()),
        "cls_loss": float(total_cls.item()),
        "box_loss": float(total_box.item()),
        "total_loss": float(total_loss.item()),
        "positive_slots": total_pos,
    }
    return total_loss, metrics
