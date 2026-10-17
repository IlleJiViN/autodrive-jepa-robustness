"""
Foundation ViT-based Robust Object Detector v2 (Architecture Overhaul).
Fixes 5 critical architectural bottlenecks from v1:
1. Multi-Layer ViT intermediate token extraction [2, 5, 8, 11] + Top-Down FPN (P3 stride 8, P4 stride 16, P5 stride 32).
2. Decoupled Anchor-Free Detection Head (SLOTS = 1) with separate 2x Conv3x3 classification & regression branches.
3. Multi-Cell Target Assignment (~5x positive density) + Soft Scale Routing with boundary overlap.
4. Complete IoU (CIoU) Box Loss + Focal Objectness Loss + Aligned Class Probabilities.
5. Vectorized GPU Batched NMS (decode_v2) using torchvision.ops.batched_nms.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from .data import CLASS_NAMES

SLOTS = 1
VALUES = 5 + len(CLASS_NAMES)  # objectness (1), class logits (10), box offsets (4)
STRIDES = (8, 16, 32)
LEVEL_NAMES = ("P3", "P4", "P5")


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) between two batches of boxes (N, 4) in (x1, y1, x2, y2) format.
    Returns CIoU values clamped between [-1.0, 1.0].
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)

    w1 = (b1_x2 - b1_x1).clamp(min=0.0)
    h1 = (b1_y2 - b1_y1).clamp(min=0.0)
    w2 = (b2_x2 - b2_x1).clamp(min=0.0)
    h2 = (b2_y2 - b2_y1).clamp(min=0.0)

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter = inter_w * inter_h

    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    cw = (c_x2 - c_x1).clamp(min=0.0)
    ch = (c_y2 - c_y1).clamp(min=0.0)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2.0, (b1_y1 + b1_y2) / 2.0
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2.0, (b2_y1 + b2_y2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio penalty term
    v = (4.0 / (math.pi ** 2)) * torch.pow(
        torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
    )
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou.clamp(min=-1.0, max=1.0)


class ViTDetFPNAdapter(nn.Module):
    """
    Top-Down Feature Pyramid Adapter for Plain ViT Backbones.
    Extracts intermediate representations from 4 ViT blocks [2, 5, 8, 11] (each stride 16, 24x40)
    and constructs multi-scale feature pyramids P3 (stride 8, 48x80), P4 (stride 16, 24x40),
    and P5 (stride 32, 12x20) via lateral projections and top-down fusion.
    """
    def __init__(self, dim: int = 384, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        # P5 (stride 32, 12x20): lateral 1x1 from block 11 + stride-2 3x3 conv
        self.lateral_p5 = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.p5_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        # P4 (stride 16, 24x40): lateral 1x1 from blocks 8 & 5 + top-down from P5 + 3x3 conv
        self.lateral_p4_deep = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.lateral_p4_mid = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.p4_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

        # P3 (stride 8, 48x80): ConvTranspose2d (stride 2) from block 2 + top-down from P4 + 3x3 conv
        self.p3_trans = nn.Sequential(
            nn.ConvTranspose2d(dim, out_channels, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )
        self.p3_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            features: [F0, F1, F2, F3] from ViT blocks [2, 5, 8, 11], each shape (B, dim, 24, 40)
        Returns:
            [P3 (B, C, 48, 80), P4 (B, C, 24, 40), P5 (B, C, 12, 20)]
        """
        f0, f1, f2, f3 = features

        # P5 computation
        lat5 = self.lateral_p5(f3)
        p5 = self.p5_conv(lat5)  # (B, C, 12, 20)

        # P4 computation with Top-Down connection from P5
        lat4 = self.lateral_p4_deep(f2) + self.lateral_p4_mid(f1)
        p5_up = F.interpolate(p5, size=lat4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.p4_conv(lat4 + p5_up)  # (B, C, 24, 40)

        # P3 computation with Top-Down connection from P4
        lat3 = self.p3_trans(f0)  # (B, C, 48, 80)
        p4_up = F.interpolate(p4, size=lat3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.p3_conv(lat3 + p4_up)  # (B, C, 48, 80)

        return [p3, p4, p5]


class DecoupledScaleHead(nn.Module):
    """
    Decoupled Anchor-Free Detection Head with SLOTS = 1.
    Features separate classification (2x Conv3x3) and regression/objectness (2x Conv3x3) branches.
    Prior biases are initialized for training stability (focal loss ~0.01 prior).
    """
    def __init__(self, in_channels: int, num_classes: int = len(CLASS_NAMES), head_channels: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.values = 5 + num_classes

        # Classification branch: 2x Conv3x3 with GroupNorm and SiLU
        self.cls_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_channels),
            nn.SiLU(inplace=True),
        )
        self.cls_out = nn.Conv2d(head_channels, num_classes, kernel_size=1)

        # Regression branch: 2x Conv3x3 with GroupNorm and SiLU
        self.reg_branch = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, head_channels),
            nn.SiLU(inplace=True),
        )
        self.obj_out = nn.Conv2d(head_channels, 1, kernel_size=1)
        self.box_out = nn.Conv2d(head_channels, 4, kernel_size=1)

        # Prior bias initialization
        with torch.no_grad():
            self.obj_out.bias.fill_(-4.6)   # prior p ~ 0.01
            self.cls_out.bias.fill_(-4.6)   # prior p ~ 0.01
            self.box_out.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        cls_feat = self.cls_branch(x)
        cls_logits = self.cls_out(cls_feat)  # (B, num_classes, H, W)

        reg_feat = self.reg_branch(x)
        obj_logit = self.obj_out(reg_feat)   # (B, 1, H, W)
        box_preds = self.box_out(reg_feat)   # (B, 4, H, W)

        # Concatenate: (B, 1 + num_classes + 4, H, W)
        out = torch.cat([obj_logit, cls_logits, box_preds], dim=1)
        # Reshape to (B, SLOTS=1, H, W, VALUES)
        return out.view(batch, 1, self.values, height, width).permute(0, 1, 3, 4, 2).contiguous()


class FoundationViTDetectorV2(nn.Module):
    """
    End-to-End Multi-Scale Vision Transformer Detector v2.
    Combines:
    - Pretrained ViT-S/16 backbone with 4-layer intermediate token extraction [2, 5, 8, 11]
    - ViTDet Top-Down Feature Pyramid Adapter (P3: stride 8, P4: stride 16, P5: stride 32)
    - Decoupled anchor-free heads with SLOTS = 1
    """
    def __init__(
        self,
        model_name: str = "vit_small_patch16_384",
        pretrained: bool = True,
        freeze_backbone_blocks: int = 0,
        num_classes: int = len(CLASS_NAMES),
        img_size: tuple[int, int] = (384, 640),
        feature_indices: tuple[int, ...] = (2, 5, 8, 11),
        fpn_channels: int = 256,
        head_channels: int = 128,
    ):
        super().__init__()
        self.backbone_name = model_name
        self.img_size = img_size
        self.grid = (img_size[0] // 16, img_size[1] // 16)  # (24, 40)
        self.feature_indices = tuple(feature_indices)

        # ViT backbone with rectangular resolution (384, 640)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        self.embed_dim = self.backbone.embed_dim

        # FPN Adapter
        self.adapter = ViTDetFPNAdapter(dim=self.embed_dim, out_channels=fpn_channels)

        # Decoupled heads for P3, P4, P5
        self.heads = nn.ModuleList([
            DecoupledScaleHead(in_channels=fpn_channels, num_classes=num_classes, head_channels=head_channels)
            for _ in range(3)
        ])

        if freeze_backbone_blocks > 0:
            self._freeze_early_blocks(freeze_backbone_blocks)

    def _freeze_early_blocks(self, num_blocks: int):
        if hasattr(self.backbone, "patch_embed"):
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        if hasattr(self.backbone, "pos_embed") and isinstance(self.backbone.pos_embed, nn.Parameter):
            self.backbone.pos_embed.requires_grad = False
        if hasattr(self.backbone, "blocks"):
            for i in range(min(num_blocks, len(self.backbone.blocks))):
                for p in self.backbone.blocks[i].parameters():
                    p.requires_grad = False

    def forward_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-layer intermediate spatial feature maps (B, C, H, W)."""
        if hasattr(self.backbone, "forward_intermediates"):
            return self.backbone.forward_intermediates(
                x, indices=list(self.feature_indices), intermediates_only=True, norm=True
            )
        else:
            b = x.shape[0]
            x = self.backbone.patch_embed(x)
            x = self.backbone._pos_embed(x)
            if hasattr(self.backbone, "norm_pre"):
                x = self.backbone.norm_pre(x)
            features = []
            for i, blk in enumerate(self.backbone.blocks):
                x = blk(x)
                if i in self.feature_indices:
                    tokens = x[:, 1:, :] if x.shape[1] == 1 + self.grid[0] * self.grid[1] else x
                    feat = tokens.transpose(1, 2).reshape(b, self.embed_dim, self.grid[0], self.grid[1])
                    features.append(feat)
            return features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass yielding multi-scale detection outputs:
        Returns:
            [P3 (B, 1, 48, 80, V), P4 (B, 1, 24, 40, V), P5 (B, 1, 12, 20, V)]
        """
        feats = self.forward_backbone(x)
        fpn_feats = self.adapter(feats)
        return [head(f) for head, f in zip(self.heads, fpn_feats)]


def assign_targets_v2(
    records: list[dict],
    indices: Sequence[int],
    shapes: list[tuple[int, int]],
    image_size: tuple[int, int],
) -> tuple[list[dict[str, torch.Tensor]], int]:
    """
    Multi-Cell Target Assignment with Soft Scale Routing.
    - Soft Scale Routing: Allows overlap across scale boundaries (P3: side <= 48, P4: 24 <= side <= 96, P5: side >= 48).
    - Multi-Cell Assignment: Assigns up to a 3x3 neighborhood (center cell + up to 4 nearest neighbor cells
      inside the GT box within radius 1.5 * stride).
    - Targets store GT xyxy boxes in pixel coordinates directly for CIoU supervision.
    """
    batch = len(indices)
    targets = []
    assigned_areas = []

    for height, width in shapes:
        targets.append({
            "present": torch.zeros((batch, SLOTS, height, width), dtype=torch.bool),
            "classes": torch.zeros((batch, SLOTS, height, width), dtype=torch.long),
            "boxes": torch.zeros((batch, SLOTS, height, width, 4), dtype=torch.float32),
        })
        assigned_areas.append(torch.full((batch, height, width), float("inf")))

    image_height, image_width = image_size
    total_assigned = 0

    for batch_index, record_index in enumerate(indices):
        record = records[int(record_index)]
        sx = image_width / record["width"]
        sy = image_height / record["height"]

        for box, category in zip(record.get("boxes", []), record.get("labels", [])):
            x1, y1, x2, y2 = box
            x1, x2 = min(x1, x2) * sx, max(x1, x2) * sx
            y1, y2 = min(y1, y2) * sy, max(y1, y2) * sy

            x1 = max(0.0, min(float(image_width), x1))
            y1 = max(0.0, min(float(image_height), y1))
            x2 = max(0.0, min(float(image_width), x2))
            y2 = max(0.0, min(float(image_height), y2))

            box_w = x2 - x1
            box_h = y2 - y1
            if box_w <= 1.0 or box_h <= 1.0:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            side = math.sqrt(box_w * box_h)
            box_area = box_w * box_h

            # Soft scale routing with overlap at boundaries
            levels = []
            if side <= 48.0:
                levels.append(0)  # P3 (stride 8)
            if 24.0 <= side <= 96.0:
                levels.append(1)  # P4 (stride 16)
            if side >= 48.0:
                levels.append(2)  # P5 (stride 32)
            if not levels:
                levels.append(1)

            for level in levels:
                stride = STRIDES[level]
                height, width = shapes[level]

                c_col = min(width - 1, max(0, int(cx / stride)))
                c_row = min(height - 1, max(0, int(cy / stride)))

                # Check 3x3 neighborhood around center cell
                candidates = []
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        r = c_row + dr
                        c = c_col + dc
                        if 0 <= r < height and 0 <= c < width:
                            cell_x = (c + 0.5) * stride
                            cell_y = (r + 0.5) * stride
                            dist = math.sqrt((cell_x - cx) ** 2 + (cell_y - cy) ** 2)
                            if dist <= 1.5 * stride:
                                # Cell center must fall inside the GT box
                                if x1 <= cell_x <= x2 and y1 <= cell_y <= y2:
                                    candidates.append((dist, r, c))

                # Ensure at least the center cell is assigned
                if not candidates:
                    candidates.append((0.0, c_row, c_col))

                # Sort by distance to box center and take up to 5 cells
                candidates.sort(key=lambda item: item[0])
                selected = candidates[:5]

                for _, r, c in selected:
                    # Smaller box takes precedence when cells overlap
                    if box_area < assigned_areas[level][batch_index, r, c]:
                        assigned_areas[level][batch_index, r, c] = box_area
                        targets[level]["present"][batch_index, 0, r, c] = True
                        targets[level]["classes"][batch_index, 0, r, c] = int(category)
                        targets[level]["boxes"][batch_index, 0, r, c] = torch.tensor(
                            [x1, y1, x2, y2], dtype=torch.float32
                        )
                        total_assigned += 1

    return targets, total_assigned


def compute_detection_loss_v2(
    predictions: list[torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    negative_weight: float = 0.05,
    box_weight: float = 2.0,
    cls_weight: float = 1.0,
    obj_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Computes Anchor-Free detection loss across P3, P4, P5 scales:
    - Focal Objectness Loss across all cells
    - Category Cross-Entropy loss on positive cells
    - Complete IoU (CIoU) Loss on decoded xyxy bounding boxes
    """
    device = predictions[0].device
    total_obj = torch.zeros((), device=device)
    total_cls = torch.zeros((), device=device)
    total_box = torch.zeros((), device=device)
    total_pos = 0

    for level_idx, (pred, target) in enumerate(zip(predictions, targets)):
        stride = STRIDES[level_idx]
        batch, slots, height, width, values = pred.shape
        num_classes = values - 5

        obj_pred = pred[..., 0].float()
        cls_pred = pred[..., 1:1 + num_classes].float()
        box_pred = pred[..., 1 + num_classes:].float()

        present = target["present"].to(device)  # (B, 1, H, W) bool
        num_pos = int(present.sum().item())
        total_pos += num_pos

        # Focal Objectness Loss
        obj_target = present.float()
        bce_obj = F.binary_cross_entropy_with_logits(obj_pred, obj_target, reduction="none")
        p_obj = torch.sigmoid(obj_pred)

        # Focal weighting: alpha=0.25 on positive, 0.75 on negative, gamma=2.0
        focal_w = torch.where(present, 0.25 * ((1.0 - p_obj) ** 2.0), 0.75 * (p_obj ** 2.0))
        scale_obj = (focal_w * bce_obj).sum() / max(num_pos, 1)
        total_obj = total_obj + scale_obj

        if num_pos > 0:
            # Classification loss on positive cells
            pos_cls_pred = cls_pred[present]  # (N_pos, num_classes)
            pos_cls_target = target["classes"].to(device)[present]  # (N_pos,)
            scale_cls = F.cross_entropy(pos_cls_pred, pos_cls_target, reduction="mean")
            total_cls = total_cls + scale_cls

            # Decode predicted boxes on positive cells
            gy, gx = torch.meshgrid(
                torch.arange(height, device=device, dtype=torch.float32),
                torch.arange(width, device=device, dtype=torch.float32),
                indexing="ij",
            )
            gx = gx.view(1, 1, height, width).expand(batch, 1, height, width)
            gy = gy.view(1, 1, height, width).expand(batch, 1, height, width)

            pos_gx = gx[present]
            pos_gy = gy[present]
            pos_box_pred = box_pred[present]

            # Expanded center offsets and exponential sizes
            pos_cx = (pos_gx + 1.5 * torch.sigmoid(pos_box_pred[:, 0]) - 0.25) * stride
            pos_cy = (pos_gy + 1.5 * torch.sigmoid(pos_box_pred[:, 1]) - 0.25) * stride
            pos_w = torch.exp(pos_box_pred[:, 2].clamp(-4.0, 4.5)) * stride
            pos_h = torch.exp(pos_box_pred[:, 3].clamp(-4.0, 4.5)) * stride

            pred_xyxy = torch.stack([
                pos_cx - pos_w / 2.0,
                pos_cy - pos_h / 2.0,
                pos_cx + pos_w / 2.0,
                pos_cy + pos_h / 2.0,
            ], dim=-1)

            target_xyxy = target["boxes"].to(device)[present]
            ciou = bbox_ciou(pred_xyxy, target_xyxy)
            scale_box = (1.0 - ciou).mean()
            total_box = total_box + scale_box

    total_loss = (obj_weight * total_obj) + (cls_weight * total_cls) + (box_weight * total_box)
    metrics = {
        "obj_loss": float(total_obj.item()),
        "cls_loss": float(total_cls.item()),
        "box_loss": float(total_box.item()),
        "total_loss": float(total_loss.item()),
        "positive_slots": total_pos,
    }
    return total_loss, metrics


class DetectionsResult(list):
    """
    List of detection dicts for 100% compatibility with project coco_ap (for item in detections),
    while also supporting dict indexing: det["boxes"], det["scores"], det["labels"].
    """
    def __init__(
        self,
        detections: list[dict],
        boxes_arr: np.ndarray | None = None,
        scores_arr: np.ndarray | None = None,
        labels_arr: np.ndarray | None = None,
    ):
        super().__init__(detections)
        self.boxes_arr = boxes_arr
        self.scores_arr = scores_arr
        self.labels_arr = labels_arr

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key in ("boxes", "box"):
                return self.boxes_arr if self.boxes_arr is not None else np.array([d["box"] for d in self])
            elif key in ("scores", "score"):
                return self.scores_arr if self.scores_arr is not None else np.array([d["score"] for d in self])
            elif key in ("labels", "label", "classes", "class"):
                return self.labels_arr if self.labels_arr is not None else np.array([d["class"] for d in self])
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


@torch.no_grad()
def decode_v2(
    predictions: list[torch.Tensor],
    image_size: tuple[int, int] = (384, 640),
    max_detections: int = 100,
    score_floor: float = 0.01,
    nms_iou: float = 0.60,
    pre_nms_candidates: int = 1000,
) -> list[DetectionsResult]:
    """
    Vectorized GPU Batched NMS Decoding for Foundation ViT Detector v2.
    Decodes predictions across all scales, computes aligned class-conditioned scores,
    and applies torchvision.ops.batched_nms on GPU.
    Returns per-image DetectionsResult matching coco_ap interface.
    """
    image_height, image_width = image_size
    batch_size = predictions[0].shape[0]
    device = predictions[0].device
    num_classes = len(CLASS_NAMES)

    level_boxes = []
    level_scores = []
    level_classes = []

    for level_idx, pred in enumerate(predictions):
        stride = STRIDES[level_idx]
        b, s, h, w, v = pred.shape
        pred_flat = pred.view(b, s * h, w, v) if s > 1 else pred.squeeze(1)

        obj_prob = torch.sigmoid(pred_flat[..., 0].float())
        cls_logits = pred_flat[..., 1:1 + num_classes].float()
        cls_prob = F.softmax(cls_logits, dim=-1)
        box_reg = pred_flat[..., 1 + num_classes:].float()

        # Combined class scores: obj * cls_prob
        scores_all = obj_prob.unsqueeze(-1) * cls_prob
        best_scores, best_classes = scores_all.max(dim=-1)

        # Coordinate grid
        gy, gx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        gx = gx.view(1, h, w).expand(b, h, w)
        gy = gy.view(1, h, w).expand(b, h, w)

        cx = (gx + 1.5 * torch.sigmoid(box_reg[..., 0]) - 0.25) * stride
        cy = (gy + 1.5 * torch.sigmoid(box_reg[..., 1]) - 0.25) * stride
        bw = torch.exp(box_reg[..., 2].clamp(-4.0, 4.5)) * stride
        bh = torch.exp(box_reg[..., 3].clamp(-4.0, 4.5)) * stride

        x1 = (cx - bw / 2.0).clamp(0.0, float(image_width))
        y1 = (cy - bh / 2.0).clamp(0.0, float(image_height))
        x2 = (cx + bw / 2.0).clamp(0.0, float(image_width))
        y2 = (cy + bh / 2.0).clamp(0.0, float(image_height))
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        level_boxes.append(boxes.view(b, -1, 4))
        level_scores.append(best_scores.view(b, -1))
        level_classes.append(best_classes.view(b, -1))

    all_boxes = torch.cat(level_boxes, dim=1)      # (B, N_cells, 4)
    all_scores = torch.cat(level_scores, dim=1)    # (B, N_cells)
    all_classes = torch.cat(level_classes, dim=1)  # (B, N_cells)

    batch_results = []
    for bi in range(batch_size):
        b_boxes = all_boxes[bi]
        b_scores = all_scores[bi]
        b_classes = all_classes[bi]

        valid_mask = b_scores >= score_floor
        c_boxes = b_boxes[valid_mask]
        c_scores = b_scores[valid_mask]
        c_classes = b_classes[valid_mask]

        if c_scores.numel() == 0:
            batch_results.append(DetectionsResult(
                [],
                boxes_arr=np.empty((0, 4), dtype=np.float32),
                scores_arr=np.empty((0,), dtype=np.float32),
                labels_arr=np.empty((0,), dtype=np.int64),
            ))
            continue

        # Top-K pre-NMS selection
        if c_scores.numel() > pre_nms_candidates:
            topk_idx = torch.topk(c_scores, k=min(pre_nms_candidates, c_scores.shape[0]))[1]
            c_boxes = c_boxes[topk_idx]
            c_scores = c_scores[topk_idx]
            c_classes = c_classes[topk_idx]

        # Vectorized GPU Batched NMS
        keep = ops.batched_nms(c_boxes, c_scores, c_classes, nms_iou)
        keep = keep[:max_detections]

        k_boxes = c_boxes[keep].cpu().numpy()
        k_scores = c_scores[keep].cpu().numpy()
        k_classes = c_classes[keep].cpu().numpy()

        items = []
        for i in range(len(keep)):
            box_lst = k_boxes[i].tolist()
            sc = float(k_scores[i])
            cl = int(k_classes[i])
            items.append({
                "box": box_lst,
                "boxes": box_lst,
                "score": sc,
                "scores": sc,
                "class": cl,
                "label": cl,
                "labels": cl,
            })

        batch_results.append(DetectionsResult(
            items,
            boxes_arr=k_boxes,
            scores_arr=k_scores,
            labels_arr=k_classes,
        ))

    return batch_results
