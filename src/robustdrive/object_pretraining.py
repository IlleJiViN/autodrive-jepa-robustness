"""Box-guided object targets; boxes guide learning, without cropping away context.

This is a custom experiment, not a reproduction of an official JEPA or MAE.
Class labels are deliberately absent from mask generation and pretraining loss.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import stable_seed
from .data import DrivingPairs
from .models import MaskedPredictor, VisionEncoder, select_tokens


def box_patch_weights(boxes: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Area intersection of normalized xyxy boxes and grid cells, normalized per box."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if not torch.isfinite(boxes).all() or ((boxes < 0) | (boxes > 1)).any():
        raise ValueError("Boxes must be finite normalized coordinates in [0, 1]")
    if (boxes[:, 2:] <= boxes[:, :2]).any():
        raise ValueError("Boxes must have positive width and height")
    h, w = grid
    if min(h, w) < 1:
        raise ValueError("Grid dimensions must be positive")
    ys, xs = torch.meshgrid(torch.arange(h, device=boxes.device),
                            torch.arange(w, device=boxes.device), indexing="ij")
    lo = torch.stack((xs.flatten() / w, ys.flatten() / h), dim=1)
    hi = torch.stack(((xs.flatten() + 1) / w, (ys.flatten() + 1) / h), dim=1)
    intersection = (torch.minimum(boxes[:, None, 2:], hi) -
                    torch.maximum(boxes[:, None, :2], lo)).clamp_min(0).prod(dim=-1)
    if len(boxes) and (intersection.sum(dim=1) <= 0).any():
        raise ValueError("Box intersection vanished; coordinates are below float precision")
    return intersection / intersection.sum(dim=1, keepdim=True).clamp_min(1e-30)


@dataclass
class ObjectMask:
    visible: torch.Tensor
    masked: torch.Tensor
    object_weights: torch.Tensor  # [batch, max_objects, masked_tokens]
    valid_objects: torch.Tensor
    selected_objects: list[list[int]]
    oversized_objects: list[int]

    def to(self, device):
        return ObjectMask(self.visible.to(device), self.masked.to(device),
                          self.object_weights.to(device), self.valid_objects.to(device),
                          self.selected_objects, self.oversized_objects)


def make_object_mask(boxes: list[torch.Tensor], grid: tuple[int, int], mask_ratio: float,
                     seeds: list[int], mode="objects", max_objects=4) -> ObjectMask:
    """Hide complete box regions before attention, then fill to an exact token budget.

    Both objectives call this same function with sample/epoch seeds. No class label,
    image pixels, model output, or global RNG state can influence target selection.
    Overlarge objects are skipped, rather than partially hidden and called whole objects.
    """
    count = grid[0] * grid[1]
    if not 0 < mask_ratio < 1 or count < 2 or max_objects < 1:
        raise ValueError("Require 0 < mask_ratio < 1, at least two tokens and max_objects >= 1")
    if mode not in ("objects", "random") or len(boxes) != len(seeds) or not boxes:
        raise ValueError("Invalid mask mode or batch/seeds")
    visible_count = max(1, min(count - 1, int(count * (1 - mask_ratio))))
    masked_count = count - visible_count
    visible_rows, masked_rows, weight_rows, valid_rows, selected, oversized = [], [], [], [], [], []
    for sample_boxes, seed in zip(boxes, seeds):
        weights = box_patch_weights(torch.as_tensor(sample_boxes).cpu(), grid)
        background_order = np.random.default_rng(stable_seed(seed, "background")).permutation(count)
        chosen = np.zeros(count, dtype=bool)
        object_ids, too_large = [], 0
        if mode == "objects":
            order = np.random.default_rng(stable_seed(seed, "objects")).permutation(len(weights))
            for obj in order:
                support = weights[int(obj)].numpy() > 0
                if int(support.sum()) > masked_count:
                    too_large += 1
                    continue
                candidate = chosen | support
                if int(candidate.sum()) <= masked_count and len(object_ids) < max_objects:
                    chosen = candidate
                    object_ids.append(int(obj))
        remaining = background_order[~chosen[background_order]][:masked_count - int(chosen.sum())]
        chosen[remaining] = True
        target_ids = torch.from_numpy(np.flatnonzero(chosen)).long()
        visible_rows.append(torch.from_numpy(np.flatnonzero(~chosen)).long())
        masked_rows.append(target_ids)
        object_weights = torch.zeros(max_objects, masked_count)
        valid = torch.zeros(max_objects, dtype=torch.bool)
        for slot, obj in enumerate(object_ids):
            object_weights[slot] = weights[obj, target_ids]
            valid[slot] = True
        weight_rows.append(object_weights)
        valid_rows.append(valid)
        selected.append(object_ids)
        oversized.append(too_large)
    return ObjectMask(torch.stack(visible_rows), torch.stack(masked_rows),
                      torch.stack(weight_rows), torch.stack(valid_rows), selected, oversized)


class ObjectPairs(DrivingPairs):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.records[index]
        if row.get("annotation_status") not in ("available_legacy_bdd100k", "available_yolo_proposals"):
            raise ValueError("Box-guided pilot requires human boxes or explicit YOLO proposals")
        boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
        boxes /= torch.tensor([row["width"], row["height"], row["width"], row["height"]])
        return {**sample, "boxes": boxes, "sample_id": str(row["scene_id"])}


def collate_object_pairs(samples):
    return {"clean": torch.stack([s["clean"] for s in samples]),
            "view": torch.stack([s["view"] for s in samples]),
            "boxes": [s["boxes"] for s in samples],
            "sample_id": [s["sample_id"] for s in samples]}


def normalized_pixel_patches(images, patch_size):
    batch, channels, height, width = images.shape
    patches = images.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch_size * patch_size * channels)
    return (patches - patches.mean(-1, keepdim=True)) / torch.sqrt(
        patches.var(-1, keepdim=True, unbiased=False) + 1e-6)


class ObjectPretrainer(nn.Module):
    def __init__(self, config):
        super().__init__()
        spec, experiment = config["model"], config["object_pretraining"]
        self.objective = experiment["objective"]
        if self.objective not in ("jepa", "mae"):
            raise ValueError("objective must be jepa or mae")
        self.online = VisionEncoder(config["data"]["image_size"], spec["patch_size"],
                                    spec["dim"], spec["depth"], spec["heads"])
        count = self.online.grid[0] * self.online.grid[1]
        self.predictor = MaskedPredictor(count, spec["dim"], spec["predictor_dim"],
                                         spec["predictor_depth"], spec["predictor_heads"])
        self.patch_size = spec["patch_size"]
        if self.objective == "mae":
            self.predictor.output = nn.Linear(spec["predictor_dim"], self.patch_size ** 2 * 3)
            self.target = None
        else:
            self.target = copy.deepcopy(self.online).requires_grad_(False).eval()
        self.region_weight = float(experiment["object_loss_weight"])
        self.variance_floor = config["training"]["variance_floor"]
        self.variance_weight = config["training"]["variance_weight"]
        if self.region_weight < 0:
            raise ValueError("object_loss_weight must be nonnegative")

    def train(self, mode=True):
        super().train(mode)
        if self.target is not None:
            self.target.eval()
        return self

    def forward(self, clean, view, mask: ObjectMask):
        context = self.online(view, mask.visible)
        prediction = self.predictor(context, mask.visible, mask.masked).float()
        with torch.no_grad():
            if self.objective == "jepa":
                target = F.layer_norm(select_tokens(self.target(clean), mask.masked).float(),
                                      (prediction.shape[-1],))
            else:
                target = select_tokens(normalized_pixel_patches(clean.float(), self.patch_size), mask.masked)
        errors = (F.smooth_l1_loss(prediction, target, reduction="none") if self.objective == "jepa"
                  else F.mse_loss(prediction, target, reduction="none")).mean(dim=-1)
        patch_loss = errors.mean()
        if self.objective == "jepa":
            object_prediction = torch.einsum("bkm,bmd->bkd", mask.object_weights, prediction)
            object_target = torch.einsum("bkm,bmd->bkd", mask.object_weights, target)
            object_target = F.layer_norm(object_target, (target.shape[-1],))
            object_errors = F.smooth_l1_loss(object_prediction, object_target, reduction="none").mean(-1)
        else:
            object_errors = torch.einsum("bkm,bm->bk", mask.object_weights, errors)
        object_loss = (object_errors * mask.valid_objects).sum() / mask.valid_objects.sum().clamp_min(1)
        context_std = context.float().flatten(0, 1).std(dim=0, unbiased=False).mean()
        variance_loss = (F.relu(self.variance_floor - context.float().flatten(0, 1).std(dim=0, unbiased=False)).mean()
                         if self.objective == "jepa" else patch_loss.new_zeros(()))
        loss = patch_loss + self.region_weight * object_loss + self.variance_weight * variance_loss
        return loss, {"prediction_loss": patch_loss.detach(), "object_loss": object_loss.detach(),
                      "variance_loss": variance_loss.detach(), "context_std": context_std.detach(),
                      "objects_per_image": mask.valid_objects.float().sum(1).mean(),
                      "images_with_object_target": mask.valid_objects.any(1).float().mean()}

    @torch.no_grad()
    def update_target(self, momentum):
        if self.target is not None:
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.mul_(momentum).add_(online, alpha=1 - momentum)
            for target, online in zip(self.target.buffers(), self.online.buffers()):
                target.copy_(online)
