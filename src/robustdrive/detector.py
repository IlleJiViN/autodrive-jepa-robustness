"""Small anchor-free, three-scale detector for frozen JEPA feature probes."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import CLASS_NAMES
from .models import YOLOFeatureAdapter

SLOTS = 3
VALUES = 5 + len(CLASS_NAMES)  # objectness, class logits, centre offsets, log width/height
STRIDES = (8, 16, 32)
LEVEL_NAMES = ("P3", "P4", "P5")


class ScaleHead(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, 128, 3, padding=1, bias=False),
            nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, SLOTS * VALUES, 1),
        )
        with torch.no_grad():
            bias = self.layers[-1].bias.view(SLOTS, VALUES)
            bias[:, 0].fill_(-5.0)

    def forward(self, feature):
        batch, _, height, width = feature.shape
        value = self.layers(feature)
        return value.view(batch, SLOTS, VALUES, height, width).permute(0, 1, 3, 4, 2).contiguous()


class FrozenFeatureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = YOLOFeatureAdapter(dim=384, channels=(128, 256, 512))
        self.heads = nn.ModuleList([ScaleHead(c) for c in (128, 256, 512)])

    def forward(self, tokens, grid=(24, 40)):
        return [head(feature) for head, feature in zip(self.heads, self.adapter(tokens, grid))]


def assign_targets(records, indices, shapes, image_size):
    batch = len(indices)
    targets = []
    for _, height, width in shapes:
        targets.append({"present": torch.zeros((batch, SLOTS, height, width), dtype=torch.bool),
                        "classes": torch.zeros((batch, SLOTS, height, width), dtype=torch.long),
                        "boxes": torch.zeros((batch, SLOTS, height, width, 4), dtype=torch.float32)})
    dropped = 0
    image_height, image_width = image_size
    for batch_index, record_index in enumerate(indices):
        record = records[int(record_index)]
        sx, sy = image_width / record["width"], image_height / record["height"]
        for box, category in zip(record["boxes"], record["labels"]):
            x1, y1, x2, y2 = box
            x1, x2 = x1 * sx, x2 * sx
            y1, y2 = y1 * sy, y2 * sy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            box_width, box_height = x2 - x1, y2 - y1
            side = math.sqrt(box_width * box_height)
            level = 0 if side <= 4 * STRIDES[0] else 1 if side <= 4 * STRIDES[1] else 2
            slots, height, width = shapes[level]
            stride = STRIDES[level]
            column = min(width - 1, max(0, int(cx / stride)))
            row = min(height - 1, max(0, int(cy / stride)))
            free = torch.nonzero(~targets[level]["present"][batch_index, :, row, column], as_tuple=False)
            if free.numel() == 0:
                dropped += 1
                continue
            slot = int(free[0, 0])
            targets[level]["present"][batch_index, slot, row, column] = True
            targets[level]["classes"][batch_index, slot, row, column] = int(category)
            targets[level]["boxes"][batch_index, slot, row, column] = torch.tensor([
                cx / stride - column, cy / stride - row,
                math.log(max(box_width / stride, math.exp(-4))),
                math.log(max(box_height / stride, math.exp(-4))),
            ])
    return targets, dropped


def detector_loss(predictions, targets, negative_weight=0.025):
    object_loss = predictions[0].new_zeros(())
    class_loss = predictions[0].new_zeros(())
    box_loss = predictions[0].new_zeros(())
    positive_count = 0
    for prediction, target in zip(predictions, targets):
        present = target["present"].to(prediction.device)
        objectness = prediction[..., 0].float()
        obj_loss = F.binary_cross_entropy_with_logits(objectness, present.float(), reduction="none")
        positive, negative = obj_loss[present], obj_loss[~present]
        if positive.numel():
            object_loss = object_loss + positive.mean() + negative_weight * negative.mean()
            positive_count += positive.numel()
            cls = target["classes"].to(prediction.device)[present]
            class_loss = class_loss + F.cross_entropy(prediction[..., 1:1 + len(CLASS_NAMES)][present].float(), cls)
            truth = target["boxes"].to(prediction.device)[present]
            predicted = prediction[..., 1 + len(CLASS_NAMES):][present].float()
            predicted = torch.cat((predicted[..., :2].sigmoid(), predicted[..., 2:]), dim=-1)
            box_loss = box_loss + F.smooth_l1_loss(predicted, truth)
    if positive_count == 0:
        raise ValueError("Detector batch has no assigned objects")
    return object_loss + class_loss + box_loss, {
        "object_loss": float(object_loss.detach()), "class_loss": float(class_loss.detach()),
        "box_loss": float(box_loss.detach()), "positive_assignments": positive_count,
    }


@torch.no_grad()
def decode(predictions, image_size, max_detections=100, score_floor=1e-5, nms_iou=0.65,
           pre_nms_candidates=300):
    """Return per-image class-aware-NMS detections in resized-image xyxy pixels."""
    image_height, image_width = image_size
    batch_size = predictions[0].shape[0]
    results = [[] for _ in range(batch_size)]
    candidate_boxes = [[] for _ in range(batch_size)]
    candidate_scores = [[] for _ in range(batch_size)]
    candidate_classes = [[] for _ in range(batch_size)]
    for prediction in predictions:
        _, _, height, width, _ = prediction.shape
        stride_y, stride_x = image_height / height, image_width / width
        detached = prediction.float()
        objectness = detached[..., 0].sigmoid()
        category = detached[..., 1:1 + len(CLASS_NAMES)].sigmoid()
        offsets = detached[..., 1 + len(CLASS_NAMES):]
        rows, columns = torch.meshgrid(torch.arange(height, device=prediction.device),
                                       torch.arange(width, device=prediction.device), indexing="ij")
        centre_x = (columns[None, None] + offsets[..., 0].sigmoid()) * stride_x
        centre_y = (rows[None, None] + offsets[..., 1].sigmoid()) * stride_y
        box_width = offsets[..., 2].clamp(-4, 4.5).exp() * stride_x
        box_height = offsets[..., 3].clamp(-4, 4.5).exp() * stride_y
        boxes = torch.stack((centre_x - box_width / 2, centre_y - box_height / 2,
                             centre_x + box_width / 2, centre_y + box_height / 2), dim=-1)
        boxes[..., 0::2].clamp_(0, image_width)
        boxes[..., 1::2].clamp_(0, image_height)
        scores, classes = (objectness[..., None] * category).max(dim=-1)
        for batch_index in range(batch_size):
            flat_boxes = boxes[batch_index].reshape(-1, 4)
            flat_scores = scores[batch_index].reshape(-1)
            flat_classes = classes[batch_index].reshape(-1)
            keep = flat_scores >= score_floor
            if keep.any():
                candidate_boxes[batch_index].append(flat_boxes[keep])
                candidate_scores[batch_index].append(flat_scores[keep])
                candidate_classes[batch_index].append(flat_classes[keep])
    for index in range(batch_size):
        if not candidate_boxes[index]:
            continue
        boxes = torch.cat(candidate_boxes[index])
        scores = torch.cat(candidate_scores[index])
        classes = torch.cat(candidate_classes[index])
        if len(scores) > pre_nms_candidates:
            selected = torch.argsort(scores, descending=True, stable=True)[:pre_nms_candidates]
            boxes, scores, classes = boxes[selected], scores[selected], classes[selected]
        boxes, scores, classes = boxes.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()
        order = np.argsort(-scores, kind="stable")
        kept = []
        suppressed = np.zeros(len(order), dtype=bool)
        for position, chosen in enumerate(order):
            if suppressed[position]:
                continue
            kept.append(chosen)
            if len(kept) >= max_detections:
                break
            remaining_positions = np.arange(position + 1, len(order))
            remaining = order[remaining_positions]
            same_class = classes[remaining] == classes[chosen]
            if not same_class.any():
                continue
            overlap_candidates = remaining_positions[same_class]
            other = boxes[order[overlap_candidates]]
            intersection_left = np.maximum(boxes[chosen, :2], other[:, :2])
            intersection_right = np.minimum(boxes[chosen, 2:], other[:, 2:])
            intersection = np.maximum(intersection_right - intersection_left, 0).prod(axis=1)
            area_a = np.maximum(boxes[chosen, 2:] - boxes[chosen, :2], 0).prod()
            area_b = np.maximum(other[:, 2:] - other[:, :2], 0).prod(axis=1)
            iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
            suppressed[overlap_candidates[iou > nms_iou]] = True
        results[index] = [{"box": boxes[item].tolist(), "score": float(scores[item]), "class": int(classes[item])}
                          for item in kept]
    return results


def coco_ap(predictions, ground_truths, train_class_counts, val_class_counts, classes=10):
    """COCO-style 101-point AP averaged across IoU .50:.95, class by class."""
    thresholds = np.arange(0.50, 0.951, 0.05)
    per_class = []
    for category in range(classes):
        image_gt = {image: boxes[labels == category] for image, (boxes, labels) in ground_truths.items()}
        positives = sum(len(boxes) for boxes in image_gt.values())
        if positives == 0 or train_class_counts[category] == 0:
            continue
        candidates = [(item["score"], image, np.asarray(item["box"], dtype=np.float32))
                      for image, detections in predictions.items() for item in detections if item["class"] == category]
        candidates.sort(key=lambda item: item[0], reverse=True)
        aps = []
        for threshold in thresholds:
            matched = {image: np.zeros(len(boxes), dtype=bool) for image, boxes in image_gt.items()}
            true_positive, false_positive = np.zeros(len(candidates)), np.zeros(len(candidates))
            for index, (_, image, box) in enumerate(candidates):
                boxes = image_gt.get(image)
                if boxes is None or len(boxes) == 0:
                    false_positive[index] = 1
                    continue
                intersection_left = np.maximum(boxes[:, :2], box[:2])
                intersection_right = np.minimum(boxes[:, 2:], box[2:])
                intersection = np.maximum(intersection_right - intersection_left, 0).prod(axis=1)
                area_a = np.maximum(box[2:] - box[:2], 0).prod()
                area_b = np.maximum(boxes[:, 2:] - boxes[:, :2], 0).prod(axis=1)
                iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
                iou[matched[image]] = -1
                best = int(iou.argmax())
                if iou[best] >= threshold:
                    true_positive[index] = 1
                    matched[image][best] = True
                else:
                    false_positive[index] = 1
            if not candidates:
                aps.append(0.0)
                continue
            recall = np.cumsum(true_positive) / positives
            precision = np.cumsum(true_positive) / np.maximum(np.cumsum(true_positive + false_positive), 1e-12)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            aps.append(float(np.mean([envelope[recall >= point].max() if np.any(recall >= point) else 0.0
                                      for point in np.linspace(0, 1, 101)])))
        per_class.append({"class": category, "validation_ground_truth_boxes": positives,
                          "ap50": aps[0], "ap50_95": float(np.mean(aps))})
    if not per_class:
        raise ValueError("No detection classes have both train and validation ground truth")
    return {"map50": float(np.mean([item["ap50"] for item in per_class])),
            "map50_95": float(np.mean([item["ap50_95"] for item in per_class])),
            "supported_classes": len(per_class), "per_class": per_class,
            "iou_thresholds": [round(float(value), 2) for value in thresholds],
            "ap_method": "101-point interpolated precision, one-to-one greedy IoU matching"}
