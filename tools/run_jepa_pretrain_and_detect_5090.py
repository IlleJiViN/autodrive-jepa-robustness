"""
Integrated RTX 5090 Natural-Noise I-JEPA Pretraining & Downstream Detection Pipeline.
Steps:
1. Natural-Noise I-JEPA Pretraining (FoundationCorruptionJEPA on BDD100K 70k)
   - Online student with strict context masking before attention (50% visible patches)
   - Teacher EMA target with full clean image (requires_grad=False)
   - Smooth L1 + VICReg Variance & Covariance decorrelation loss
   - Measure Representation Quality (Effective Rank, Invariance Cosine, JEPA loss) before & after
2. Downstream ViT Detection with JEPA-Pretrained Encoder
   - Part A: Controlled 2,048-image 10-epoch A/B test vs Pure Supervised ViT-S (16.16%)
   - Part B: 70,000-image Full-Scale Detector training & official 10k validation benchmark
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.common import stable_seed
from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.corruptions import corrupt, FAMILIES
from robustdrive.detector import coco_ap
from robustdrive.foundation_jepa import FoundationCorruptionJEPA, ema_momentum
from robustdrive.foundation_v2 import (
    STRIDES,
    FoundationViTDetectorV2,
    assign_targets_v2,
    compute_detection_loss_v2,
    decode_v2,
)
from robustdrive.metrics import compute_effective_rank, compute_uniformity


# ==============================================================================
# Phase 1: Natural-Noise I-JEPA Dataset & Representation Quality
# ==============================================================================

class BDDJEPAPretrainDataset(Dataset):
    """
    On-the-fly Clean-Corrupted Pair Dataset for Natural-Noise I-JEPA.
    Teacher receives the clean image; Student receives dynamically corrupted view
    (Fog, Rain, Motion Blur, Low-Light at Severity 1..4).
    """
    def __init__(self, records: list[dict], image_size: tuple[int, int] = (384, 640)):
        self.records = records
        self.image_size = image_size
        self.height, self.width = image_size
        self.families = ("fog", "rain", "motion_blur", "low_light")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        img_path = rec["image_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        # 1. Clean view for EMA Teacher
        clean_tensor = to_tensor(image)

        # 2. Corrupted natural-noise view for Student Context Encoder
        family = self.families[idx % len(self.families)]
        severity = (idx % 4) + 1  # 1, 2, 3, 4
        seed = (idx * 7919) % 10000000
        corrupted = corrupt(image, family, severity, seed)
        view_tensor = to_tensor(corrupted)

        return clean_tensor, view_tensor


def evaluate_representation_quality(
    model: FoundationCorruptionJEPA,
    val_records: list[dict],
    device: str = "cuda",
    max_images: int = 256,
) -> dict[str, float]:
    """
    Evaluates Representation Quality Metrics:
    - Effective Rank (singular value entropy of representations)
    - Clean-vs-Corrupted Invariance (mean cosine similarity across weather corruptions)
    - Latent JEPA Prediction Loss
    """
    model.eval()
    eval_recs = val_records[:max_images]
    image_size = (384, 640)

    clean_features = []
    corrupt_features = {fam: [] for fam in FAMILIES}
    jepa_losses = []

    with torch.no_grad():
        for idx, rec in enumerate(eval_recs):
            with Image.open(rec["image_path"]) as img:
                image = img.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

            clean_t = to_tensor(image).unsqueeze(0).to(device)

            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                # Full unmasked token extraction from online encoder
                feat_clean = model.extract_features(clean_t)  # (1, 960, 384)
                clean_features.append(feat_clean.mean(dim=1).squeeze(0).float().cpu())

                for fam in FAMILIES:
                    c_seed = stable_seed(42, str(rec["image_path"]), fam, 2)
                    c_img = corrupt(image, fam, severity=2, seed=c_seed)
                    c_t = to_tensor(c_img).unsqueeze(0).to(device)
                    feat_c = model.extract_features(c_t)
                    corrupt_features[fam].append(feat_c.mean(dim=1).squeeze(0).float().cpu())

                # Validation JEPA prediction loss on corrupted pair
                loss, metrics = model(clean_t, c_t)
                jepa_losses.append(metrics["prediction_loss"].item())

    z_clean = torch.stack(clean_features)
    erank = compute_effective_rank(z_clean)
    uniformity = compute_uniformity(z_clean)

    invariances = {}
    for fam, feats in corrupt_features.items():
        z_c = torch.stack(feats)
        cos_sim = F.cosine_similarity(z_clean, z_c, dim=-1).mean().item()
        invariances[fam] = cos_sim

    mean_inv = float(np.mean(list(invariances.values())))
    mean_jepa_loss = float(np.mean(jepa_losses))

    return {
        "effective_rank": erank,
        "uniformity": uniformity,
        "mean_invariance_cosine": mean_inv,
        "invariance_fog": invariances["fog"],
        "invariance_rain": invariances["rain"],
        "invariance_low_light": invariances["low_light"],
        "invariance_motion_blur": invariances["motion_blur"],
        "val_jepa_prediction_loss": mean_jepa_loss,
    }


def train_jepa_pretrain_phase(
    train_records: list[dict],
    val_records: list[dict],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1.5e-4,
    workers: int = 4,
    device: str = "cuda",
    output_dir: Path = ROOT / "runs/rtx5090-jepa-pretrained",
) -> Path:
    """Phase 1: Pretrain FoundationCorruptionJEPA on BDD100K images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("PHASE 1: NATURAL-NOISE I-JEPA PRETRAINING (RTX 5090)")
    print(f"Dataset: {len(train_records):,} images | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 80)

    # Initialize FoundationCorruptionJEPA
    model = FoundationCorruptionJEPA(
        model_name="vit_small_patch16_384",
        img_size=(384, 640),
        pretrained=True,
        mask_ratio=0.50,
        predictor_dim=192,
        predictor_depth=4,
        predictor_heads=6,
        variance_floor=1.0,
        variance_weight=0.1,
        covariance_weight=0.05,
    ).to(device)

    # Evaluate Baseline Representation Quality (Epoch 0 before training)
    print("[Phase 1] Evaluating Baseline ViT-S Representation Quality (Epoch 0)...")
    baseline_metrics = evaluate_representation_quality(model, val_records, device=device, max_images=256)
    print(f"  >>> Baseline Effective Rank: {baseline_metrics['effective_rank']:.2f}")
    print(f"  >>> Baseline Clean-vs-Corrupt Invariance Cosine: {baseline_metrics['mean_invariance_cosine']:.4f}")
    print(f"  >>> Baseline Latent Prediction Loss: {baseline_metrics['val_jepa_prediction_loss']:.4f}")

    train_dataset = BDDJEPAPretrainDataset(train_records, image_size=(384, 640))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
    )

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)

    trainable_params = [p for p in model.online.parameters() if p.requires_grad] + list(model.predictor.parameters())
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=0.05, betas=(0.9, 0.98))

    print(f"[Phase 1] Trainable params: {sum(p.numel() for p in trainable_params):,}")
    print(f"[Phase 1] Steps per epoch: {steps_per_epoch} | Total steps: {total_steps} (Warmup: {warmup_steps})")

    global_step = 0
    start_time = time.time()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_pred, ep_var, ep_cov = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step, (clean_imgs, view_imgs) in enumerate(train_loader):
            global_step += 1

            # Cosine LR schedule with linear warmup
            if global_step < warmup_steps:
                cur_lr = lr * max(0.01, float(global_step) / max(1, warmup_steps))
            else:
                progress = float(global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cur_lr = lr * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            clean_imgs = clean_imgs.to(device, non_blocking=True)
            view_imgs = view_imgs.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                loss, metrics = model(clean_imgs, view_imgs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()

            # Target EMA update: cosine momentum 0.996 -> 1.0
            momentum = ema_momentum(0.996, global_step, total_steps)
            model.update_target(momentum)

            ep_loss += metrics["total_loss"].item()
            ep_pred += metrics["prediction_loss"].item()
            ep_var += metrics["variance_loss"].item()
            ep_cov += metrics["covariance_loss"].item()

            if (step + 1) % 50 == 0 or (step + 1) == steps_per_epoch:
                elapsed = time.time() - ep_start
                rate = (step + 1) / max(elapsed, 0.01)
                img_rate = rate * batch_size
                print(
                    f"  [JEPA Ep {epoch:02d}/{epochs:02d} | Step {step+1:04d}/{steps_per_epoch}] "
                    f"Loss: {metrics['total_loss'].item():.4f} (Pred: {metrics['prediction_loss'].item():.4f}, "
                    f"Var: {metrics['variance_loss'].item():.4f}, Cov: {metrics['covariance_loss'].item():.4f}) | "
                    f"Ctx Std: {metrics['context_std'].item():.3f} | {rate:.1f} it/s ({img_rate:.0f} img/s) | lr: {cur_lr:.2e}"
                )

        ep_time = time.time() - ep_start
        avg_loss = ep_loss / steps_per_epoch
        avg_pred = ep_pred / steps_per_epoch
        print(f"  >>> [Epoch {epoch:02d} Complete in {ep_time:.1f}s] Avg Loss: {avg_loss:.4f} | Avg Pred Loss: {avg_pred:.4f} <<<")

        history.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "avg_pred_loss": avg_pred,
            "epoch_seconds": ep_time,
        })

    total_pretrain_time = time.time() - start_time
    print(f"\n[Phase 1] Pretraining finished in {total_pretrain_time/60:.2f} min ({total_pretrain_time:.1f}s)!")

    # Post-Pretraining Representation Quality Evaluation
    print("\n[Phase 1] Evaluating Post-JEPA Representation Quality (Epoch 5)...")
    post_metrics = evaluate_representation_quality(model, val_records, device=device, max_images=256)
    print(f"  >>> Post-JEPA Effective Rank: {post_metrics['effective_rank']:.2f} (Baseline: {baseline_metrics['effective_rank']:.2f})")
    print(f"  >>> Post-JEPA Clean-vs-Corrupt Invariance Cosine: {post_metrics['mean_invariance_cosine']:.4f} (Baseline: {baseline_metrics['mean_invariance_cosine']:.4f})")
    print(f"  >>> Post-JEPA Latent Prediction Loss: {post_metrics['val_jepa_prediction_loss']:.4f} (Baseline: {baseline_metrics['val_jepa_prediction_loss']:.4f})")

    # Save student encoder weights
    encoder_path = output_dir / "jepa_student_encoder.pt"
    torch.save(model.get_student_encoder().state_dict(), encoder_path)
    print(f"[Phase 1] JEPA Student Encoder saved to {encoder_path}")

    # Save summary report
    summary = {
        "pretraining_time_seconds": total_pretrain_time,
        "epochs": epochs,
        "batch_size": batch_size,
        "baseline_metrics": baseline_metrics,
        "post_metrics": post_metrics,
        "delta": {
            "effective_rank_gain": post_metrics["effective_rank"] - baseline_metrics["effective_rank"],
            "invariance_gain": post_metrics["mean_invariance_cosine"] - baseline_metrics["mean_invariance_cosine"],
            "prediction_loss_reduction": baseline_metrics["val_jepa_prediction_loss"] - post_metrics["val_jepa_prediction_loss"],
        },
        "history": history,
    }
    summary_path = output_dir / "jepa_pretrain_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Phase 1] Summary metrics saved to {summary_path}")

    return encoder_path


# ==============================================================================
# Phase 2: Downstream Detection Dataset & Training
# ==============================================================================

class BDD100KDetectorDataset(Dataset):
    """BDD100K Dataset with horizontal flip and weather augmentations for detector training."""
    def __init__(self, records: list[dict], image_size: tuple[int, int] = (384, 640), augment: bool = True):
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.height, self.width = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img_path = rec["image_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        orig_w = rec.get("width", image.width)
        orig_h = rec.get("height", image.height)
        boxes = [list(b) for b in rec.get("boxes", [])]
        labels = list(rec.get("labels", []))

        # Geometric Augmentation: Random Horizontal Flip (50%)
        if self.augment and random.random() < 0.50:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[orig_w - b[2], b[1], orig_w - b[0], b[3]] for b in boxes]

        # Weather Augmentation (30%)
        if self.augment and random.random() < 0.30:
            c_fam = random.choice(["fog", "rain", "motion_blur", "low_light"])
            c_sev = random.randint(1, 3)
            c_seed = random.randint(0, 1000000)
            image = corrupt(image, c_fam, c_sev, seed=c_seed)

        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        tensor_img = to_tensor(image)
        sx = self.width / orig_w
        sy = self.height / orig_h
        scaled_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]

        return {
            "image": tensor_img,
            "boxes": scaled_boxes,
            "labels": labels,
            "width": self.width,
            "height": self.height,
        }


def detector_collate_fn(batch: list[dict], shapes: list[tuple[int, int]], image_size: tuple[int, int]):
    images = torch.stack([item["image"] for item in batch])
    records = [
        {"boxes": item["boxes"], "labels": item["labels"], "width": item["width"], "height": item["height"]}
        for item in batch
    ]
    targets, _ = assign_targets_v2(records, list(range(len(batch))), shapes, image_size)
    return images, targets


def evaluate_detector_quick(
    model: nn.Module,
    val_records: list[dict],
    train_records: list[dict],
    device: str = "cuda",
    max_images: int = 256,
) -> tuple[float, float]:
    """Fast validation evaluation using vectorized GPU batched NMS."""
    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)

    train_counts = np.zeros(num_classes, dtype=int)
    for r in train_records[:500]:
        for c in r.get("labels", []):
            if c < num_classes:
                train_counts[c] += 1

    val_counts = np.zeros(num_classes, dtype=int)
    eval_recs = val_records[:max_images]
    for r in eval_recs:
        for c in r.get("labels", []):
            if c < num_classes:
                val_counts[c] += 1

    predictions = {}
    ground_truths = {}

    with torch.no_grad():
        for idx, rec in enumerate(eval_recs):
            img_id = rec.get("scene_id", f"img_{idx}")
            with Image.open(rec["image_path"]) as img:
                image = img.convert("RGB")
            orig_w, orig_h = rec.get("width", image.width), rec.get("height", image.height)
            if image.size != (image_size[1], image_size[0]):
                image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

            inp = to_tensor(image).unsqueeze(0).to(device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                preds = model(inp)

            decoded = decode_v2(
                preds,
                image_size=image_size,
                max_detections=100,
                score_floor=0.01,
                nms_iou=0.60,
                pre_nms_candidates=1000,
            )
            predictions[img_id] = decoded[0]

            sx, sy = image_size[1] / orig_w, image_size[0] / orig_h
            gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
            gt_labels = [int(c) for c in rec.get("labels", [])]

            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    return res["map50"] * 100.0, res["map50_95"] * 100.0


def run_comprehensive_weather_benchmark(
    model: nn.Module,
    val_records: list[dict],
    train_records: list[dict],
    device: str = "cuda",
    max_images: int = 256,
    severities: Sequence[int] = (2, 3),
    seeds: Sequence[int] = (42, 100, 2024),
) -> dict[str, Any]:
    """
    Evaluates detector across Clean and Adverse Weather (Fog, Rain, Low-light, Motion blur)
    at multiple severity levels (e.g. Sev 2 and Sev 3) across multiple seeds.
    """
    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)
    eval_recs = val_records[:max_images]

    train_counts = np.zeros(num_classes, dtype=int)
    for r in train_records[:500]:
        for c in r.get("labels", []):
            if c < num_classes:
                train_counts[c] += 1

    val_counts = np.zeros(num_classes, dtype=int)
    for r in eval_recs:
        for c in r.get("labels", []):
            if c < num_classes:
                val_counts[c] += 1

    results = {}

    # 1. Clean condition
    clean_map50_seeds, clean_map50_95_seeds = [], []
    for s in seeds:
        preds, gts = {}, {}
        with torch.no_grad():
            for idx, rec in enumerate(eval_recs):
                img_id = rec.get("scene_id", f"img_{idx}")
                with Image.open(rec["image_path"]) as img:
                    image = img.convert("RGB")
                orig_w, orig_h = rec.get("width", image.width), rec.get("height", image.height)
                if image.size != (image_size[1], image_size[0]):
                    image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

                inp = to_tensor(image).unsqueeze(0).to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    p = model(inp)
                preds[img_id] = decode_v2(p, image_size=image_size)[0]

                sx, sy = image_size[1] / orig_w, image_size[0] / orig_h
                gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
                gts[img_id] = (
                    np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                    np.array(rec.get("labels", []), dtype=np.int64),
                )
        res = coco_ap(preds, gts, train_counts, val_counts, classes=num_classes)
        clean_map50_seeds.append(res["map50"] * 100.0)
        clean_map50_95_seeds.append(res["map50_95"] * 100.0)

    clean_mean_50 = float(np.mean(clean_map50_seeds))
    clean_mean_50_95 = float(np.mean(clean_map50_95_seeds))
    results["clean"] = {
        "condition": "Clean",
        "severity": 0,
        "map50_mean": clean_mean_50,
        "map50_std": float(np.std(clean_map50_seeds)),
        "map50_95_mean": clean_mean_50_95,
        "map50_95_std": float(np.std(clean_map50_95_seeds)),
    }
    print(f"  Condition: Clean                | mAP50: {clean_mean_50:.2f}% | mAP50:95: {clean_mean_50_95:.2f}%")

    # 2. Adverse weather conditions
    for sev in severities:
        for fam in ("fog", "rain", "low_light", "motion_blur"):
            cond_key = f"{fam}_s{sev}"
            cond_map50_seeds, cond_map50_95_seeds = [], []

            for s in seeds:
                preds, gts = {}, {}
                with torch.no_grad():
                    for idx, rec in enumerate(eval_recs):
                        img_id = rec.get("scene_id", f"img_{idx}")
                        with Image.open(rec["image_path"]) as img:
                            image = img.convert("RGB")
                        orig_w, orig_h = rec.get("width", image.width), rec.get("height", image.height)

                        c_seed = stable_seed(s, str(rec["image_path"]), fam, sev)
                        image = corrupt(image, fam, sev, c_seed)
                        if image.size != (image_size[1], image_size[0]):
                            image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

                        inp = to_tensor(image).unsqueeze(0).to(device)
                        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                            p = model(inp)
                        preds[img_id] = decode_v2(p, image_size=image_size)[0]

                        sx, sy = image_size[1] / orig_w, image_size[0] / orig_h
                        gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
                        gts[img_id] = (
                            np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                            np.array(rec.get("labels", []), dtype=np.int64),
                        )
                res = coco_ap(preds, gts, train_counts, val_counts, classes=num_classes)
                cond_map50_seeds.append(res["map50"] * 100.0)
                cond_map50_95_seeds.append(res["map50_95"] * 100.0)

            c_mean_50 = float(np.mean(cond_map50_seeds))
            c_mean_50_95 = float(np.mean(cond_map50_95_seeds))
            retention = (c_mean_50 / max(clean_mean_50, 1e-6)) * 100.0

            results[cond_key] = {
                "condition": f"{fam.capitalize()} (Sev {sev})",
                "family": fam,
                "severity": sev,
                "map50_mean": c_mean_50,
                "map50_std": float(np.std(cond_map50_seeds)),
                "map50_95_mean": c_mean_50_95,
                "map50_95_std": float(np.std(cond_map50_95_seeds)),
                "retention_rate": retention,
            }
            print(f"  Condition: {fam.capitalize():<12} (Sev {sev}) | mAP50: {c_mean_50:.2f} ± {np.std(cond_map50_seeds):.2f}% | mAP50:95: {c_mean_50_95:.2f}% | Retention: {retention:.1f}%")

    # Compute overall robust averages
    for sev in severities:
        sev_keys = [f"{fam}_s{sev}" for fam in ("fog", "rain", "low_light", "motion_blur")]
        m50 = float(np.mean([results[k]["map50_mean"] for k in sev_keys]))
        m50_95 = float(np.mean([results[k]["map50_95_mean"] for k in sev_keys]))
        ret = (m50 / max(clean_mean_50, 1e-6)) * 100.0
        results[f"mean_adverse_s{sev}"] = {
            "map50_mean": m50,
            "map50_95_mean": m50_95,
            "retention_rate": ret,
        }
        print(f"  >>> Overall Adverse Sev {sev} Mean: mAP50: {m50:.2f}% | mAP50:95: {m50_95:.2f}% | Retention: {ret:.1f}% <<<")

    return results


def run_official_10k_validation(
    model: nn.Module,
    val_10k_manifest_path: Path,
    train_records: list[dict],
    device: str = "cuda",
) -> dict[str, float]:
    """Runs clean validation evaluation over the full 10,000-image official validation set."""
    print(f"\n[Val 10k] Loading full 10,000-image official validation manifest...")
    val_records = [json.loads(l) for l in val_10k_manifest_path.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    total_imgs = len(val_records)
    print(f"[Val 10k] Loaded {total_imgs:,} validation records. Starting vectorized inference...")

    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)

    train_counts = np.zeros(num_classes, dtype=int)
    for r in train_records[:500]:
        for c in r.get("labels", []):
            if c < num_classes:
                train_counts[c] += 1

    val_counts = np.zeros(num_classes, dtype=int)
    for r in val_records:
        for c in r.get("labels", []):
            if c < num_classes:
                val_counts[c] += 1

    predictions = {}
    ground_truths = {}
    start_t = time.time()

    with torch.no_grad():
        for idx, rec in enumerate(val_records):
            img_id = rec.get("scene_id", f"img_{idx}")
            with Image.open(rec["image_path"]) as img:
                image = img.convert("RGB")
            orig_w, orig_h = rec.get("width", image.width), rec.get("height", image.height)
            if image.size != (image_size[1], image_size[0]):
                image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

            inp = to_tensor(image).unsqueeze(0).to(device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                p = model(inp)

            decoded = decode_v2(
                p,
                image_size=image_size,
                max_detections=100,
                score_floor=0.01,
                nms_iou=0.60,
                pre_nms_candidates=1000,
            )
            predictions[img_id] = decoded[0]

            sx, sy = image_size[1] / orig_w, image_size[0] / orig_h
            gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(rec.get("labels", []), dtype=np.int64),
            )

            if (idx + 1) % 1000 == 0 or (idx + 1) == total_imgs:
                elapsed = time.time() - start_t
                fps = (idx + 1) / max(elapsed, 0.01)
                print(f"  [Val 10k: {idx+1:05d}/{total_imgs}] {fps:.1f} img/s | Elapsed: {elapsed:.1f}s")

    infer_time = time.time() - start_t
    print(f"[Val 10k] Computing official COCO AP across {len(ground_truths):,} ground truth images...")
    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    map50 = res["map50"] * 100.0
    map50_95 = res["map50_95"] * 100.0

    print(f"\n[Val 10k Result] Official 10k Val mAP50: {map50:.2f}% | mAP50:95: {map50_95:.2f}% (Total time: {infer_time:.1f}s)")
    return {
        "val_images": total_imgs,
        "clean_map50": map50,
        "clean_map50_95": map50_95,
        "inference_seconds": infer_time,
    }


# ==============================================================================
# Phase 2A: Controlled 2,048-image 10-epoch A/B Test
# ==============================================================================

def train_ab_2k_phase(
    jepa_encoder_path: Path,
    train_records: list[dict],
    val_records: list[dict],
    epochs: int = 10,
    batch_size: int = 32,
    backbone_lr: float = 2e-5,
    head_lr: float = 2e-4,
    workers: int = 4,
    device: str = "cuda",
    output_dir: Path = ROOT / "runs/rtx5090-v2-jepa-2k-ab",
) -> Path:
    """Trains FoundationViTDetectorV2 initialized with JEPA student weights on 2k pilot set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("PHASE 2A: CONTROLLED 2,048-IMAGE 10-EPOCH A/B TEST (JEPA vs SUPERVISED)")
    print("=" * 80)

    shapes = [(48, 80), (24, 40), (12, 20)]
    image_size = (384, 640)

    # Instantiate detector with JEPA student weights
    model = FoundationViTDetectorV2(
        model_name="vit_small_patch16_384",
        pretrained=False,
    ).to(device)

    print(f"[Phase 2A] Loading JEPA-pretrained student encoder from {jepa_encoder_path}...")
    student_sd = torch.load(jepa_encoder_path, map_location=device)
    model.backbone.load_state_dict(student_sd)
    print("[Phase 2A] JEPA student weights loaded successfully into detector backbone!")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [*model.adapter.parameters(), *model.heads.parameters()]

    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": head_lr, "weight_decay": 1e-4},
    ])

    train_dataset = BDD100KDetectorDataset(train_records, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=lambda b: detector_collate_fn(b, shapes, image_size),
    )

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_steps)

    best_map50 = 0.0
    history = []
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_obj, ep_cls, ep_box = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step, (images, targets) in enumerate(train_loader):
            global_step = (epoch - 1) * steps_per_epoch + step + 1
            if global_step < warmup_steps:
                lr_mult = max(0.01, float(global_step) / max(1, warmup_steps))
            else:
                progress = float(global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_mult = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            optimizer.param_groups[0]["lr"] = backbone_lr * lr_mult
            optimizer.param_groups[1]["lr"] = head_lr * lr_mult

            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss_v2(
                    preds, targets, negative_weight=0.05, box_weight=2.0, cls_weight=1.0, obj_weight=1.0
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([*backbone_params, *head_params], max_norm=5.0)
            optimizer.step()

            ep_loss += metrics["total_loss"]
            ep_obj += metrics["obj_loss"]
            ep_cls += metrics["cls_loss"]
            ep_box += metrics["box_loss"]

        ep_time = time.time() - ep_start
        avg_loss = ep_loss / steps_per_epoch

        val_map50, val_map50_95 = evaluate_detector_quick(
            model, val_records, train_records, device=device, max_images=256
        )
        print(
            f"  [Ep {epoch:02d}/{epochs:02d} | {ep_time:.1f}s] Loss: {avg_loss:.4f} | "
            f"Val mAP50: {val_map50:.2f}% | mAP50:95: {val_map50_95:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_map50": val_map50,
            "val_map50_95": val_map50_95,
            "epoch_seconds": ep_time,
        })

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_map50": val_map50,
                "val_map50_95": val_map50_95,
            }, output_dir / "best_checkpoint.pt")

    print(f"\n[Phase 2A] 2k Pilot training completed in {time.time() - start_t:.1f}s! Best mAP50: {best_map50:.2f}%")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Comprehensive Multi-Seed Adverse Weather Benchmark (Sev 2 & Sev 3)
    print("\n[Phase 2A Benchmark] Running Comprehensive Multi-Seed Adverse Weather Benchmark (Sev 2 & Sev 3)...")
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    benchmark_report = run_comprehensive_weather_benchmark(
        model,
        val_records,
        train_records,
        device=device,
        max_images=256,
        severities=(2, 3),
        seeds=(42, 100, 2024),
    )

    report_path = output_dir / "comprehensive_benchmark_report.json"
    report_path.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    print(f"[Phase 2A] Benchmark report saved to {report_path}")

    # A/B Comparison Printout
    pure_supervised_clean = 16.16
    pure_supervised_s2 = 13.34
    jepa_clean = benchmark_report["clean"]["map50_mean"]
    jepa_s2 = benchmark_report["mean_adverse_s2"]["map50_mean"]
    jepa_s3 = benchmark_report["mean_adverse_s3"]["map50_mean"]

    print("\n" + "=" * 80)
    print("CONTROLLED 2,048-IMAGE A/B COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Configuration':<35} | {'Clean mAP50':<12} | {'Adverse S=2 mAP50':<18} | {'Adverse S=3 mAP50':<18}")
    print("-" * 80)
    print(f"{'Pure Supervised ViT-S Baseline':<35} | {pure_supervised_clean:.2f}%{' '*6} | {pure_supervised_s2:.2f}%{' '*12} | {'--':<18}")
    print(f"{'Natural-Noise JEPA ViT-S (Ours)':<35} | {jepa_clean:.2f}%{' '*6} | {jepa_s2:.2f}%{' '*12} | {jepa_s3:.2f}%{' '*12}")
    print(f"{'JEPA Delta':<35} | {jepa_clean - pure_supervised_clean:+.2f}%p{' '*5} | {jepa_s2 - pure_supervised_s2:+.2f}%p{' '*11} | {'--':<18}")
    print("=" * 80 + "\n")

    return output_dir / "best_checkpoint.pt"


# ==============================================================================
# Phase 2B: Full-Scale 70,000-image Detector Training & Official 10k Val
# ==============================================================================

def train_full_70k_phase(
    jepa_encoder_path: Path,
    train_70k_records: list[dict],
    val_records: list[dict],
    val_10k_manifest_path: Path,
    epochs: int = 12,
    batch_size: int = 32,
    backbone_lr: float = 2e-5,
    head_lr: float = 2e-4,
    workers: int = 4,
    device: str = "cuda",
    output_dir: Path = ROOT / "runs/rtx5090-v2-jepa-full70k",
) -> Path:
    """Phase 2B: Full-Scale Detector Training on 70,000 images with JEPA Backbone."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80)
    print("PHASE 2B: FULL-SCALE 70,000-IMAGE DETECTOR TRAINING WITH JEPA BACKBONE")
    print(f"Dataset: {len(train_70k_records):,} images | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 80)

    shapes = [(48, 80), (24, 40), (12, 20)]
    image_size = (384, 640)

    # Initialize detector
    model = FoundationViTDetectorV2(
        model_name="vit_small_patch16_384",
        pretrained=False,
    ).to(device)

    print(f"[Phase 2B] Loading JEPA student encoder weights from {jepa_encoder_path}...")
    student_sd = torch.load(jepa_encoder_path, map_location=device)
    model.backbone.load_state_dict(student_sd)
    print("[Phase 2B] JEPA student weights transferred successfully into full-scale detector!")

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [*model.adapter.parameters(), *model.heads.parameters()]

    optimizer = AdamW([
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": head_lr, "weight_decay": 1e-4},
    ])

    train_dataset = BDD100KDetectorDataset(train_70k_records, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=2 if workers > 0 else None,
        collate_fn=lambda b: detector_collate_fn(b, shapes, image_size),
    )

    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_steps)

    best_map50 = 0.0
    history = []
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss, ep_obj, ep_cls, ep_box = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step, (images, targets) in enumerate(train_loader):
            global_step = (epoch - 1) * steps_per_epoch + step + 1
            if global_step < warmup_steps:
                lr_mult = max(0.01, float(global_step) / max(1, warmup_steps))
            else:
                progress = float(global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                lr_mult = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            optimizer.param_groups[0]["lr"] = backbone_lr * lr_mult
            optimizer.param_groups[1]["lr"] = head_lr * lr_mult

            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss_v2(
                    preds, targets, negative_weight=0.05, box_weight=2.0, cls_weight=1.0, obj_weight=1.0
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([*backbone_params, *head_params], max_norm=5.0)
            optimizer.step()

            ep_loss += metrics["total_loss"]
            ep_obj += metrics["obj_loss"]
            ep_cls += metrics["cls_loss"]
            ep_box += metrics["box_loss"]

            if (step + 1) % 100 == 0 or (step + 1) == steps_per_epoch:
                elapsed = time.time() - ep_start
                rate = (step + 1) / max(elapsed, 0.01)
                img_rate = rate * batch_size
                print(
                    f"  [Ep {epoch:02d}/{epochs:02d} | Step {step+1:04d}/{steps_per_epoch}] "
                    f"Loss: {metrics['total_loss']:.3f} (Obj: {metrics['obj_loss']:.3f}, "
                    f"Cls: {metrics['cls_loss']:.3f}, Box: {metrics['box_loss']:.3f}) | "
                    f"{rate:.2f} it/s ({img_rate:.0f} img/s) | lr: {optimizer.param_groups[1]['lr']:.2e}"
                )

        ep_time = time.time() - ep_start
        avg_loss = ep_loss / steps_per_epoch

        val_map50, val_map50_95 = evaluate_detector_quick(
            model, val_records, train_70k_records, device=device, max_images=256
        )
        print(
            f"  >>> [Val @ Epoch {epoch:02d} | {ep_time:.1f}s] "
            f"mAP50: {val_map50:.2f}% | mAP50:95: {val_map50_95:.2f}% (Loss: {avg_loss:.4f}) <<<"
        )

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_map50": val_map50,
            "val_map50_95": val_map50_95,
            "epoch_seconds": ep_time,
        })

        if val_map50 > best_map50:
            best_map50 = val_map50
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_map50": val_map50,
                "val_map50_95": val_map50_95,
            }, output_dir / "best_checkpoint.pt")
            print(f"  *** New Best Checkpoint saved to {output_dir / 'best_checkpoint.pt'} (mAP50: {best_map50:.2f}%) ***")

    print(f"\n[Phase 2B] Full-Scale Training Completed: {epochs} Epochs in {(time.time() - start_t)/60:.2f} min!")
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Rigorous Benchmark on best checkpoint
    print("\n[Phase 2B Benchmark] Running Rigorous Multi-Seed Adverse Weather Benchmark...")
    best_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    benchmark_report = run_comprehensive_weather_benchmark(
        model,
        val_records,
        train_70k_records,
        device=device,
        max_images=256,
        severities=(2, 3),
        seeds=(42, 100, 2024),
    )
    (output_dir / "rigorous_evaluation_report.json").write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")

    # Official 10,000-Image Full Validation Evaluation
    print("\n[Phase 2B Benchmark] Running Official 10,000-Image Full Validation Benchmark...")
    eval_10k_dir = output_dir / "eval_10k"
    eval_10k_dir.mkdir(parents=True, exist_ok=True)
    summary_10k = run_official_10k_validation(model, val_10k_manifest_path, train_70k_records, device=device)
    (eval_10k_dir / "summary_10k.json").write_text(json.dumps(summary_10k, indent=2), encoding="utf-8")

    return output_dir / "best_checkpoint.pt"


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Natural-Noise I-JEPA Pretrain & Downstream Detection on RTX 5090")
    parser.add_argument("--phase", type=str, default="pretrain_and_ab",
                        choices=["all", "pretrain", "ab_2k", "full_70k", "pretrain_and_ab"],
                        help="Execution mode: pretrain_and_ab runs Phase 1 & Phase 2A; full_70k runs Phase 2B; all runs all.")
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch-size", type=int, default=64)
    parser.add_argument("--ab-epochs", type=int, default=10)
    parser.add_argument("--ab-batch-size", type=int, default=32)
    parser.add_argument("--full-epochs", type=int, default=12)
    parser.add_argument("--full-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jepa-encoder-path", type=str, default=str(ROOT / "runs/rtx5090-jepa-pretrained/jepa_student_encoder.pt"))
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 80)
    print(f"AUTODRIVE I-JEPA PIPELINE ON {args.device.upper()} ({gpu_name})")
    print(f"Target Phase: {args.phase}")
    print("=" * 80)

    # Manifest paths
    train_70k_manifest = ROOT / "data/manifests/bdd100k_train_70k_local.jsonl"
    train_2k_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    val_10k_manifest = ROOT / "data/manifests/bdd100k_val_10k_local.jsonl"

    train_70k_records = [json.loads(l) for l in train_70k_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    train_2k_records = [json.loads(l) for l in train_2k_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    print(f"[Data] Loaded 70k train: {len(train_70k_records):,} | 2k train: {len(train_2k_records):,} | 256 val: {len(val_records):,}")

    jepa_path = Path(args.jepa_encoder_path)

    # 1. Phase 1: Natural-Noise I-JEPA Pretraining
    if args.phase in ["all", "pretrain", "pretrain_and_ab"]:
        jepa_path = train_jepa_pretrain_phase(
            train_records=train_70k_records,
            val_records=val_records,
            epochs=args.pretrain_epochs,
            batch_size=args.pretrain_batch_size,
            workers=args.workers,
            device=args.device,
        )

    # 2. Phase 2A: Controlled 2,048-image 10-epoch A/B Test
    if args.phase in ["all", "ab_2k", "pretrain_and_ab"]:
        train_ab_2k_phase(
            jepa_encoder_path=jepa_path,
            train_records=train_2k_records,
            val_records=val_records,
            epochs=args.ab_epochs,
            batch_size=args.ab_batch_size,
            workers=args.workers,
            device=args.device,
        )

    # 3. Phase 2B: Full-Scale 70,000-image Detector Training & Official 10k Val
    if args.phase in ["all", "full_70k"]:
        train_full_70k_phase(
            jepa_encoder_path=jepa_path,
            train_70k_records=train_70k_records,
            val_records=val_records,
            val_10k_manifest_path=val_10k_manifest,
            epochs=args.full_epochs,
            batch_size=args.full_batch_size,
            workers=args.workers,
            device=args.device,
        )

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
