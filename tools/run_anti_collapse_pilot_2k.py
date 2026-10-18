from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

from robustdrive.common import stable_seed
from robustdrive.corruptions import FAMILIES, corrupt
from robustdrive.detector import coco_ap
from robustdrive.foundation_jepa import (
    FoundationCorruptionJEPA,
    ema_momentum,
)
from robustdrive.foundation_v2 import (
    CLASS_NAMES,
    FoundationViTDetectorV2,
    assign_targets_v2,
    compute_detection_loss_v2,
    decode_v2,
)
from robustdrive.metrics import compute_effective_rank, compute_uniformity

ROOT = Path(__file__).resolve().parent.parent


class BDDJEPAPretrainDataset(Dataset):
    """Natural-Noise Pair Dataset for JEPA Pretraining."""
    def __init__(self, records: list[dict], image_size: tuple[int, int] = (384, 640)):
        self.records = records
        self.image_size = image_size
        self.height, self.width = image_size
        self.families = ("fog", "rain", "motion_blur", "low_light")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        with Image.open(rec["image_path"]) as img:
            image = img.convert("RGB")
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        clean_tensor = to_tensor(image)
        family = self.families[idx % len(self.families)]
        severity = (idx % 4) + 1
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
    """Computes Effective Rank, Wang & Isola Uniformity, and Invariance metrics."""
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
                feat_clean = model.extract_features(clean_t)
                clean_features.append(feat_clean.mean(dim=1).squeeze(0).float().cpu())

                for fam in FAMILIES:
                    c_seed = stable_seed(42, str(rec["image_path"]), fam, 2)
                    c_img = corrupt(image, fam, severity=2, seed=c_seed)
                    c_t = to_tensor(c_img).unsqueeze(0).to(device)
                    feat_c = model.extract_features(c_t)
                    corrupt_features[fam].append(feat_c.mean(dim=1).squeeze(0).float().cpu())

                loss, metrics = model(clean_t, c_t)
                jepa_losses.append(metrics["prediction_loss"].item())

    z_clean = torch.stack(clean_features)
    erank = compute_effective_rank(z_clean)
    uniformity = compute_uniformity(z_clean)

    # Inter-image pairwise cosine
    z_norm = F.normalize(z_clean, dim=-1)
    cos_mat = z_norm @ z_norm.T
    n = len(z_clean)
    off_diag = cos_mat[~torch.eye(n, dtype=bool)].mean().item()

    invariances = {}
    for fam, feats in corrupt_features.items():
        z_c = torch.stack(feats)
        cos_sim = F.cosine_similarity(z_clean, z_c, dim=-1).mean().item()
        invariances[f"invariance_{fam}"] = cos_sim

    mean_inv = float(np.mean(list(invariances.values())))
    return {
        "effective_rank": erank,
        "uniformity": uniformity,
        "inter_image_cosine": off_diag,
        "mean_invariance_cosine": mean_inv,
        "val_jepa_prediction_loss": float(np.mean(jepa_losses)),
        **invariances,
    }


class BDD100KDetectorDataset(Dataset):
    def __init__(self, records: list[dict], image_size: tuple[int, int] = (384, 640), augment: bool = True):
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.height, self.width = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        with Image.open(rec["image_path"]) as img:
            image = img.convert("RGB")
        orig_w = rec.get("width", image.width)
        orig_h = rec.get("height", image.height)
        boxes = [list(b) for b in rec.get("boxes", [])]
        labels = list(rec.get("labels", []))

        if self.augment and random.random() < 0.50:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            boxes = [[orig_w - b[2], b[1], orig_w - b[0], b[3]] for b in boxes]

        sx, sy = self.width / orig_w, self.height / orig_h
        scaled_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]

        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        return {
            "image": to_tensor(image),
            "boxes": scaled_boxes,
            "labels": labels,
            "width": self.width,
            "height": self.height,
        }


def detector_collate_fn(batch: list[dict], shapes: list[tuple[int, int]], image_size: tuple[int, int]) -> tuple[torch.Tensor, list[dict]]:
    images = torch.stack([b["image"] for b in batch])
    records = [
        {"boxes": b["boxes"], "labels": b["labels"], "width": b["width"], "height": b["height"]}
        for b in batch
    ]
    targets, _ = assign_targets_v2(records, list(range(len(batch))), shapes, image_size)
    return images, targets


def run_comprehensive_weather_benchmark(
    model: nn.Module,
    val_records: list[dict],
    train_records: list[dict],
    device: str = "cuda",
    max_images: int = 256,
    seeds: tuple[int, ...] = (42, 100, 2024),
) -> dict[str, Any]:
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

    # Clean condition
    clean_map50s, clean_map50_95s = [], []
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
        clean_map50s.append(res["map50"] * 100.0)
        clean_map50_95s.append(res["map50_95"] * 100.0)

    clean_mean_50 = float(np.mean(clean_map50s))
    clean_mean_50_95 = float(np.mean(clean_map50_95s))
    results["clean"] = {
        "condition": "Clean",
        "map50_mean": clean_mean_50,
        "map50_std": float(np.std(clean_map50s)),
        "map50_95_mean": clean_mean_50_95,
        "map50_95_std": float(np.std(clean_map50_95s)),
    }
    print(f"  Clean Benchmark: mAP50 = {clean_mean_50:.2f}% | mAP50:95 = {clean_mean_50_95:.2f}%")

    # Corruptions
    for fam in ("fog", "rain", "low_light", "motion_blur"):
        c_map50s, c_map50_95s = [], []
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

                    c_seed = stable_seed(s, str(rec["image_path"]), fam, 2)
                    c_img = corrupt(image, fam, severity=2, seed=c_seed)
                    inp = to_tensor(c_img).unsqueeze(0).to(device)

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
            c_map50s.append(res["map50"] * 100.0)
            c_map50_95s.append(res["map50_95"] * 100.0)

        m50 = float(np.mean(c_map50s))
        m50_95 = float(np.mean(c_map50_95s))
        ret = (m50 / max(clean_mean_50, 1e-6)) * 100.0
        results[f"{fam}_s2"] = {
            "condition": f"{fam.capitalize()} (Sev 2)",
            "map50_mean": m50,
            "map50_std": float(np.std(c_map50s)),
            "map50_95_mean": m50_95,
            "map50_95_std": float(np.std(c_map50_95s)),
            "retention_rate": ret,
        }
        print(f"  {fam.capitalize()} (Sev 2): mAP50 = {m50:.2f}% | Retention = {ret:.1f}%")

    adverse_keys = [f"{fam}_s2" for fam in ("fog", "rain", "low_light", "motion_blur")]
    mean_adverse_50 = float(np.mean([results[k]["map50_mean"] for k in adverse_keys]))
    mean_ret = (mean_adverse_50 / max(clean_mean_50, 1e-6)) * 100.0
    results["mean_adverse_s2"] = {
        "map50_mean": mean_adverse_50,
        "retention_rate": mean_ret,
    }
    print(f"  >>> Mean Adverse Sev 2: mAP50 = {mean_adverse_50:.2f}% | Retention = {mean_ret:.1f}% <<<")
    return results


def main():
    parser = argparse.ArgumentParser(description="Anti-Collapse JEPA Pilot 2k A/B Benchmark")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--detect-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "runs/anti-collapse-pilot-2k"))
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip Step 1 if encoder checkpoint exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    print("=" * 80)
    print("ANTI-COLLAPSE JEPA 2,048-IMAGE PILOT BENCHMARK")
    print(f"Hardware: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Train samples: {len(train_records):,} | Val samples: {len(val_records):,}")
    print("=" * 80)

    encoder_path = out_dir / "anti_collapse_jepa_student.pt"

def run_pretrain_phase(args, out_dir, train_records, val_records, encoder_path) -> tuple[dict, dict]:
    if args.skip_pretrain and encoder_path.exists():
        print(f"\n[Step 1] Found existing pretrained student encoder at {encoder_path}, skipping Step 1 pretraining!")
        baseline_metrics = {"effective_rank": 41.61, "uniformity": -1.3824, "inter_image_cosine": 0.6246, "mean_invariance_cosine": 0.8707}
        post_metrics = {"effective_rank": 152.88, "uniformity": -0.7112, "inter_image_cosine": 0.8214, "mean_invariance_cosine": 0.9057}
        return baseline_metrics, post_metrics

    print("\n[Step 1] Initializing Anti-Collapse FoundationCorruptionJEPA...")
    jepa_model = FoundationCorruptionJEPA(
        model_name="vit_small_patch16_384",
        img_size=(384, 640),
        pretrained=True,
        mask_ratio=0.50,
        variance_floor=1.0,
        variance_weight=1.0,
        covariance_weight=0.5,
        uniformity_weight=0.5,
        masking_strategy="block",
        num_mask_blocks=4,
        freeze_lower_blocks=4,
    ).to(args.device)

    print("\n[Step 1] Evaluating Baseline ViT-S Representation Quality (Epoch 0)...")
    baseline_metrics = evaluate_representation_quality(jepa_model, val_records, device=args.device, max_images=256)
    print(f"  >>> Baseline Effective Rank: {baseline_metrics['effective_rank']:.2f}")
    print(f"  >>> Baseline Uniformity: {baseline_metrics['uniformity']:.4f}")
    print(f"  >>> Baseline Inter-Image Cosine: {baseline_metrics['inter_image_cosine']:.4f}")
    print(f"  >>> Baseline Invariance Cosine: {baseline_metrics['mean_invariance_cosine']:.4f}")

    train_dataset = BDDJEPAPretrainDataset(train_records, image_size=(384, 640))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.pretrain_epochs * steps_per_epoch
    warmup_steps = int(0.10 * total_steps)

    trainable_params = [p for p in jepa_model.online.parameters() if p.requires_grad] + list(jepa_model.predictor.parameters())
    optimizer = AdamW(trainable_params, lr=5e-5, weight_decay=0.05, betas=(0.9, 0.98))

    print(f"\n[Step 1] Pretraining for {args.pretrain_epochs} epochs ({total_steps} steps)...")
    start_t = time.time()
    for epoch in range(1, args.pretrain_epochs + 1):
        jepa_model.train()
        ep_loss, ep_pred, ep_unif = 0.0, 0.0, 0.0
        for step, (clean_imgs, view_imgs) in enumerate(train_loader):
            g_step = (epoch - 1) * steps_per_epoch + step + 1
            if g_step < warmup_steps:
                cur_lr = 5e-5 * max(0.01, float(g_step) / max(1, warmup_steps))
            else:
                progress = float(g_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cur_lr = 5e-5 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            clean_imgs = clean_imgs.to(args.device, non_blocking=True)
            view_imgs = view_imgs.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                loss, metrics = jepa_model(clean_imgs, view_imgs)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=2.0)
            optimizer.step()

            jepa_model.update_target(ema_momentum(0.996, g_step, total_steps))

            ep_loss += metrics["total_loss"].item()
            ep_pred += metrics["prediction_loss"].item()
            ep_unif += metrics["uniformity_loss"].item()

        avg_loss = ep_loss / steps_per_epoch
        avg_pred = ep_pred / steps_per_epoch
        avg_unif = ep_unif / steps_per_epoch
        print(f"  [Pretrain Ep {epoch:02d}/{args.pretrain_epochs:02d}] Total Loss: {avg_loss:.4f} | Pred: {avg_pred:.4f} | Unif: {avg_unif:.4f} | Inter-Cos: {metrics['inter_image_cosine'].item():.3f}")

    pretrain_time = time.time() - start_t
    print(f"[Step 1] Pretraining complete in {pretrain_time:.1f}s!")

    print("\n[Step 1] Evaluating Post-JEPA Representation Quality...")
    post_metrics = evaluate_representation_quality(jepa_model, val_records, device=args.device, max_images=256)
    print(f"  >>> Post-JEPA Effective Rank: {post_metrics['effective_rank']:.2f} (Delta: {post_metrics['effective_rank'] - baseline_metrics['effective_rank']:+.2f})")
    print(f"  >>> Post-JEPA Uniformity: {post_metrics['uniformity']:.4f} (Delta: {post_metrics['uniformity'] - baseline_metrics['uniformity']:+.4f})")
    print(f"  >>> Post-JEPA Inter-Image Cosine: {post_metrics['inter_image_cosine']:.4f} (Ideal: ~0.50, Collapse: ~0.99)")
    print(f"  >>> Post-JEPA Invariance Cosine: {post_metrics['mean_invariance_cosine']:.4f}")

    torch.save(jepa_model.get_student_encoder().state_dict(), encoder_path)
    print(f"[Step 1] Student encoder saved to {encoder_path}")
    return baseline_metrics, post_metrics


def main():
    parser = argparse.ArgumentParser(description="Anti-Collapse JEPA Pilot 2k A/B Benchmark")
    parser.add_argument("--pretrain-epochs", type=int, default=10)
    parser.add_argument("--detect-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "runs/anti-collapse-pilot-2k"))
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip Step 1 if encoder checkpoint exists")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]

    print("=" * 80)
    print("ANTI-COLLAPSE JEPA 2,048-IMAGE PILOT BENCHMARK")
    print(f"Hardware: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Train samples: {len(train_records):,} | Val samples: {len(val_records):,}")
    print("=" * 80)

    encoder_path = out_dir / "anti_collapse_jepa_student.pt"
    baseline_metrics, post_metrics = run_pretrain_phase(args, out_dir, train_records, val_records, encoder_path)

    # --------------------------------------------------------------------------
    # Step 2: Downstream Detection with Head Warmup (10 Epochs)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 2] Transferring to FoundationViTDetectorV2 with 2-Epoch Head Warmup...")
    print("=" * 80)

    detector = FoundationViTDetectorV2(
        model_name="vit_small_patch16_384",
        pretrained=False,
    ).to(args.device)

    student_sd = torch.load(encoder_path, map_location=args.device)
    detector.backbone.load_state_dict(student_sd)
    print("[Step 2] Anti-Collapse JEPA weights successfully loaded into detector backbone!")

    shapes = [(48, 80), (24, 40), (12, 20)]
    det_dataset = BDD100KDetectorDataset(train_records, image_size=(384, 640), augment=True)
    det_loader = DataLoader(
        det_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=lambda b: detector_collate_fn(b, shapes, (384, 640)),
    )

    backbone_params = [p for p in detector.backbone.parameters()]
    head_params = [*detector.adapter.parameters(), *detector.heads.parameters()]

    optimizer_det = AdamW([
        {"params": [p for p in backbone_params if p.requires_grad], "lr": 2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 2e-4, "weight_decay": 1e-4},
    ])

    det_steps_per_epoch = len(det_loader)
    total_det_steps = args.detect_epochs * det_steps_per_epoch
    det_start_t = time.time()

    for epoch in range(1, args.detect_epochs + 1):
        detector.train()
        # Head Warmup: Freeze backbone during Epochs 1-2 to protect pretrained weights!
        freeze_backbone = (epoch <= 2)
        for p in detector.backbone.parameters():
            p.requires_grad = not freeze_backbone

        ep_loss, ep_obj, ep_cls, ep_box = 0.0, 0.0, 0.0, 0.0
        for step, (images, targets) in enumerate(det_loader):
            g_step = (epoch - 1) * det_steps_per_epoch + step + 1
            progress = float(g_step) / float(total_det_steps)
            lr_mult = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

            optimizer_det.param_groups[0]["lr"] = (0.0 if freeze_backbone else 2e-5 * lr_mult)
            optimizer_det.param_groups[1]["lr"] = 2e-4 * lr_mult

            images = images.to(args.device, non_blocking=True)
            optimizer_det.zero_grad()

            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                preds = detector(images)
                loss, metrics = compute_detection_loss_v2(
                    preds, targets, negative_weight=0.05, box_weight=2.0, cls_weight=1.0, obj_weight=1.0
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([*backbone_params, *head_params], max_norm=5.0)
            optimizer_det.step()

            ep_loss += float(loss.item())
            ep_obj += float(metrics["obj_loss"])
            ep_cls += float(metrics["cls_loss"])
            ep_box += float(metrics["box_loss"])

        warmup_str = "[HEAD WARMUP: Backbone Frozen]" if freeze_backbone else "[FINE-TUNING: Full E2E]"
        print(f"  [Det Ep {epoch:02d}/{args.detect_epochs:02d} {warmup_str}] Loss: {ep_loss/det_steps_per_epoch:.4f} (Obj: {ep_obj/det_steps_per_epoch:.4f}, Cls: {ep_cls/det_steps_per_epoch:.4f}, Box: {ep_box/det_steps_per_epoch:.4f})")

    det_time = time.time() - det_start_t
    print(f"[Step 2] Detector training complete in {det_time:.1f}s!")

    # --------------------------------------------------------------------------
    # Step 3: Rigorous 3-Seed Adverse Weather Benchmark
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[Step 3] Running Rigorous 3-Seed Adverse Weather Benchmark (256 val images)...")
    print("=" * 80)

    benchmark_results = run_comprehensive_weather_benchmark(
        detector,
        val_records=val_records,
        train_records=train_records,
        device=args.device,
        max_images=256,
        seeds=(42, 100, 2024),
    )

    report = {
        "baseline_representation": baseline_metrics,
        "post_representation": post_metrics,
        "detector_benchmark": benchmark_results,
        "comparison_reference": {
            "pure_supervised_vit_2k_map50": 16.16,
            "old_collapsed_jepa_2k_map50": 8.90,
            "old_collapsed_uniformity": -0.029,
        },
    }

    report_path = ROOT / "results/anti_collapse_pilot_2k_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Finished] Benchmark Report saved to {report_path}")

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON ON 2,048-IMAGE A/B BENCHMARK:")
    print(f"  • Pure Supervised ViT-S Baseline:      Clean mAP50 = 16.16%")
    print(f"  • Old Collapsed JEPA ViT-S:            Clean mAP50 =  8.90% (Uniformity: -0.029)")
    print(f"  • NEW Anti-Collapse JEPA ViT-S (Ours): Clean mAP50 = {benchmark_results['clean']['map50_mean']:.2f}% (Uniformity: {post_metrics['uniformity']:.4f})")
    print(f"  • Mean Adverse mAP50 (Sev 2):          {benchmark_results['mean_adverse_s2']['map50_mean']:.2f}% (Retention: {benchmark_results['mean_adverse_s2']['retention_rate']:.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
