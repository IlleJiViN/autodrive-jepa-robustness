"""
Full-Scale RTX 5090 High-Performance Training & Evaluation for Foundation ViT Detector v2.
Features:
- FoundationViTDetectorV2 with ViTDet Top-Down FPN and Decoupled Single-Slot Heads
- Multi-Worker PyTorch DataLoader (num_workers=8, pin_memory=True, persistent_workers=True)
- Random Horizontal Flip + Background Multi-Process Weather Augmentation (Fog, Rain, Blur, Low-Light)
- Linear LR Warmup (first 10% of total steps) + Cosine Annealing LR Schedule
- Vectorized GPU Batched NMS (decode_v2) and Complete IoU (CIoU) Loss
- Automatic 256-image Validation Evaluation every epoch
- Rigorous Post-Training / Standalone Multi-Seed Evaluation (Clean, Fog, Rain, Low-Light, Motion Blur)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.common import stable_seed
from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.corruptions import corrupt
from robustdrive.detector import coco_ap
from robustdrive.foundation_v2 import (
    STRIDES,
    FoundationViTDetectorV2,
    assign_targets_v2,
    compute_detection_loss_v2,
    decode_v2,
)


class BDD100KFullScaleDataset(Dataset):
    """
    High-throughput BDD100K Dataset with Geometric and Corruptive Augmentations:
    - Random Horizontal Flip (50% probability)
    - Dynamic Weather Corruptions (30% probability across fog, rain, motion blur, low light)
    - Fast resizing to rectangular resolution (384, 640)
    """
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
        if self.augment and np.random.rand() < 0.50:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            flipped_boxes = []
            for b in boxes:
                x1, y1, x2, y2 = b
                flipped_boxes.append([orig_w - x2, y1, orig_w - x1, y2])
            boxes = flipped_boxes

        # Weather Augmentation: Corruptions (30%)
        if self.augment and np.random.rand() < 0.30:
            c_fam = np.random.choice(["fog", "rain", "motion_blur", "low_light"])
            c_sev = int(np.random.randint(1, 4))
            c_seed = int(np.random.randint(0, 1000000))
            image = corrupt(image, c_fam, c_sev, seed=c_seed)

        # Resizing to rectangular resolution
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


def collate_fn_v2(batch: list[dict], shapes: list[tuple[int, int]], image_size: tuple[int, int]):
    images = torch.stack([item["image"] for item in batch])
    records = [
        {"boxes": item["boxes"], "labels": item["labels"], "width": item["width"], "height": item["height"]}
        for item in batch
    ]
    targets, _ = assign_targets_v2(records, list(range(len(batch))), shapes, image_size)
    return images, targets


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int, min_ratio: float = 0.01) -> float:
    """Computes Linear Warmup + Cosine Annealing learning rate multiplier."""
    if step < warmup_steps:
        # Linear warmup from min_ratio to 1.0
        return max(min_ratio, float(step) / max(1, warmup_steps))
    else:
        # Cosine decay from 1.0 to min_ratio
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine


def evaluate_val_quick_v2(
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
                image = img.convert("RGB").resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

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

            orig_w, orig_h = rec.get("width", 1280), rec.get("height", 720)
            sx, sy = image_size[1] / orig_w, image_size[0] / orig_h
            gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
            gt_labels = [int(c) for c in rec.get("labels", [])]

            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    return res["map50"] * 100.0, res["map50_95"] * 100.0


def run_rigorous_evaluation_v2(
    model: nn.Module,
    val_records: list[dict],
    train_records: list[dict],
    device: str = "cuda",
    max_images: int = 256,
    seeds: Sequence[int] = (42, 100, 2024),
) -> dict[str, Any]:
    """
    Multi-Seed Rigorous Benchmark across 5 conditions:
    Clean (sev 0), Fog (sev 2), Rain (sev 2), Low-Light (sev 2), Motion Blur (sev 2).
    """
    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)
    conditions = [
        ("clean", 0),
        ("fog", 2),
        ("rain", 2),
        ("low_light", 2),
        ("motion_blur", 2),
    ]

    train_counts = np.zeros(num_classes, dtype=int)
    for r in train_records[:500]:
        for c in r.get("labels", []):
            if c < num_classes:
                train_counts[c] += 1

    eval_recs = val_records[:max_images]
    val_counts = np.zeros(num_classes, dtype=int)
    for r in eval_recs:
        for c in r.get("labels", []):
            if c < num_classes:
                val_counts[c] += 1

    print("\n" + "=" * 80)
    print(f"RIGOROUS MULTI-SEED EVALUATION ({len(seeds)} Seeds x {len(conditions)} Conditions, N={len(eval_recs)})")
    print("=" * 80)

    results_table = []
    condition_summaries = {}

    for family, severity in conditions:
        seed_map50 = []
        seed_map50_95 = []

        for seed in seeds:
            predictions = {}
            ground_truths = {}

            with torch.no_grad():
                for idx, rec in enumerate(eval_recs):
                    img_id = rec.get("scene_id", f"img_{idx}")
                    img_path = rec["image_path"]
                    with Image.open(img_path) as img:
                        image = img.convert("RGB")

                    if family != "clean" and severity > 0:
                        c_seed = stable_seed(seed, str(img_path), family, severity)
                        image = corrupt(image, family, severity, c_seed)

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
            seed_map50.append(res["map50"] * 100.0)
            seed_map50_95.append(res["map50_95"] * 100.0)

        mean_50 = float(np.mean(seed_map50))
        std_50 = float(np.std(seed_map50))
        mean_50_95 = float(np.mean(seed_map50_95))
        std_50_95 = float(np.std(seed_map50_95))

        cond_name = f"{family.capitalize()} (Sev {severity})" if severity > 0 else "Clean"
        condition_summaries[family] = {
            "condition": cond_name,
            "severity": severity,
            "map50_mean": mean_50,
            "map50_std": std_50,
            "map50_95_mean": mean_50_95,
            "map50_95_std": std_50_95,
            "per_seed_map50": seed_map50,
            "per_seed_map50_95": seed_map50_95,
        }
        results_table.append((cond_name, f"{mean_50:.2f} ± {std_50:.2f}%", f"{mean_50_95:.2f} ± {std_50_95:.2f}%"))
        print(f"  Condition: {cond_name:<20} | mAP50: {mean_50:.2f} ± {std_50:.2f}% | mAP50:95: {mean_50_95:.2f} ± {std_50_95:.2f}%")

    corrupt_keys = [c for c in condition_summaries if c != "clean"]
    mean_corrupt_50 = float(np.mean([condition_summaries[k]["map50_mean"] for k in corrupt_keys]))
    mean_corrupt_50_95 = float(np.mean([condition_summaries[k]["map50_95_mean"] for k in corrupt_keys]))
    clean_50 = condition_summaries["clean"]["map50_mean"]
    clean_50_95 = condition_summaries["clean"]["map50_95_mean"]

    print("-" * 80)
    print(f"  Clean Benchmark:       mAP50: {clean_50:.2f}% | mAP50:95: {clean_50_95:.2f}%")
    print(f"  Robustness (Mean Corrupt): mAP50: {mean_corrupt_50:.2f}% | mAP50:95: {mean_corrupt_50_95:.2f}%")
    print("=" * 80 + "\n")

    return {
        "conditions": condition_summaries,
        "clean_map50": clean_50,
        "clean_map50_95": clean_50_95,
        "robust_map50": mean_corrupt_50,
        "robust_map50_95": mean_corrupt_50_95,
        "seeds": list(seeds),
    }


def main():
    parser = argparse.ArgumentParser(description="Full-Scale RTX 5090 Foundation ViT Detector v2 Trainer")
    parser.add_argument("--train-manifest", type=str, default=None,
                        help="Path to training manifest. Defaults to bdd100k_train_70k_local.jsonl if present, fallback bdd100k_train_local.jsonl")
    parser.add_argument("--val-manifest", type=str, default=None,
                        help="Path to validation manifest. Defaults to bdd100k_val_local.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-pct", type=float, default=0.10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "runs/rtx5090-v2-full-scale"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--freeze-backbone-blocks", type=int, default=0)
    parser.add_argument("--fpn-channels", type=int, default=256)
    parser.add_argument("--head-channels", type=int, default=128)
    parser.add_argument("--val-max-images", type=int, default=256)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve train manifest: bdd100k_train_70k_local.jsonl -> fallback bdd100k_train_local.jsonl
    if args.train_manifest is not None:
        train_manifest = Path(args.train_manifest)
    else:
        m_70k = ROOT / "data/manifests/bdd100k_train_70k_local.jsonl"
        train_manifest = m_70k if m_70k.exists() else (ROOT / "data/manifests/bdd100k_train_local.jsonl")

    # Resolve val manifest: bdd100k_val_local.jsonl
    if args.val_manifest is not None:
        val_manifest = Path(args.val_manifest)
    else:
        val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"

    print(f"[Init] Loading manifests:")
    print(f"  Train: {train_manifest}")
    print(f"  Val:   {val_manifest}")

    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    print(f"[Init] Loaded {len(train_records):,} train records and {len(val_records):,} val records.")

    shapes = [(48, 80), (24, 40), (12, 20)]  # P3, P4, P5 grid dimensions
    image_size = (384, 640)

    # Initialize FoundationViTDetectorV2
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Init] Instantiating FoundationViTDetectorV2 on {args.device} ({gpu_name})...")
    model = FoundationViTDetectorV2(
        model_name="vit_small_patch16_384",
        pretrained=True,
        freeze_backbone_blocks=args.freeze_backbone_blocks,
        num_classes=len(CLASS_NAMES),
        img_size=image_size,
        feature_indices=(2, 5, 8, 11),
        fpn_channels=args.fpn_channels,
        head_channels=args.head_channels,
    ).to(args.device)

    if args.checkpoint is not None and Path(args.checkpoint).is_file():
        print(f"[Init] Loading checkpoint from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        state_dict = ckpt.get("model_state", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print("[Init] Checkpoint loaded successfully.")

    if args.evaluate_only:
        print("[Mode] Running Evaluate-Only Mode...")
        eval_report = run_rigorous_evaluation_v2(
            model, val_records, train_records, device=args.device, max_images=args.val_max_images
        )
        report_file = out_dir / "rigorous_evaluation_report.json"
        report_file.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
        print(f"[Done] Report saved to {report_file}")
        return

    # Differential learning rate parameter groups
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [*model.adapter.parameters(), *model.heads.parameters()]

    print(f"[Init] Backbone trainable params: {sum(p.numel() for p in backbone_params):,} (base lr={args.backbone_lr})")
    print(f"[Init] Adapter & Head trainable params: {sum(p.numel() for p in head_params):,} (base lr={args.head_lr})")

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])

    train_dataset = BDD100KFullScaleDataset(train_records, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=lambda b: collate_fn_v2(b, shapes, image_size),
    )

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_pct * total_steps)

    print(f"[Train] Multi-Worker DataLoader ({args.workers} workers, batch_size={args.batch_size}) ready.")
    print(f"[Train] Total Epochs: {args.epochs} | Steps/epoch: {steps_per_epoch} | Total Steps: {total_steps} (Warmup: {warmup_steps} steps)")

    history = []
    best_map50 = 0.0
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_obj, epoch_cls, epoch_box, epoch_tot = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step, (images, targets) in enumerate(train_loader):
            global_step += 1
            # Update learning rates with warmup + cosine decay
            lr_mult = get_lr_multiplier(global_step, total_steps, warmup_steps)
            optimizer.param_groups[0]["lr"] = args.backbone_lr * lr_mult
            optimizer.param_groups[1]["lr"] = args.head_lr * lr_mult

            images = images.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss_v2(
                    preds, targets, negative_weight=0.05, box_weight=2.0, cls_weight=1.0, obj_weight=1.0
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_([*backbone_params, *head_params], max_norm=args.grad_clip)
            optimizer.step()

            epoch_obj += metrics["obj_loss"]
            epoch_cls += metrics["cls_loss"]
            epoch_box += metrics["box_loss"]
            epoch_tot += metrics["total_loss"]

            if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                elapsed = time.time() - ep_start
                rate = (step + 1) / max(elapsed, 0.01)
                img_rate = rate * args.batch_size
                cur_head_lr = optimizer.param_groups[1]["lr"]
                print(
                    f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {step+1:03d}/{steps_per_epoch}] "
                    f"Loss: {metrics['total_loss']:.3f} (Obj: {metrics['obj_loss']:.3f}, "
                    f"Cls: {metrics['cls_loss']:.3f}, Box: {metrics['box_loss']:.3f}) | "
                    f"{rate:.2f} it/s ({img_rate:.0f} img/s) | lr: {cur_head_lr:.2e}"
                )

        ep_time = time.time() - ep_start
        avg_obj = epoch_obj / steps_per_epoch
        avg_cls = epoch_cls / steps_per_epoch
        avg_box = epoch_box / steps_per_epoch
        avg_tot = epoch_tot / steps_per_epoch

        # Validation evaluation
        val_map50, val_map50_95 = evaluate_val_quick_v2(
            model, val_records, train_records, device=args.device, max_images=args.val_max_images
        )
        print(
            f"  >>> [Val @ Epoch {epoch:02d} | {ep_time:.1f}s] "
            f"mAP50: {val_map50:.2f}% | mAP50:95: {val_map50_95:.2f}% (Train Loss: {avg_tot:.4f}) <<<"
        )

        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_map50": val_map50,
            "val_map50_95": val_map50_95,
            "loss": avg_tot,
        }, ckpt_path)

        if val_map50 > best_map50:
            best_map50 = val_map50
            best_path = out_dir / "best_checkpoint.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_map50": val_map50,
                "val_map50_95": val_map50_95,
                "loss": avg_tot,
            }, best_path)
            print(f"  *** New Best Checkpoint saved to {best_path} (mAP50: {best_map50:.2f}%) ***")

        history.append({
            "epoch": epoch,
            "total_loss": avg_tot,
            "obj_loss": avg_obj,
            "cls_loss": avg_cls,
            "box_loss": avg_box,
            "val_map50": val_map50,
            "val_map50_95": val_map50_95,
            "epoch_seconds": ep_time,
        })

    total_time = time.time() - start_time
    print(f"\n[Done] Training completed: {args.epochs} Epochs in {total_time/60:.2f} min ({total_time:.1f}s)!")
    history_file = out_dir / "training_history.json"
    history_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[Done] Training history saved to {history_file}")

    # Rigorous Multi-Seed Benchmark on best checkpoint
    print("\n[Benchmark] Running Post-Training Rigorous 5-Condition Evaluation on Best Checkpoint...")
    best_ckpt = torch.load(out_dir / "best_checkpoint.pt", map_location=args.device)
    model.load_state_dict(best_ckpt["model_state"])
    eval_report = run_rigorous_evaluation_v2(
        model, val_records, train_records, device=args.device, max_images=args.val_max_images
    )
    report_file = out_dir / "rigorous_evaluation_report.json"
    report_file.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    print(f"[Done] Rigorous Evaluation report saved to {report_file}")


if __name__ == "__main__":
    main()
