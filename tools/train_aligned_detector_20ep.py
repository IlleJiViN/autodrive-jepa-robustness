"""
Fast 20-Epoch Head Alignment Training for Foundation ViT Detector.
With the stride unpacking bug fixed and positive-aware loss applied,
trains the FPN adapter and anchor-free heads to achieve real mAP convergence.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.detector import STRIDES, assign_targets, decode, coco_ap
from robustdrive.corruptions import corrupt
from robustdrive.foundation import FoundationViTDetector, compute_detection_loss


def build_batch(records: list[dict], device: str, shapes: list[tuple[int, int, int]], image_size=(384, 640), augment: bool = True):
    images = []
    scaled_records = []
    image_height, image_width = image_size

    for rec in records:
        img_path = Path(rec["image_path"])
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Weather augmentation with 30% probability
        if augment and np.random.rand() < 0.30:
            c_fam = np.random.choice(["fog", "rain", "motion_blur", "low_light"])
            c_sev = int(np.random.randint(1, 4))
            image = corrupt(image, c_fam, c_sev, seed=int(np.random.randint(0, 10000)))

        sx = image_width / rec["width"]
        sy = image_height / rec["height"]
        if image.size != (image_width, image_height):
            image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)

        images.append(to_tensor(image))
        scaled_boxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in rec.get("boxes", [])]
        scaled_records.append({
            "boxes": scaled_boxes,
            "labels": rec.get("labels", []),
            "width": image_width,
            "height": image_height,
        })

    targets, _ = assign_targets(scaled_records, list(range(len(records))), shapes, image_size)
    tensor_images = torch.stack(images).to(device)
    return tensor_images, targets


def evaluate_val_quick(model, val_records, train_records, device="cuda", max_images=100):
    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)
    train_counts = np.zeros(num_classes, dtype=int)
    for r in train_records[:500]:
        for c in r.get("labels", []):
            if c < num_classes:
                train_counts[c] += 1
    val_counts = np.zeros(num_classes, dtype=int)
    for r in val_records[:max_images]:
        for c in r.get("labels", []):
            if c < num_classes:
                val_counts[c] += 1

    predictions = {}
    ground_truths = {}
    eval_recs = val_records[:max_images]

    with torch.no_grad():
        for idx, rec in enumerate(eval_recs):
            img_id = rec.get("scene_id", f"img_{idx}")
            with Image.open(rec["image_path"]) as img:
                image = img.convert("RGB").resize((640, 384))
            inp = to_tensor(image).unsqueeze(0).to(device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                preds = model(inp)
            decoded = decode(preds, image_size=image_size, max_detections=100, score_floor=0.01, nms_iou=0.6)
            predictions[img_id] = decoded[0]

            sx, sy = 640 / rec["width"], 384 / rec["height"]
            gt_boxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in rec.get("boxes", [])]
            gt_labels = [int(c) for c in rec.get("labels", [])]
            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    return res["map50"] * 100.0, res["map50_95"] * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "runs/aligned-vit-detector-20ep"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Init] Initializing Foundation ViT Detector on {args.device}...")
    model = FoundationViTDetector(
        model_name="vit_small_patch16_384",
        pretrained=True,
        num_classes=len(CLASS_NAMES),
        img_size=(384, 640),
        freeze_backbone_blocks=12,  # Freeze all 12 ViT blocks for ultra-fast head alignment
    ).to(args.device)

    # Trainable parameters: Adapter + Heads only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load manifests
    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines()]

    shapes = [(3, 48, 80), (3, 24, 40), (3, 12, 20)]
    num_samples = len(train_records)
    steps_per_epoch = math.ceil(num_samples / args.batch_size)

    print(f"[Train] Training {len(trainable_params)} tensor groups for {args.epochs} epochs ({steps_per_epoch} steps/epoch)...")
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = np.random.permutation(num_samples)
        epoch_obj, epoch_cls, epoch_box, epoch_tot = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step in range(steps_per_epoch):
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_recs = [train_records[i] for i in batch_idx]
            images, targets = build_batch(batch_recs, args.device, shapes, image_size=(384, 640), augment=True)

            optimizer.zero_grad()
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss(preds, targets, negative_weight=0.025)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()

            epoch_obj += metrics["obj_loss"]
            epoch_cls += metrics["cls_loss"]
            epoch_box += metrics["box_loss"]
            epoch_tot += metrics["total_loss"]

            if (step + 1) % 40 == 0 or (step + 1) == steps_per_epoch:
                elapsed = time.time() - ep_start
                rate = (step + 1) / max(elapsed, 0.01)
                print(f"  [Ep {epoch:02d}/{args.epochs:02d} | Step {step+1:03d}/{steps_per_epoch}] "
                      f"Loss: {metrics['total_loss']:.3f} (Obj: {metrics['obj_loss']:.3f}, "
                      f"Cls: {metrics['cls_loss']:.3f}, Box: {metrics['box_loss']:.3f}) | {rate:.2f} it/s")

        scheduler.step()
        avg_obj = epoch_obj / steps_per_epoch
        avg_cls = epoch_cls / steps_per_epoch
        avg_box = epoch_box / steps_per_epoch
        avg_tot = epoch_tot / steps_per_epoch

        # Periodic validation every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            val_map50, val_map50_95 = evaluate_val_quick(model, val_records, train_records, device=args.device, max_images=100)
            print(f"  >>> [Val @ Epoch {epoch}] mAP50: {val_map50:.2f}% | mAP50:95: {val_map50_95:.2f}% <<<")
            
            # Save checkpoint
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_map50": val_map50,
                "val_map50_95": val_map50_95,
                "loss": avg_tot,
            }, ckpt_path)

            history.append({
                "epoch": epoch, "total_loss": avg_tot, "obj_loss": avg_obj,
                "cls_loss": avg_cls, "box_loss": avg_box,
                "val_map50": val_map50, "val_map50_95": val_map50_95,
            })
        else:
            history.append({
                "epoch": epoch, "total_loss": avg_tot, "obj_loss": avg_obj,
                "cls_loss": avg_cls, "box_loss": avg_box,
            })

    total_time = time.time() - start_time
    print(f"\n[Done] 20 Epochs completed in {total_time/60:.2f} minutes!")
    summary_file = out_dir / "training_history.json"
    summary_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[Done] History saved to {summary_file}")


if __name__ == "__main__":
    main()
