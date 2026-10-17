"""
Ablation Training: 20-Epoch Head Alignment WITHOUT weather augmentations (Clean-only).
Identical architecture, optimizer, and hyperparameters as train_aligned_detector_20ep.py,
with augment=False strictly enforced to establish the non-robust control baseline.
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
from robustdrive.foundation import FoundationViTDetector, compute_detection_loss


def build_batch_clean_only(records: list[dict], device: str, shapes: list[tuple[int, int, int]], image_size=(384, 640)):
    images = []
    scaled_records = []
    image_height, image_width = image_size

    for rec in records:
        img_path = Path(rec["image_path"])
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # STRICTLY NO WEATHER AUGMENTATION (Clean only)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "runs/aligned-vit-no-aug-control-20ep"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Ablation Init] Initializing Clean-Only Foundation ViT Detector on {args.device}...")
    model = FoundationViTDetector(
        model_name="vit_small_patch16_384",
        pretrained=True,
        num_classes=len(CLASS_NAMES),
        img_size=(384, 640),
        freeze_backbone_blocks=12,
    ).to(args.device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines()]

    shapes = [(3, 48, 80), (3, 24, 40), (3, 12, 20)]
    num_samples = len(train_records)
    steps_per_epoch = math.ceil(num_samples / args.batch_size)

    print(f"[Ablation Train] Training Clean-Only Control for {args.epochs} epochs ({steps_per_epoch} steps/epoch)...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        indices = np.random.permutation(num_samples)
        epoch_tot = 0.0

        for step in range(steps_per_epoch):
            batch_idx = indices[step * args.batch_size : (step + 1) * args.batch_size]
            batch_recs = [train_records[i] for i in batch_idx]
            images, targets = build_batch_clean_only(batch_recs, args.device, shapes, image_size=(384, 640))

            optimizer.zero_grad()
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss(preds, targets, negative_weight=0.025)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()
            epoch_tot += metrics["total_loss"]

        scheduler.step()
        avg_loss = epoch_tot / steps_per_epoch
        print(f"  [Clean-Control Ep {epoch:02d}/{args.epochs:02d}] Avg Loss: {avg_loss:.4f}")

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)

    total_time = time.time() - start_time
    print(f"\n[Ablation Done] Clean-Only Control Training completed in {total_time/60:.2f} minutes!")


if __name__ == "__main__":
    main()
