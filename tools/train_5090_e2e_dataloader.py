"""
High-Performance Multi-Worker Full e2e Training for RTX 5090 + Ryzen 9 9950X3D.
Features:
- PyTorch Dataset + DataLoader with num_workers=8, pin_memory=True, persistent_workers=True
- Background multi-process on-the-fly weather synthesis (Fog, Rain, Blur, Low-light)
- ViT-S 12-block Full e2e (freeze=0) with differential learning rate
- Automatic 256-image validation evaluation
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.detector import STRIDES, assign_targets, decode, coco_ap
from robustdrive.corruptions import corrupt
from robustdrive.foundation import FoundationViTDetector, compute_detection_loss


class BDD100KDetectionDataset(Dataset):
    def __init__(self, records: list[dict], image_size=(384, 640), augment: bool = True):
        self.records = records
        self.image_size = image_size
        self.augment = augment
        self.height, self.width = image_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = rec["image_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Weather augmentation with 30% probability
        if self.augment and np.random.rand() < 0.30:
            c_fam = np.random.choice(["fog", "rain", "motion_blur", "low_light"])
            c_sev = int(np.random.randint(1, 4))
            image = corrupt(image, c_fam, c_sev, seed=int(np.random.randint(0, 100000)))

        sx = self.width / rec["width"]
        sy = self.height / rec["height"]
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.Resampling.BILINEAR)

        tensor_img = to_tensor(image)
        scaled_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]

        return {
            "image": tensor_img,
            "boxes": scaled_boxes,
            "labels": rec.get("labels", []),
            "width": self.width,
            "height": self.height,
        }


def collate_fn(batch, shapes, image_size):
    images = torch.stack([item["image"] for item in batch])
    records = [{"boxes": item["boxes"], "labels": item["labels"], "width": item["width"], "height": item["height"]} for item in batch]
    targets, _ = assign_targets(records, list(range(len(batch))), shapes, image_size)
    return images, targets


def evaluate_val_quick(model, val_records, train_records, device="cuda", max_images=256):
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
            gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
            gt_labels = [int(c) for c in rec.get("labels", [])]
            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    return res["map50"] * 100.0, res["map50_95"] * 100.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm-start", default=str(ROOT / "runs/unfreeze-top4-vit-detector-10ep/checkpoint_epoch_10.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "runs/rtx5090-e2e-vit-detector-20ep"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[Init] Initializing Full e2e Foundation ViT Detector on {args.device} ({gpu_name})...")
    model = FoundationViTDetector(
        model_name="vit_small_patch16_384",
        pretrained=True,
        num_classes=len(CLASS_NAMES),
        img_size=(384, 640),
        freeze_backbone_blocks=0,  # FULL e2e: all 12 blocks unfrozen!
    ).to(args.device)

    # All parameters are trainable
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [*model.adapter.parameters(), *model.heads.parameters()]

    print(f"[Init] Backbone trainable parameters: {sum(p.numel() for p in backbone_params):,} (lr={args.backbone_lr})")
    print(f"[Init] Adapter & Head trainable parameters: {sum(p.numel() for p in head_params):,} (lr={args.head_lr})")

    optimizer = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 1e-4},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Load manifests
    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines()]
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines()]

    shapes = [(3, 48, 80), (3, 24, 40), (3, 12, 20)]
    image_size = (384, 640)

    train_dataset = BDD100KDetectionDataset(train_records, image_size=image_size, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=lambda b: collate_fn(b, shapes, image_size),
    )

    steps_per_epoch = len(train_loader)
    print(f"[Train] Multi-Worker DataLoader ({args.num_workers} workers) ready. {steps_per_epoch} steps/epoch...")
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_obj, epoch_cls, epoch_box, epoch_tot = 0.0, 0.0, 0.0, 0.0
        ep_start = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(args.device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(args.device.split(":")[0], dtype=torch.bfloat16, enabled=args.device.startswith("cuda")):
                preds = model(images)
                loss, metrics = compute_detection_loss(preds, targets, negative_weight=0.025)

            loss.backward()
            torch.nn.utils.clip_grad_norm_([*backbone_params, *head_params], max_norm=5.0)
            optimizer.step()

            epoch_obj += metrics["obj_loss"]
            epoch_cls += metrics["cls_loss"]
            epoch_box += metrics["box_loss"]
            epoch_tot += metrics["total_loss"]

            if (step + 1) % 20 == 0 or (step + 1) == steps_per_epoch:
                elapsed = time.time() - ep_start
                rate = (step + 1) / max(elapsed, 0.01)
                img_rate = rate * args.batch_size
                print(f"  [5090 e2e Ep {epoch:02d}/{args.epochs:02d} | Step {step+1:03d}/{steps_per_epoch}] "
                      f"Loss: {metrics['total_loss']:.3f} (Obj: {metrics['obj_loss']:.3f}, "
                      f"Cls: {metrics['cls_loss']:.3f}, Box: {metrics['box_loss']:.3f}) | {rate:.2f} it/s ({img_rate:.0f} img/s)")

        scheduler.step()
        ep_time = time.time() - ep_start
        avg_obj = epoch_obj / steps_per_epoch
        avg_cls = epoch_cls / steps_per_epoch
        avg_box = epoch_box / steps_per_epoch
        avg_tot = epoch_tot / steps_per_epoch

        val_map50, val_map50_95 = evaluate_val_quick(model, val_records, train_records, device=args.device, max_images=256)
        print(f"  >>> [Val @ Ep {epoch:02d} | {ep_time:.1f}s] Full 256-img mAP50: {val_map50:.2f}% | mAP50:95: {val_map50_95:.2f}% (Loss: {avg_tot:.4f}) <<<")

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
            "epoch_seconds": ep_time,
        })

    total_time = time.time() - start_time
    print(f"\n[Done] 5090 Full e2e 20 Epochs completed in {total_time/60:.2f} minutes ({total_time:.1f}s)!")
    summary_file = out_dir / "training_history.json"
    summary_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[Done] History saved to {summary_file}")


if __name__ == "__main__":
    main()
