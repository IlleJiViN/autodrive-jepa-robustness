"""
Fine-tuning CLI for Foundation ViT-based Robust Object Detector.
Leverages Pretrained ViT-S backbone from timm with on-the-fly weather corruption augmentation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.common import seed_all, stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.detector import assign_targets, SLOTS
from robustdrive.foundation import FoundationViTDetector, compute_detection_loss


class BDDDetectionDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        image_size: tuple[int, int] = (384, 640),
        augment_corruptions: bool = True,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.augment_corruptions = augment_corruptions
        self.seed = seed
        self.families = ["fog", "rain", "motion_blur", "low_light"]
        
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.records = [json.loads(line) for line in lines]
        # Filter for records whose image actually exists locally
        self.valid_records = []
        for r in self.records:
            img_p = Path(r["image_path"])
            # Fallback check relative to data/bdd100k/images/100k
            if not img_p.exists():
                fallback = ROOT / "data/bdd100k/images/100k" / r["split"] / img_p.name
                if fallback.exists():
                    r["image_path"] = str(fallback)
                    self.valid_records.append(r)
            else:
                self.valid_records.append(r)

        print(f"[Dataset] Loaded {len(self.valid_records)} available images from {path.name}")

    def __len__(self):
        return len(self.valid_records)

    def __getitem__(self, idx: int):
        rec = self.valid_records[idx]
        img_path = Path(rec["image_path"])
        with Image.open(img_path) as img:
            image = img.convert("RGB")

        # Apply random on-the-fly weather corruption with 35% probability during training
        if self.augment_corruptions and random.random() < 0.35:
            fam = random.choice(self.families)
            sev = random.randint(1, 3)
            c_seed = stable_seed(self.seed, str(img_path), fam, sev)
            image = corrupt(image, fam, sev, c_seed)

        # Resize to fixed resolution (384, 640)
        h_orig, w_orig = rec["height"], rec["width"]
        h_target, w_target = self.image_size
        if image.size != (w_target, h_target):
            image = image.resize((w_target, h_target), Image.Resampling.BILINEAR)

        img_tensor = to_tensor(image)
        return img_tensor, idx


def collate_fn(batch, dataset: BDDDetectionDataset, shapes, image_size):
    images = torch.stack([item[0] for item in batch])
    indices = [item[1] for item in batch]
    # Assign target labels and boxes across P3, P4, P5 shapes
    targets, dropped = assign_targets(dataset.valid_records, indices, shapes, image_size)
    return images, targets


def train_one_epoch(model, loader, optimizer, device, epoch, max_batches=None):
    from robustdrive.detector import detector_loss
    model.train()
    total_loss, total_pos = 0.0, 0
    start = time.time()

    for step, (images, targets) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        images = images.to(device)

        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(images)
            loss, metrics = detector_loss(outputs, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_pos += metrics.get("positive_slots", 0)

        if (step + 1) % 10 == 0 or (step + 1) == len(loader) or (max_batches and (step + 1) == max_batches):
            print(
                f"  Epoch [{epoch}] Step [{step+1}/{len(loader)}] | "
                f"Loss: {float(loss.item()):.4f} (Obj: {float(metrics.get('object_loss', 0)):.4f}, "
                f"Cls: {float(metrics.get('class_loss', 0)):.4f}, Box: {float(metrics.get('box_loss', 0)):.4f}) | "
                f"Speed: {(step+1)/(time.time()-start):.2f} batch/s"
            )

    return total_loss / (step + 1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Foundation ViT on BDD100K")
    parser.add_argument("--backbone", default="vit_small_patch16_384")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--freeze-blocks", type=int, default=6, help="Freeze early ViT layers for stability")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "runs/foundation-vit-detector"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"[Fine-tune] Initializing Foundation ViT Detector ({args.backbone})")
    print(f"=======================================================")

    # Initialize model (Pretrained ViT backbone from timm)
    # If internet or cache available, pretrained=True; else graceful fallback
    try:
        model = FoundationViTDetector(
            model_name=args.backbone,
            pretrained=True,
            freeze_backbone_blocks=args.freeze_blocks,
            img_size=(384, 640),
        )
        print("[Fine-tune] Successfully loaded pretrained weights from timm!")
    except Exception as e:
        print(f"[Fine-tune] Note: loading timm without pretrain cache: {e}")
        model = FoundationViTDetector(
            model_name=args.backbone,
            pretrained=False,
            freeze_backbone_blocks=args.freeze_blocks,
            img_size=(384, 640),
        )

    model = model.to(device)

    # Multi-scale feature grid shapes: (1/8, 1/16, 1/32)
    shapes = [(3, 48, 80), (3, 24, 40), (3, 12, 20)]
    image_size = (384, 640)

    # Dataset & Dataloader
    train_manifest = ROOT / "data/manifests/bdd100k_train_labeled.jsonl"
    val_manifest = ROOT / "data/manifests/bdd100k_val_labeled.jsonl"

    train_ds = BDDDetectionDataset(str(train_manifest), image_size=image_size, augment_corruptions=True)
    if len(train_ds) == 0:
        print("[Fine-tune] Waiting for local images or none found. Exiting.")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, train_ds, shapes, image_size),
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": args.lr * 0.1},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad], "lr": args.lr},
    ], weight_decay=args.weight_decay)

    print(f"[Fine-tune] Starting training for {args.epochs} epochs on {len(train_ds)} images...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_batches=args.max_train_batches,
        )
        ckpt_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "loss": epoch_loss,
            "backbone": args.backbone,
        }, ckpt_path)
        print(f"[Fine-tune] Epoch {epoch} complete. Checkpoint saved: {ckpt_path}\n")


if __name__ == "__main__":
    main()
