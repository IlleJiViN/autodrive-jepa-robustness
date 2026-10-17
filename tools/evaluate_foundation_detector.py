"""
Evaluate trained Foundation ViT Detector on validation set (256 images).
Uses the project's official coco_ap function to calculate exact mAP50 and mAP50:95.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.common import stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.detector import decode, coco_ap
from robustdrive.foundation import FoundationViTDetector


def evaluate_detection(
    model: FoundationViTDetector,
    records: list[dict],
    train_records: list[dict],
    device: str = "cuda",
    family: str = "clean",
    severity: int = 0,
) -> dict:
    model.eval()
    image_size = (384, 640)
    num_classes = len(CLASS_NAMES)

    # Count class instances in train and val
    train_class_counts = np.zeros(num_classes, dtype=int)
    for r in train_records:
        for cat in r.get("labels", []):
            if cat < num_classes:
                train_class_counts[cat] += 1

    val_class_counts = np.zeros(num_classes, dtype=int)
    for r in records:
        for cat in r.get("labels", []):
            if cat < num_classes:
                val_class_counts[cat] += 1

    predictions = {}
    ground_truths = {}

    with torch.no_grad():
        for idx, rec in enumerate(records):
            img_id = rec.get("scene_id", f"img_{idx}")
            img_path = Path(rec["image_path"])
            with Image.open(img_path) as img:
                image = img.convert("RGB")

            if family != "clean" and severity > 0:
                c_seed = stable_seed(42, str(img_path), family, severity)
                image = corrupt(image, family, severity, c_seed)

            orig_w, orig_h = rec["width"], rec["height"]
            scale_x, scale_y = image_size[1] / orig_w, image_size[0] / orig_h
            if image.size != (image_size[1], image_size[0]):
                image = image.resize((image_size[1], image_size[0]), Image.Resampling.BILINEAR)

            inp = to_tensor(image).unsqueeze(0).to(device)
            with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                preds = model(inp)

            decoded = decode(preds, image_size=image_size, max_detections=100, score_floor=0.001, nms_iou=0.6)
            predictions[img_id] = decoded[0]

            # Ground truth in resized coordinates
            gt_boxes = []
            gt_labels = []
            for b, cat in zip(rec.get("boxes", []), rec.get("labels", [])):
                gt_boxes.append([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y])
                gt_labels.append(int(cat))

            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

    ap_metrics = coco_ap(
        predictions=predictions,
        ground_truths=ground_truths,
        train_class_counts=train_class_counts,
        val_class_counts=val_class_counts,
        classes=num_classes,
    )

    return {
        "condition": f"{family}_s{severity}",
        "map50_pct": ap_metrics["map50"] * 100.0,
        "map50_95_pct": ap_metrics["map50_95"] * 100.0,
        "per_class": ap_metrics["per_class"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/foundation-vit-detector/checkpoint_epoch_3.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()

    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"

    val_lines = val_manifest.read_text(encoding="utf-8").strip().splitlines()
    val_records = [json.loads(l) for l in val_lines][:args.max_images]

    train_lines = train_manifest.read_text(encoding="utf-8").strip().splitlines()
    train_records = [json.loads(l) for l in train_lines][:500]

    print(f"\n[Eval] Loading Foundation ViT Detector from {args.checkpoint}...")
    model = FoundationViTDetector(model_name="vit_small_patch16_384", pretrained=False, img_size=(384, 640))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(args.device)

    print(f"[Eval] Evaluating on {len(val_records)} validation images...")
    clean_res = evaluate_detection(model, val_records, train_records, device=args.device, family="clean", severity=0)
    fog_res = evaluate_detection(model, val_records, train_records, device=args.device, family="fog", severity=3)
    rain_res = evaluate_detection(model, val_records, train_records, device=args.device, family="rain", severity=3)

    print("\n=======================================================")
    print("      Foundation ViT 실측 검출 성능 평가 결과 (COCO mAP)")
    print("=======================================================")
    print(f"  [Clean]     mAP50: {clean_res['map50_pct']:.4f}% | mAP50:95: {clean_res['map50_95_pct']:.4f}%")
    print(f"  [Fog s=3]   mAP50: {fog_res['map50_pct']:.4f}% | mAP50:95: {fog_res['map50_95_pct']:.4f}%")
    print(f"  [Rain s=3]  mAP50: {rain_res['map50_pct']:.4f}% | mAP50:95: {rain_res['map50_95_pct']:.4f}%")
    print("=======================================================\n")

    summary = {
        "checkpoint": args.checkpoint,
        "clean": clean_res,
        "fog_s3": fog_res,
        "rain_s3": rain_res,
    }
    out_file = ROOT / "runs/foundation_vit_eval_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Eval] Results saved to: {out_file}")


if __name__ == "__main__":
    main()
