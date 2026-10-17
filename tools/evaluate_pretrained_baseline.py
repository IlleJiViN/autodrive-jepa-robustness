"""
Step 3: Standard Pretrained Detector Baseline Evaluation (Torchvision Faster R-CNN / FCOS).
Evaluates an off-the-shelf COCO-pretrained object detector on the exact same 256 BDD100K
validation images and corruption conditions using the project's coco_ap metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision.models.detection as tv_det

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.corruptions import corrupt
from robustdrive.detector import coco_ap

# COCO 91-class index to BDD100K 10-class index mapping
# BDD100K: 0:pedestrian, 1:rider, 2:car, 3:truck, 4:bus, 5:train, 6:motorcycle, 7:bicycle, 8:traffic light, 9:traffic sign
COCO_TO_BDD = {
    1: 0,   # person -> pedestrian
    2: 7,   # bicycle -> bicycle
    3: 2,   # car -> car
    4: 6,   # motorcycle -> motorcycle
    6: 4,   # bus -> bus
    7: 5,   # train -> train
    8: 3,   # truck -> truck
    10: 8,  # traffic light -> traffic light
    13: 9,  # stop sign -> traffic sign
}


def evaluate_baseline_condition(
    model,
    val_records: list[dict],
    train_records: list[dict],
    family: str,
    severity: int,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
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

    ground_truths = {}
    predictions = {}

    with torch.no_grad():
        for idx, rec in enumerate(val_records):
            img_id = rec.get("scene_id", f"img_{idx}")
            sx, sy = 640 / rec["width"], 384 / rec["height"]
            gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
            gt_labels = [int(c) for c in rec.get("labels", [])]
            ground_truths[img_id] = (
                np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
                np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
            )

            with Image.open(rec["image_path"]) as img:
                image = img.convert("RGB")

            if family != "clean" and severity > 0:
                image = corrupt(image, family, severity, seed=seed + idx)

            if image.size != (640, 384):
                image = image.resize((640, 384), Image.Resampling.BILINEAR)

            inp = to_tensor(image).to(device)
            outputs = model([inp])[0]

            boxes = outputs["boxes"].cpu().numpy()
            scores = outputs["scores"].cpu().numpy()
            labels = outputs["labels"].cpu().numpy()

            dets = []
            for box, score, label in zip(boxes, scores, labels):
                if score < 0.05:
                    continue
                if int(label) in COCO_TO_BDD:
                    dets.append({
                        "box": box.tolist(),
                        "score": float(score),
                        "class": COCO_TO_BDD[int(label)],
                    })
            predictions[img_id] = dets

    res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
    return {
        "family": family,
        "severity": severity,
        "map50": float(res["map50"] * 100.0),
        "map50_95": float(res["map50_95"] * 100.0),
        "per_class": res["per_class"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "runs/pretrained_baseline_summary.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[Baseline Eval] Loading Torchvision Faster R-CNN ResNet50 FPN v2 (COCO Pretrained) on {args.device}...")
    weights = tv_det.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = tv_det.fasterrcnn_resnet50_fpn_v2(weights=weights).to(args.device)
    model.eval()

    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines()]
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines()]

    conditions = [
        ("clean", 0),
        ("fog", 3),
        ("rain", 3),
        ("low_light", 3),
        ("motion_blur", 3),
    ]

    results = {}
    for fam, sev in conditions:
        print(f"  Testing {fam} (severity {sev}) on 256 images...")
        res = evaluate_baseline_condition(model, val_records, train_records, fam, sev, seed=42, device=args.device)
        results[f"{fam}_s{sev}"] = res
        print(f"    --> mAP50: {res['map50']:.2f}% | mAP50:95: {res['map50_95']:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Baseline Eval Done] Saved to {out_path}")


if __name__ == "__main__":
    main()
