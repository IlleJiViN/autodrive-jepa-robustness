"""
Rigorous Multi-Seed & Full 256-Image Evaluation Script.
Evaluates model on all 256 validation images across Clean, Fog, Rain, and Low-light
using 3 distinct corruption random seeds to establish rigorous mean and standard deviation.
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

from robustdrive.data import CLASS_NAMES, to_tensor
from robustdrive.corruptions import corrupt
from robustdrive.detector import decode, coco_ap
from robustdrive.foundation import FoundationViTDetector


def evaluate_condition_multiseed(
    model,
    val_records: list[dict],
    train_records: list[dict],
    family: str,
    severity: int,
    seeds: list[int],
    device: str = "cuda",
) -> dict:
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

    map50_list = []
    map50_95_list = []

    # Ground truth in resized coordinates (seed-independent)
    ground_truths = {}
    for idx, rec in enumerate(val_records):
        img_id = rec.get("scene_id", f"img_{idx}")
        sx, sy = 640 / rec["width"], 384 / rec["height"]
        gt_boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in rec.get("boxes", [])]
        gt_labels = [int(c) for c in rec.get("labels", [])]
        ground_truths[img_id] = (
            np.array(gt_boxes, dtype=np.float32) if gt_boxes else np.empty((0, 4), dtype=np.float32),
            np.array(gt_labels, dtype=np.int64) if gt_labels else np.empty((0,), dtype=np.int64),
        )

    # If clean, severity is 0 and only run once
    run_seeds = [0] if family == "clean" else seeds

    for seed in run_seeds:
        predictions = {}
        with torch.no_grad():
            for idx, rec in enumerate(val_records):
                img_id = rec.get("scene_id", f"img_{idx}")
                with Image.open(rec["image_path"]) as img:
                    image = img.convert("RGB")

                if family != "clean" and severity > 0:
                    image = corrupt(image, family, severity, seed=seed + idx)

                if image.size != (640, 384):
                    image = image.resize((640, 384), Image.Resampling.BILINEAR)

                inp = to_tensor(image).unsqueeze(0).to(device)
                with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    preds = model(inp)

                decoded = decode(preds, image_size=image_size, max_detections=100, score_floor=0.01, nms_iou=0.6)
                predictions[img_id] = decoded[0]

        res = coco_ap(predictions, ground_truths, train_counts, val_counts, classes=num_classes)
        map50_list.append(res["map50"] * 100.0)
        map50_95_list.append(res["map50_95"] * 100.0)

    return {
        "family": family,
        "severity": severity,
        "map50_mean": float(np.mean(map50_list)),
        "map50_std": float(np.std(map50_list)) if len(map50_list) > 1 else 0.0,
        "map50_95_mean": float(np.mean(map50_95_list)),
        "map50_95_std": float(np.std(map50_95_list)) if len(map50_95_list) > 1 else 0.0,
        "raw_map50": map50_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "runs/aligned-vit-detector-20ep/checkpoint_epoch_20.pt"))
    parser.add_argument("--output", default=str(ROOT / "runs/rigorous_benchmark_summary.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"[Rigorous Eval] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = FoundationViTDetector(
        model_name="vit_small_patch16_384",
        pretrained=False,
        num_classes=len(CLASS_NAMES),
        img_size=(384, 640),
    ).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_manifest = ROOT / "data/manifests/bdd100k_val_local.jsonl"
    train_manifest = ROOT / "data/manifests/bdd100k_train_local.jsonl"
    val_records = [json.loads(l) for l in val_manifest.read_text(encoding="utf-8").strip().splitlines()]
    train_records = [json.loads(l) for l in train_manifest.read_text(encoding="utf-8").strip().splitlines()]

    print(f"[Rigorous Eval] Evaluating full {len(val_records)} validation images...")
    seeds = [42, 100, 2026]

    conditions = [
        ("clean", 0),
        ("fog", 3),
        ("rain", 3),
        ("low_light", 3),
        ("motion_blur", 3),
    ]

    results = {}
    for fam, sev in conditions:
        print(f"  Testing {fam} (severity {sev}) with 3 seeds...")
        res = evaluate_condition_multiseed(model, val_records, train_records, fam, sev, seeds, device=args.device)
        results[f"{fam}_s{sev}"] = res
        print(f"    --> mAP50: {res['map50_mean']:.2f}% (±{res['map50_std']:.2f}%) | mAP50:95: {res['map50_95_mean']:.2f}%")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Rigorous Eval Done] Saved results to {out_path}")


if __name__ == "__main__":
    main()
