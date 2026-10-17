"""Generate object-target proposals from a specified local YOLO checkpoint on clean images.

Human labels remain separate and are used only by the downstream semantic evaluator.
This script requires existing images and weights; it does not select or download weights.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_json, load_config
from robustdrive.data import CLASS_NAMES, read_manifest, validate_data

NAME_MAP = {"person": "pedestrian", "pedestrian": "pedestrian", "rider": "rider",
            "car": "car", "truck": "truck", "bus": "bus", "train": "train",
            "motorcycle": "motorcycle", "bicycle": "bicycle", "traffic light": "traffic light",
            "traffic sign": "traffic sign", "stop sign": "traffic sign"}


def proposal_row(row, result, teacher_sha256):
    if result.boxes is None:
        raise ValueError("The supplied YOLO weights must produce detection boxes")
    if tuple(result.orig_shape) != (row["height"], row["width"]):
        raise ValueError("YOLO proposal coordinates must match original image geometry")
    if Path(result.path).resolve() != Path(row["image_path"]).resolve():
        raise ValueError("YOLO result/image ordering mismatch")
    output = {k: v for k, v in row.items() if k not in
              ("boxes", "labels", "object_attributes", "annotation_source_member", "annotation_sha256")}
    output.update(boxes=[], labels=[], proposal_confidence=[], annotation_status="available_yolo_proposals",
                  teacher_sha256=teacher_sha256, proposal_image_condition="clean")
    for box, label, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist()):
        if not all(math.isfinite(value) for value in [*box, label, confidence]) or not 0 <= confidence <= 1:
            raise ValueError("YOLO produced nonfinite coordinates or invalid confidence")
        category = NAME_MAP.get(result.names[int(label)])
        if category is None:
            continue
        x1, y1, x2, y2 = box
        box = [max(0., min(row["width"], x1)), max(0., min(row["height"], y1)),
               max(0., min(row["width"], x2)), max(0., min(row["height"], y2))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        output["boxes"].append(box)
        output["labels"].append(CLASS_NAMES.index(category))
        output["proposal_confidence"].append(float(confidence))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/object_pilot.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--confidence", type=float, default=.25)
    args = parser.parse_args()
    weights, output = Path(args.weights).resolve(), Path(args.output_dir).resolve()
    if not weights.is_file():
        raise FileNotFoundError("Supply an existing local YOLO checkpoint")
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size < 1 or not 0 <= args.confidence <= 1:
        raise ValueError("Invalid batch size or confidence")
    config = load_config(args.config)
    validate_data(config)
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("YOLO proposal generation requires the ultralytics package") from error
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    teacher = YOLO(str(weights))
    predictions, counts = {}, {}
    started = time.perf_counter()
    for split in ("train", "val"):
        rows = read_manifest(config["data"][split + "_manifest"])
        converted = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            results = teacher.predict(source=[row["image_path"] for row in batch], imgsz=640,
                                      conf=args.confidence, iou=.7, max_det=100, device=args.device,
                                      verbose=False, save=False, augment=False)
            if len(results) != len(batch):
                raise ValueError("YOLO returned a different number of results than inputs")
            converted.extend(proposal_row(row, result, digest) for row, result in zip(batch, results))
        predictions[split] = converted
        counts[split] = {"images": len(converted), "proposals": sum(len(row["boxes"]) for row in converted),
                         "images_without_proposals": sum(not row["boxes"] for row in converted)}
    output.mkdir(parents=True)
    for split, rows in predictions.items():
        path = output / (split + ".jsonl")
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        config["data"][split + "_manifest"] = str(path)
    config["object_pretraining"].update(annotation_status="available_yolo_proposals",
                                        box_source="YOLO_clean_image_teacher_sha256:" + digest)
    atomic_json(output / "training_config.json", config)
    atomic_json(output / "provenance.json", {"teacher_checkpoint": str(weights), "teacher_sha256": digest,
                "ultralytics_version": ultralytics.__version__, "confidence": args.confidence,
                "iou": .7, "max_det": 100, "splits": counts, "category_mapping": NAME_MAP,
                "wall_seconds": time.perf_counter() - started,
                "supervision": "external_pretrained_detector_proposals; no class loss in JEPA/MAE"})
    print(json.dumps({"training_config": str(output / "training_config.json"), "splits": counts}), flush=True)


if __name__ == "__main__":
    main()
