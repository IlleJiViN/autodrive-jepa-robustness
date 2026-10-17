import json
import math
import time
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path("/home/coder/autodrive-jepa-robustness")
LABELS_ZIP = ROOT / "data/bdd100k/annotations/bdd100k_labels.zip"
IMAGES_ROOT = ROOT / "data/bdd100k/images"
MANIFESTS_ROOT = ROOT / "data/manifests"

CLASS_NAMES = [
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign"
]
ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}

def parse_member_batch(args):
    # args: (members_list, split)
    members, split = args
    records = []
    stats = {
        "total": len(members),
        "class_counts": Counter(),
        "dropped_crowd": 0,
        "dropped_non_det": 0,
        "dropped_invalid_area": 0,
        "clipped_boxes": 0,
        "empty_boxes_images": 0,
    }

    img_dir = IMAGES_ROOT / "100k" / split
    with zipfile.ZipFile(LABELS_ZIP) as zf:
        for member in members:
            stem = Path(member).stem
            img_path = img_dir / f"{stem}.jpg"
            if not img_path.is_file():
                raise FileNotFoundError(f"Extracted image not found: {img_path}")

            raw = zf.read(member)
            entry = json.loads(raw)
            frames = entry.get("frames", [])
            if len(frames) != 1:
                frames = [f for f in frames if f.get("timestamp") == 10000]
            if len(frames) != 1:
                raise ValueError(f"Ambiguous frames for {member}: {len(frames)}")

            frame = frames[0]
            entry_attrs = entry.get("attributes") or {}
            weather = entry_attrs.get("weather", "undefined")
            timeofday = entry_attrs.get("timeofday", "undefined")
            scene = entry_attrs.get("scene", "undefined")

            boxes = []
            labels = []
            obj_attrs = []

            for obj in frame.get("objects", []):
                category = obj.get("category", "").replace("_", " ")
                category = ALIASES.get(category, category)
                box = obj.get("box2d")
                if category not in CLASS_NAMES or not box:
                    stats["dropped_non_det"] += 1
                    continue
                if obj.get("attributes", {}).get("crowd", False):
                    stats["dropped_crowd"] += 1
                    continue

                try:
                    x1 = float(box["x1"])
                    y1 = float(box["y1"])
                    x2 = float(box["x2"])
                    y2 = float(box["y2"])
                except (KeyError, ValueError, TypeError):
                    stats["dropped_invalid_area"] += 1
                    continue

                if not (math.isfinite(x1) and math.isfinite(y1) and math.isfinite(x2) and math.isfinite(y2)):
                    stats["dropped_invalid_area"] += 1
                    continue

                # Clip to 1280x720
                bx1 = min(max(x1, 0.0), 1280.0)
                by1 = min(max(y1, 0.0), 720.0)
                bx2 = min(max(x2, 0.0), 1280.0)
                by2 = min(max(y2, 0.0), 720.0)

                if (bx1, by1, bx2, by2) != (x1, y1, x2, y2):
                    stats["clipped_boxes"] += 1

                if bx2 <= bx1 or by2 <= by1:
                    stats["dropped_invalid_area"] += 1
                    continue

                label_idx = CLASS_NAMES.index(category)
                boxes.append([bx1, by1, bx2, by2])
                labels.append(label_idx)
                obj_attrs.append(obj.get("attributes") or {})
                stats["class_counts"][category] += 1

            if not boxes:
                stats["empty_boxes_images"] += 1

            record = {
                "image_path": str(img_path.resolve()),
                "scene_id": stem,
                "split": split,
                "boxes": boxes,
                "labels": labels,
                "annotation_status": "available_legacy_bdd100k",
                "width": 1280,
                "height": 720,
                "weather": weather,
                "timeofday": timeofday,
                "scene": scene,
                "attributes": entry_attrs,
                "object_attributes": obj_attrs,
                "annotation_source_member": member,
            }
            records.append(record)

    return records, stats

def build_manifest_for_split(split, output_file, num_workers=24):
    print(f"\n==========================================", flush=True)
    print(f"Building manifest for {split} split...", flush=True)
    t0 = time.time()
    
    with zipfile.ZipFile(LABELS_ZIP) as zf:
        members = [n for n in zf.namelist() if n.startswith(f"100k/{split}/") and n.endswith(".json")]
    print(f"Found {len(members)} annotations in {LABELS_ZIP.name} for {split}.", flush=True)
    
    # Sort members for determinism
    members.sort()
    
    batch_size = 500
    batches = [(members[i:i+batch_size], split) for i in range(0, len(members), batch_size)]
    
    all_records = []
    aggregated_stats = {
        "total": 0,
        "class_counts": Counter(),
        "dropped_crowd": 0,
        "dropped_non_det": 0,
        "dropped_invalid_area": 0,
        "clipped_boxes": 0,
        "empty_boxes_images": 0,
    }
    
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for records, stats in pool.map(parse_member_batch, batches):
            all_records.extend(records)
            aggregated_stats["total"] += stats["total"]
            aggregated_stats["class_counts"].update(stats["class_counts"])
            aggregated_stats["dropped_crowd"] += stats["dropped_crowd"]
            aggregated_stats["dropped_non_det"] += stats["dropped_non_det"]
            aggregated_stats["dropped_invalid_area"] += stats["dropped_invalid_area"]
            aggregated_stats["clipped_boxes"] += stats["clipped_boxes"]
            aggregated_stats["empty_boxes_images"] += stats["empty_boxes_images"]

    # Sort final records by scene_id
    all_records.sort(key=lambda r: r["scene_id"])

    # Write out atomically
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    tmp_path.replace(output_path)

    elapsed = time.time() - t0
    total_boxes = sum(len(r["boxes"]) for r in all_records)
    print(f"Successfully wrote {len(all_records)} records to {output_path} in {elapsed:.2f}s!", flush=True)
    print(f"Total boxes: {total_boxes} across {len(all_records)} images ({total_boxes/max(len(all_records),1):.2f} boxes/img)")
    print(f"Images without boxes: {aggregated_stats['empty_boxes_images']}")
    print(f"Clipped boxes: {aggregated_stats['clipped_boxes']}")
    print(f"Dropped objects: crowd={aggregated_stats['dropped_crowd']}, non-detection={aggregated_stats['dropped_non_det']}, invalid_area={aggregated_stats['dropped_invalid_area']}")
    print(f"Class breakdown:")
    for cls in CLASS_NAMES:
        print(f"  {cls:15s}: {aggregated_stats['class_counts'][cls]:7d}")
        
    return len(all_records), total_boxes

def main():
    print("Starting BDD100K Full Manifest Build...", flush=True)
    t_start = time.time()
    
    train_out = MANIFESTS_ROOT / "bdd100k_train_70k_local.jsonl"
    val_out = MANIFESTS_ROOT / "bdd100k_val_10k_local.jsonl"
    
    train_records, train_boxes = build_manifest_for_split("train", train_out)
    val_records, val_boxes = build_manifest_for_split("val", val_out)
    
    total_elapsed = time.time() - t_start
    print("\n==========================================", flush=True)
    print("FULL BDD100K MANIFEST GENERATION COMPLETE!", flush=True)
    print(f"Train 70k: {train_records} records, {train_boxes} boxes -> {train_out}")
    print(f"Val 10k:   {val_records} records, {val_boxes} boxes -> {val_out}")
    print(f"Total execution time: {total_elapsed:.2f}s")

if __name__ == "__main__":
    main()
