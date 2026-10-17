"""Acquire legacy per-image BDD labels and match the existing pilot frames."""
import collections
import hashlib
import json
import math
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_json, load_config
from robustdrive.data import CLASS_NAMES, read_manifest, validate_data

URL = "http://128.32.162.150/bdd100k/bdd100k_labels.zip"
EXPECTED_BYTES = 189638612
ALIASES = {"person": "pedestrian", "bike": "bicycle", "motor": "motorcycle"}


def download(path):
    if path.exists():
        if path.stat().st_size != EXPECTED_BYTES:
            raise ValueError("Existing annotation archive has an unexpected size")
        return
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(6):
        offset = temporary.stat().st_size if temporary.exists() else 0
        if offset == EXPECTED_BYTES:
            temporary.replace(path)
            return
        if offset > EXPECTED_BYTES:
            raise ValueError("Partial archive exceeds expected size")
        headers = {"Range": f"bytes={offset}-{EXPECTED_BYTES - 1}"} if offset else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=30) as response:
                if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor archive resume offset")
                with temporary.open("ab") as output:
                    last_report = offset // (8 * 1024 * 1024)
                    while block := response.read(1024 * 1024):
                        output.write(block)
                        offset += len(block)
                        if offset > EXPECTED_BYTES:
                            raise ValueError("Annotation download larger than source metadata")
                        if offset // (8 * 1024 * 1024) > last_report:
                            last_report = offset // (8 * 1024 * 1024)
                            print(json.dumps({"event": "annotation_download", "bytes": offset, "total": EXPECTED_BYTES}), flush=True)
            if offset != EXPECTED_BYTES:
                raise ValueError("Incomplete annotation download")
            temporary.replace(path)
            return
        except (OSError, TimeoutError) as error:
            if attempt == 5:
                raise
            print(json.dumps({"event": "download_retry", "attempt": attempt + 1, "error": str(error)}), flush=True)
            time.sleep(min(2 ** attempt, 16))


def write_checked(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise FileExistsError(f"Existing annotation export differs: {path}")
    path.write_bytes(content)


def main():
    start = time.time()
    folder = ROOT / "data/bdd100k/annotations"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "bdd100k_labels.zip"
    download(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    license_bytes = urllib.request.urlopen("http://128.32.162.150/bdd100k/LICENSE.txt", timeout=20).read()
    write_checked(folder / "SOURCE_LICENSE.txt", license_bytes)
    summary = {}
    yolo_root = ROOT / "data/bdd100k/yolo_pilot"
    with zipfile.ZipFile(archive) as source:
        available = collections.Counter()
        for member in source.namelist():
            parts = Path(member).parts
            if len(parts) == 3 and parts[0] == "100k" and parts[-1].endswith(".json"):
                available[parts[1]] += 1
        for split in ("train", "val"):
            records = read_manifest(ROOT / f"data/manifests/bdd100k_{split}.jsonl")
            output, categories, dropped, clipped = [], collections.Counter(), collections.Counter(), 0
            for row in records:
                stem = Path(row["image_path"]).stem
                member = f"100k/{split}/{stem}.json"
                metadata = source.getinfo(member)
                if metadata.file_size > 2_000_000:
                    raise ValueError(f"Unexpected annotation size: {member}")
                raw = source.read(member)  # ZipFile verifies CRC during read.
                entry = json.loads(raw)
                if Path(entry["name"]).stem != stem:
                    raise ValueError(f"Annotation/image identity differs: {member}")
                frames = entry["frames"]
                if len(frames) != 1:
                    frames = [frame for frame in frames if frame.get("timestamp") == 10000]
                if len(frames) != 1:
                    raise ValueError(f"Ambiguous keyframe annotation: {member}")
                boxes, labels, attrs = [], [], []
                for obj in frames[0]["objects"]:
                    category = obj.get("category", "").replace("_", " ")
                    category = ALIASES.get(category, category)
                    box = obj.get("box2d")
                    if category not in CLASS_NAMES or not box:
                        dropped["non_detection_object"] += 1
                        continue
                    if obj.get("attributes", {}).get("crowd", False):
                        dropped["crowd"] += 1
                        continue
                    xyxy = [float(box[key]) for key in ("x1", "y1", "x2", "y2")]
                    if not all(math.isfinite(value) for value in xyxy):
                        raise ValueError(f"Nonfinite box: {member}")
                    bounded = [min(max(value, 0), row["width"] if index % 2 == 0 else row["height"])
                               for index, value in enumerate(xyxy)]
                    clipped += bounded != xyxy
                    if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                        dropped["invalid_area"] += 1
                        continue
                    boxes.append(bounded)
                    labels.append(CLASS_NAMES.index(category))
                    attrs.append(obj.get("attributes", {}))
                    categories[category] += 1
                labeled = {**row, "boxes": boxes, "labels": labels, "annotation_status": "available_legacy_bdd100k",
                           "attributes": entry.get("attributes", {}), "object_attributes": attrs,
                           "annotation_source_member": member, "annotation_sha256": hashlib.sha256(raw).hexdigest()}
                output.append(labeled)
                link = yolo_root / f"images/{split}/{stem}.jpg"
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.exists() or link.is_symlink():
                    if link.resolve() != Path(row["image_path"]).resolve():
                        raise FileExistsError(f"Existing image link differs: {link}")
                else:
                    link.symlink_to(row["image_path"])
                text = "".join(f"{label} {(box[0] + box[2]) / (2 * row['width']):.8f} {(box[1] + box[3]) / (2 * row['height']):.8f} {(box[2] - box[0]) / row['width']:.8f} {(box[3] - box[1]) / row['height']:.8f}\n"
                               for label, box in zip(labels, boxes))
                write_checked(yolo_root / f"labels/{split}/{stem}.txt", text.encode())
            write_checked(ROOT / f"data/manifests/bdd100k_{split}_labeled.jsonl",
                          "".join(json.dumps(row) + "\n" for row in output).encode())
            summary[split] = {"images_matched": len(output), "boxes": sum(len(r["boxes"]) for r in output),
                              "images_without_boxes": sum(not r["boxes"] for r in output),
                              "class_counts": {name: categories[name] for name in CLASS_NAMES},
                              "discarded": dict(dropped), "clipped_boxes": clipped}
    yaml = f"path: {json.dumps(str(yolo_root))}\ntrain: images/train\nval: images/val\nnames: {json.dumps(CLASS_NAMES)}\n"
    write_checked(yolo_root / "dataset.yaml", yaml.encode())
    config = load_config(ROOT / "configs/pilot_corruption.json")
    for split in ("train", "val"):
        config["data"][f"{split}_manifest"] = str(ROOT / f"data/manifests/bdd100k_{split}_labeled.jsonl")
    validation = validate_data(config)
    provenance = {"status": "pilot_detection_annotations_ready", "source_url": URL,
                  "source_transport": "http_public_download_endpoint", "archive_bytes": archive.stat().st_size,
                  "archive_sha256": digest.hexdigest(), "source_published_hash_available": False,
                  "format": "legacy_per_image_frames_objects_not_detection_2020",
                  "archive_split_counts": dict(available), "selected_split_counts": summary,
                  "category_aliases": ALIASES, "image_hash_and_box_validation": validation,
                  "yolo_dataset": str(yolo_root / "dataset.yaml"), "seconds": time.time() - start,
                  "existing_jepa_manifests_unchanged": True}
    atomic_json(ROOT / "data/bdd100k/labels_provenance.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
