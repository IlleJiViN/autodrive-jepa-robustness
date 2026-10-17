from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import stable_seed
from .corruptions import FAMILIES, corrupt

CLASS_NAMES = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest missing: {path}. Import local BDD100K data first.")
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("image_path", "scene_id", "split", "boxes", "labels"):
            if key not in row:
                raise ValueError(f"{path}:{line_no}: missing {key}")
        p = Path(row["image_path"])
        row["image_path"] = str(p if p.is_absolute() else path.parent / p)
        if len(row["boxes"]) != len(row["labels"]):
            raise ValueError(f"{path}:{line_no}: boxes/labels length mismatch")
        records.append(row)
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def validate_data(config: dict, check_images: bool = True) -> dict:
    height, width = config["data"]["image_size"]
    patch = config["model"]["patch_size"]
    if min(height, width, patch) <= 0 or height % patch or width % patch:
        raise ValueError("Image dimensions must be positive multiples of patch_size")
    model = config["model"]
    if not 0 < model["mask_ratio"] < 1:
        raise ValueError("mask_ratio must be between 0 and 1")
    if model["dim"] % model["heads"] or model["predictor_dim"] % model["predictor_heads"]:
        raise ValueError("Embedding dimensions must be divisible by attention heads")
    for family in config["corruption"]["families"]:
        if family not in FAMILIES:
            raise ValueError(f"Unsupported corruption family: {family}")
    train_families = config["corruption"].get("train_families", config["corruption"]["families"])
    if not train_families or any(family not in ("clean", *FAMILIES) for family in train_families):
        raise ValueError("train_families must contain clean or supported corruptions")
    for key in ("train_severities", "eval_severities"):
        if not config["corruption"][key] or any(s not in range(1, 6) for s in config["corruption"][key]):
            raise ValueError(f"Invalid {key}")
    summaries, split_sets = {}, {}
    for split, key in (("train", "train_manifest"), ("val", "val_manifest")):
        records = read_manifest(config["data"][key])
        paths, groups, hashes = set(), set(), set()
        hash_checked, unlabeled = 0, 0
        for row in records:
            image = Path(row["image_path"]).resolve()
            if row["split"] != split or not str(row["scene_id"]).strip():
                raise ValueError(f"Bad split/scene_id: {image}")
            if str(image) in paths:
                raise ValueError(f"Duplicate image path inside {split}: {image}")
            paths.add(str(image))
            groups.add(str(row["scene_id"]))
            if not image.is_file():
                raise FileNotFoundError(image)
            if row.get("annotation_status") == "unavailable_image_only":
                unlabeled += 1
            if row.get("image_sha256"):
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest != row["image_sha256"]:
                    raise ValueError(f"Image hash differs from manifest: {image}")
                hashes.add(digest)
                hash_checked += 1
            if check_images:
                with Image.open(image) as im:
                    img_width, img_height = im.size
                    im.verify()
                for box, label in zip(row["boxes"], row["labels"]):
                    if len(box) != 4 or not all(np.isfinite(box)) or not 0 <= int(label) < len(CLASS_NAMES):
                        raise ValueError(f"Bad box/label: {image}")
                    x1, y1, x2, y2 = box
                    if not 0 <= x1 < x2 <= img_width or not 0 <= y1 < y2 <= img_height:
                        raise ValueError(f"Box outside image: {image}: {box}")
        split_sets[split] = (paths, groups, hashes)
        summaries[split] = {"images": len(records), "scene_groups": len(groups),
                            "image_hashes_checked": hash_checked, "images_without_detection_annotations": unlabeled}
    if split_sets["train"][0] & split_sets["val"][0] or split_sets["train"][1] & split_sets["val"][1]:
        raise ValueError("Train/validation image paths or scene groups overlap")
    if split_sets["train"][2] & split_sets["val"][2]:
        raise ValueError("Identical image bytes occur across train/validation")
    return {"status": "manifest_valid", "splits": summaries, "image_headers_checked": check_images,
            "content_duplicates_checked": all(s["image_hashes_checked"] == s["images"] for s in summaries.values())}


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0


class DrivingPairs(Dataset):
    def __init__(self, manifest: str, image_size: list[int], families: list[str], severities: list[int], seed: int):
        self.records = read_manifest(manifest)
        self.image_size = image_size
        self.families, self.severities, self.seed = families, severities, seed
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def clean_image(self, index: int) -> Image.Image:
        with Image.open(self.records[index]["image_path"]) as raw:
            return raw.convert("RGB").resize(tuple(reversed(self.image_size)), Image.Resampling.BILINEAR)

    def __getitem__(self, index: int):
        clean = self.clean_image(index)
        seed = stable_seed(self.seed, self.epoch, self.records[index]["image_path"])
        rng = np.random.default_rng(seed)
        family = str(rng.choice(self.families))
        severity = 0 if family == "clean" else int(rng.choice(self.severities))
        view = corrupt(clean, family, severity, seed)
        return {"clean": to_tensor(clean), "view": to_tensor(view), "family": family, "severity": severity}


def import_bdd(annotations: str, images: str, output: str, split: str) -> dict:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {destination}")
    source = json.loads(Path(annotations).read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError("Expected BDD100K detection JSON list")
    records = []
    for entry in source:
        path = (Path(images) / entry["name"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Annotation image missing: {path}")
        with Image.open(path) as im:
            width, height = im.size
        boxes, labels = [], []
        ignored = 0
        for obj in entry.get("labels", []):
            category = obj.get("category", "").replace("_", " ")
            box = obj.get("box2d")
            if category not in CLASS_NAMES or not box or obj.get("attributes", {}).get("crowd", False):
                ignored += 1
                continue
            xyxy = [max(0.0, float(box["x1"])), max(0.0, float(box["y1"])),
                    min(float(width), float(box["x2"])), min(float(height), float(box["y2"]))]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                ignored += 1
                continue
            boxes.append(xyxy)
            labels.append(CLASS_NAMES.index(category))
        records.append({"image_path": str(path), "scene_id": Path(entry["name"]).stem,
                        "split": split, "boxes": boxes, "labels": labels, "width": width, "height": height,
                        "attributes": entry.get("attributes", {}), "ignored_annotations": ignored})
    if not records:
        raise ValueError("No images in annotation file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    temp.replace(destination)
    return {"manifest": str(destination), "images": len(records), "boxes": sum(len(r["boxes"]) for r in records),
            "split": split, "class_names": CLASS_NAMES}
