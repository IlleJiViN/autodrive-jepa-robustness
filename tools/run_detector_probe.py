"""Train equal-budget, frozen-encoder detection heads and score actual boxes."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import sys
import time
import traceback

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_checkpoint, atomic_json, load_config, seed_all, stable_seed
from robustdrive.corruptions import FAMILIES, corrupt
from robustdrive.data import CLASS_NAMES, DrivingPairs, to_tensor, validate_data
from robustdrive.detector import LEVEL_NAMES, STRIDES, FrozenFeatureDetector, assign_targets, coco_ap, decode, detector_loss
from robustdrive.models import VisionEncoder

RUN = ROOT / "runs/detector-probe-seed17-v2"
STATUS = ROOT / "runs/detector_probe_status.json"
START = time.time()


def status(stage, state="running", **extra):
    atomic_json(STATUS, {"status": state, "stage": stage, "pid": os.getpid(), "updated_unix": time.time(), **extra})
    print(json.dumps({"status": state, "stage": stage, **extra}), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ImageRows(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, to_tensor(self.dataset.clean_image(index))


def encoder_for(name, config):
    spec = config["model"]
    seed_all(config["seed"])
    encoder = VisionEncoder(config["data"]["image_size"], spec["patch_size"], spec["dim"], spec["depth"], spec["heads"])
    source_hash = None
    source_step = 0
    if name != "random":
        source = ROOT / f"runs/continue10-{name}-seed17/final_encoder.pt"
        saved = torch.load(source, map_location="cpu", weights_only=True)
        if saved["selection_criterion"] != "final_completed_schedule" or saved["step"] != 2560:
            raise ValueError(f"Expected final 2560-step encoder: {source}")
        if saved["config"]["model"] != config["model"] or saved["config"]["data"]["image_size"] != config["data"]["image_size"]:
            raise ValueError("Encoder geometry differs from detector configuration")
        encoder.load_state_dict(saved["encoder_state"], strict=True)
        source_hash = sha256(source)
        source_step = saved["step"] + saved["warm_start"]["ancestor_optimizer_steps"]
        del saved
    else:
        digest = hashlib.sha256()
        for value in encoder.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        source_hash = digest.hexdigest()
    encoder.cuda().eval().requires_grad_(False)
    return encoder, source_hash, source_step


def make_features(name, encoder, dataset, output, config, encoder_hash, manifest_hash):
    marker = output.with_name(output.name + ".complete.json")
    partial = output.with_name(output.name + ".partial")
    if output.exists() or marker.exists():
        if not output.is_file() or not marker.is_file():
            raise FileExistsError(f"Incomplete feature cache artifacts must be preserved: {output}")
        metadata = json.loads(marker.read_text())
        expected = {"encoder_sha256": encoder_hash, "manifest_sha256": manifest_hash,
                    "shape": [len(dataset), 24 * 40, 384], "dtype": "float16"}
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Existing feature cache identity mismatch: {output}")
        if metadata.get("cache_sha256") != sha256(output) or metadata.get("complete") is not True:
            raise ValueError(f"Existing feature cache checksum or completion marker mismatch: {output}")
        array = np.load(output, mmap_mode="r")
        if array.shape != tuple(expected["shape"]) or array.dtype != np.float16:
            raise ValueError(f"Existing feature cache has wrong shape or type: {output}")
        return array
    if partial.exists():
        raise FileExistsError(f"Incomplete feature cache must be preserved: {partial}")
    output.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16,
                                     shape=(len(dataset), 24 * 40, 384))
    loader = DataLoader(ImageRows(dataset), batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=False)
    for batch_index, (indices, images) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tokens = encoder(images).float()
        if not torch.isfinite(tokens).all():
            raise FloatingPointError(f"Nonfinite {name} encoder features at batch {batch_index}")
        array[indices.numpy()] = tokens.cpu().numpy().astype(np.float16, copy=False)
        if batch_index % 64 == 0 or batch_index + 1 == len(loader):
            array.flush()
            status(f"{name}_feature_cache", images_done=min((batch_index + 1) * 8, len(dataset)),
                   images_total=len(dataset), feature_cache=str(output))
    array.flush()
    del array
    partial.replace(output)
    atomic_json(marker, {"encoder_sha256": encoder_hash, "manifest_sha256": manifest_hash,
                         "shape": [len(dataset), 24 * 40, 384], "dtype": "float16",
                         "cache_sha256": sha256(output), "complete": True})
    return np.load(output, mmap_mode="r")


def targets_for_manifest(dataset, image_size, shapes):
    cached = []
    collisions = 0
    for index in range(len(dataset)):
        targets, dropped = assign_targets(dataset.records, [index], shapes, image_size)
        cached.append(targets)
        collisions += dropped
    return cached, collisions


def batch_targets(cached, indices):
    targets = []
    for level in range(3):
        targets.append({key: torch.cat([cached[int(index)][level][key] for index in indices], dim=0)
                        for key in ("present", "classes", "boxes")})
    return targets


def train_head(name, cache, dataset, config, shapes, cached_targets, collision_drops,
               dataset_hash, encoder_hash, encoder_steps):
    experiment = RUN / name
    experiment.mkdir(parents=True, exist_ok=True)
    seed_all(1717)
    model = FrozenFeatureDetector().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs, batch_size = config["epochs"], config["batch_size"]
    history = []
    final_path = experiment / "detector.pt"
    latest_path = experiment / "training_latest.pt"
    identity = {"encoder_name": name, "encoder_sha256": encoder_hash,
                "encoder_cumulative_steps": encoder_steps, "labeled_manifest_sha256": dataset_hash,
                "config": config, "epochs": epochs}
    if final_path.is_file():
        saved = torch.load(final_path, map_location="cpu", weights_only=True)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Existing detector checkpoint identity mismatch: {final_path}")
        model.load_state_dict(saved["detector"], strict=True)
        model.cuda().eval()
        return model, saved
    start_epoch = 0
    if latest_path.is_file():
        saved = torch.load(latest_path, map_location="cpu", weights_only=True)
        if any(saved.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Existing training checkpoint identity mismatch: {latest_path}")
        model.load_state_dict(saved["detector"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["training_history"]
        start_epoch = int(saved["epoch_completed"])
        if not 0 <= start_epoch <= epochs:
            raise ValueError(f"Invalid resumable epoch {start_epoch} in {latest_path}")
        status(f"{name}_head_resume", epoch_completed=start_epoch, epochs=epochs)
    metrics_path = experiment / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as log:
        for row in history:
            log.write(json.dumps(row) + "\n")
        log.flush()
        for epoch in range(start_epoch, epochs):
            model.train()
            generator = torch.Generator().manual_seed(config["seed"] + epoch)
            order = torch.randperm(len(dataset), generator=generator).tolist()
            epoch_loss, steps, assigned = 0.0, 0, 0
            for start in range(0, len(order), batch_size):
                indices = order[start:start + batch_size]
                tokens = torch.from_numpy(np.array(cache[indices], copy=True)).cuda(non_blocking=True)
                targets = batch_targets(cached_targets, indices)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predictions = model(tokens)
                    loss, parts = detector_loss(predictions, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite detector loss at epoch {epoch + 1}, step {steps + 1}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss += float(loss.detach())
                steps += 1
                assigned += parts["positive_assignments"]
            row = {"epoch": epoch + 1, "steps": steps, "loss": epoch_loss / max(steps, 1),
                   "positive_assignments": assigned, "train_images": len(dataset)}
            if not np.isfinite(row["loss"]):
                raise FloatingPointError("Nonfinite epoch detector loss")
            history.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            status(f"{name}_head_training", epoch=epoch + 1, epochs=epochs, loss=row["loss"])
            resumable = {**identity, "format_version": 1,
                         "detector": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                         "optimizer": optimizer.state_dict(), "epoch_completed": epoch + 1,
                         "training_history": history}
            atomic_checkpoint(latest_path, resumable)
    checkpoint = {**identity, "format_version": 1,
                  "detector": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "detector_class": "FrozenFeatureDetector", "encoder_frozen": True,
                  "encoder_grid": [24, 40], "feature_cache_sha256": sha256(cache.filename),
                  "training_history": history, "selection": "fixed_final_epoch_no_validation_selection",
                  "positive_slot_collisions_dropped": collision_drops}
    atomic_checkpoint(final_path, checkpoint)
    model.cuda().eval()
    return model, checkpoint


@torch.no_grad()
def extract_val_tokens(encoder, images):
    images = images.cuda(non_blocking=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return encoder(images).float()


@torch.no_grad()
def predict_condition(name, encoder, model, dataset, family, severity, config, shapes):
    model.eval()
    predictions, ground_truths = {}, {}
    loader = DataLoader(ImageRows(dataset), batch_size=8, shuffle=False, num_workers=4,
                        pin_memory=True, persistent_workers=False)
    for batch_index, (indices, clean_batch) in enumerate(loader):
        images = []
        for local, index_tensor in enumerate(indices):
            index = int(index_tensor)
            if family == "clean":
                image = dataset.clean_image(index)
            else:
                image = dataset.clean_image(index)
                seed = stable_seed(17, dataset.records[index]["image_path"], family, severity)
                image = corrupt(image, family, severity, seed)
            images.append(to_tensor(image))
        tokens = extract_val_tokens(encoder, torch.stack(images))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(tokens)
        decoded = decode(output, config["image_size"], config["max_detections_per_image"],
                         nms_iou=config["nms_iou"],
                         pre_nms_candidates=config["pre_nms_candidates_per_image"])
        for local, index_tensor in enumerate(indices):
            index = int(index_tensor)
            predictions[index] = decoded[local]
            record = dataset.records[index]
            sx, sy = config["image_size"][1] / record["width"], config["image_size"][0] / record["height"]
            ground_truths[index] = (np.asarray([[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
                                               for b in record["boxes"]], dtype=np.float32).reshape(-1, 4),
                                    np.asarray(record["labels"], dtype=np.int64))
        if batch_index % 32 == 0:
            status(f"{name}_eval_{family}_{severity}", images_done=min((batch_index + 1) * 8, len(dataset)),
                   images_total=len(dataset))
    train_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for record in _TRAIN_RECORDS:
        if record["labels"]:
            train_counts[np.asarray(record["labels"], dtype=np.int64)] += 1
    val_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for _, labels in ground_truths.values():
        if len(labels):
            val_counts[np.unique(labels)] += 1
    metrics = coco_ap(predictions, ground_truths, train_counts, val_counts)
    return {"family": family, "severity": severity, "images": len(dataset), "train_class_images": train_counts.tolist(),
            "validation_class_images": val_counts.tolist(), "prediction_count": sum(map(len, predictions.values())), **metrics}


_TRAIN_RECORDS = []


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This detector probe requires the Coder CUDA device")
    torch.set_num_threads(4)
    config = json.loads((ROOT / "configs/detector_pilot.json").read_text())
    jepa_config = load_config(ROOT / "configs/continue10_corruption.json")
    jepa_config["data"]["train_manifest"] = str(ROOT / "data/manifests/bdd100k_train_labeled.jsonl")
    jepa_config["data"]["val_manifest"] = str(ROOT / "data/manifests/bdd100k_val_labeled.jsonl")
    validation = validate_data(jepa_config)
    train_data = DrivingPairs(jepa_config["data"]["train_manifest"], config["image_size"], ["clean"], [1], 17)
    val_data = DrivingPairs(jepa_config["data"]["val_manifest"], config["image_size"], ["clean"], [1], 17)
    _TRAIN_RECORDS.extend(train_data.records)
    for item in train_data.records + val_data.records:
        if item.get("annotation_status") != "available_legacy_bdd100k":
            raise ValueError("Detector probe requires matched detection boxes")
    shapes = [(STRIDES[i], config["image_size"][0] // stride, config["image_size"][1] // stride)
              for i, stride in enumerate(STRIDES)]
    train_targets, train_drops = targets_for_manifest(train_data, config["image_size"], shapes)
    val_drops = targets_for_manifest(val_data, config["image_size"], shapes)[1]
    train_hash = sha256(jepa_config["data"]["train_manifest"])
    val_hash = sha256(jepa_config["data"]["val_manifest"])
    protocol = {**config, "task": "object_detection", "head": "anchor_free_three_scale_3_slots_per_cell",
                "pyramid_levels": list(LEVEL_NAMES), "strides": list(STRIDES),
                "head_output": "objectness_class_xy_offset_log_wh", "assignment": "box_area_scale_and_center_cell",
                "box_loss_transform": "sigmoid_xy_offsets_linear_log_wh",
                "training_split": "labeled_train_2048_clean_images", "validation_split": "labeled_val_256",
                "train_images": len(train_data), "val_images": len(val_data), "train_assignment_collisions_dropped": train_drops,
                "val_assignment_collisions_dropped": val_drops, "data_validation": validation,
                "train_manifest_sha256": train_hash, "val_manifest_sha256": val_hash,
                "evaluation": "class-aware-NMS, per-class greedy IoU match, AP50 and mean AP50:95",
                "confidence_filter": 1e-5, "reporting_scope": "single_seed_small_labeled_subset_not_driving_safety"}
    RUN.mkdir(parents=True, exist_ok=True)
    protocol_path = RUN / "protocol.json"
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f"Existing detector protocol differs from requested run: {protocol_path}")
    elif any(RUN.iterdir()):
        raise FileExistsError(f"Run has artifacts but no protocol; preserving them: {RUN}")
    else:
        atomic_json(protocol_path, protocol)
    cache_dir = RUN / "feature-cache"
    cache_dir.mkdir(exist_ok=True)
    results = {}
    for name in ("random", "clean", "corruption"):
        status(f"{name}_encoder_load")
        encoder, source_hash, source_steps = encoder_for(name, jepa_config)
        cache_path = cache_dir / f"{name}-train-tokens.npy"
        train_cache = make_features(name, encoder, train_data, cache_path, config, source_hash, train_hash)
        model, checkpoint = train_head(name, train_cache, train_data, config, shapes, train_targets, train_drops,
                                       train_hash, source_hash, source_steps)
        experiment = RUN / name
        atomic_json(experiment / "checkpoint_metadata.json", {k: v for k, v in checkpoint.items() if k != "detector"})
        model_hash = sha256(experiment / "detector.pt")
        evaluations = []
        conditions = [("clean", 0)] + [(family, severity) for family in FAMILIES
                                        for severity in config["eval_corruption_severities"]]
        for family, severity in conditions:
            eval_path = experiment / f"eval-{family}-{severity}.json"
            if eval_path.is_file():
                measured = json.loads(eval_path.read_text())
                expected = {"family": family, "severity": severity, "encoder_sha256": source_hash,
                            "detector_checkpoint_sha256": model_hash}
                if any(measured.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Existing evaluation identity mismatch: {eval_path}")
                evaluations.append(measured)
                continue
            status(f"{name}_evaluate_{family}_{severity}")
            measured = predict_condition(name, encoder, model, val_data, family, severity, config, shapes)
            measured["encoder_sha256"] = source_hash
            measured["detector_checkpoint_sha256"] = model_hash
            evaluations.append(measured)
            atomic_json(eval_path, measured)
        results[name] = {"encoder_sha256": source_hash, "encoder_cumulative_steps": source_steps,
                         "detector_checkpoint_sha256": model_hash, "checkpoint_bytes": (experiment / "detector.pt").stat().st_size,
                         "training_epochs": config["epochs"], "clean_map50": evaluations[0]["map50"],
                         "clean_map50_95": evaluations[0]["map50_95"],
                         "corrupt_mean_map50": float(np.mean([x["map50"] for x in evaluations[1:]])),
                         "corrupt_mean_map50_95": float(np.mean([x["map50_95"] for x in evaluations[1:]])),
                         "conditions": evaluations, "training_final_loss": checkpoint["training_history"][-1]["loss"]}
        atomic_json(experiment / "results.json", results[name])
        del encoder, model, train_cache
        torch.cuda.empty_cache()
    summary = {"status": "three_scale_detector_probe_complete", "protocol": protocol, "results": results,
               "comparisons": {"corruption_minus_clean_corrupt_map50": results["corruption"]["corrupt_mean_map50"] - results["clean"]["corrupt_mean_map50"],
                               "corruption_minus_random_corrupt_map50": results["corruption"]["corrupt_mean_map50"] - results["random"]["corrupt_mean_map50"],
                               "corruption_minus_clean_corrupt_map50_95": results["corruption"]["corrupt_mean_map50_95"] - results["clean"]["corrupt_mean_map50_95"],
                               "corruption_minus_random_corrupt_map50_95": results["corruption"]["corrupt_mean_map50_95"] - results["random"]["corrupt_mean_map50_95"]},
               "total_wall_seconds": time.time() - START,
               "interpretation": "Actual legacy-box detection on val256 with one seed and a trained detector; not full BDD100K, multiple-seed evidence, real-weather validation or driving safety."}
    atomic_json(ROOT / "runs/detector_probe_summary.json", summary)
    status("complete", state="complete", summary=str(ROOT / "runs/detector_probe_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=str(error))
        traceback.print_exc()
        raise
