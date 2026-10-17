"""Frozen image-level class-presence probes, never detector mAP."""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_checkpoint, atomic_json, load_config, seed_all, stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.data import CLASS_NAMES, DrivingPairs, to_tensor, validate_data
from robustdrive.models import VisionEncoder

START = time.time()
STATUS = ROOT / "runs/presence_probe_status.json"
OUT = ROOT / "runs/presence-probe-seed17"


def status(stage, state="running", **extra):
    atomic_json(STATUS, {"status": state, "stage": stage, "pid": os.getpid(),
                         "updated_unix": time.time(), **extra})
    print(json.dumps({"stage": stage, "status": state, **extra}), flush=True)


def target_labels(dataset):
    labels = torch.zeros(len(dataset), len(CLASS_NAMES))
    for index, row in enumerate(dataset.records):
        if row.get("annotation_status") != "available_legacy_bdd100k":
            raise ValueError("Probe requires real matched annotations")
        if row["labels"]:
            labels[index, list(set(row["labels"]))] = 1
    return labels


@torch.no_grad()
def features(encoder, dataset, family="clean", severity=0):
    result = []
    for start in range(0, len(dataset), 8):
        batch = []
        for index in range(start, min(start + 8, len(dataset))):
            clean = dataset.clean_image(index)
            seed = stable_seed(17, dataset.records[index]["image_path"], family, severity)
            batch.append(to_tensor(corrupt(clean, family, severity, seed)))
        images = torch.stack(batch).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pooled = encoder(images).float().mean(dim=1)
        result.append(pooled.cpu())
    result = torch.cat(result)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite frozen features")
    return result


def fit_head(train_features, labels, active):
    seed_all(17)
    mean, std = train_features.mean(dim=0), train_features.std(dim=0, unbiased=False).clamp_min(1e-5)
    inputs = (train_features - mean) / std
    head = nn.Linear(inputs.shape[1], len(CLASS_NAMES))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    positives = labels.sum(dim=0)
    pos_weight = ((len(labels) - positives) / positives.clamp_min(1)).clamp(0.1, 10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight[active])
    losses = []
    head.train()
    for epoch in range(50):
        order = torch.randperm(len(inputs), generator=torch.Generator().manual_seed(17 + epoch))
        accumulated = 0.0
        for start in range(0, len(order), 128):
            index = order[start:start + 128]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(inputs[index])[:, active], labels[index][:, active])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite probe loss")
            loss.backward()
            optimizer.step()
            accumulated += float(loss.detach()) * len(index)
        losses.append(accumulated / len(inputs))
    head.eval()
    return head, mean, std, losses


@torch.no_grad()
def evaluate_head(head, features, mean, std, labels, active):
    scores = head((features - mean) / std).sigmoid().numpy()
    per_class = []
    for index, name in enumerate(CLASS_NAMES):
        positives = int(labels[:, index].sum())
        supported = bool(active[index]) and 0 < positives < len(labels)
        per_class.append({"class": name, "validation_positive_images": positives,
                          "supported": supported,
                          "average_precision": float(average_precision_score(labels[:, index].numpy(), scores[:, index])) if supported else None})
    values = [row["average_precision"] for row in per_class if row["supported"]]
    if not values:
        raise ValueError("No class has both positive and negative validation images")
    return {"class_presence_macro_ap": float(np.mean(values)), "supported_classes": len(values),
            "per_class": per_class}


def main():
    status("waiting_for_labels_and_diagnostics")
    ready = ROOT / "data/bdd100k/labels_provenance.json"
    while True:
        state_file = ROOT / "runs/diagnostics_status.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        if state.get("status") == "failed":
            raise RuntimeError("Representation diagnostics failed")
        if ready.exists() and state.get("status") == "complete":
            break
        if time.time() - START > 1800:
            raise TimeoutError("Labels/diagnostics not ready after 30 minutes")
        time.sleep(5)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Probe run already has artifacts")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    config = load_config(ROOT / "configs/continue10_corruption.json")
    for split in ("train", "val"):
        config["data"][f"{split}_manifest"] = str(ROOT / f"data/manifests/bdd100k_{split}_labeled.jsonl")
    data_validation = validate_data(config)
    datasets = {split: DrivingPairs(config["data"][f"{split}_manifest"], config["data"]["image_size"],
                                   ["clean"], [1], 17) for split in ("train", "val")}
    labels = {split: target_labels(dataset) for split, dataset in datasets.items()}
    positives = labels["train"].sum(dim=0)
    active = (positives > 0) & (positives < len(labels["train"]))
    protocol = {"task": "image_level_multilabel_class_presence", "not_detector_map": True,
                "encoder_frozen": True, "encoder_checkpoint_selection": "final_last_pt_step2560_plus_prior200",
                "train_images": len(datasets["train"]), "validation_images": len(datasets["val"]),
                "head": "standardized_384_to_10_linear_sigmoid", "standardization_fit_split": "train_only",
                "epochs": 50, "batch_size": 128, "lr": 0.001, "weight_decay": 0.01,
                "loss": "BCEWithLogitsLoss_class_balancing_clamped_0.1_to_10", "seed": 17,
                "head_selection": "fixed_final_epoch_without_validation_tuning",
                "train_positive_images": {name: int(positives[i]) for i, name in enumerate(CLASS_NAMES)},
                "torch": str(torch.__version__), "sklearn": sklearn.__version__,
                "annotation_provenance": json.loads(ready.read_text()), "data_validation": data_validation,
                "manifest_sha256": {split: hashlib.sha256(Path(config["data"][f"{split}_manifest"]).read_bytes()).hexdigest()
                                    for split in ("train", "val")}}
    atomic_json(OUT / "protocol.json", protocol)
    results = {}
    for experiment in ("corruption", "clean", "random"):
        status(f"{experiment}_extract_train")
        seed_all(17)
        spec = config["model"]
        encoder = VisionEncoder(config["data"]["image_size"], spec["patch_size"], spec["dim"], spec["depth"], spec["heads"])
        source_hash = None
        if experiment != "random":
            checkpoint = ROOT / f"runs/continue10-{experiment}-seed17/last.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            encoder.load_state_dict({k.removeprefix("online."): v for k, v in saved["model"].items() if k.startswith("online.")})
            with checkpoint.open("rb") as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            source_hash = digest.hexdigest()
            del saved
        encoder.cuda().eval().requires_grad_(False)
        train_features = features(encoder, datasets["train"])
        head, mean, std, losses = fit_head(train_features, labels["train"], active)
        atomic_checkpoint(OUT / f"{experiment}_head.pt", {"format_version": 1, "head": head.state_dict(),
                          "feature_mean": mean, "feature_std": std, "classes": CLASS_NAMES,
                          "encoder_checkpoint_sha256": source_hash, "protocol": protocol})
        atomic_checkpoint(OUT / f"{experiment}_train_features.pt", {"features": train_features,
                          "labels": labels["train"], "encoder_checkpoint_sha256": source_hash})
        conditions = []
        for family, severity in [("clean", 0)] + [(f, s) for f in config["corruption"]["families"] for s in range(1, 6)]:
            status(f"{experiment}_{family}_{severity}")
            val_features = features(encoder, datasets["val"], family, severity)
            row = {"family": family, "severity": severity,
                   **evaluate_head(head, val_features, mean, std, labels["val"], active)}
            conditions.append(row)
            print(json.dumps({"experiment": experiment, **row}), flush=True)
        results[experiment] = {"encoder_checkpoint_sha256": source_hash, "conditions": conditions,
                               "final_train_loss": losses[-1], "head_training_losses": losses,
                               "clean_class_presence_macro_ap": conditions[0]["class_presence_macro_ap"],
                               "corrupt_class_presence_macro_ap": float(np.mean([r["class_presence_macro_ap"] for r in conditions[1:]]))}
        atomic_json(OUT / f"{experiment}_results.json", results[experiment])
        del encoder
        torch.cuda.empty_cache()
    summary = {"status": "frozen_class_presence_probe_complete", "protocol": protocol,
               "results": results, "wall_seconds_including_wait": time.time() - START,
               "interpretation": "Image-level category ranking on a small legacy-labeled subset; does not measure box localization, detector mAP or driving safety."}
    atomic_json(ROOT / "runs/presence_probe_summary.json", summary)
    status("complete", state="complete", summary=str(ROOT / "runs/presence_probe_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=str(error))
        traceback.print_exc()
        raise
