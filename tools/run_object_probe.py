"""Known-box semantic retention probe. This does not measure detection or control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_checkpoint, atomic_json, load_config, seed_all, stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.data import CLASS_NAMES, DrivingPairs, to_tensor, validate_data
from robustdrive.models import VisionEncoder
from robustdrive.object_pretraining import box_patch_weights


def object_metadata(dataset, image_size):
    labels, geometry, cases, groups = [], [], [], []
    for index, row in enumerate(dataset.records):
        if row.get("annotation_status") != "available_legacy_bdd100k":
            raise ValueError("Semantic labels must be human annotations, not YOLO pseudo-labels")
        for obj, (box, label) in enumerate(zip(row["boxes"], row["labels"])):
            x1, y1, x2, y2 = np.array(box) / [row["width"], row["height"], row["width"], row["height"]]
            width, height = x2 - x1, y2 - y1
            geometry.append([(x1 + x2) / 2, (y1 + y2) / 2, width, height, np.log(width * height + 1e-8)])
            attributes = row.get("object_attributes", [])
            attr = attributes[obj] if obj < len(attributes) else {}
            cases.append({"night": row.get("attributes", {}).get("timeofday") == "night",
                          "small": width * image_size[1] * height * image_size[0] < 32 ** 2,
                          "occluded": attr.get("occluded") is True})
            labels.append(label)
            groups.append(index)
    return torch.tensor(labels, dtype=torch.long), torch.tensor(geometry, dtype=torch.float32), cases, np.array(groups)


@torch.no_grad()
def extract(encoder, dataset, config, device, family="clean", severity=0):
    features = []
    batch_size = config["training"]["batch_size"]
    for start in range(0, len(dataset), batch_size):
        indices = range(start, min(start + batch_size, len(dataset)))
        images = []
        for index in indices:
            seed = stable_seed(config["seed"], dataset.records[index]["scene_id"], family, severity)
            images.append(to_tensor(corrupt(dataset.clean_image(index), family, severity, seed)))
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and config["training"]["amp_bfloat16"]):
            tokens = encoder(torch.stack(images).to(device)).float()
        for local, index in enumerate(indices):
            row = dataset.records[index]
            boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
            boxes /= torch.tensor([row["width"], row["height"], row["width"], row["height"]])
            if len(boxes):
                weights = box_patch_weights(boxes, encoder.grid).to(device)
                features.append((weights @ tokens[local]).cpu())
    result = torch.cat(features)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite object embeddings")
    return result


def fit_head(features, labels, active, seed, epochs):
    seed_all(seed)
    keep = torch.isin(labels, active)
    x, y = features[keep], labels[keep]
    mapping = torch.full((len(CLASS_NAMES),), -1, dtype=torch.long)
    mapping[active] = torch.arange(len(active))
    y = mapping[y]
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x - mean) / std
    head = nn.Linear(x.shape[1], len(active))
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001, weight_decay=0.01)
    counts = torch.bincount(y, minlength=len(active)).float()
    weights = counts.clamp_min(1).rsqrt()
    weights /= weights.mean()
    for epoch in range(epochs):
        order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + epoch))
        for start in range(0, len(order), 256):
            indices = order[start:start + 256]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(x[indices]), y[indices], weight=weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite semantic probe loss")
            loss.backward()
            optimizer.step()
    return head.eval(), mean, std


@torch.no_grad()
def probabilities(head, features, mean, std):
    return head((features - mean) / std).softmax(-1).numpy()


def metrics(scores, labels, active):
    active = active.numpy() if torch.is_tensor(active) else np.asarray(active)
    labels = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)
    supported = np.isin(labels, active)
    scores, labels = scores[supported], labels[supported]
    if not len(labels):
        return {"objects": 0}
    predictions = active[scores.argmax(1)]
    correct = predictions == labels
    per_class = []
    for slot, label in enumerate(active):
        positive = labels == label
        n = int(positive.sum())
        per_class.append({"class": CLASS_NAMES[int(label)], "objects": n,
            "recall": float(correct[positive].mean()) if n else None,
            "ap": float(average_precision_score(positive, scores[:, slot])) if 0 < n < len(labels) else None})
    confidence = scores.max(1)
    ece = 0.0
    for lo in np.linspace(0, 1, 16)[:-1]:
        include = (confidence >= lo) & (confidence < lo + 1 / 15 + (1e-8 if lo > .93 else 0))
        if include.any():
            ece += include.mean() * abs(confidence[include].mean() - correct[include].mean())
    onehot = (labels[:, None] == active[None, :]).astype(np.float32)
    aps = [p["ap"] for p in per_class if p["ap"] is not None]
    return {"objects": len(labels), "accuracy": float(correct.mean()),
            "balanced_accuracy": float(np.mean([p["recall"] for p in per_class if p["recall"] is not None])),
            "macro_ap": float(np.mean(aps)) if aps else None, "ece_15_bins": float(ece),
            "brier_score": float(np.square(scores - onehot).sum(1).mean()), "per_class": per_class}


def case_metrics(scores, labels, active, cases):
    labels = labels.numpy()
    return {name: metrics(scores[mask], labels[mask], active)
            for name in ("night", "small", "occluded")
            for mask in [np.array([case[name] for case in cases], dtype=bool)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/object_pilot.json")
    parser.add_argument("--checkpoint", action="append", default=[], help="name=path, repeated per learned encoder")
    parser.add_argument("--include-random", action="store_true")
    parser.add_argument("--run", required=True)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.checkpoint and not args.include_random or args.probe_epochs < 1:
        parser.error("At least one encoder and positive probe-epochs are required")
    config = load_config(args.config)
    validate_data(config)
    output, device = Path(args.run), torch.device(args.device)
    if output.exists():
        raise FileExistsError(output)
    datasets = {split: DrivingPairs(config["data"][split + "_manifest"], config["data"]["image_size"],
                                    ["clean"], [1], config["seed"]) for split in ("train", "val")}
    metadata = {split: object_metadata(data, config["data"]["image_size"]) for split, data in datasets.items()}
    train_labels, train_geometry, _, _ = metadata["train"]
    val_labels, val_geometry, cases, _ = metadata["val"]
    active = torch.tensor(sorted(set(train_labels.tolist()) & set(val_labels.tolist())), dtype=torch.long)
    if len(active) < 2:
        raise ValueError("Semantic classification requires at least two supported classes")
    names = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        if name in names or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name):
            raise ValueError("Use unique lowercase names without path separators")
        names[name] = Path(path).resolve()
    if args.include_random:
        if "random" in names:
            raise ValueError("random is reserved for the untrained control")
        names["random"] = None
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    protocol = {"task": "oracle_box_conditioned_object_classification_not_detection_or_driving_actions",
        "training_scope": "clean-only frozen-encoder linear object classifier; class-balanced cross entropy",
        "probe_epochs": args.probe_epochs, "seed": config["seed"],
        "supported_classes": [CLASS_NAMES[int(i)] for i in active],
        "train_images": len(datasets["train"]), "val_images": len(datasets["val"]),
        "train_objects": len(train_labels), "val_objects": len(val_labels),
        "small_definition": "box area < 32^2 pixels after resizing; boxes do not change with corruption",
        "checkpoint_selection": "final_completed_schedule", "synthetic_corruptions_only": True}
    atomic_json(output / "protocol.json", protocol)
    geom_head, mean, std = fit_head(train_geometry, train_labels, active, config["seed"], args.probe_epochs)
    geometry_result = metrics(probabilities(geom_head, val_geometry, mean, std), val_labels, active)
    results = {"protocol": protocol, "geometry_only_oracle_control": geometry_result, "encoders": {}}
    conditions = [("clean", 0)] + [(f, s) for f in config["corruption"]["families"]
                                   for s in config["corruption"]["eval_severities"]]
    for name, path in names.items():
        seed_all(config["seed"])
        spec = config["model"]
        encoder = VisionEncoder(config["data"]["image_size"], spec["patch_size"], spec["dim"], spec["depth"], spec["heads"])
        digest = None
        if path:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if saved["config"]["model"] != spec or saved["config"]["data"]["image_size"] != config["data"]["image_size"]:
                raise ValueError("Every encoder must use the same model and image geometry")
            encoder.load_state_dict(saved["encoder_state"], strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        encoder.to(device).eval().requires_grad_(False)
        train_features = extract(encoder, datasets["train"], config, device)
        head, mean, std = fit_head(train_features, train_labels, active, config["seed"], args.probe_epochs)
        atomic_checkpoint(output / (name + "_head.pt"), {"head": head.state_dict(), "mean": mean,
                          "std": std, "classes": active, "encoder_sha256": digest, "protocol": protocol})
        clean_features = None
        clean_correct = None
        rows, raw_scores = [], {}
        for family, severity in conditions:
            features = extract(encoder, datasets["val"], config, device, family, severity)
            scores = probabilities(head, features, mean, std)
            correct = active.numpy()[scores.argmax(1)] == val_labels.numpy()
            row = {"family": family, "severity": severity, **metrics(scores, val_labels, active),
                   "cases": case_metrics(scores, val_labels, active, cases)}
            if family == "clean":
                clean_features, clean_correct = features, correct
                centered = features - features.mean(0)
                spectrum = torch.linalg.svdvals(centered).square()
                distribution = spectrum / spectrum.sum().clamp_min(1e-12)
                row["centered_effective_rank"] = float(torch.exp(-(distribution * distribution.clamp_min(1e-12).log()).sum()))
            else:
                row["clean_corrupt_object_cosine"] = float(F.cosine_similarity(clean_features, features).mean())
                row["accuracy_on_clean_correct_objects"] = float(correct[clean_correct].mean()) if clean_correct.any() else None
                row["clean_correct_objects"] = int(clean_correct.sum())
            rows.append(row)
            raw_scores[f"{family}-{severity}"] = torch.from_numpy(scores)
            atomic_json(output / (name + ".json"), {"encoder_sha256": digest, "conditions": rows})
            print(json.dumps({"encoder": name, "family": family, "severity": severity,
                              "accuracy": row["accuracy"], "macro_ap": row["macro_ap"]}), flush=True)
        atomic_checkpoint(output / (name + "_scores.pt"), {"scores": raw_scores, "labels": val_labels,
                          "classes": active, "image_groups": torch.from_numpy(metadata["val"][3])})
        results["encoders"][name] = {"encoder_sha256": digest, "conditions": rows}
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    atomic_json(output / "summary.json", results)


if __name__ == "__main__":
    main()
