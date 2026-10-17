from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import atomic_json, stable_seed
from .corruptions import corrupt
from .data import DrivingPairs, to_tensor
from .models import VisionEncoder


@torch.no_grad()
def evaluate(config: dict, checkpoint: str, output: str, max_images: int = 512, device_name="cuda"):
    if Path(output).exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output}")
    if max_images < 0:
        raise ValueError("max_images must be >=0 (0 means all validation images)")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select --device cpu explicitly")
    checkpoint = Path(checkpoint).resolve()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trained_config = saved["config"]
    if config["data"]["image_size"] != trained_config["data"]["image_size"]:
        raise ValueError("Evaluation image size must match the trained positional embedding")
    spec = trained_config["model"]
    encoder = VisionEncoder(config["data"]["image_size"], spec["patch_size"], spec["dim"], spec["depth"], spec["heads"])
    if "encoder_state" in saved:
        state = saved["encoder_state"]
    else:
        state = {key.removeprefix("online."): value for key, value in saved["model"].items() if key.startswith("online.")}
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    dataset = DrivingPairs(config["data"]["val_manifest"], config["data"]["image_size"],
                           config["corruption"]["families"], [1], config["seed"])
    rng = np.random.default_rng(config["seed"])
    indices = np.arange(len(dataset))
    if max_images > 0 and max_images < len(indices):
        indices = np.sort(rng.choice(indices, max_images, replace=False))
    if len(indices) < 2:
        raise ValueError("Representation health evaluation needs at least two images")
    batch_size = config["training"]["batch_size"]

    def embed(family="clean", severity=0):
        vectors = []
        for start in range(0, len(indices), batch_size):
            batch = []
            for index in indices[start:start + batch_size]:
                clean = dataset.clean_image(int(index))
                seed = stable_seed(config["seed"], dataset.records[int(index)]["image_path"], family, severity)
                batch.append(to_tensor(corrupt(clean, family, severity, seed)))
            images = torch.stack(batch).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and config["training"]["amp_bfloat16"]):
                pooled = encoder(images).float().mean(dim=1)
            vectors.append(pooled.cpu())
        vectors = torch.cat(vectors)
        if not torch.isfinite(vectors).all():
            raise FloatingPointError("Nonfinite encoder embeddings")
        return vectors

    clean = embed()
    normalized = F.normalize(clean, dim=-1)
    clean_std = clean.std(dim=0, unbiased=False).mean()
    centered = clean - clean.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centered).square()
    proportions = spectrum / spectrum.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(proportions * proportions.clamp_min(1e-12).log()).sum())
    health = {"pooled_std_mean": float(clean_std), "pooled_norm_mean": float(clean.norm(dim=-1).mean()),
              "different_image_cosine_mean": float((normalized * normalized.roll(1, dims=0)).sum(dim=-1).mean()),
              "collapse_warning": bool(clean_std < 0.001), "warning_threshold_is_heuristic": True,
              "centered_effective_rank": float(effective_rank),
              "identity_retrieval_candidates": len(indices), "identity_retrieval_chance_top1": 1 / len(indices)}
    conditions = []
    for family in config["corruption"]["families"]:
        for severity in config["corruption"]["eval_severities"]:
            changed = embed(family, severity)
            cosine = (normalized * F.normalize(changed, dim=-1)).sum(dim=-1)
            similarities = F.normalize(changed, dim=-1) @ normalized.T
            identity_top1 = similarities.argmax(dim=1).eq(torch.arange(len(indices))).float().mean()
            other = similarities.clone()
            other.fill_diagonal_(-float("inf"))
            margins = similarities.diag() - other.max(dim=1).values
            conditions.append({"family": family, "severity": severity, "images": len(indices),
                "cosine_mean": float(cosine.mean()), "cosine_p10": float(torch.quantile(cosine, 0.1)),
                "identity_retrieval_top1": float(identity_top1), "identity_retrieval_margin_median": float(margins.median()),
                "feature_drift_rms": float((changed - clean).square().mean().sqrt())})
            print(conditions[-1], flush=True)
    if not conditions:
        raise ValueError("No evaluation conditions configured")
    result = {"status": "representation_robustness_measured", "checkpoint": str(checkpoint),
              "checkpoint_step": saved.get("step"), "checkpoint_warm_start": saved.get("warm_start"),
              "checkpoint_selection": saved.get("selection_criterion", "full_training_checkpoint"),
              "checkpoint_training_manifest_fingerprint": saved.get("manifest_fingerprint"),
              "validation_manifest_sha256": hashlib.sha256(Path(config["data"]["val_manifest"]).read_bytes()).hexdigest(),
              "scope": "full_validation_manifest" if len(indices) == len(dataset) else "seeded_validation_subset",
              "images": len(indices), "total_validation_images": len(dataset), "seed": config["seed"],
              "health": health, "conditions": conditions,
              "worst_mean_cosine": min(row["cosine_mean"] for row in conditions),
              "interpretation": "Cosine can be high after collapse. Identity retrieval matches an image to its clean counterpart among this validation subset; it is not semantic classification, detector mAP or driving safety."}
    atomic_json(output, result)
    return result
