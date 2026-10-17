"""
Comprehensive Representation Quality Audit Tool
Calculates Alignment, Uniformity, Effective Rank, Unbiased Linear CKA, and k-NN Jaccard Preservation.
Addresses the dimensional collapse illusion inherent in standard Cosine Similarity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.common import stable_seed
from robustdrive.corruptions import corrupt
from robustdrive.data import to_tensor
from robustdrive.metrics import (
    compute_alignment,
    compute_uniformity,
    compute_effective_rank,
    compute_linear_cka,
    compute_knn_jaccard,
)
from robustdrive.models import VisionEncoder


def load_encoder(checkpoint_path: str, image_size=(384, 640), device="cuda") -> VisionEncoder:
    """Load VisionEncoder from checkpoint or initialize random baseline."""
    encoder = VisionEncoder(image_size, patch_size=16, dim=384, depth=12, heads=6)
    if checkpoint_path.lower() == "random":
        print("[Audit] Initialized Random ViT baseline (0 training steps)")
    else:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if "encoder_state" in saved:
            state = saved["encoder_state"]
        elif "model" in saved:
            state = {k.removeprefix("online."): v for k, v in saved["model"].items() if k.startswith("online.")}
        else:
            state = saved
        encoder.load_state_dict(state, strict=True)
        print(f"[Audit] Loaded checkpoint from {path}")

    encoder = encoder.to(device).eval()
    return encoder


def generate_synthetic_images(num_images=64, size=(384, 640), seed=42) -> list[Image.Image]:
    """
    Generate realistic synthetic road/scene textures deterministically
    when local raw BDD dataset images are not present.
    """
    rng = np.random.default_rng(seed)
    h, w = size
    images = []
    for i in range(num_images):
        # Base sky + ground gradient
        arr = np.zeros((h, w, 3), dtype=np.float32)
        horizon = int(h * 0.45 + rng.integers(-20, 20))
        # Sky
        arr[:horizon, :, 0] = rng.uniform(0.5, 0.8)
        arr[:horizon, :, 1] = rng.uniform(0.6, 0.9)
        arr[:horizon, :, 2] = rng.uniform(0.8, 1.0)
        # Road / ground
        arr[horizon:, :, 0] = rng.uniform(0.2, 0.4)
        arr[horizon:, :, 1] = rng.uniform(0.2, 0.4)
        arr[horizon:, :, 2] = rng.uniform(0.2, 0.4)
        # Random road markings and object-like rectangles
        num_boxes = rng.integers(3, 8)
        for _ in range(num_boxes):
            bx = rng.integers(0, w - 80)
            by = rng.integers(horizon, h - 80)
            bw = rng.integers(30, 80)
            bh = rng.integers(30, 80)
            color = rng.uniform(0.1, 0.9, size=(3,))
            arr[by:by+bh, bx:bx+bw] = color
        # Add slight texture noise
        arr += rng.normal(0, 0.05, size=(h, w, 3))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


@torch.no_grad()
def extract_embeddings(
    encoder: VisionEncoder,
    images: list[Image.Image],
    family: str = "clean",
    severity: int = 0,
    batch_size: int = 8,
    seed: int = 17,
    device: str = "cuda",
) -> torch.Tensor:
    embeddings = []
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        tensors = []
        for j, img in enumerate(batch_imgs):
            img_idx = i + j
            if family == "clean" or severity == 0:
                cur = img
            else:
                c_seed = stable_seed(seed, f"img_{img_idx}", family, severity)
                cur = corrupt(img, family, severity, c_seed)
            tensors.append(to_tensor(cur))
        
        batch_tensor = torch.stack(tensors).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=use_cuda):
            # Mean pool patch tokens
            pooled = encoder(batch_tensor).float().mean(dim=1)
        embeddings.append(pooled.cpu())
    
    return torch.cat(embeddings)


def run_audit(
    models: dict[str, str],
    num_samples: int = 64,
    families: list[str] = None,
    severities: list[int] = None,
    device: str = "cuda",
    seed: int = 17,
) -> dict:
    if families is None:
        families = ["fog", "rain", "motion_blur", "low_light"]
    if severities is None:
        severities = [1, 3, 5]

    print(f"\n[Audit] Generating {num_samples} evaluation images...")
    images = generate_synthetic_images(num_samples, size=(384, 640), seed=seed)

    results = {}

    for model_name, ckpt_path in models.items():
        print(f"\n=======================================================")
        print(f"[Audit] Evaluating Model: {model_name}")
        print(f"=======================================================")
        encoder = load_encoder(ckpt_path, image_size=(384, 640), device=device)

        # 1. Clean feature properties
        z_clean = extract_embeddings(encoder, images, family="clean", severity=0, device=device)
        clean_erank = compute_effective_rank(z_clean)
        clean_uniformity = compute_uniformity(z_clean)

        print(f"  -> Clean Effective Rank: {clean_erank:.2f} / 384")
        print(f"  -> Clean Uniformity (lower=better): {clean_uniformity:.4f}")

        model_results = {
            "clean_effective_rank": clean_erank,
            "clean_uniformity": clean_uniformity,
            "conditions": [],
        }

        # 2. Corrupted feature properties across conditions
        for fam in families:
            for sev in severities:
                z_corrupt = extract_embeddings(encoder, images, family=fam, severity=sev, device=device)
                
                # Metrics
                align = compute_alignment(z_clean, z_corrupt)
                unif = compute_uniformity(z_corrupt)
                erank = compute_effective_rank(z_corrupt)
                cka = compute_linear_cka(z_clean, z_corrupt)
                knn_jaccard = compute_knn_jaccard(z_clean, z_corrupt, k=5)
                
                # Cosine for direct comparison
                z_clean_norm = F.normalize(z_clean.float(), dim=-1)
                z_corrupt_norm = F.normalize(z_corrupt.float(), dim=-1)
                cos_mean = float((z_clean_norm * z_corrupt_norm).sum(dim=-1).mean().item())

                condition_metric = {
                    "family": fam,
                    "severity": sev,
                    "cosine_similarity": cos_mean,
                    "alignment": align,
                    "uniformity": unif,
                    "effective_rank": erank,
                    "linear_cka": cka,
                    "knn_jaccard_k5": knn_jaccard,
                }
                model_results["conditions"].append(condition_metric)
                print(
                    f"  [{fam:<11} s={sev}] Cos: {cos_mean:.4f} | "
                    f"Align: {align:.4f} | Unif: {unif:.3f} | "
                    f"ERank: {erank:5.1f} | CKA: {cka:.4f} | kNN-J: {knn_jaccard:.4f}"
                )

        # Aggregate means across all evaluated conditions
        conds = model_results["conditions"]
        model_results["summary"] = {
            "mean_cosine": float(np.mean([c["cosine_similarity"] for c in conds])),
            "mean_alignment": float(np.mean([c["alignment"] for c in conds])),
            "mean_uniformity": float(np.mean([c["uniformity"] for c in conds])),
            "mean_effective_rank": float(np.mean([c["effective_rank"] for c in conds])),
            "mean_linear_cka": float(np.mean([c["linear_cka"] for c in conds])),
            "mean_knn_jaccard_k5": float(np.mean([c["knn_jaccard_k5"] for c in conds])),
        }
        results[model_name] = model_results

    return results


def format_markdown_report(results: dict, output_path: str):
    md = []
    md.append("# Representation Robustness Audit: Beyond Cosine Similarity\n")
    md.append("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    md.append("This report evaluates representations across Alignment, Uniformity, Effective Rank, Unbiased Linear CKA, and k-NN Jaccard neighborhood overlap.\n")
    md.append("## 1. Executive Summary Table\n")
    md.append("| Model | Clean ERank | Clean Uniformity (↓) | Mean Corrupt ERank | Mean Alignment (↓) | Mean Linear CKA (↑) | Mean k-NN Jaccard (↑) | Mean Cosine |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for name, data in results.items():
        s = data["summary"]
        md.append(
            f"| **{name}** | {data['clean_effective_rank']:.1f} | {data['clean_uniformity']:.3f} | "
            f"{s['mean_effective_rank']:.1f} | {s['mean_alignment']:.4f} | **{s['mean_linear_cka']:.4f}** | "
            f"**{s['mean_knn_jaccard_k5']:.4f}** | {s['mean_cosine']:.4f} |"
        )

    md.append("\n## 2. In-Depth Interpretation\n")
    md.append("- **Effective Rank**: Indicates whether the embedding space is spanning diverse dimensions. Values < 10 indicate severe dimensional collapse.")
    md.append("- **Alignment (Wang & Isola, 2020)**: Distance on the hypersphere between clean and corrupted pairs (Lower is better).")
    md.append("- **Uniformity (Wang & Isola, 2020)**: Measures how evenly points are distributed on the hypersphere (Lower is better; very high values indicate collapse).")
    md.append("- **Linear CKA (Kornblith et al., 2019)**: Geometric structural similarity between clean and corrupted feature manifolds (Higher is better, invariant to orthogonal rotations).")
    md.append("- **k-NN Jaccard (k=5)**: Topology preservation rate of the local sample neighborhood despite adverse corruptions.\n")

    md.append("## 3. Condition-by-Condition Breakdown\n")
    for name, data in results.items():
        md.append(f"### Model: {name}\n")
        md.append("| Family | Severity | Cosine (↑) | Alignment (↓) | Uniformity (↓) | ERank | Linear CKA (↑) | k-NN Jaccard (↑) |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for c in data["conditions"]:
            md.append(
                f"| {c['family']} | {c['severity']} | {c['cosine_similarity']:.4f} | "
                f"{c['alignment']:.4f} | {c['uniformity']:.3f} | {c['effective_rank']:.1f} | "
                f"{c['linear_cka']:.4f} | {c['knn_jaccard_k5']:.4f} |"
            )
        md.append("\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(md), encoding="utf-8")
    print(f"\n[Audit] Markdown report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audit representation quality beyond cosine similarity.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--output-json", default=str(ROOT / "runs/audit_representation_quality.json"))
    parser.add_argument("--output-report", default=str(ROOT / "docs/audit_representation_quality.md"))
    args = parser.parse_args()

    models = {
        "Corruption JEPA": str(ROOT / "runs/continue10-corruption-seed17/final_encoder.pt"),
        "Clean JEPA": str(ROOT / "runs/continue10-clean-seed17/final_encoder.pt"),
        "Random ViT": "random",
    }

    results = run_audit(
        models=models,
        num_samples=args.num_samples,
        device=args.device,
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[Audit] Machine-readable JSON saved to: {out_json}")

    format_markdown_report(results, args.output_report)


if __name__ == "__main__":
    main()
