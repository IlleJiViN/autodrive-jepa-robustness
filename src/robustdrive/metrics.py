from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def compute_alignment(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0) -> float:
    """
    Wang & Isola (ICML 2020) Alignment Metric.
    Measures expected distance between positive pairs on the unit hypersphere.
    Lower is better (closer representations).
    """
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")
    if len(z1) == 0:
        return 0.0
    z1_norm = F.normalize(z1.float(), dim=-1)
    z2_norm = F.normalize(z2.float(), dim=-1)
    diff = (z1_norm - z2_norm).norm(p=2, dim=-1)
    return float(diff.pow(alpha).mean().item())


def compute_uniformity(z: torch.Tensor, t: float = 2.0) -> float:
    """
    Wang & Isola (ICML 2020) Uniformity Metric.
    Log of average Gaussian potential over all distinct pairs.
    Lower is better (asymptotically approaches uniform distribution on hypersphere).
    A high value indicates dimensional or full representation collapse.
    """
    if len(z) < 2:
        return 0.0
    z_norm = F.normalize(z.float(), dim=-1)
    sq_pdist = torch.pdist(z_norm, p=2).pow(2)
    val = sq_pdist.mul(-t).exp().mean().clamp_min(1e-12)
    return float(val.log().item())


def compute_effective_rank(z: torch.Tensor) -> float:
    """
    Entropy of the singular value distribution of centered features (Roy & Vetterli, 2007).
    Measures the number of effective dimensions spanned by the representations.
    Maximum is min(N, D). Very low value (e.g. < 5 for D=384) indicates severe dimensional collapse.
    """
    if len(z) < 2:
        return 1.0
    z_float = z.float()
    centered = z_float - z_float.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered).square()
    total = singular_values.sum().clamp_min(1e-12)
    p = singular_values / total
    entropy = -(p * p.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def _unbiased_hsic(k: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC estimator from Song et al. (2012) and Kornblith et al. (2019)."""
    n = k.shape[0]
    k_zero = k.clone().fill_diagonal_(0.0)
    l_zero = l.clone().fill_diagonal_(0.0)
    kl = (k_zero * l_zero).sum()
    k_sum = k_zero.sum()
    l_sum = l_zero.sum()
    k_row = k_zero.sum(dim=1)
    l_row = l_zero.sum(dim=1)
    hsic = kl + (k_sum * l_sum) / ((n - 1) * (n - 2)) - 2.0 * (k_row * l_row).sum() / (n - 2)
    return hsic / (n * (n - 3))


def compute_linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Centered Kernel Alignment (CKA) with Linear Kernel (Kornblith et al., ICML 2019).
    Uses the unbiased HSIC estimator to prevent small-sample or high-dimension bias (N < D).
    Returns value typically between 0.0 and 1.0. Higher is more structurally similar.
    """
    if len(x) != len(y):
        raise ValueError(f"Batch size mismatch: {len(x)} vs {len(y)}")
    if len(x) < 4:
        raise ValueError("Unbiased CKA requires at least N >= 4 samples")
    
    x_f = x.float()
    y_f = y.float()
    k = x_f @ x_f.T
    l = y_f @ y_f.T

    hsic_kl = _unbiased_hsic(k, l)
    hsic_kk = _unbiased_hsic(k, k)
    hsic_ll = _unbiased_hsic(l, l)

    denom = (hsic_kk * hsic_ll).clamp_min(1e-12).sqrt()
    cka = hsic_kl / denom
    return float(torch.clamp(cka, 0.0, 1.0).item())


def compute_knn_jaccard(z1: torch.Tensor, z2: torch.Tensor, k: int = 5) -> float:
    """
    Neighborhood Preservation Metric:
    Calculates average Jaccard similarity of top-k nearest neighbors between two spaces (e.g. Clean vs Corrupt).
    Higher is better (local neighborhood topology is preserved despite corruption).
    """
    n = len(z1)
    if n <= k or k <= 0:
        return 1.0
    
    z1_norm = F.normalize(z1.float(), dim=-1)
    z2_norm = F.normalize(z2.float(), dim=-1)

    sim1 = z1_norm @ z1_norm.T
    sim2 = z2_norm @ z2_norm.T

    sim1.fill_diagonal_(-float("inf"))
    sim2.fill_diagonal_(-float("inf"))

    _, topk1 = sim1.topk(k, dim=1)
    _, topk2 = sim2.topk(k, dim=1)

    jaccard_scores = []
    for i in range(n):
        set1 = set(topk1[i].tolist())
        set2 = set(topk2[i].tolist())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        jaccard_scores.append(intersection / union if union > 0 else 1.0)

    return float(sum(jaccard_scores) / len(jaccard_scores))
