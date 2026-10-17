import unittest
from pathlib import Path
import sys
import torch
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.metrics import (
    compute_alignment,
    compute_uniformity,
    compute_effective_rank,
    compute_linear_cka,
    compute_knn_jaccard,
)


class TestRepresentationMetrics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_alignment_identical(self):
        z = torch.randn(50, 384)
        score = compute_alignment(z, z)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_alignment_orthogonal(self):
        # Two orthogonal unit vectors: ||u - v||^2 = ||u||^2 + ||v||^2 = 2.0
        z1 = torch.zeros(10, 2)
        z1[:, 0] = 1.0
        z2 = torch.zeros(10, 2)
        z2[:, 1] = 1.0
        score = compute_alignment(z1, z2)
        self.assertAlmostEqual(score, 2.0, places=5)

    def test_uniformity_collapsed_vs_uniform(self):
        # Collapsed: all vectors are nearly identical -> pairwise distance ~ 0 -> exp(-2*0)=1 -> log(1)=0
        collapsed = torch.ones(100, 384) + torch.randn(100, 384) * 1e-4
        score_collapsed = compute_uniformity(collapsed)

        # Uniform: random Gaussian in high dimensions spreads on hypersphere
        uniform = torch.randn(100, 384)
        score_uniform = compute_uniformity(uniform)

        # Uniformity: lower is better (more uniform). Collapsed is much higher (near 0)
        self.assertGreater(score_collapsed, -0.1)
        self.assertLess(score_uniform, -1.5)

    def test_effective_rank_collapsed_vs_isotropic(self):
        # 1-D collapsed data: all variation in 1 direction
        t = torch.randn(200, 1)
        collapsed = t * torch.ones(1, 384)
        rank_collapsed = compute_effective_rank(collapsed)
        self.assertAlmostEqual(rank_collapsed, 1.0, delta=0.1)

        # High-dimensional isotropic Gaussian data
        isotropic = torch.randn(1000, 50)
        rank_isotropic = compute_effective_rank(isotropic)
        # Should span close to 50 dimensions
        self.assertGreater(rank_isotropic, 40.0)

    def test_linear_cka(self):
        x = torch.randn(100, 384)
        # Identical
        cka_self = compute_linear_cka(x, x)
        self.assertAlmostEqual(cka_self, 1.0, places=4)

        # Orthogonal rotation: CKA is invariant to orthogonal transformations
        q, _ = torch.linalg.qr(torch.randn(384, 384))
        rotated_x = x @ q
        cka_rot = compute_linear_cka(x, rotated_x)
        self.assertAlmostEqual(cka_rot, 1.0, places=4)

        # Independent random noise
        y = torch.randn(100, 384)
        cka_indep = compute_linear_cka(x, y)
        self.assertLess(cka_indep, 0.1)

    def test_knn_jaccard(self):
        z1 = torch.randn(100, 64)
        # Identical should have perfect neighborhood overlap
        jaccard_self = compute_knn_jaccard(z1, z1, k=5)
        self.assertAlmostEqual(jaccard_self, 1.0, places=5)

        # Independent noise should have very low neighbor overlap (random chance is ~ 5 / 99 = 0.05)
        z2 = torch.randn(100, 64)
        jaccard_indep = compute_knn_jaccard(z1, z2, k=5)
        self.assertLess(jaccard_indep, 0.2)


if __name__ == "__main__":
    unittest.main()
