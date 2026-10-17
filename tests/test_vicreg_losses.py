import unittest
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.losses import (
    covariance_decorrelation_loss,
    variance_regularization_loss,
    RobustJEPALoss,
)


class TestVICRegLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_covariance_decorrelation_uncorrelated(self):
        # N=1000, D=32: Independent standard normal vectors
        x = torch.randn(1000, 32)
        loss = covariance_decorrelation_loss(x)
        # Off-diagonal covariance should be close to 0
        self.assertLess(loss.item(), 0.05)

    def test_covariance_decorrelation_collapsed(self):
        # All 32 dimensions copy the same 1-D signal -> maximum covariance
        signal = torch.randn(200, 1)
        x_collapsed = signal.repeat(1, 32)
        loss_collapsed = covariance_decorrelation_loss(x_collapsed)
        
        # Loss should be roughly: (1/D) * D * (D-1) * Var^2 ~ (D-1) * 1.0 ~ 31
        self.assertGreater(loss_collapsed.item(), 20.0)

    def test_variance_regularization(self):
        # 1. Zero variance data (constant)
        x_const = torch.zeros(100, 16)
        loss_const = variance_regularization_loss(x_const, target_std=1.0)
        self.assertAlmostEqual(loss_const.item(), 1.0, delta=0.01)

        # 2. Ample variance data (std = 2.0 > target_std = 1.0)
        x_ample = torch.randn(100, 16) * 2.0
        loss_ample = variance_regularization_loss(x_ample, target_std=1.0)
        self.assertAlmostEqual(loss_ample.item(), 0.0, places=3)

    def test_robust_jepa_loss_backward(self):
        criterion = RobustJEPALoss(variance_weight=0.1, variance_floor=0.5, covariance_weight=0.05)
        
        b, n_mask, n_vis, d = 4, 10, 15, 32
        prediction = torch.randn(b, n_mask, d, requires_grad=True)
        target = torch.randn(b, n_mask, d)
        context = torch.randn(b, n_vis, d, requires_grad=True)

        total_loss, metrics = criterion(prediction, target, context)
        
        self.assertTrue(torch.isfinite(total_loss))
        self.assertIn("prediction_loss", metrics)
        self.assertIn("variance_loss", metrics)
        self.assertIn("covariance_loss", metrics)

        # Ensure gradient flows properly to both context and prediction
        total_loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNotNone(context.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertTrue(torch.isfinite(context.grad).all())


if __name__ == "__main__":
    unittest.main()
