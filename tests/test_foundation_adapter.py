import unittest
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.foundation import FoundationViTDetector, compute_detection_loss, SLOTS, VALUES


class TestFoundationViTDetector(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_shapes(self):
        # Instantiate without downloading pretrained weights for fast testing
        model = FoundationViTDetector(model_name="vit_small_patch16_384", pretrained=False)
        model.eval()

        # Batch 2, RGB 3, 384x640
        x = torch.randn(2, 3, 384, 640)
        with torch.no_grad():
            outputs = model(x)

        self.assertEqual(len(outputs), 3)
        p3, p4, p5 = outputs

        # P3 (stride 8): 384/8 = 48, 640/8 = 80
        self.assertEqual(p3.shape, (2, SLOTS, 48, 80, VALUES))
        # P4 (stride 16): 384/16 = 24, 640/16 = 40
        self.assertEqual(p4.shape, (2, SLOTS, 24, 40, VALUES))
        # P5 (stride 32): 384/32 = 12, 640/32 = 20
        self.assertEqual(p5.shape, (2, SLOTS, 12, 20, VALUES))

    def test_loss_and_backward(self):
        model = FoundationViTDetector(model_name="vit_small_patch16_384", pretrained=False)
        model.train()

        x = torch.randn(2, 3, 384, 640)
        outputs = model(x)

        # Mock dummy targets for P3, P4, P5
        shapes = [(48, 80), (24, 40), (12, 20)]
        targets = []
        for h, w in shapes:
            present = torch.zeros(2, SLOTS, h, w, dtype=torch.bool)
            present[0, 0, 5, 5] = True  # 1 positive target
            classes = torch.zeros(2, SLOTS, h, w, dtype=torch.long)
            classes[0, 0, 5, 5] = 1  # class index 1
            boxes = torch.zeros(2, SLOTS, h, w, 4, dtype=torch.float32)
            boxes[0, 0, 5, 5] = torch.tensor([0.5, 0.5, 1.2, 0.8])
            targets.append({"present": present, "classes": classes, "boxes": boxes})

        loss, metrics = compute_detection_loss(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["positive_slots"], 3)

        loss.backward()
        # Verify gradient flow into detection heads and adapter
        self.assertIsNotNone(model.heads[0].layers[0].weight.grad)
        self.assertIsNotNone(model.adapter.projections[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
