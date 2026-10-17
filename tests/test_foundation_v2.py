import math
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from robustdrive.data import CLASS_NAMES
from robustdrive.foundation_v2 import (
    SLOTS,
    STRIDES,
    VALUES,
    DecoupledScaleHead,
    DetectionsResult,
    FoundationViTDetectorV2,
    ViTDetFPNAdapter,
    assign_targets_v2,
    bbox_ciou,
    compute_detection_loss_v2,
    decode_v2,
)


class TestFoundationViTDetectorV2(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_bbox_ciou_identical_boxes(self):
        """Identical boxes should yield CIoU of ~1.0."""
        boxes1 = torch.tensor([[10.0, 10.0, 50.0, 50.0], [0.0, 0.0, 100.0, 200.0]], dtype=torch.float32)
        boxes2 = boxes1.clone()
        ciou = bbox_ciou(boxes1, boxes2)
        self.assertEqual(ciou.shape, (2,))
        for val in ciou:
            self.assertAlmostEqual(val.item(), 1.0, places=4)

    def test_bbox_ciou_disjoint_boxes(self):
        """Disjoint boxes should yield CIoU within [-1.0, 1.0]."""
        boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        boxes2 = torch.tensor([[100.0, 100.0, 120.0, 120.0]], dtype=torch.float32)
        ciou = bbox_ciou(boxes1, boxes2)
        self.assertEqual(ciou.shape, (1,))
        self.assertLess(ciou.item(), 0.0)
        self.assertGreaterEqual(ciou.item(), -1.0)

    def test_vit_det_fpn_adapter(self):
        """Test FPN Adapter shapes and top-down fusion across P3, P4, P5."""
        batch = 2
        dim = 384
        out_channels = 256
        adapter = ViTDetFPNAdapter(dim=dim, out_channels=out_channels)

        # 4 intermediate features from ViT blocks [2, 5, 8, 11] at stride 16 (24x40)
        f0 = torch.randn(batch, dim, 24, 40, requires_grad=True)
        f1 = torch.randn(batch, dim, 24, 40, requires_grad=True)
        f2 = torch.randn(batch, dim, 24, 40, requires_grad=True)
        f3 = torch.randn(batch, dim, 24, 40, requires_grad=True)

        pyramid = adapter([f0, f1, f2, f3])
        self.assertEqual(len(pyramid), 3)
        p3, p4, p5 = pyramid

        # P3: stride 8 -> 48x80
        self.assertEqual(p3.shape, (batch, out_channels, 48, 80))
        # P4: stride 16 -> 24x40
        self.assertEqual(p4.shape, (batch, out_channels, 24, 40))
        # P5: stride 32 -> 12x20
        self.assertEqual(p5.shape, (batch, out_channels, 12, 20))

        # Test gradient propagation
        loss = p3.sum() + p4.sum() + p5.sum()
        loss.backward()
        self.assertIsNotNone(f0.grad)
        self.assertIsNotNone(f3.grad)

    def test_decoupled_scale_head(self):
        """Test decoupled classification and regression head."""
        batch = 2
        in_channels = 256
        num_classes = len(CLASS_NAMES)
        head = DecoupledScaleHead(in_channels=in_channels, num_classes=num_classes, head_channels=128)

        # Check prior bias initialization
        self.assertAlmostEqual(head.obj_out.bias.item(), -4.6, places=1)
        self.assertAlmostEqual(head.cls_out.bias[0].item(), -4.6, places=1)
        self.assertTrue(torch.all(head.box_out.bias == 0.0))

        feat = torch.randn(batch, in_channels, 24, 40)
        out = head(feat)

        # Shape should be (B, SLOTS=1, H, W, VALUES)
        expected_values = 5 + num_classes
        self.assertEqual(out.shape, (batch, 1, 24, 40, expected_values))

    def test_foundation_detector_v2_forward_and_freeze(self):
        """Test full model forward pass with pretrained=False."""
        model = FoundationViTDetectorV2(
            model_name="vit_small_patch16_384",
            pretrained=False,
            img_size=(384, 640),
            freeze_backbone_blocks=2,
        )
        model.eval()

        # Verify frozen early blocks
        for p in model.backbone.patch_embed.parameters():
            self.assertFalse(p.requires_grad)
        for p in model.backbone.blocks[0].parameters():
            self.assertFalse(p.requires_grad)
        for p in model.backbone.blocks[1].parameters():
            self.assertFalse(p.requires_grad)
        # Block 2 should remain trainable
        self.assertTrue(any(p.requires_grad for p in model.backbone.blocks[2].parameters()))

        x = torch.randn(2, 3, 384, 640)
        with torch.no_grad():
            outputs = model(x)

        self.assertEqual(len(outputs), 3)
        p3, p4, p5 = outputs

        num_classes = len(CLASS_NAMES)
        values = 5 + num_classes
        self.assertEqual(p3.shape, (2, 1, 48, 80, values))
        self.assertEqual(p4.shape, (2, 1, 24, 40, values))
        self.assertEqual(p5.shape, (2, 1, 12, 20, values))

    def test_assign_targets_v2(self):
        """Test multi-cell target assignment with soft scale routing."""
        image_size = (384, 640)
        shapes = [(48, 80), (24, 40), (12, 20)]

        records = [
            {
                "width": 1280,
                "height": 720,
                # Small box (car) -> should route to P3
                # Medium/large box (bus) -> should route to P4/P5
                "boxes": [
                    [100, 100, 150, 140],   # side ~ sqrt(50 * 40 * sx * sy)
                    [400, 200, 800, 600],   # large side
                ],
                "labels": [0, 2],
            }
        ]

        targets, total_assigned = assign_targets_v2(records, indices=[0], shapes=shapes, image_size=image_size)

        self.assertEqual(len(targets), 3)
        self.assertGreater(total_assigned, 0)

        for lvl, (h, w) in enumerate(shapes):
            t = targets[lvl]
            self.assertIn("present", t)
            self.assertIn("classes", t)
            self.assertIn("boxes", t)
            self.assertEqual(t["present"].shape, (1, 1, h, w))
            self.assertEqual(t["classes"].shape, (1, 1, h, w))
            self.assertEqual(t["boxes"].shape, (1, 1, h, w, 4))

        # Check positive assignment count across levels
        num_p3_pos = targets[0]["present"].sum().item()
        num_p5_pos = targets[2]["present"].sum().item()
        self.assertGreater(num_p3_pos, 0)
        self.assertGreater(num_p5_pos, 0)

    def test_compute_detection_loss_v2_and_backward(self):
        """Test loss computation and gradient backpropagation."""
        model = FoundationViTDetectorV2(
            model_name="vit_small_patch16_384",
            pretrained=False,
            img_size=(384, 640),
        )
        model.train()

        x = torch.randn(2, 3, 384, 640)
        outputs = model(x)

        image_size = (384, 640)
        shapes = [(48, 80), (24, 40), (12, 20)]
        records = [
            {"width": 640, "height": 384, "boxes": [[50, 50, 100, 100]], "labels": [1]},
            {"width": 640, "height": 384, "boxes": [[200, 150, 450, 300]], "labels": [0]},
        ]
        targets, total_assigned = assign_targets_v2(records, indices=[0, 1], shapes=shapes, image_size=image_size)

        loss, metrics = compute_detection_loss_v2(outputs, targets)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("obj_loss", metrics)
        self.assertIn("cls_loss", metrics)
        self.assertIn("box_loss", metrics)
        self.assertIn("total_loss", metrics)
        self.assertEqual(metrics["positive_slots"], total_assigned)

        loss.backward()
        # Verify gradient flow into adapter and heads
        self.assertIsNotNone(model.adapter.lateral_p5.weight.grad)
        self.assertIsNotNone(model.heads[0].cls_out.weight.grad)
        self.assertIsNotNone(model.heads[0].box_out.weight.grad)

    def test_decode_v2_empty_and_valid(self):
        """Test decode_v2 with empty low-score background and with strong detection."""
        batch_size = 2
        shapes = [(48, 80), (24, 40), (12, 20)]
        num_classes = len(CLASS_NAMES)
        values = 5 + num_classes

        # 1. Background tensors (low scores)
        preds_empty = []
        for h, w in shapes:
            pred = torch.full((batch_size, 1, h, w, values), -10.0, dtype=torch.float32)
            preds_empty.append(pred)

        results_empty = decode_v2(preds_empty, image_size=(384, 640), score_floor=0.05)
        self.assertEqual(len(results_empty), batch_size)
        for res in results_empty:
            self.assertEqual(len(res), 0)
            self.assertEqual(res.boxes_arr.shape, (0, 4))
            self.assertEqual(res.scores_arr.shape, (0,))
            self.assertEqual(res.labels_arr.shape, (0,))

        # 2. Inject high confidence target into batch 0, P4 (stride 16)
        # Cell (r=12, c=20)
        preds_active = [p.clone() for p in preds_empty]
        # P4 is index 1
        preds_active[1][0, 0, 12, 20, 0] = 5.0     # high objectness
        preds_active[1][0, 0, 12, 20, 1 + 3] = 6.0  # class 3
        # box regression: cx offset, cy offset, w log-scale, h log-scale
        preds_active[1][0, 0, 12, 20, 1 + num_classes + 0] = 0.0
        preds_active[1][0, 0, 12, 20, 1 + num_classes + 1] = 0.0
        preds_active[1][0, 0, 12, 20, 1 + num_classes + 2] = 1.0
        preds_active[1][0, 0, 12, 20, 1 + num_classes + 3] = 1.0

        results = decode_v2(preds_active, image_size=(384, 640), score_floor=0.05)
        self.assertEqual(len(results), batch_size)

        det0 = results[0]
        self.assertGreaterEqual(len(det0), 1)
        self.assertEqual(det0[0]["class"], 3)
        self.assertGreater(det0[0]["score"], 0.8)

        # Verify DetectionsResult indexing compatibility with both list and dict-like callers
        self.assertEqual(len(det0["boxes"]), len(det0))
        self.assertEqual(len(det0["scores"]), len(det0))
        self.assertEqual(len(det0["labels"]), len(det0))
        self.assertIn("box", det0[0])
        self.assertIn("score", det0[0])
        self.assertIn("class", det0[0])


if __name__ == "__main__":
    unittest.main()
