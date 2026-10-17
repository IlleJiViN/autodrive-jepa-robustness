import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from export_yolo_object_targets import proposal_row
from run_object_probe import metrics


class ProbeChecks(unittest.TestCase):
    def test_collapsed_classifier_cannot_pass_semantic_accuracy(self):
        labels = torch.tensor([0, 2, 9, 0, 2, 9])
        active = torch.tensor([0, 2, 9])
        correct = np.eye(3)[[0, 1, 2, 0, 1, 2]]
        collapsed = np.tile([0., 1., 0.], (6, 1))
        self.assertEqual(metrics(correct, labels, active)["balanced_accuracy"], 1.)
        self.assertEqual(metrics(correct, labels, active)["ece_15_bins"], 0.)
        self.assertAlmostEqual(metrics(collapsed, labels, active)["balanced_accuracy"], 1 / 3)
        self.assertGreater(metrics(collapsed, labels, active)["ece_15_bins"], .6)

    def test_yolo_proposals_do_not_inherit_ground_truth_object_annotations(self):
        row = {"image_path": str(ROOT / "fixture.jpg"), "height": 16, "width": 24,
               "boxes": [[0., 0., 24., 16.]], "labels": [9], "object_attributes": [{"occluded": True}],
               "annotation_sha256": "groundtruth", "annotation_status": "available_legacy_bdd100k"}
        before = copy.deepcopy(row)
        result = SimpleNamespace(orig_shape=(16, 24), path=row["image_path"], names={0: "person", 1: "dog"},
                    boxes=SimpleNamespace(xyxy=torch.tensor([[-1., -2., 8., 8.], [1., 1., 3., 3.]]),
                                          cls=torch.tensor([0., 1.]), conf=torch.tensor([.8, .9])))
        out = proposal_row(row, result, "teacherhash")
        self.assertEqual(out["annotation_status"], "available_yolo_proposals")
        self.assertEqual(out["boxes"], [[0., 0., 8., 8.]])
        self.assertEqual(out["labels"], [0])
        self.assertNotIn("object_attributes", out)
        self.assertNotIn("annotation_sha256", out)
        self.assertEqual(row, before)
        result.orig_shape = (32, 48)
        with self.assertRaisesRegex(ValueError, "original image geometry"):
            proposal_row(row, result, "teacherhash")


if __name__ == "__main__":
    unittest.main()
