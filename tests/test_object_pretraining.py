"""Check object-target boundaries, information leakage and exact resumption."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from robustdrive.object_pretraining import ObjectPretrainer, box_patch_weights, make_object_mask
import run_object_pretraining as runner


def tiny_config():
    config = json.loads((ROOT / "configs/object_pilot.json").read_text())
    config["data"].update(image_size=[16, 24], num_workers=0)
    config["model"].update(patch_size=8, dim=32, depth=2, heads=4,
                           predictor_dim=16, predictor_depth=1, predictor_heads=4)
    config["training"].update(batch_size=1, epochs=2, validation_batches=2,
                              save_every_steps=1, amp_bfloat16=False)
    return config


class ObjectTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_complete_regions_and_equal_mask_budget_in_both_modes(self):
        boxes = [torch.tensor([[.05, .05, .45, .45], [.6, .6, .8, .8]])]
        object_mask = make_object_mask(boxes, (4, 6), .6, [17])
        random_mask = make_object_mask(boxes, (4, 6), .6, [17], "random")
        self.assertEqual(object_mask.masked.shape, random_mask.masked.shape)
        self.assertEqual(object_mask.visible.shape, random_mask.visible.shape)
        weights = box_patch_weights(boxes[0], (4, 6))
        for obj in object_mask.selected_objects[0]:
            self.assertFalse((weights[obj, object_mask.visible[0]] > 0).any())
        self.assertTrue(torch.allclose(object_mask.object_weights.sum(-1)[object_mask.valid_objects],
                                       torch.ones(int(object_mask.valid_objects.sum()))))
        self.assertEqual(set(object_mask.visible[0].tolist()) | set(object_mask.masked[0].tolist()), set(range(24)))
        self.assertFalse(torch.isin(object_mask.visible, object_mask.masked).any())

    def test_tiny_objects_empty_boxes_and_oversized_regions(self):
        small = torch.tensor([[.1, .1, .1001, .1001]])
        weights = box_patch_weights(small, (4, 6))
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        mask = make_object_mask([small, torch.empty(0, 4), torch.tensor([[0., 0., 1., 1.]])], (4, 6), .6, [17, 23, 29])
        self.assertTrue(mask.valid_objects[0].any())
        self.assertFalse(mask.valid_objects[1:].any())
        self.assertEqual(mask.oversized_objects[2], 1)
        self.assertEqual(mask.masked.shape, (3, 15))
        for invalid in ([[0., 0., 0., .1]], [[-1., 0., .2, .3]], [[float("nan"), 0., .2, .3]]):
            with self.assertRaises(ValueError):
                box_patch_weights(torch.tensor(invalid), (4, 6))

    def test_mask_is_independent_of_model_rng_and_shared_between_objectives(self):
        boxes = [torch.tensor([[.1, .1, .3, .3]])]
        first = make_object_mask(boxes, (4, 6), .6, [23])
        torch.rand(1000)
        np.random.rand(1000)
        second = make_object_mask(boxes, (4, 6), .6, [23])
        self.assertTrue(torch.equal(first.masked, second.masked))
        self.assertTrue(torch.equal(first.object_weights, second.object_weights))

    def test_hidden_object_pixels_cannot_enter_context_or_predictor(self):
        config = tiny_config()
        torch.manual_seed(17)
        model = ObjectPretrainer(config).eval()
        mask = make_object_mask([torch.tensor([[0., 0., 1 / 3, .5]])], (2, 3), .6, [17])
        clean = torch.rand(1, 3, 16, 24)
        view = clean.clone().requires_grad_(True)
        changed = view.detach().clone()
        for index in mask.masked[0]:
            y, x = divmod(int(index), 3)
            changed[:, :, y * 8:(y + 1) * 8, x * 8:(x + 1) * 8] = torch.rand(1, 3, 8, 8)
        loss, _ = model(clean, view, mask)
        changed_loss, _ = model(clean, changed, mask)
        self.assertTrue(torch.equal(loss, changed_loss))
        loss.backward()
        self.assertEqual(float(view.grad[:, :, :8, :8].abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in model.target.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_both_objectives_cuda_bfloat16_have_finite_loss_and_gradients(self):
        config = tiny_config()
        clean = torch.rand(2, 3, 16, 24, device="cuda")
        mask = make_object_mask([torch.tensor([[0., 0., 1 / 3, .5]])] * 2, (2, 3), .6, [17, 23]).to("cuda")
        for objective in ("jepa", "mae"):
            config["object_pretraining"]["objective"] = objective
            model = ObjectPretrainer(config).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, metrics = model(clean, clean * .7, mask)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.online.patch.weight.grad.abs().sum()), 0)
            self.assertTrue(torch.isfinite(model.predictor.mask_token.grad).all())
            self.assertEqual(float(metrics["objects_per_image"]), 1)

    def test_exact_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = tiny_config()
            rng = np.random.default_rng(17)
            for split in ("train", "val"):
                rows = []
                for i in range(2):
                    path = root / f"{split}{i}.png"
                    Image.fromarray(rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)).save(path)
                    rows.append({"image_path": str(path), "scene_id": path.stem, "split": split,
                                 "boxes": [[0., 0., 8., 8.]], "labels": [2], "width": 24, "height": 16,
                                 "annotation_status": "available_legacy_bdd100k"})
                manifest = root / (split + ".jsonl")
                manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
                config["data"][split + "_manifest"] = str(manifest)
            original = runner.atomic_checkpoint

            def interrupt(path, state):
                original(path, state)
                if Path(path).name == "last.pt" and state["step"] == 1:
                    raise RuntimeError("intentional interruption")

            with mock.patch.object(runner, "atomic_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "intentional interruption"):
                    runner.train(copy.deepcopy(config), root / "resumed", 3, "cpu")
            runner.train(copy.deepcopy(config), root / "resumed", 3, "cpu", root / "resumed/last.pt")
            runner.train(copy.deepcopy(config), root / "control", 3, "cpu")
            a = torch.load(root / "resumed/last.pt", weights_only=True)
            b = torch.load(root / "control/last.pt", weights_only=True)
            self.assertEqual(a["step"], 3)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            a["source_fingerprint"] = "0" * 64
            torch.save(a, root / "resumed/last.pt")
            with self.assertRaisesRegex(ValueError, "identical training source"):
                runner.train(copy.deepcopy(config), root / "resumed", 3, "cpu", root / "resumed/last.pt")


if __name__ == "__main__":
    unittest.main()
