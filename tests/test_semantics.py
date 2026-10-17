"""Critical correctness checks; synthetic fixtures are not research results."""
import copy
import hashlib
import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from robustdrive.corruptions import FAMILIES, corrupt
from robustdrive.data import validate_data
from robustdrive.models import VisionEncoder, CorruptionJEPA

train_module = importlib.import_module("robustdrive.train")


class SemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_target_pixels_cannot_enter_visible_attention(self):
        torch.manual_seed(17)
        encoder = VisionEncoder([32, 48], 8, 32, 2, 4).eval()
        images = torch.rand(1, 3, 32, 48)
        visible = torch.tensor([[0, 1, 2]])
        changed = images.clone()
        changed[:, :, 8:, :] = torch.rand_like(changed[:, :, 8:, :])
        changed[:, :, :8, 24:] = torch.rand_like(changed[:, :, :8, 24:])
        with torch.no_grad():
            self.assertTrue(torch.equal(encoder(images, visible), encoder(changed, visible)))
        images.requires_grad_(True)
        encoder(images, visible)[0, 0, 0].backward()
        self.assertEqual(float(images.grad[:, :, 8:, :].abs().sum()), 0.0)
        self.assertEqual(float(images.grad[:, :, :8, 24:].abs().sum()), 0.0)
        self.assertGreater(float(images.grad[:, :, :8, :24].abs().sum()), 0.0)

    def test_corruptions_are_seeded_and_do_not_mutate_input(self):
        image = Image.fromarray(np.random.default_rng(17).integers(0, 256, (32, 48, 3), dtype=np.uint8))
        before = image.tobytes()
        for family in FAMILIES:
            a, b = corrupt(image, family, 3, 17), corrupt(image, family, 3, 17)
            self.assertEqual(a.size, image.size)
            self.assertEqual(a.tobytes(), b.tobytes())
            self.assertNotEqual(a.tobytes(), before)
        self.assertEqual(image.tobytes(), before)

    def config(self, root):
        template = Path(__file__).resolve().parents[1] / "configs/phase1_vits.json"
        config = json.loads(template.read_text())
        config["data"] = {"train_manifest": str(root / "train.jsonl"), "val_manifest": str(root / "val.jsonl"),
                          "image_size": [32, 48], "num_workers": 0}
        config["model"].update(patch_size=8, dim=32, depth=2, heads=4, predictor_dim=16, predictor_depth=1, predictor_heads=4)
        config["training"].update(epochs=2, batch_size=1, save_every_steps=1, validation_batches=2, amp_bfloat16=False)
        generator = np.random.default_rng(17)
        for split in ("train", "val"):
            rows = []
            for index in range(2):
                path = root / f"{split}-{index}.png"
                Image.fromarray(generator.integers(0, 256, (32, 48, 3), dtype=np.uint8)).save(path)
                rows.append({"image_path": str(path), "scene_id": path.stem, "split": split,
                             "boxes": [], "labels": [], "annotation_status": "unavailable_image_only",
                             "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (root / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        return config

    def test_content_duplicate_with_different_names_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            rows = [json.loads(line) for line in (root / "val.jsonl").read_text().splitlines()]
            path = Path(rows[0]["image_path"])
            path.write_bytes((root / "train-0.png").read_bytes())
            rows[0]["image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "val.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "Identical image bytes"):
                validate_data(config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16_forward_backward_and_mask_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            model = CorruptionJEPA(config).cuda().train()
            clean = torch.rand(2, 3, 32, 48, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = model(clean, clean * 0.7)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertIsNotNone(model.predictor.mask_token.grad)
            self.assertTrue(torch.isfinite(model.predictor.mask_token.grad).all())
            self.assertGreater(float(model.online.patch.weight.grad.abs().sum()), 0.0)
            self.assertTrue(all(parameter.grad is None for parameter in model.target.parameters()))

    def test_interrupted_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root)
            original_save = train_module.atomic_checkpoint

            def interrupt_after_first_save(path, value):
                original_save(path, value)
                if Path(path).name == "last.pt" and value["step"] == 1:
                    raise RuntimeError("simulated interruption")

            with mock.patch.object(train_module, "atomic_checkpoint", side_effect=interrupt_after_first_save):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_module.train(copy.deepcopy(config), str(root / "resumed"), 3, device_name="cpu")
            train_module.train(copy.deepcopy(config), str(root / "resumed"), 3,
                               str(root / "resumed/last.pt"), "cpu")
            train_module.train(copy.deepcopy(config), str(root / "control"), 3, device_name="cpu")
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)
            control = torch.load(root / "control/last.pt", weights_only=True)
            self.assertEqual(resumed["step"], control["step"])
            for key, value in resumed["model"].items():
                self.assertTrue(torch.equal(value, control["model"][key]), key)


if __name__ == "__main__":
    unittest.main()
