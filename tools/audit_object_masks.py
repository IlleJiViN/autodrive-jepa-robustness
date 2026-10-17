"""Audit real box geometry and whole-object mask coverage without loading images."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_json, load_config, stable_seed
from robustdrive.data import read_manifest
from robustdrive.object_pretraining import make_object_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/object_pilot.json")
    parser.add_argument("--output", default="runs/object_mask_audit.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    config = load_config(args.config)
    grid = tuple(v // config["model"]["patch_size"] for v in config["data"]["image_size"])
    summaries = {}
    torch.set_num_threads(2)
    for split in ("train", "val"):
        rows = read_manifest(config["data"][split + "_manifest"])
        selected, oversized, failed = [], [], 0
        for row in rows:
            if row.get("annotation_status") != config["object_pretraining"].get("annotation_status", "available_legacy_bdd100k"):
                raise ValueError("Audit box source must match the declared experiment")
            boxes = torch.tensor(row["boxes"], dtype=torch.float32).reshape(-1, 4)
            boxes /= torch.tensor([row["width"], row["height"], row["width"], row["height"]])
            mask = make_object_mask([boxes], grid, config["model"]["mask_ratio"],
                        [stable_seed(config["seed"], split, 0, row["scene_id"], "object-mask")],
                        max_objects=config["object_pretraining"]["max_objects"])
            selected.append(len(mask.selected_objects[0]))
            oversized.append(mask.oversized_objects[0])
            failed += int(not mask.valid_objects.any())
        summaries[split] = {"images": len(rows), "annotated_boxes": sum(len(r["boxes"]) for r in rows),
                            "images_without_selected_object": failed,
                            "mean_selected_whole_objects": float(np.mean(selected)),
                            "oversized_objects_skipped": int(sum(oversized)),
                            "visible_tokens_per_image": mask.visible.shape[1],
                            "masked_tokens_per_image": mask.masked.shape[1]}
    result = {"scope": "annotation_geometry_only_not_training_or_model_quality", "seed": config["seed"],
              "epoch": 0, "box_source": config["object_pretraining"]["box_source"], "splits": summaries}
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
