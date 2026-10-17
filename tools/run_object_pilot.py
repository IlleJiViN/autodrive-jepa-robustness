"""Run the four matched pretraining arms, then a shared clean-trained semantic probe."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_json, load_config
from robustdrive.data import read_manifest, validate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", required=True, help="YOLO proposal config or explicit oracle-box control")
    parser.add_argument("--semantic-config", default="configs/object_pilot.json", help="Human ground-truth boxes and labels")
    parser.add_argument("--run", required=True)
    parser.add_argument("--steps", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-mask-only-jepa", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    run = Path(args.run).resolve()
    if run.exists():
        raise FileExistsError(run)
    training, semantic = load_config(args.training_config), load_config(args.semantic_config)
    validate_data(training)
    validate_data(semantic)
    for split in ("train", "val"):
        a = read_manifest(training["data"][split + "_manifest"])
        b = read_manifest(semantic["data"][split + "_manifest"])
        if not all(r.get("image_sha256") for r in a + b):
            raise ValueError("Matched pilot requires an image hash for every input")
        if [(r["scene_id"], r["image_sha256"], str(Path(r["image_path"]).resolve())) for r in a] != [
                (r["scene_id"], r["image_sha256"], str(Path(r["image_path"]).resolve())) for r in b]:
            raise ValueError("Proposal learning and semantic evaluation must refer to identical image splits")
    if training["model"] != semantic["model"] or training["data"]["image_size"] != semantic["data"]["image_size"]:
        raise ValueError("Training and semantic geometry must match")
    arms = [("random_jepa", "jepa", "random", 0.), ("object_jepa", "jepa", "objects", 1.),
            ("random_mae", "mae", "random", 0.), ("object_mae", "mae", "objects", 1.)]
    if args.include_mask_only_jepa:
        arms.append(("mask_only_jepa", "jepa", "objects", 0.))
    run.mkdir(parents=True)
    atomic_json(run / "protocol.json", {"seed": args.seed, "max_steps_per_arm": args.steps,
                "box_source": training["object_pretraining"]["box_source"], "arms": arms,
                "semantic_probe_config": str(Path(args.semantic_config).resolve()),
                "action_pilot": "not_run_requires_true_action_labels_or_closed_loop_simulator"})
    for name, objective, mask, weight in arms:
        command = [sys.executable, str(ROOT / "tools/run_object_pretraining.py"),
                   "--config", args.training_config, "--objective", objective, "--mask", mask,
                   "--object-loss-weight", str(weight), "--seed", str(args.seed),
                   "--run", str(run / name), "--max-steps", str(args.steps), "--device", args.device]
        with (run / (name + ".stdout.log")).open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"arm": name, "status": "training_complete"}), flush=True)
    command = [sys.executable, str(ROOT / "tools/run_object_probe.py"), "--config", args.semantic_config,
               "--run", str(run / "semantic_probe"), "--device", args.device, "--include-random"]
    for name, _, _, _ in arms:
        command.extend(["--checkpoint", name + "=" + str(run / name / "final_encoder.pt")])
    with (run / "semantic_probe.stdout.log").open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"status": "pretraining_and_semantic_probe_complete", "run": str(run)}), flush=True)


if __name__ == "__main__":
    main()
