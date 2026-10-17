from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import load_config
from .data import import_bdd, validate_data


def main():
    parser = argparse.ArgumentParser(description="Driving corruption-aware JEPA: local data only")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Show settings and data availability without constructing a model")
    plan.add_argument("--config", default="configs/phase1_vits.json")
    prepare = commands.add_parser("import-bdd", help="Convert existing BDD100K detection labels to JSONL")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--images", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--split", required=True, choices=["train", "val", "test"])
    validation = commands.add_parser("validate", help="Check manifests, image headers, boxes and scene split")
    validation.add_argument("--config", default="configs/phase1_vits.json")
    training = commands.add_parser("train", help="Start or resume JEPA representation training")
    training.add_argument("--config", default="configs/phase1_vits.json")
    training.add_argument("--run", required=True)
    training.add_argument("--max-steps", type=int)
    training.add_argument("--resume")
    training.add_argument("--warm-start", help="Load full JEPA model state into a new optimizer and schedule")
    training.add_argument("--device", default="cuda")
    evaluation = commands.add_parser("evaluate", help="Measure held-out embedding robustness and collapse indicators")
    evaluation.add_argument("--config", default="configs/phase1_vits.json")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--max-images", type=int, default=512, help="0 evaluates all validation images")
    evaluation.add_argument("--device", default="cuda")
    args = parser.parse_args()
    try:
        if args.command == "import-bdd":
            result = import_bdd(args.annotations, args.images, args.output, args.split)
        else:
            config = load_config(args.config)
            if args.command == "plan":
                ready = {name: Path(config["data"][name]).is_file() for name in ("train_manifest", "val_manifest")}
                result = {"status": "plan_only", "config": config, "manifest_exists": ready,
                          "training_started": False, "download_behavior": "none"}
            elif args.command == "validate":
                result = validate_data(config)
            elif args.command == "train":
                from .train import train
                result = train(config, args.run, args.max_steps, args.resume, args.device, args.warm_start)
            elif args.command == "evaluate":
                from .evaluate import evaluate
                result = evaluate(config, args.checkpoint, args.output, args.max_images, args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
