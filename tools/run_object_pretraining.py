"""Matched random/box-guided JEPA and denoising pixel objectives, with exact resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_checkpoint, atomic_json, load_config, manifest_fingerprint, seed_all, stable_seed
from robustdrive.data import validate_data
from robustdrive.models import ema_momentum
from robustdrive.object_pretraining import ObjectPairs, ObjectPretrainer, collate_object_pairs, make_object_mask
from robustdrive.train import lr_at_step, restore_rng, rng_state


def mask_for_batch(batch, model, config, epoch, device, validation=False):
    spec = config["object_pretraining"]
    seeds = [stable_seed(config["seed"], "val" if validation else "train", epoch, key, "object-mask")
             for key in batch["sample_id"]]
    return make_object_mask(batch["boxes"], model.online.grid, config["model"]["mask_ratio"],
                            seeds, spec["mask_mode"], spec["max_objects"]).to(device)


def data_loader(dataset, config, epoch, shuffle):
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=shuffle,
                      num_workers=config["data"]["num_workers"], collate_fn=collate_object_pairs,
                      generator=torch.Generator().manual_seed(config["seed"] + epoch))


@torch.no_grad()
def validate(model, dataset, config, device):
    model.eval()
    sums, count = {}, 0
    for batch_no, batch in enumerate(data_loader(dataset, config, 0, False)):
        if batch_no >= config["training"]["validation_batches"]:
            break
        mask = mask_for_batch(batch, model, config, 0, device, validation=True)
        with torch.autocast(device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda" and config["training"]["amp_bfloat16"]):
            loss, metrics = model(batch["clean"].to(device), batch["view"].to(device), mask)
        n = len(batch["clean"])
        for key, value in {"loss": loss, **metrics}.items():
            sums[key] = sums.get(key, 0) + float(value) * n
        count += n
    model.train()
    return {**{key: value / count for key, value in sums.items()}, "images": count}


def train(config, run, max_steps=None, device_name="cuda", resume=None):
    data_validation = validate_data(config)
    run, device = Path(run).resolve(), torch.device(device_name)
    if run.exists() and any(run.iterdir()) and not resume:
        raise FileExistsError(f"Preserving existing run: {run}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    spec, options = config["object_pretraining"], config["training"]
    if spec["mask_mode"] not in ("random", "objects") or options["epochs"] < 1 or options["batch_size"] < 1:
        raise ValueError("Invalid experiment configuration")
    families = config["corruption"].get("train_families", config["corruption"]["families"])
    train_data = ObjectPairs(config["data"]["train_manifest"], config["data"]["image_size"],
                             families, config["corruption"]["train_severities"], config["seed"])
    val_data = ObjectPairs(config["data"]["val_manifest"], config["data"]["image_size"],
                           config["corruption"]["families"], [2], config["seed"] + 1)
    for dataset in (train_data, val_data):
        expected = spec.get("annotation_status", "available_legacy_bdd100k")
        if any(row.get("annotation_status") != expected for row in dataset.records):
            raise ValueError("Manifest box provenance must match the declared experiment source in every arm")
    total_steps = min(math.ceil(len(train_data) / options["batch_size"]) * options["epochs"],
                      max_steps or math.inf)
    seed_all(config["seed"])
    model = ObjectPretrainer(config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=options["learning_rate"], weight_decay=options["weight_decay"])
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    data_hash = manifest_fingerprint(config)
    source_hash = hashlib.sha256()
    for relative in ("src/robustdrive/object_pretraining.py", "src/robustdrive/models.py",
                     "src/robustdrive/data.py", "src/robustdrive/corruptions.py",
                     "src/robustdrive/common.py", "src/robustdrive/train.py", "tools/run_object_pretraining.py"):
        source_hash.update(relative.encode())
        source_hash.update((ROOT / relative).read_bytes())
    source_fingerprint = source_hash.hexdigest()
    step, epoch_start, skip = 0, 0, 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (state["config_fingerprint"], state["manifest_fingerprint"], state["total_steps"]) != (config_hash, data_hash, total_steps):
            raise ValueError("Exact resume requires identical configuration, manifests and step schedule")
        if state["source_fingerprint"] != source_fingerprint:
            raise ValueError("Exact resume requires identical training source")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        step, epoch_start, skip = state["step"], state["next_epoch"], state["next_batch"]
        restore_rng(state["rng"])
        if step >= total_steps:
            raise ValueError("Schedule already complete")
    run.mkdir(parents=True, exist_ok=True)
    atomic_json(run / "config.json", config)
    atomic_json(run / "data_validation.json", data_validation)
    atomic_json(run / "environment.json", {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "total_steps": total_steps, "box_source": spec["box_source"],
                "source_fingerprint": source_fingerprint,
                "class_labels_used_in_pretraining": False, "whole_image_context": True})
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def save(next_epoch, next_batch):
        atomic_checkpoint(run / "last.pt", {"format_version": 1, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "total_steps": total_steps,
            "next_epoch": next_epoch, "next_batch": next_batch, "rng": rng_state(),
            "config_fingerprint": config_hash, "manifest_fingerprint": data_hash,
            "source_fingerprint": source_fingerprint})

    for epoch in range(epoch_start, options["epochs"]):
        train_data.epoch = epoch
        loader = data_loader(train_data, config, epoch, True)
        for batch_no, batch in enumerate(loader):
            if epoch == epoch_start and batch_no < skip:
                continue
            mask = mask_for_batch(batch, model, config, epoch, device)
            lr = lr_at_step(options["learning_rate"], step, total_steps, options["warmup_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda" and options["amp_bfloat16"]):
                loss, metrics = model(batch["clean"].to(device), batch["view"].to(device), mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            model.update_target(ema_momentum(options["ema_start"], step, total_steps))
            row = {"event": "train", "step": step, "epoch": epoch, "lr": lr,
                   "loss": float(loss.detach()), **{k: float(v) for k, v in metrics.items()}}
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            next_epoch, next_batch = ((epoch + 1, 0) if batch_no + 1 == len(loader) else (epoch, batch_no + 1))
            if step % options["save_every_steps"] == 0 or step >= total_steps:
                save(next_epoch, next_batch)
            if step >= total_steps:
                break
        val = {"event": "validation", "step": step, "epoch": epoch, **validate(model, val_data, config, device)}
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(val, allow_nan=False) + "\n")
        print(json.dumps(val), flush=True)
        save(next_epoch, next_batch)
        skip = 0
        if step >= total_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    atomic_checkpoint(run / "final_encoder.pt", {"format_version": 1, "config": config,
                      "encoder_state": model.online.state_dict(), "step": step,
                      "objective": spec["objective"], "mask_mode": spec["mask_mode"],
                      "manifest_fingerprint": data_hash, "source_fingerprint": source_fingerprint,
                      "selection_criterion": "final_completed_schedule"})
    summary = {"status": "training_schedule_complete", "steps": step, "experiment": spec,
               "wall_seconds_this_invocation": time.perf_counter() - started,
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
               "final_encoder": str(run / "final_encoder.pt"),
               "performance_claim": "Training losses do not establish semantics or driving safety"}
    atomic_json(run / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/object_pilot.json")
    parser.add_argument("--objective", choices=["jepa", "mae"], required=True)
    parser.add_argument("--mask", choices=["random", "objects"], required=True)
    parser.add_argument("--object-loss-weight", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config = load_config(args.config)
    config["object_pretraining"].update(objective=args.objective, mask_mode=args.mask)
    if args.mask == "random":
        config["object_pretraining"]["object_loss_weight"] = 0.0
    if args.object_loss_weight is not None:
        config["object_pretraining"]["object_loss_weight"] = args.object_loss_weight
    if args.seed is not None:
        config["seed"] = args.seed
    print(json.dumps(train(config, args.run, args.max_steps, args.device, args.resume)), flush=True)


if __name__ == "__main__":
    main()
