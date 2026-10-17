from __future__ import annotations

import json
import hashlib
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import atomic_checkpoint, atomic_json, architecture_fingerprint, manifest_fingerprint, seed_all
from .data import DrivingPairs, validate_data
from .models import CorruptionJEPA, ema_momentum


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def lr_at_step(base, step, total, warmup_fraction):
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * min((step - warmup) / max(total - warmup, 1), 1)))


@torch.no_grad()
def validate_model(model, dataset, config, device):
    model.eval()
    options = config["training"]
    loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=False,
                        num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config["seed"]))
    generator = torch.Generator(device=device).manual_seed(config["seed"] + 10_000)
    total, samples, batches = {}, 0, 0
    for index, batch in enumerate(loader):
        if index >= options.get("validation_batches", 64):
            break
        clean, view = batch["clean"].to(device), batch["view"].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and options["amp_bfloat16"]):
            loss, metrics = model(clean, view, generator=generator)
        for key, value in {"loss": loss, **metrics}.items():
            total[key] = total.get(key, 0) + float(value) * len(clean)
        samples += len(clean)
        batches += 1
    model.train()
    return {**{key: value / samples for key, value in total.items()}, "images": samples, "batches": batches,
            "scope": "fixed_validation_subset" if samples < len(dataset) else "full_validation_manifest"}


def train(config: dict, run: str, max_steps: int | None = None, resume: str | None = None, device_name="cuda", warm_start: str | None = None):
    if resume and warm_start:
        raise ValueError("Choose either exact resume or warm start")
    validation = validate_data(config)
    options = config["training"]
    if options["batch_size"] < 1 or options["epochs"] < 1 or max_steps is not None and max_steps < 1:
        raise ValueError("epochs, batch_size and max_steps must be positive")
    if options.get("validation_batches", 64) < 1 or options["save_every_steps"] < 1:
        raise ValueError("validation_batches and save_every_steps must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    run = Path(run).resolve()
    if not resume and run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {run}. Use a new directory or --resume.")
    run.mkdir(parents=True, exist_ok=True)
    seed_all(config["seed"])
    families = config["corruption"]["families"]
    train_families = config["corruption"].get("train_families", families)
    train_data = DrivingPairs(config["data"]["train_manifest"], config["data"]["image_size"], train_families,
                              config["corruption"]["train_severities"], config["seed"])
    val_data = DrivingPairs(config["data"]["val_manifest"], config["data"]["image_size"], families, [2], config["seed"] + 1)
    batches_per_epoch = math.ceil(len(train_data) / options["batch_size"])
    total_steps = min(batches_per_epoch * options["epochs"], max_steps or batches_per_epoch * options["epochs"])
    model = CorruptionJEPA(config).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=options["learning_rate"], weight_decay=options["weight_decay"])
    dataset_hash, architecture_hash = manifest_fingerprint(config), architecture_fingerprint(config)
    step, start_epoch, skip_batches, best = 0, 0, 0, float("inf")
    warm_start_info = None
    if warm_start:
        source = Path(warm_start).resolve()
        state = torch.load(source, map_location="cpu", weights_only=True)
        source_config = state["config"]
        if source_config["model"] != config["model"] or source_config["data"]["image_size"] != config["data"]["image_size"]:
            raise ValueError("Warm start requires identical model and image geometry")
        if "model" not in state:
            raise ValueError("Warm start requires a full JEPA checkpoint including target and predictor")
        model.load_state_dict(state["model"], strict=True)
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        ancestor_steps = state.get("warm_start", {}).get("ancestor_optimizer_steps", 0) if state.get("warm_start") else 0
        warm_start_info = {"source_checkpoint": str(source), "source_sha256": digest.hexdigest(),
                           "source_step": state["step"], "ancestor_optimizer_steps": ancestor_steps + state["step"],
                           "source_manifest_fingerprint": state["manifest_fingerprint"],
                           "model_state_loaded_strictly": True, "optimizer_reset": True,
                           "rng_reset_to_config_seed": True, "epoch_and_learning_rate_schedule_reset": True}
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["manifest_fingerprint"] != dataset_hash or state["architecture_fingerprint"] != architecture_hash:
            raise ValueError("Resume requires identical manifests, model, seed, corruption and training configuration")
        if state["total_steps"] != total_steps:
            raise ValueError("Resume requires the original --max-steps schedule")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, start_epoch, skip_batches, best = state["step"], state["next_epoch"], state["next_batch"], state["best_val"]
        warm_start_info = state.get("warm_start")
        restore_rng(state["rng"])
        if step >= total_steps:
            raise ValueError("The requested training schedule is already complete")
    atomic_json(run / "config.json", config)
    atomic_json(run / "data_validation.json", validation)
    atomic_json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "manifest_fingerprint": dataset_hash, "architecture_fingerprint": architecture_hash,
                "total_steps": total_steps, "resume": str(resume) if resume else None,
                "warm_start": warm_start_info})

    def checkpoint(next_epoch, next_batch, filename="last.pt"):
        atomic_checkpoint(run / filename, {"format_version": 1, "config": config, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": step, "next_epoch": next_epoch, "next_batch": next_batch,
            "best_val": best, "total_steps": total_steps, "rng": rng_state(),
            "warm_start": warm_start_info,
            "manifest_fingerprint": dataset_hash, "architecture_fingerprint": architecture_hash})

    started = time.perf_counter()
    model.train()
    for epoch in range(start_epoch, options["epochs"]):
        train_data.epoch = epoch
        loader = DataLoader(train_data, batch_size=options["batch_size"], shuffle=True,
                            num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda",
                            generator=torch.Generator().manual_seed(config["seed"] + epoch))
        next_epoch, next_batch = epoch, 0
        for batch_no, batch in enumerate(loader):
            if epoch == start_epoch and batch_no < skip_batches:
                continue
            clean, view = batch["clean"].to(device, non_blocking=True), batch["view"].to(device, non_blocking=True)
            lr = lr_at_step(options["learning_rate"], step, total_steps, options["warmup_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and options["amp_bfloat16"]):
                loss, metrics = model(clean, view)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}; previous successful checkpoint retained")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                          options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            momentum = ema_momentum(options["ema_start"], step, total_steps)
            model.update_target(momentum)
            row = {"event": "train", "step": step, "epoch": epoch, "loss": float(loss.detach()), "lr": lr,
                   "ema": momentum, **{key: float(value) for key, value in metrics.items()}}
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 20 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            next_epoch, next_batch = ((epoch + 1, 0) if batch_no + 1 == len(loader) else (epoch, batch_no + 1))
            if step % options["save_every_steps"] == 0:
                checkpoint(next_epoch, next_batch)
            if step >= total_steps:
                break
        val = validate_model(model, val_data, config, device)
        val_row = {"event": "validation", "step": step, "epoch": epoch, **val}
        if not all(math.isfinite(value) for key, value in val.items() if isinstance(value, float)):
            raise FloatingPointError("Validation returned nonfinite metrics")
        with (run / "metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(val_row, allow_nan=False) + "\n")
        print(json.dumps(val_row), flush=True)
        if val["prediction_loss"] < best:
            best = val["prediction_loss"]
            checkpoint(next_epoch, next_batch, "best.pt")
            atomic_checkpoint(run / "encoder.pt", {"format_version": 1, "config": config,
                "encoder_state": model.online.state_dict(),
                "step": step, "manifest_fingerprint": dataset_hash, "architecture_fingerprint": architecture_hash,
                "warm_start": warm_start_info,
                "selection_criterion": "heldout_masked_prediction_loss", "val": val})
        checkpoint(next_epoch, next_batch)
        skip_batches = 0
        if step >= total_steps:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    atomic_checkpoint(run / "final_encoder.pt", {"format_version": 1, "config": config,
        "encoder_state": model.online.state_dict(), "step": step,
        "manifest_fingerprint": dataset_hash, "architecture_fingerprint": architecture_hash,
        "warm_start": warm_start_info, "selection_criterion": "final_completed_schedule"})
    summary = {"status": "training_schedule_complete", "steps": step, "best_val_prediction_loss": best,
               "warm_start": warm_start_info,
               "optimizer_steps_including_ancestors": step + (warm_start_info["ancestor_optimizer_steps"] if warm_start_info else 0),
               "wall_seconds_this_invocation": time.perf_counter() - started,
               "checkpoint": str(run / "last.pt"), "encoder": str(run / "encoder.pt"),
               "final_encoder": str(run / "final_encoder.pt"),
               "performance_claim": "prediction_loss_is_not_detection_accuracy_or_driving_safety"}
    if device.type == "cuda":
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    atomic_json(run / "summary.json", summary)
    return summary
