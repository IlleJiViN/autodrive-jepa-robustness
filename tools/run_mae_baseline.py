"""Train a matched-budget masked-autoencoder baseline for the JEPA cost study."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robustdrive.common import atomic_checkpoint, atomic_json, load_config, seed_all
from robustdrive.data import DrivingPairs, validate_data
from robustdrive.models import VisionEncoder, select_tokens
from robustdrive.train import lr_at_step


class MaskedAutoencoder(nn.Module):
    """ViT-S encoder with a small pixel decoder, using the JEPA mask ratio."""

    def __init__(self, config: dict):
        super().__init__()
        data, spec = config["data"], config["model"]
        self.patch_size = spec["patch_size"]
        self.encoder = VisionEncoder(data["image_size"], spec["patch_size"], spec["dim"],
                                     spec["depth"], spec["heads"])
        count = self.encoder.grid[0] * self.encoder.grid[1]
        self.visible_count = max(1, min(count - 1, int(count * (1 - spec["mask_ratio"]))))
        decoder_dim = spec["predictor_dim"]
        self.decoder_projection = nn.Linear(spec["dim"], decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position = nn.Parameter(torch.zeros(1, count, decoder_dim))
        layer = nn.TransformerEncoderLayer(decoder_dim, spec["predictor_heads"], decoder_dim * 4,
                                           dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, spec["predictor_depth"],
                                             norm=nn.LayerNorm(decoder_dim), enable_nested_tensor=False)
        self.pixel_prediction = nn.Linear(decoder_dim, self.patch_size * self.patch_size * 3)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_position, std=0.02)
        for block in self.decoder.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)

    def forward(self, images: torch.Tensor):
        batch = images.shape[0]
        count = self.encoder.grid[0] * self.encoder.grid[1]
        permutation = torch.rand(batch, count, device=images.device).argsort(dim=1)
        visible, masked = permutation[:, :self.visible_count], permutation[:, self.visible_count:]
        context = self.encoder(images, visible)
        encoded = self.decoder_projection(context)
        tokens = self.mask_token.to(dtype=encoded.dtype).expand(batch, count, -1).clone()
        tokens.scatter_(1, visible.unsqueeze(-1).expand_as(encoded), encoded)
        decoded = self.decoder(tokens + self.decoder_position.to(dtype=encoded.dtype))
        prediction = select_tokens(self.pixel_prediction(decoded), masked)
        return prediction, masked


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = images.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Image geometry must be divisible by patch_size")
    grid_h, grid_w = height // patch_size, width // patch_size
    patches = images.reshape(batch, channels, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, grid_h * grid_w, -1)
    mean = patches.mean(dim=-1, keepdim=True)
    variance = patches.var(dim=-1, keepdim=True, unbiased=False)
    return (patches - mean) / torch.sqrt(variance + 1.0e-6)


def mae_loss(model: MaskedAutoencoder, input_images: torch.Tensor, clean_targets: torch.Tensor):
    prediction, masked = model(input_images)
    target = select_tokens(patchify(clean_targets, model.patch_size), masked)
    return F.mse_loss(prediction.float(), target.float())


@torch.no_grad()
def validate(model, dataset, config, device):
    model.eval()
    options = config["training"]
    loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=False,
                        num_workers=config["data"]["num_workers"], pin_memory=device.type == "cuda")
    losses, images_seen = 0.0, 0
    for index, batch in enumerate(loader):
        if index >= options.get("validation_batches", 64):
            break
        input_images = batch["view"].to(device, non_blocking=True)
        clean = batch["clean"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = mae_loss(model, input_images, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError("MAE validation loss is nonfinite")
        losses += float(loss) * len(clean)
        images_seen += len(clean)
    model.train()
    return {"masked_pixel_mse": losses / images_seen, "images": images_seen,
            "batches": min(len(loader), options.get("validation_batches", 64))}


def train(config: dict, run: pathlib.Path):
    options = config["training"]
    if run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Run directory is not empty; preserving existing artifacts: {run}")
    run.mkdir(parents=True, exist_ok=True)
    validation_info = validate_data(config)
    seed_all(config["seed"])
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The matched cost experiment requires the Coder CUDA GPU")
    train_families = config["corruption"].get("train_families", config["corruption"]["families"])
    train_data = DrivingPairs(config["data"]["train_manifest"], config["data"]["image_size"],
                              train_families, config["corruption"]["train_severities"], config["seed"])
    val_data = DrivingPairs(config["data"]["val_manifest"], config["data"]["image_size"],
                            ["clean"], [1], config["seed"] + 1)
    batches_per_epoch = math.ceil(len(train_data) / options["batch_size"])
    total_steps = batches_per_epoch * options["epochs"]
    model = MaskedAutoencoder(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"],
                                  weight_decay=options["weight_decay"])
    atomic_json(run / "config.json", config)
    atomic_json(run / "data_validation.json", validation_info)
    atomic_json(run / "environment.json", {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                "device": torch.cuda.get_device_name(device), "seed": config["seed"],
                "steps": total_steps, "objective": "masked_patch_pixel_reconstruction",
                "encoder_geometry": config["model"], "mask_ratio": config["model"]["mask_ratio"]})
    started = time.perf_counter()
    step, best_val = 0, float("inf")
    torch.cuda.reset_peak_memory_stats(device)
    metrics_path = run / "metrics.jsonl"
    model.train()
    for epoch in range(options["epochs"]):
        train_data.epoch = epoch
        loader = DataLoader(train_data, batch_size=options["batch_size"], shuffle=True,
                            num_workers=config["data"]["num_workers"], pin_memory=True,
                            generator=torch.Generator().manual_seed(config["seed"] + epoch))
        for batch in loader:
            input_images = batch["view"].to(device, non_blocking=True)
            clean_targets = batch["clean"].to(device, non_blocking=True)
            lr = lr_at_step(options["learning_rate"], step, total_steps, options["warmup_fraction"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=options["amp_bfloat16"]):
                loss = mae_loss(model, input_images, clean_targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite MAE loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {"event": "train", "step": step, "epoch": epoch, "loss": float(loss.detach()), "lr": lr}
            with metrics_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, allow_nan=False) + "\n")
            if step % 20 == 0 or step == 1:
                print(json.dumps(row), flush=True)
            if step % options["save_every_steps"] == 0:
                atomic_checkpoint(run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "epoch": epoch, "config": config})
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=options["amp_bfloat16"]):
            val = validate(model, val_data, config, device)
        row = {"event": "validation", "step": step, "epoch": epoch, **val}
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        if val["masked_pixel_mse"] < best_val:
            best_val = val["masked_pixel_mse"]
        atomic_checkpoint(run / "last.pt", {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "step": step, "epoch": epoch + 1, "config": config})
    torch.cuda.synchronize(device)
    atomic_checkpoint(run / "final_encoder.pt", {"format_version": 1, "config": config,
        "encoder_state": model.encoder.state_dict(), "step": step,
        "selection_criterion": "final_completed_schedule", "objective": "masked_patch_pixel_reconstruction"})
    summary = {"status": "training_schedule_complete", "steps": step, "best_val_masked_pixel_mse": best_val,
               "wall_seconds_this_invocation": time.perf_counter() - started,
               "final_encoder": str(run / "final_encoder.pt"), "objective": "masked_patch_pixel_reconstruction",
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    atomic_json(run / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continue10_corruption.json")
    parser.add_argument("--run", default="runs/cost-mae-vits-seed17")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    train(config, ROOT / args.run)


if __name__ == "__main__":
    main()
