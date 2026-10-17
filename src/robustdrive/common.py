from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def stable_seed(*parts) -> int:
    raw = json.dumps(parts, ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(raw.encode()).digest()[:8], "little") % (2**63 - 1)


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("train_manifest", "val_manifest"):
        p = Path(config["data"][key])
        config["data"][key] = str(p if p.is_absolute() else ROOT / p)
    return config


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_checkpoint(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def manifest_fingerprint(config: dict) -> str:
    digest = hashlib.sha256()
    for key in ("train_manifest", "val_manifest"):
        digest.update(key.encode())
        digest.update(Path(config["data"][key]).read_bytes())
    return digest.hexdigest()


def architecture_fingerprint(config: dict) -> str:
    settings = {"model": config["model"], "image_size": config["data"]["image_size"],
                "seed": config["seed"], "corruption": config["corruption"],
                "training": config["training"]}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
