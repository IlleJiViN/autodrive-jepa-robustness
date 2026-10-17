"""Back up matched cost-benchmark evidence and final model checkpoints from Coder."""
from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/home/coder/autodrive-jepa-robustness"
CODER = ["coder", "--url", "https://computeshare-access.tail641d9d.ts.net",
         "ssh", "--disable-autostart", "owner/jwj-01", "--"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Backup destination escaped the project root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def transfer_evidence(names: list[str]) -> list[str]:
    source = ("import base64,json,pathlib\n"
              f"root=pathlib.Path({REMOTE!r})\n"
              f"names={names!r}\n"
              "print(json.dumps({n:base64.b64encode((root/n).read_bytes()).decode('ascii') "
              "for n in names if (root/n).is_file()}))\n")
    result = subprocess.run(CODER + ["python3"], input=source.encode("utf-8"),
                            capture_output=True, check=True)
    payload = json.loads(result.stdout)
    for name, encoded in payload.items():
        target = local_path(name)
        data = base64.b64decode(encoded, validate=True)
        if target.exists():
            if target.read_bytes() != data:
                raise FileExistsError(f"Existing local evidence differs: {target}")
        else:
            target.write_bytes(data)
    return list(payload)


def copy_checkpoint(remote_relative: str, local_relative: str, expected_bytes: int, expected_hash: str):
    target = local_path(local_relative)
    if target.exists():
        if target.stat().st_size != expected_bytes or sha256(target) != expected_hash:
            raise FileExistsError(f"Existing local checkpoint differs: {target}")
    else:
        temporary = target.with_suffix(target.suffix + ".part")
        with temporary.open("wb") as output:
            subprocess.run(CODER + ["cat", f"{REMOTE}/{remote_relative}"], stdin=subprocess.DEVNULL,
                           stdout=output, check=True)
        if temporary.stat().st_size != expected_bytes or sha256(temporary) != expected_hash:
            raise ValueError(f"Checkpoint backup hash mismatch: {temporary}")
        temporary.replace(target)
    return {"local_path": str(target), "remote_path": f"{REMOTE}/{remote_relative}",
            "bytes": expected_bytes, "sha256": expected_hash, "local_sha256_verified": True}


def remote_size(remote_relative: str) -> int:
    result = subprocess.run(CODER + ["stat", "-c", "%s", f"{REMOTE}/{remote_relative}"],
                            capture_output=True, check=True, text=True)
    return int(result.stdout.strip())


def main():
    names = [
        "runs/cost_benchmark_summary.json", "runs/detector_cost_probe_summary.json",
        "runs/detector_cost_probe_status.json", "runs/detector-cost-probe-seed17-v1/protocol.json",
        "runs/cost-benchmark-seed17/idle-gpu-samples.csv",
    ]
    stages = ("clean-jepa", "corruption-jepa", "mae-vits-torch2141")
    for stage in stages:
        names += [f"runs/cost-benchmark-seed17/{stage}-process.json",
                  f"runs/cost-benchmark-seed17/{stage}-gpu-samples.csv",
                  f"runs/cost-benchmark-seed17/{stage}.stdout.log"]
    # Keep the first MAE run for audit; its torch 2.14.0 runtime excludes it from the matched comparison.
    names += ["runs/cost-benchmark-seed17/mae-vits-process.json",
              "runs/cost-benchmark-seed17/mae-vits-gpu-samples.csv",
              "runs/cost-benchmark-seed17/mae-vits.stdout.log"]
    pretraining_runs = {
        "clean": "cost-clean-jepa-seed17",
        "corruption": "cost-corruption-jepa-seed17",
        "mae": "cost-mae-vits-seed17-torch2141",
    }
    for run in pretraining_runs.values():
        names += [f"runs/{run}/{filename}" for filename in
                  ("summary.json", "config.json", "environment.json", "data_validation.json", "metrics.jsonl")]
    detector_root = "runs/detector-cost-probe-seed17-v1"
    for method in ("random", "clean", "corruption", "mae"):
        names += [f"{detector_root}/{method}/{filename}" for filename in
                  ("metrics.jsonl", "checkpoint_metadata.json", "results.json")]
        names += [f"{detector_root}/{method}/eval-{family}-{severity}.json"
                  for family, severity in [("clean", 0)] +
                  [(family, severity) for family in ("fog", "rain", "motion_blur", "low_light")
                   for severity in range(1, 6)]]

    copied_files = transfer_evidence(names)
    summary = json.loads(local_path("runs/detector_cost_probe_summary.json").read_text(encoding="utf-8"))
    checkpoints = []
    for method, run_name in pretraining_runs.items():
        result = summary["results"][method]
        relative = f"runs/{run_name}/final_encoder.pt"
        checkpoints.append(copy_checkpoint(relative, relative, remote_size(relative), result["encoder_sha256"]))
    for method in ("random", "clean", "corruption", "mae"):
        result = summary["results"][method]
        checkpoints.append(copy_checkpoint(f"{detector_root}/{method}/detector.pt",
                            f"{detector_root}/{method}/detector.pt", result["checkpoint_bytes"],
                            result["detector_checkpoint_sha256"]))
    receipt = {"workspace": "owner/jwj-01", "remote_root": REMOTE, "experiment": "cost_benchmark",
               "created_unix": time.time(), "evidence_files": len(copied_files), "checkpoints": checkpoints,
               "images_location": "VM only; labeled manifests and image provenance are backed up locally"}
    receipt_path = local_path("runs/cost_benchmark_local_backup.json")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "backup_complete", "evidence_files": len(copied_files),
                      "verified_checkpoints": len(checkpoints), "receipt": str(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()
