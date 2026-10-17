"""Run matched pretraining jobs while sampling GPU-board power."""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/cost-benchmark-seed17"
POWER_QUERY = ["nvidia-smi", "--query-gpu=timestamp,power.draw,utilization.gpu",
               "--format=csv,noheader", "-lms", "500"]


def start_sampler(path: pathlib.Path):
    handle = path.open("x", encoding="ascii")
    process = subprocess.Popen(POWER_QUERY, stdout=handle, stderr=subprocess.DEVNULL, text=True)
    return process, handle


def stop_sampler(process, handle):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=3)
    handle.close()


def idle_baseline():
    samples = OUT / "idle-gpu-samples.csv"
    if samples.exists():
        raise FileExistsError(f"Preserving existing power samples: {samples}")
    process, handle = start_sampler(samples)
    time.sleep(15)
    stop_sampler(process, handle)


def measure(name: str, command: list[str]):
    log_path = OUT / f"{name}.stdout.log"
    sample_path = OUT / f"{name}-gpu-samples.csv"
    metadata_path = OUT / f"{name}-process.json"
    if any(path.exists() for path in (log_path, sample_path, metadata_path)):
        raise FileExistsError(f"Preserving prior benchmark artifacts for {name}")
    sampler, handle = start_sampler(sample_path)
    time.sleep(1)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, check=False)
    wall_seconds = time.perf_counter() - started
    stop_sampler(sampler, handle)
    metadata = {"name": name, "command": command, "exit_code": result.returncode,
                "process_wall_seconds": wall_seconds, "power_samples": str(sample_path)}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)
    if result.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        print("\n".join(tail), file=sys.stderr, flush=True)
        raise SystemExit(result.returncode)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    idle_baseline()
    measure("clean-jepa", ["./run.sh", "train", "--config", "configs/continue10_clean.json",
                            "--run", "runs/cost-clean-jepa-seed17", "--max-steps", "2560"])
    measure("corruption-jepa", ["./run.sh", "train", "--config", "configs/continue10_corruption.json",
                                 "--run", "runs/cost-corruption-jepa-seed17", "--max-steps", "2560"])
    measure("mae-vits", [sys.executable, "tools/run_mae_baseline.py", "--config",
                          "configs/continue10_corruption.json", "--run", "runs/cost-mae-vits-seed17"])
    print("Pretraining cost runs complete. Detector evaluation is a separate stage.", flush=True)


if __name__ == "__main__":
    main()
