"""Run the authorized data/validation/GPU pilot and paired baseline sequentially."""
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
STATUS = RUNS / "pilot_status.json"
START = time.time()


def status(stage, state="running", **extra):
    value = {"status": state, "stage": stage, "pid": os.getpid(), "started_unix": START,
             "updated_unix": time.time(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def command(stage, args):
    status(stage)
    log = RUNS / f"{stage}.log"
    with log.open("wb") as output:
        process = subprocess.Popen([str(ROOT / "run.sh"), *args], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        status(stage, command=args, child_pid=process.pid, log=str(log))
        code = process.wait()
    if code:
        raise RuntimeError(f"{stage} failed with exit {code}; log: {log}")


def main():
    status("waiting_for_data")
    ready = ROOT / "data/bdd100k/pilot_provenance.json"
    while not ready.exists():
        if time.time() - START > 1800:
            raise TimeoutError("Dataset preparation exceeded 30 minutes")
        time.sleep(5)
    command("validate_pilot_data", ["validate", "--config", "configs/pilot_corruption.json"])
    experiments = [("corruption", "configs/pilot_corruption.json"), ("clean", "configs/pilot_clean.json")]
    for name, config in experiments:
        run = f"runs/pilot-{name}-seed17"
        command(f"train_{name}", ["train", "--config", config, "--run", run, "--max-steps", "200"])
        command(f"evaluate_{name}", ["evaluate", "--config", config, "--checkpoint", run + "/encoder.pt",
                                     "--output", run + "/robustness-val256.json", "--max-images", "0"])
    status("summarize")
    results = {name: json.loads((RUNS / f"pilot-{name}-seed17/robustness-val256.json").read_text()) for name, _ in experiments}
    rows = []
    for augmented, baseline in zip(results["corruption"]["conditions"], results["clean"]["conditions"]):
        if (augmented["family"], augmented["severity"]) != (baseline["family"], baseline["severity"]):
            raise ValueError("Paired evaluation conditions differ")
        rows.append({"family": augmented["family"], "severity": augmented["severity"],
                     "corruption_jepa_cosine": augmented["cosine_mean"], "clean_jepa_cosine": baseline["cosine_mean"],
                     "cosine_delta": augmented["cosine_mean"] - baseline["cosine_mean"]})
    fingerprints = {}
    for name, _ in experiments:
        fingerprints[name] = {}
        for filename in ("last.pt", "best.pt", "encoder.pt"):
            path = RUNS / f"pilot-{name}-seed17" / filename
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            fingerprints[name][filename] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    summary = {"status": "paired_gpu_pilot_complete", "dataset": json.loads(ready.read_text()),
               "training": {name: json.loads((RUNS / f"pilot-{name}-seed17/summary.json").read_text()) for name, _ in experiments},
               "health": {name: value["health"] for name, value in results.items()},
               "conditions": rows, "mean_cosine_delta": sum(row["cosine_delta"] for row in rows) / len(rows),
               "checkpoints": fingerprints, "total_wall_seconds": time.time() - START,
               "scope": "one_seed_200_steps_each_2048_train_images_256_validation_images",
               "interpretation": "Pilot pipeline evidence only; cosine agreement is not mAP, semantic quality or driving safety. No detection labels were available."}
    (RUNS / "paired_pilot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    status("complete", state="complete", summary=str(RUNS / "paired_pilot_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=str(error))
        traceback.print_exc()
        raise
