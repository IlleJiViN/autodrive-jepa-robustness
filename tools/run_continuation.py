"""Continue both seed17 JEPA pilots with a new ten-epoch schedule."""
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
START = time.time()
STATUS = RUNS / "continuation_status.json"


def status(stage, state="running", **extra):
    value = {"status": state, "stage": stage, "pid": os.getpid(), "started_unix": START,
             "updated_unix": time.time(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(STATUS)
    print(json.dumps(value), flush=True)


def command(stage, args):
    log = RUNS / f"{stage}.log"
    with log.open("wb") as output:
        process = subprocess.Popen([str(ROOT / "run.sh"), *args], cwd=ROOT,
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        status(stage, child_pid=process.pid, command=args, log=str(log))
        code = process.wait()
    if code:
        raise RuntimeError(f"{stage} exited {code}; see {log}")


def main():
    status("starting")
    initial = json.loads((RUNS / "paired_pilot_summary.json").read_text())
    results, training, checkpoints = {}, {}, {}
    for name in ("corruption", "clean"):
        config = f"configs/continue10_{name}.json"
        run = f"runs/continue10-{name}-seed17"
        source = f"runs/pilot-{name}-seed17/last.pt"
        command(f"continue10_train_{name}", ["train", "--config", config, "--run", run,
                                              "--warm-start", source])
        command(f"continue10_evaluate_{name}", ["evaluate", "--config", config,
                                                 "--checkpoint", run + "/encoder.pt",
                                                 "--output", run + "/robustness-val256.json", "--max-images", "0"])
        folder = ROOT / run
        results[name] = json.loads((folder / "robustness-val256.json").read_text())
        training[name] = json.loads((folder / "summary.json").read_text())
        checkpoints[name] = {}
        for filename in ("last.pt", "best.pt", "encoder.pt", "final_encoder.pt"):
            path = folder / filename
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            checkpoints[name][filename] = {"path": str(path), "bytes": path.stat().st_size,
                                            "sha256": digest.hexdigest()}
    conditions = []
    for a, b in zip(results["corruption"]["conditions"], results["clean"]["conditions"]):
        if (a["family"], a["severity"]) != (b["family"], b["severity"]):
            raise ValueError("Paired conditions differ")
        conditions.append({"family": a["family"], "severity": a["severity"],
                           "corruption_jepa_cosine": a["cosine_mean"], "clean_jepa_cosine": b["cosine_mean"],
                           "cosine_delta": a["cosine_mean"] - b["cosine_mean"]})
    summary = {"status": "paired_continuation_complete", "dataset": initial["dataset"],
               "training": training, "health": {n: r["health"] for n, r in results.items()},
               "conditions": conditions,
               "mean_cosine_delta": sum(r["cosine_delta"] for r in conditions) / len(conditions),
               "pilot_mean_cosine_delta": initial["mean_cosine_delta"], "checkpoints": checkpoints,
               "total_wall_seconds": time.time() - START,
               "scope": "same_2048_train_256_validation_seed17_each_200_prior_plus_2560_new_steps",
               "interpretation": "Subset representation experiment, not full BDD100K training or detector mAP."}
    (RUNS / "paired_continuation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    status("complete", state="complete", summary=str(RUNS / "paired_continuation_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=str(error))
        traceback.print_exc()
        raise
