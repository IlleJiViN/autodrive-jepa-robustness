"""Evaluate selected and final continuation states with instance diagnostics."""
import json
import os
import pathlib
import subprocess
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parents[1]
START = time.time()
STATUS = ROOT / "runs/diagnostics_status.json"


def status(stage, state="running", **extra):
    value = {"stage": stage, "status": state, "pid": os.getpid(), "updated_unix": time.time(), **extra}
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(STATUS)
    print(json.dumps(value), flush=True)


def main():
    status("waiting_for_continuation")
    while True:
        p = ROOT / "runs/continuation_status.json"
        s = json.loads(p.read_text()) if p.exists() else {}
        if s.get("status") == "failed":
            raise RuntimeError("Continuation failed")
        if s.get("status") == "complete":
            break
        if time.time() - START > 3600:
            raise TimeoutError("Continuation did not finish within an hour")
        time.sleep(5)
    evaluations = {}
    for experiment in ("corruption", "clean"):
        evaluations[experiment] = {}
        for selection, checkpoint in (("selected", "encoder.pt"), ("final", "last.pt")):
            run = f"runs/continue10-{experiment}-seed17"
            result = f"{run}/diagnostics-{selection}-val256.json"
            if not (ROOT / result).exists():
                args = [str(ROOT / "run.sh"), "evaluate", "--config", f"configs/continue10_{experiment}.json",
                        "--checkpoint", f"{run}/{checkpoint}", "--output", result, "--max-images", "0"]
                log = ROOT / f"runs/diagnostics_{experiment}_{selection}.log"
                with log.open("wb") as output:
                    p = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
                    status(f"{experiment}_{selection}", child_pid=p.pid, log=str(log))
                    code = p.wait()
                if code:
                    raise RuntimeError(f"Evaluation exited {code}: {log}")
            evaluations[experiment][selection] = json.loads((ROOT / result).read_text())
    output = ROOT / "runs/continuation_diagnostics_summary.json"
    output.write_text(json.dumps({"status": "continuation_diagnostics_complete", "evaluations": evaluations,
                                 "wall_seconds": time.time() - START}, indent=2) + "\n")
    status("complete", state="complete", summary=str(output))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", state="failed", error=str(error))
        traceback.print_exc()
        raise
