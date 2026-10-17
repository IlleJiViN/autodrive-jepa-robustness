"""Back up this project's pilot evidence and verified checkpoints from Coder."""
import argparse
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


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def destination(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Backup destination outside project")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Also back up resumable last.pt files")
    parser.add_argument("--experiment", choices=("pilot", "continue10", "detector"), default="pilot")
    args = parser.parse_args()
    prefix = args.experiment
    summary_name = {"pilot": "paired_pilot_summary.json", "continue10": "paired_continuation_summary.json",
                    "detector": "detector_probe_summary.json"}[prefix]
    names = ["runs/paired_pilot_summary.json", "runs/pilot_status.json", "runs/checkpoint_audit.json",
             "runs/unit_checks.log", "runs/validate_pilot_data.log", "runs/pilot_orchestrator_retry.log",
             "data/bdd100k/pilot_provenance.json", "data/bdd100k/SOURCE_LICENSE_NOTICE.md",
             "data/bdd100k/SOURCE_README.md", "data/manifests/bdd100k_train.jsonl",
             "data/manifests/bdd100k_val.jsonl"]
    for experiment in ("corruption", "clean"):
        names += [f"runs/train_{experiment}.log", f"runs/evaluate_{experiment}.log"]
        names += [f"runs/pilot-{experiment}-seed17/{name}" for name in
                  ("summary.json", "metrics.jsonl", "config.json", "environment.json",
                   "data_validation.json", "robustness-val256.json")]
    if prefix == "continue10":
        names = ["runs/paired_continuation_summary.json", "runs/continuation_status.json",
                 "runs/continuation_checkpoint_audit.json", "runs/continuation_diagnostics_summary.json",
                 "runs/diagnostics_status.json", "runs/presence_probe_status.json", "runs/presence_probe_summary.json",
                 "runs/presence_probe_orchestrator.log", "runs/prepare_labels.log",
                 "data/bdd100k/labels_provenance.json", "data/bdd100k/annotations/SOURCE_LICENSE.txt",
                 "data/manifests/bdd100k_train_labeled.jsonl", "data/manifests/bdd100k_val_labeled.jsonl"]
        for experiment in ("corruption", "clean"):
            names += [f"runs/continue10_train_{experiment}.log", f"runs/continue10_evaluate_{experiment}.log"]
            names += [f"runs/continue10-{experiment}-seed17/{name}" for name in
                      ("summary.json", "metrics.jsonl", "config.json", "environment.json", "data_validation.json",
                       "robustness-val256.json", "diagnostics-selected-val256.json", "diagnostics-final-val256.json")]
        names += [f"runs/presence-probe-seed17/{name}" for name in
                  ("protocol.json", "corruption_results.json", "clean_results.json", "random_results.json",
                   "corruption_head.pt", "clean_head.pt", "random_head.pt")]
    elif prefix == "detector":
        names = ["runs/detector_probe_summary.json", "runs/detector_probe_status.json",
                 "runs/detector_probe_orchestrator_v2.log", "runs/detector-probe-seed17-v2/protocol.json"]
        for experiment in ("random", "clean", "corruption"):
            names += [f"runs/detector-probe-seed17-v2/{experiment}/{name}" for name in
                      ("metrics.jsonl", "checkpoint_metadata.json", "results.json")]
            names += [f"runs/detector-probe-seed17-v2/{experiment}/eval-{family}-{severity}.json"
                      for family, severity in [("clean", 0)] +
                      [(family, severity) for family in ("fog", "rain", "motion_blur", "low_light")
                       for severity in range(1, 6)]]
    script = ("import base64,json,pathlib\n"
              f"root=pathlib.Path({REMOTE!r})\n"
              f"names={names!r}\n"
              "print(json.dumps({name:base64.b64encode((root/name).read_bytes()).decode('ascii') for name in names if (root/name).is_file()}))\n")
    response = subprocess.run(CODER + ["python3"], input=script.encode(), capture_output=True, check=True)
    small_files = json.loads(response.stdout)
    for name, encoded in small_files.items():
        destination(name).write_bytes(base64.b64decode(encoded, validate=True))
    if f"runs/{summary_name}" not in small_files:
        raise FileNotFoundError("Requested experiment summary is not ready")
    summary = json.loads(destination(f"runs/{summary_name}").read_text())
    copied = []
    if prefix == "detector":
        for experiment in ("random", "clean", "corruption"):
            info = summary["results"][experiment]
            filename = "detector.pt"
            expected_path = f"{REMOTE}/runs/detector-probe-seed17-v2/{experiment}/{filename}"
            target = destination(f"runs/detector-probe-seed17-v2/{experiment}/{filename}")
            if target.exists():
                if sha256(target) != info["detector_checkpoint_sha256"]:
                    raise FileExistsError(f"Existing local checkpoint differs: {target}")
            else:
                temporary = target.with_suffix(target.suffix + ".part")
                print(f"Copying {experiment}/{filename}: {info['checkpoint_bytes']} bytes", flush=True)
                with temporary.open("wb") as output:
                    subprocess.run(CODER + ["cat", expected_path], stdin=subprocess.DEVNULL,
                                   stdout=output, check=True)
                if temporary.stat().st_size != info["checkpoint_bytes"] or sha256(temporary) != info["detector_checkpoint_sha256"]:
                    raise ValueError(f"Checkpoint backup hash mismatch: {temporary}")
                temporary.replace(target)
            copied.append({"local_path": str(target), "path": expected_path,
                           "bytes": info["checkpoint_bytes"], "sha256": info["detector_checkpoint_sha256"],
                           "local_sha256_verified": True})
            print(f"Verified {experiment}/{filename}", flush=True)
    else:
        for experiment in ("corruption", "clean"):
            filenames = ["encoder.pt", "last.pt"] if args.full else ["encoder.pt"]
            if "final_encoder.pt" in summary["checkpoints"][experiment]:
                filenames.insert(1, "final_encoder.pt")
            for filename in filenames:
                info = summary["checkpoints"][experiment][filename]
                expected_path = f"{REMOTE}/runs/{prefix}-{experiment}-seed17/{filename}"
                if info["path"] != expected_path:
                    raise ValueError("Unexpected remote checkpoint path")
                target = destination(f"runs/{prefix}-{experiment}-seed17/{filename}")
                if target.exists():
                    if sha256(target) != info["sha256"]:
                        raise FileExistsError(f"Existing local checkpoint differs: {target}")
                else:
                    temporary = target.with_suffix(target.suffix + ".part")
                    print(f"Copying {experiment}/{filename}: {info['bytes']} bytes", flush=True)
                    with temporary.open("wb") as output:
                        subprocess.run(CODER + ["cat", expected_path], stdin=subprocess.DEVNULL,
                                       stdout=output, check=True)
                    if temporary.stat().st_size != info["bytes"] or sha256(temporary) != info["sha256"]:
                        raise ValueError(f"Checkpoint backup hash mismatch: {temporary}")
                    temporary.replace(target)
                copied.append({"local_path": str(target), **info, "local_sha256_verified": True})
                print(f"Verified {experiment}/{filename}", flush=True)
    receipt = {"workspace": "owner/jwj-01", "remote_root": REMOTE, "experiment": prefix, "created_unix": time.time(),
               "evidence_files": len(small_files), "checkpoints": copied,
               "images_location": "VM only; image manifests backed up locally"}
    receipt_name = {"pilot": "local_backup.json", "continue10": "continuation_local_backup.json",
                    "detector": "detector_local_backup.json"}[prefix]
    destination(f"runs/{receipt_name}").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": "backup_complete", "evidence_files": len(small_files),
                      "verified_checkpoints": len(copied)}, indent=2))


if __name__ == "__main__":
    main()
