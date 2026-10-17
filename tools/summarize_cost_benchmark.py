"""Integrate sampled GPU power and join it with training and detector results."""
from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "cost-benchmark-seed17"


def read_samples(path):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            stamp = dt.datetime.fromisoformat(row[0].strip().replace("/", "-"))
            power = float(row[1].strip().split()[0])
            utilization = float(row[2].strip().split()[0])
            rows.append((stamp, power, utilization))
    if len(rows) < 3:
        raise ValueError(f"Too few GPU power samples: {path}")
    return rows


def integrate(path, idle_power):
    rows = read_samples(path)
    energy_joules, covered, gaps, nonpositive, intervals = 0.0, 0.0, 0, 0, []
    for left, right in zip(rows, rows[1:]):
        seconds = (right[0] - left[0]).total_seconds()
        if seconds <= 0:
            nonpositive += 1
            continue
        if seconds > 2.0:
            gaps += 1
            continue
        energy_joules += (left[1] + right[1]) * 0.5 * seconds
        covered += seconds
        intervals.append(seconds)
    mean_power = energy_joules / covered
    return {"samples": len(rows), "covered_seconds": covered, "skipped_gaps_over_2s": gaps,
            "skipped_nonpositive_timestamp_intervals": nonpositive,
            "median_sample_interval_seconds": statistics.median(intervals),
            "mean_board_power_watts": mean_power, "gpu_board_energy_joules": energy_joules,
            "gpu_board_energy_wh": energy_joules / 3600,
            "idle_adjusted_energy_joules": (mean_power - idle_power) * covered,
            "idle_adjusted_energy_wh": (mean_power - idle_power) * covered / 3600,
            "mean_gpu_utilization_percent": statistics.mean(row[2] for row in rows),
            "peak_sampled_gpu_utilization_percent": max(row[2] for row in rows)}


def main():
    idle = read_samples(OUT / "idle-gpu-samples.csv")
    idle_power = statistics.mean(row[1] for row in idle)
    names = {
        "clean_jepa": ("clean-jepa", RUNS / "cost-clean-jepa-seed17"),
        "corruption_jepa": ("corruption-jepa", RUNS / "cost-corruption-jepa-seed17"),
        "corruption_mae": ("mae-vits-torch2141", RUNS / "cost-mae-vits-seed17-torch2141"),
    }
    methods = {}
    for label, (stage, run) in names.items():
        process = json.loads((OUT / f"{stage}-process.json").read_text(encoding="utf-8"))
        train = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
        methods[label] = {"run": str(run), "process": process, "training": train,
                          "environment": environment,
                          "gpu_power": integrate(OUT / f"{stage}-gpu-samples.csv", idle_power)}
    result = {"status": "complete", "gpu": "NVIDIA GeForce RTX 5090 32GB",
              "idle_sample_count": len(idle), "idle_mean_board_power_watts": idle_power,
              "methods": methods,
              "measurement_note": "Power draw was sampled every 500 ms and trapezoid-integrated; idle-adjusted energy subtracts the preceding idle mean. One seed, one run per method."}
    detector_path = RUNS / "detector_cost_probe_summary.json"
    if detector_path.is_file():
        detector = json.loads(detector_path.read_text(encoding="utf-8"))
        result["detector_scope"] = detector["protocol"]
        result["detector_results"] = {
            name: {key: values[key] for key in ("encoder_cumulative_steps", "clean_map50", "clean_map50_95",
                                                 "corrupt_mean_map50", "corrupt_mean_map50_95",
                                                 "training_final_loss") if key in values}
            for name, values in detector["results"].items()}
    output = RUNS / "cost_benchmark_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
