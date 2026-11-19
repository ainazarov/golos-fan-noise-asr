"""Forecast the sequential Golos, mixed-data and noisy mixed-data experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo


def build_forecast(golos_audit, sova_audit, benchmark, configurations, start, allowance=0.15):
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if not isinstance(start, datetime) or start.tzinfo is None or start.utcoffset() is None:
        raise ValueError("Start time must be an ISO8601 datetime with its UTC offset")
    if not isinstance(allowance, (int, float)) or isinstance(allowance, bool) or not math.isfinite(allowance) or allowance < 0:
        raise ValueError("Allowance must be finite and nonnegative")
    if benchmark.get("status") != "complete" or benchmark.get("mel_bins") != 128:
        raise ValueError("A completed 128-mel benchmark is required")
    zone = ZoneInfo("Europe/Moscow")
    phases = (
        ("baseline", "golos", golos_audit["splits"]["train"]),
        ("mixed", "mixed", sova_audit["splits"]["train_mixed"]),
        ("noise", "mixed_noise", sova_audit["splits"]["train_mixed"]),
    )
    stages = {}
    cumulative_training_seconds = 0.0
    cumulative_planning_seconds = 0.0
    total_epochs = 0
    for phase, benchmark_name, split in phases:
        measurement = benchmark["results"][benchmark_name]
        config = configurations[phase]
        epochs = config["trainer"]["max_epochs"]
        batch_size = config["model"]["train_ds"]["batch_size"]
        recordings = split["recordings"]
        for name, value in (("epochs", epochs), ("batch_size", batch_size), ("recordings", recordings)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{phase}: {name} must be a positive integer")
        if measurement.get("status") != "passed":
            raise ValueError(f"{phase}: benchmark did not pass")
        if batch_size != measurement["batch_size"] or str(config["trainer"]["precision"]) != str(measurement["precision"]):
            raise ValueError(f"{phase}: configuration batch size/precision differs from benchmark")
        if config["model"]["preprocessor"]["features"] != 128 or config["model"]["encoder"]["feat_in"] != 128:
            raise ValueError(f"{phase}: configuration must use 128 mel bins")
        if config["trainer"].get("accumulate_grad_batches", 1) != 1 or config["model"]["train_ds"].get("drop_last", False):
            raise ValueError(f"{phase}: forecast requires no gradient accumulation or dropped final batch")
        seconds_per_step = float(measurement["mean_seconds_per_step"])
        if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
            raise ValueError(f"{phase}: benchmark seconds per step must be finite and positive")
        steps = math.ceil(recordings / batch_size)
        epoch_seconds = steps * seconds_per_step
        training_seconds = epochs * epoch_seconds
        planning_seconds = training_seconds * (1 + allowance)
        cumulative_training_seconds += training_seconds
        cumulative_planning_seconds += planning_seconds
        total_epochs += epochs
        stages[phase] = {
            "epochs": epochs, "train_recordings": recordings, "train_hours": split["hours"],
            "benchmark": benchmark_name, "batch_size": batch_size, "precision": measurement["precision"],
            "steps_per_epoch": steps, "seconds_per_step": seconds_per_step,
            "epoch_training_hours": epoch_seconds / 3600,
            "training_only_hours": training_seconds / 3600, "planning_hours": planning_seconds / 3600,
            "training_only_finish_moscow": (start + timedelta(seconds=cumulative_training_seconds)).astimezone(zone).isoformat(),
            "planning_finish_moscow": (start + timedelta(seconds=cumulative_planning_seconds)).astimezone(zone).isoformat(),
        }
    return {
        "hypothetical_start_moscow": start.astimezone(zone).isoformat(),
        "actual_training_started": False, "mel_bins": 128, "vocab_size": 112,
        "phase_order": [phase for phase, _, _ in phases], "total_epochs": total_epochs,
        "operational_allowance_fraction": allowance, "stages": stages,
        "total_training_only_hours": cumulative_training_seconds / 3600,
        "total_planning_hours": cumulative_planning_seconds / 3600,
        "training_only_finish_moscow": (start + timedelta(seconds=cumulative_training_seconds)).astimezone(zone).isoformat(),
        "planning_finish_moscow": (start + timedelta(seconds=cumulative_planning_seconds)).astimezone(zone).isoformat(),
        "assumptions": [
            "Exclusive uninterrupted use of the benchmark V100 with the same model, precision, batch size and loader settings",
            "Epoch sizes use the final prepared manifests after development/test reservations",
            f"{allowance:.0%} planning allowance covers periodic validation, checkpoint writing and throughput variation",
            "Validation remains every five completed epochs and checkpoints are retained every epoch",
            "The allowance is not a statistical confidence interval or a guaranteed deadline",
            "Short benchmarks measure execution speed, not model convergence or final recognition quality",
            "Comprehensive final test/checkpoint evaluation is excluded",
            "Preparation is complete before the assumed start; future interruptions and GPU contention are not predicted",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start", help="Hypothetical ISO8601 start including UTC offset; defaults to now")
    parser.add_argument("--allowance", type=float, default=0.15)
    args = parser.parse_args()
    if not math.isfinite(args.allowance) or args.allowance < 0:
        parser.error("--allowance must be finite and nonnegative")
    from omegaconf import OmegaConf

    root = args.output_root.resolve()
    golos = json.loads((root / "reports/data_audit.json").read_text(encoding="utf-8"))
    sova = json.loads((root / "reports/sova_audit.json").read_text(encoding="utf-8"))
    benchmark = json.loads((root / "reports/benchmark128.json").read_text(encoding="utf-8"))
    configurations = {
        phase: OmegaConf.to_container(OmegaConf.load(root / "configs" / (phase + ".yaml")), resolve=True)
        for phase in ("baseline", "mixed", "noise")
    }
    report = build_forecast(golos, sova, benchmark, configurations,
                            args.start if args.start else datetime.now(timezone.utc), args.allowance)
    (root / "reports/runtime_forecast.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
