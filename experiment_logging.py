"""Persistent validation results associated with saved epoch checkpoints."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path


def scalar_metrics(metrics, prefix=None):
    result = {}
    for name, value in metrics.items():
        if prefix is not None and not name.startswith(prefix):
            continue
        try:
            scalar = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
        except (TypeError, ValueError, RuntimeError):
            continue
        result[name] = scalar if math.isfinite(scalar) else None
    return result


class EpochHistory:
    def __init__(self, run_dir, name="validation_history"):
        self.run_dir = Path(run_dir)
        self.jsonl_path = self.run_dir / (name + ".jsonl")
        self.csv_path = self.run_dir / (name + ".csv")

    def _read(self):
        if not self.jsonl_path.exists():
            return []
        return [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _atomic_write(path, content):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write(self, rows):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        fixed = ["epoch", "epoch_index", "global_step", "checkpoint", "checkpoint_bytes", "saved_utc"]
        fields = fixed + sorted(set().union(*(row.keys() for row in rows)) - set(fixed))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        self._atomic_write(self.jsonl_path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
        self._atomic_write(self.csv_path, output.getvalue())

    def resume_from_step(self, global_step):
        rows = self._read()
        retained = [row for row in rows if row["global_step"] <= global_step]
        if len(retained) != len(rows):
            self._write(retained)

    def record(self, *, epoch_index, global_step, checkpoint, metrics, validation_completed=True):
        checkpoint = Path(checkpoint).resolve()
        relative = checkpoint.relative_to(self.run_dir.resolve()).as_posix()
        row = {"epoch": int(epoch_index) + 1, "epoch_index": int(epoch_index),
               "global_step": int(global_step), "checkpoint": relative,
               "checkpoint_bytes": checkpoint.stat().st_size,
               "validation_completed": bool(validation_completed),
               "saved_utc": datetime.now(timezone.utc).isoformat()}
        if validation_completed:
            row.update(scalar_metrics(metrics, prefix="val"))
        rows = [existing for existing in self._read() if existing["epoch"] != row["epoch"]]
        rows.append(row)
        rows.sort(key=lambda existing: existing["epoch"])
        self._write(rows)


class ProgressHistory:
    fields = ["utc", "segment", "epoch", "global_step", "batch_in_epoch",
              "train_batches_per_epoch", "segment_elapsed_seconds", "interval_steps",
              "interval_seconds", "seconds_per_step", "steps_per_second", "learning_rate",
              "train_loss", "rnnt_loss", "ctc_loss", "gpu_allocated_gib", "gpu_reserved_gib",
              "train_metrics_json"]

    def __init__(self, run_dir, segment):
        self.run_dir = Path(run_dir)
        self.segment = segment
        self.directory = self.run_dir / "progress" / segment
        self.directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.directory / "progress.jsonl"
        self.csv_path = self.directory / "progress.csv"
        self.status_path = self.run_dir / "progress_status.json"

    def status(self, status, **values):
        record = {"status": status, "utc": datetime.now(timezone.utc).isoformat(),
                  "segment": self.segment, **values}
        EpochHistory._atomic_write(self.status_path, json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n")

    def append(self, record):
        row = {"utc": datetime.now(timezone.utc).isoformat(), "segment": self.segment, **record}
        metrics = row.pop("train_metrics", {})
        row["train_metrics_json"] = json.dumps(metrics, ensure_ascii=False, allow_nan=False, sort_keys=True)
        new_csv = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if new_csv:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({**row, "train_metrics": metrics}, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
        self.status("running", **{key: value for key, value in row.items() if key not in {"utc", "segment"}}, train_metrics=metrics)
