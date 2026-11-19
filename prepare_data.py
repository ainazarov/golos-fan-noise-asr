"""Prepare deterministic full-Golos ASR splits without copying source audio.

The official test sets are never sampled into training or development.  Golos
does not expose speaker IDs, so these are recording-disjoint, not certified
speaker-disjoint splits.  Only the training pool receives the 0.3--20 second
and non-empty-transcript filters.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Iterable


MIN_TRAIN_SECONDS = 0.3
MAX_TRAIN_SECONDS = 20.0
DEV_FRACTIONS = {"crowd": 0.9, "farfield": 0.1}


@dataclass(frozen=True, slots=True)
class Recording:
    id: str
    audio_filepath: str
    duration: float
    text: str
    condition: str

    def manifest_row(self) -> dict:
        return {
            "audio_filepath": self.audio_filepath,
            "duration": self.duration,
            "text": self.text,
            "id": self.id,
        }


def rank_key(seed: int, namespace: str, recording: Recording) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{namespace}\0{recording.id}".encode("utf-8")
    ).digest()


def duration_bin(seconds: float) -> str:
    if seconds <= 2:
        return "<=2s"
    if seconds <= 5:
        return "2-5s"
    if seconds <= 10:
        return "5-10s"
    if seconds <= 20:
        return "10-20s"
    return ">20s"


def statistics(records: Iterable[Recording]) -> dict:
    n = 0
    seconds = 0.0
    empty = 0
    conditions = defaultdict(lambda: {"recordings": 0, "seconds": 0.0})
    durations = Counter()
    for record in records:
        n += 1
        seconds += record.duration
        empty += not record.text
        conditions[record.condition]["recordings"] += 1
        conditions[record.condition]["seconds"] += record.duration
        durations[duration_bin(record.duration)] += 1
    return {
        "recordings": n,
        "seconds": round(seconds, 6),
        "hours": round(seconds / 3600, 6),
        "empty_transcripts": empty,
        "conditions": {
            key: {
                "recordings": value["recordings"],
                "hours": round(value["seconds"] / 3600, 6),
            }
            for key, value in sorted(conditions.items())
        },
        "duration_bins": dict(sorted(durations.items())),
    }


def read_manifest(path: Path, audio_root: Path, test_condition: str | None = None):
    """Read/validate metadata and fingerprint exactly the bytes being read."""
    digest = hashlib.sha256()
    records = []
    ids = set()
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("record must be a JSON object")
                identifier = row["id"]
                text = row["text"]
                audio = row["audio_filepath"]
                seconds = row["duration"]
                if not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("id must be a non-empty string")
                if identifier in ids:
                    raise ValueError(f"duplicate recording ID: {identifier}")
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
                if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
                    raise ValueError("duration must be numeric")
                seconds = float(seconds)
                if not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("duration must be finite and positive")
                if not isinstance(audio, str) or not audio or "\\" in audio:
                    raise ValueError("audio_filepath must be a relative POSIX path")
                relative = PurePosixPath(audio)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("audio_filepath must remain inside its source audio root")
                if not relative.parts:
                    raise ValueError("empty audio_filepath")
                condition = test_condition or relative.parts[0]
                if condition not in DEV_FRACTIONS:
                    raise ValueError(f"unexpected recording condition: {condition}")
                normalized = " ".join(text.split())
                records.append(
                    Recording(
                        identifier,
                        str(audio_root.joinpath(*relative.parts)),
                        seconds,
                        normalized,
                        condition,
                    )
                )
                ids.add(identifier)
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"No recordings in {path}")
    return records, {"path": str(path), "sha256": digest.hexdigest(), **statistics(records)}


def split_development(records: list[Recording], seed: int, dev_hours: float):
    selected = []
    quotas = {}
    for condition, fraction in DEV_FRACTIONS.items():
        target = dev_hours * 3600 * fraction
        # Every eligible recording is >=0.3 seconds. This bounded heap therefore
        # contains enough of the smallest hashes to reach the duration target.
        maximum_needed = math.ceil(target / MIN_TRAIN_SECONDS) + 1
        candidates = heapq.nsmallest(
            maximum_needed,
            (record for record in records if record.condition == condition),
            key=lambda record: rank_key(seed, "development", record),
        )
        seconds = 0.0
        count = 0
        for record in candidates:
            if seconds >= target:
                break
            selected.append(record)
            seconds += record.duration
            count += 1
        if seconds < target:
            raise ValueError(
                f"Not enough eligible {condition} audio for development: "
                f"{seconds:.3f} seconds available, {target:.3f} required"
            )
        quotas[condition] = {
            "target_seconds": target,
            "actual_seconds": round(seconds, 6),
            "recordings": count,
        }
    development_ids = {record.id for record in selected}
    train = [record for record in records if record.id not in development_ids]
    if not train:
        raise ValueError("Development reservation leaves no training recordings")
    return train, sorted(selected, key=lambda record: record.id), quotas


def stratified_sample(records: list[Recording], size: int, seed: int, namespace: str):
    """Proportional condition/duration sample, deterministic within each stratum."""
    size = min(size, len(records))
    if size <= 0:
        return []
    groups = defaultdict(list)
    for record in records:
        groups[(record.condition, duration_bin(record.duration))].append(record)
    allocations = {}
    remainders = []
    for key, members in sorted(groups.items()):
        exact = size * len(members) / len(records)
        allocations[key] = math.floor(exact)
        remainders.append((exact - allocations[key], key))
    missing = size - sum(allocations.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[:missing]:
        allocations[key] += 1
    sample = []
    for key, members in sorted(groups.items()):
        sample.extend(
            heapq.nsmallest(
                allocations[key],
                members,
                key=lambda record: rank_key(seed, namespace, record),
            )
        )
    return sorted(sample, key=lambda record: rank_key(seed, namespace, record))


def atomic_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for line in lines:
                stream.write(line)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: dict) -> None:
    atomic_lines(path, [json.dumps(value, ensure_ascii=False, indent=2)])


def prepare(args: argparse.Namespace) -> dict:
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    outputs = {
        name: output_root / "data" / f"{name}.jsonl"
        for name in [
            "train", "val_original", "test_crowd_original", "test_farfield_original",
            "benchmark_train", "benchmark_val",
        ]
    }
    corpus_path = output_root / "data" / "train_text.txt"
    report_path = output_root / "reports" / "data_audit.json"
    if not args.overwrite:
        existing = [str(path) for path in [*outputs.values(), corpus_path, report_path] if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing outputs; use --overwrite: " + ", ".join(existing))

    pool, train_source = read_manifest(source_root / "train" / "manifest.jsonl", source_root / "train")
    tests = {}
    sources = [train_source]
    for condition in DEV_FRACTIONS:
        records, source = read_manifest(
            source_root / "test" / condition / "manifest.jsonl",
            source_root / "test" / condition,
            test_condition=condition,
        )
        tests[condition] = records
        sources.append(source)

    pool_ids = {record.id for record in pool}
    test_ids = {key: {record.id for record in value} for key, value in tests.items()}
    intersections = {
        "train_pool_vs_test_crowd": pool_ids & test_ids["crowd"],
        "train_pool_vs_test_farfield": pool_ids & test_ids["farfield"],
        "test_crowd_vs_test_farfield": test_ids["crowd"] & test_ids["farfield"],
    }
    report = {
        "status": "preparing",
        "seed": args.seed,
        "sources": sources,
        "split_policy": {
            "identity": "recording id; speaker IDs are unavailable",
            "development": "smallest SHA256(seed, namespace, recording ID), per-condition duration quota",
            "development_target_hours": args.dev_hours,
            "development_condition_fractions": DEV_FRACTIONS,
            "training_min_seconds": MIN_TRAIN_SECONDS,
            "training_max_seconds": MAX_TRAIN_SECONDS,
            "training_requires_nonempty_text": True,
            "test_duration_filter": None,
            "test_empty_transcripts": "retained",
            "text_normalization": "collapse Unicode whitespace only; preserve case and punctuation",
            "tokenizer_corpus": "training split transcripts only",
        },
        "overlap_recording_counts": {key: len(value) for key, value in intersections.items()},
    }
    if any(intersections.values()):
        report["status"] = "failed_recording_overlap"
        report["overlap_examples"] = {key: sorted(value)[:10] for key, value in intersections.items() if value}
        atomic_json(report_path, report)
        raise ValueError(f"Official split recording IDs overlap; see {report_path}")

    excluded = Counter()
    eligible = []
    for record in pool:
        duration_ok = MIN_TRAIN_SECONDS <= record.duration <= MAX_TRAIN_SECONDS
        text_ok = bool(record.text)
        if not duration_ok:
            excluded["duration_out_of_range"] += 1
        if not text_ok:
            excluded["empty_transcript"] += 1
        if not duration_ok and not text_ok:
            excluded["both_duration_and_empty"] += 1
        if duration_ok and text_ok:
            eligible.append(record)
    excluded["unique_excluded"] = len(pool) - len(eligible)
    train, validation, quotas = split_development(eligible, args.seed, args.dev_hours)
    train.sort(key=lambda record: record.id)
    benchmark_train = stratified_sample(train, args.benchmark_size, args.seed, "benchmark_train")
    benchmark_val = stratified_sample(validation, 128, args.seed, "benchmark_val")
    splits = {
        "train": train,
        "val_original": validation,
        "test_crowd_original": sorted(tests["crowd"], key=lambda record: record.id),
        "test_farfield_original": sorted(tests["farfield"], key=lambda record: record.id),
        "benchmark_train": benchmark_train,
        "benchmark_val": benchmark_val,
    }
    report["excluded_training_pool"] = dict(excluded)
    report["development_quotas"] = quotas
    report["splits"] = {name: statistics(records) for name, records in splits.items()}
    report["overlap_recording_counts"]["train_vs_validation"] = len(
        {record.id for record in train} & {record.id for record in validation}
    )

    # Inspect all benchmark audio paths and deterministic samples of all full
    # splits. Reading every WAV header across a network share is a separate audit.
    checks = {}
    missing_paths = []
    for name, records in splits.items():
        checked = records if name.startswith("benchmark_") else stratified_sample(records, 128, args.seed, "audio_path_check")
        missing = [record.audio_filepath for record in checked if not Path(record.audio_filepath).is_file()]
        checks[name] = {"checked": len(checked), "missing": len(missing)}
        missing_paths.extend(missing)
    report["audio_path_checks"] = checks
    report["audio_audit_limit"] = "Existence checks only on the stated samples; no full-corpus audio decoding or content-hash audit."
    if missing_paths:
        report["status"] = "failed_missing_audio"
        report["missing_audio_examples"] = missing_paths[:20]
        atomic_json(report_path, report)
        raise FileNotFoundError(f"Sampled audio files are missing; see {report_path}")

    for name, records in splits.items():
        atomic_lines(outputs[name], (json.dumps(record.manifest_row(), ensure_ascii=False) for record in records))
    atomic_lines(corpus_path, (record.text for record in train))
    report["status"] = "complete"
    report["outputs"] = {name: str(path) for name, path in outputs.items()}
    report["outputs"]["training_text"] = str(corpus_path)
    atomic_json(report_path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/share/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--dev-hours", type=float, default=2.0)
    parser.add_argument("--benchmark-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.dev_hours) or args.dev_hours <= 0:
        parser.error("--dev-hours must be finite and positive")
    if args.benchmark_size <= 0:
        parser.error("--benchmark-size must be positive")
    return args


def main():
    report = prepare(parse_args())
    print(json.dumps({"status": report["status"], "splits": report["splits"], "audio_path_checks": report["audio_path_checks"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
