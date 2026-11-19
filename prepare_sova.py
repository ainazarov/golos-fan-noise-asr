"""Add deterministic Sova holdouts and combined manifests without altering Golos."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import heapq
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import uuid

from prepare_data import (
    MAX_TRAIN_SECONDS, MIN_TRAIN_SECONDS, Recording, atomic_json, atomic_lines,
    rank_key, statistics,
)


SOURCE_PREFIX = PurePosixPath("/root/dataset/sova")


def canonical_identity(value: str) -> str:
    value = value.removeprefix("sova:").strip()
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return value.casefold()


def mapped_audio_path(value: str, audio_root: Path) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("audio_filepath must be an absolute POSIX source path")
    source = PurePosixPath(value)
    if ".." in source.parts:
        raise ValueError("audio_filepath must not contain parent traversal")
    try:
        relative = source.relative_to(SOURCE_PREFIX)
    except ValueError as error:
        raise ValueError(f"audio_filepath must be inside {SOURCE_PREFIX}") from error
    if not relative.parts:
        raise ValueError("audio_filepath must identify a recording")
    try:
        identifier = str(uuid.UUID(source.stem))
    except ValueError as error:
        raise ValueError("Sova recording filenames must contain UUID identities") from error
    return identifier, audio_root.joinpath(*relative.parts)


def read_sova(source: Path, audio_root: Path):
    digest = hashlib.sha256()
    source_ids = set()
    records = []
    excluded = Counter()
    normalization = Counter()
    source_count = 0
    source_seconds = 0.0
    with source.open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            digest.update(raw)
            try:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("record must be a JSON object")
                identifier, path = mapped_audio_path(row["audio_filepath"], audio_root)
                if identifier in source_ids:
                    raise ValueError(f"duplicate Sova recording UUID: {identifier}")
                source_ids.add(identifier)
                duration = row["duration"]
                if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                    raise ValueError("duration must be numeric")
                duration = float(duration)
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError("duration must be finite and positive")
                if not isinstance(row["text"], str):
                    raise ValueError("text must be a string")
                text = " ".join(row["text"].split())
                replaced_characters = text.count("ё")
                normalization["source_changed_recordings"] += bool(replaced_characters)
                normalization["source_replaced_characters"] += replaced_characters
                text = text.replace("ё", "е")
                source_count += 1
                source_seconds += duration
                duration_ok = MIN_TRAIN_SECONDS <= duration <= MAX_TRAIN_SECONDS
                text_ok = bool(text)
                if not duration_ok:
                    excluded["duration_out_of_range"] += 1
                if not text_ok:
                    excluded["empty_transcript"] += 1
                if not duration_ok and not text_ok:
                    excluded["both_duration_and_empty"] += 1
                if duration_ok and text_ok:
                    normalization["eligible_changed_recordings"] += bool(replaced_characters)
                    normalization["eligible_replaced_characters"] += replaced_characters
                    records.append(Recording("sova:" + identifier, str(path), duration, text, "sova"))
            except (ValueError, TypeError, KeyError, UnicodeError) as error:
                raise ValueError(f"{source}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"No eligible Sova recordings in {source}")
    excluded["unique_excluded"] = source_count - len(records)
    audit = {
        "path": str(source), "sha256": digest.hexdigest(),
        "recordings": source_count, "hours": round(source_seconds / 3600, 6),
        "eligible": statistics(records), "excluded": dict(excluded),
        "yo_to_e_normalization": dict(normalization),
        "duplicate_uuid_policy": "reject all duplicate UUIDs, including duplicate metadata rows",
    }
    return records, source_ids, audit


def reserve_holdout(records, seed, namespace, hours):
    target = hours * 3600
    maximum_needed = math.ceil(target / MIN_TRAIN_SECONDS) + 1
    candidates = heapq.nsmallest(maximum_needed, records, key=lambda record: rank_key(seed, namespace, record))
    selected = []
    duration = 0.0
    for record in candidates:
        if duration >= target:
            break
        selected.append(record)
        duration += record.duration
    if duration < target:
        raise ValueError(f"Not enough eligible Sova audio for {namespace}: {duration:.3f}s < {target:.3f}s")
    selected_ids = {record.id for record in selected}
    remainder = [record for record in records if record.id not in selected_ids]
    return remainder, sorted(selected, key=lambda record: record.id), {
        "target_seconds": target, "actual_seconds": round(duration, 6), "recordings": len(selected),
    }


def scan_golos_manifest(path, sova_ids):
    digest = hashlib.sha256()
    overlap = set()

    def records():
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                digest.update(raw)
                try:
                    row = json.loads(raw)
                    identifier = row["id"]
                    audio = row["audio_filepath"]
                    duration = row["duration"]
                    text = row["text"]
                    if not isinstance(identifier, str) or not identifier.strip():
                        raise ValueError("id must be a nonempty string")
                    if not isinstance(audio, str) or not isinstance(text, str):
                        raise ValueError("audio_filepath and text must be strings")
                    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                        raise ValueError("duration must be numeric")
                    duration = float(duration)
                    if not math.isfinite(duration) or duration <= 0:
                        raise ValueError("duration must be finite and positive")
                    identities = {canonical_identity(identifier), canonical_identity(PurePosixPath(audio).stem)}
                    overlap.update(identities & sova_ids)
                    condition = "golos_farfield" if "/farfield/" in audio else "golos_crowd" if "/crowd/" in audio else "golos"
                    yield Recording(identifier, audio, duration, text, condition)
                except (ValueError, TypeError, KeyError, UnicodeError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error

    result = statistics(records())
    if result["recordings"] == 0:
        raise ValueError(f"Empty Golos manifest: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), **result}, overlap


def combine_statistics(*sources):
    conditions = {}
    duration_bins = Counter()
    for source in sources:
        for condition, values in source["conditions"].items():
            result = conditions.setdefault(condition, {"recordings": 0, "hours": 0.0})
            result["recordings"] += values["recordings"]
            result["hours"] += values["hours"]
        duration_bins.update(source["duration_bins"])
    seconds = sum(source["seconds"] for source in sources)
    return {
        "recordings": sum(source["recordings"] for source in sources),
        "seconds": round(seconds, 6), "hours": round(seconds / 3600, 6),
        "empty_transcripts": sum(source["empty_transcripts"] for source in sources),
        "conditions": conditions, "duration_bins": dict(sorted(duration_bins.items())),
    }


def manifest_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield line.rstrip("\r\n")


def prepare(args):
    root = args.output_root.expanduser().resolve()
    source = args.source_manifest.expanduser().resolve()
    audio_root = args.audio_root.expanduser().resolve()
    names = ("train_sova", "val_sova_original", "test_sova_original", "train_mixed", "val_mixed_original")
    outputs = {name: root / "data" / (name + ".jsonl") for name in names}
    report_path = root / "reports/sova_audit.json"
    if not args.overwrite:
        existing = [str(path) for path in (*outputs.values(), report_path) if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing Sova outputs: " + ", ".join(existing))
    golos_paths = {name: root / "data" / (name + ".jsonl")
                   for name in ("train", "val_original", "test_crowd_original", "test_farfield_original")}
    for path in golos_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    eligible, source_ids, source_audit = read_sova(source, audio_root)
    report = {
        "status": "preparing", "seed": args.seed, "sova_source": source_audit,
        "split_policy": {
            "identity": "normalized recording UUID; speaker IDs unavailable; not certified speaker-disjoint",
            "development": "smallest SHA256(seed, sova_development, prefixed UUID) until duration quota",
            "test": "smallest SHA256(seed, sova_test, prefixed UUID) among remaining records until quota",
            "development_target_hours": args.dev_hours, "test_target_hours": args.test_hours,
            "duration_range_seconds": [MIN_TRAIN_SECONDS, MAX_TRAIN_SECONDS],
            "nonempty_text_required": True,
            "text_normalization": "collapse Unicode whitespace and replace ё with е; preserve other case and punctuation",
            "source_path_prefix": str(SOURCE_PREFIX), "mapped_audio_root": str(audio_root),
            "mixture": "one copy of each Golos and Sova training recording; no oversampling",
            "tokenizer": "existing tokenizer112 and its Golos-only training corpus remain unchanged",
        },
        "golos_sources": {}, "golos_overlap_recording_counts": {},
    }
    overlaps = {}
    for name, path in golos_paths.items():
        audit, overlap = scan_golos_manifest(path, source_ids)
        report["golos_sources"][name] = audit
        report["golos_overlap_recording_counts"][name] = len(overlap)
        if overlap:
            overlaps[name] = sorted(overlap)[:10]
    if overlaps:
        report.update(status="failed_recording_overlap", overlap_examples=overlaps)
        atomic_json(report_path, report)
        raise ValueError(f"Sova recording IDs overlap Golos; see {report_path}")
    remaining, development, dev_quota = reserve_holdout(eligible, args.seed, "sova_development", args.dev_hours)
    train, test, test_quota = reserve_holdout(remaining, args.seed, "sova_test", args.test_hours)
    if not train:
        raise ValueError("Sova development/test reservations leave no training data")
    train.sort(key=lambda record: record.id)
    splits = {"train_sova": train, "val_sova_original": development, "test_sova_original": test}
    report["holdout_quotas"] = {"development": dev_quota, "test": test_quota}
    report["splits"] = {name: statistics(records) for name, records in splits.items()}
    checks = {}
    missing = []
    for name, records in splits.items():
        selected = records if args.audio_check_limit == 0 else heapq.nsmallest(
            args.audio_check_limit, records, key=lambda record: rank_key(args.seed, "sova_audio_check", record))
        failures = []
        for record in selected:
            try:
                exists = Path(record.audio_filepath).is_file()
            except PermissionError:
                exists = False
            if not exists:
                failures.append(record.audio_filepath)
        checks[name] = {"checked": len(selected), "missing": len(failures), "all_recordings_checked": len(selected) == len(records)}
        missing.extend(failures)
    report["audio_path_checks"] = checks
    report["audio_audit_limit"] = "File existence only on the stated records; no full audio decoding or content-hash deduplication."
    if missing:
        report.update(status="failed_missing_audio", missing_audio_examples=missing[:10])
        atomic_json(report_path, report)
        raise FileNotFoundError(f"Sova audio paths are missing or inaccessible; see {report_path}")
    for name, records in splits.items():
        atomic_lines(outputs[name], (json.dumps(record.manifest_row(), ensure_ascii=False) for record in records))
    for mixed_name, golos_name, sova_name in (
        ("train_mixed", "train", "train_sova"),
        ("val_mixed_original", "val_original", "val_sova_original"),
    ):
        atomic_lines(outputs[mixed_name], itertools.chain(manifest_lines(golos_paths[golos_name]), manifest_lines(outputs[sova_name])))
        report["splits"][mixed_name] = combine_statistics(report["golos_sources"][golos_name], report["splits"][sova_name])
    report["status"] = "complete"
    report["outputs"] = {name: str(path) for name, path in outputs.items()}
    atomic_json(report_path, report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=Path("/share/sova/sova3_man.json"))
    parser.add_argument("--audio-root", type=Path, default=Path("/share/sova"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=112)
    parser.add_argument("--dev-hours", type=float, default=2.0)
    parser.add_argument("--test-hours", type=float, default=2.0)
    parser.add_argument("--audio-check-limit", type=int, default=0, help="0 checks every Sova audio path; otherwise a deterministic per-split sample")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or value <= 0 for value in (args.dev_hours, args.test_hours)):
        parser.error("holdout hours must be finite and positive")
    if args.audio_check_limit < 0:
        parser.error("--audio-check-limit must be nonnegative")
    return args


def main():
    report = prepare(parse_args())
    print(json.dumps({"status": report["status"], "splits": report["splits"], "audio_path_checks": report["audio_path_checks"]}, indent=2))


if __name__ == "__main__":
    main()
