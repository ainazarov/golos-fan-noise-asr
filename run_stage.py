"""Train one authorized phase; run full test evaluation only when explicitly requested."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from checkpoint_utils import resume_run_directory


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_checkpoints(run_dir):
    run_dir = Path(run_dir).resolve()
    records = [json.loads(line) for line in (run_dir / "validation_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No completed epoch evaluations are available")
    checkpoint_history = run_dir / "checkpoint_history.jsonl"
    checkpoint_records = ([json.loads(line) for line in checkpoint_history.read_text(encoding="utf-8").splitlines() if line.strip()]
                          if checkpoint_history.is_file() else records)
    if not checkpoint_records:
        raise ValueError("No completed epoch checkpoints are available")
    for record in records + checkpoint_records:
        checkpoint = (run_dir / record["checkpoint"]).resolve()
        if checkpoint.parent != run_dir / "checkpoints" or not checkpoint.is_file():
            raise ValueError(f"Invalid/missing recorded checkpoint: {checkpoint}")
        record["checkpoint"] = str(checkpoint)
    valid = [record for record in records if isinstance(record.get("val_wer"), (int, float)) and math.isfinite(record["val_wer"])]
    if not valid:
        raise ValueError("No finite validation WER is available for checkpoint selection")
    best = min(valid, key=lambda record: (record["val_wer"], record["epoch"]))
    final = max(checkpoint_records, key=lambda record: record["epoch"])
    if best["checkpoint"] == final["checkpoint"]:
        return [{**best, "role": "best", "selected_as": ["best", "final"]}]
    return [{**best, "role": "best", "selected_as": ["best"]},
            {**final, "role": "final", "selected_as": ["final"]}]


def prepare_test_manifests(root, creator=None):
    root = Path(root).resolve()
    manifests = []
    domains = ["crowd", "farfield"]
    if (root / "data/test_sova_original.jsonl").is_file():
        domains.append("sova")
    for domain in domains:
        original = root / "data" / f"test_{domain}_original.jsonl"
        if not original.is_file():
            raise FileNotFoundError(original)
        manifests.append(original)
        for snr in (5, 10, 20):
            manifest = root / "data" / f"test_{domain}_fan_{snr}db.jsonl"
            if not manifest.is_file():
                if creator is None:
                    from prepare_eval import create_noisy
                    creator = create_noisy
                creator(root, original.name, "noise_test.jsonl", [snr])
            if not manifest.is_file():
                raise FileNotFoundError(manifest)
            manifests.append(manifest)
    return manifests


def write_results(destination, selections, results):
    write_json(destination / "summary.json", {"updated_utc": utc_now(), "selection_rule": "minimum validation WER and final completed epoch", "checkpoints": selections, "results": results})
    fields = ["selected_as", "epoch", "checkpoint", "condition", "manifest", "wer", "wer_percent", "records", "reference_words", "word_errors", "substitutions", "deletions", "insertions", "evaluation_seconds", "predictions"]
    temporary = destination / "summary.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    temporary.replace(destination / "summary.csv")


def evaluate_run(root, run_dir, runner=None, prepare=None):
    runner = runner or subprocess.run
    prepare = prepare or prepare_test_manifests
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    if run_dir.parent != root / "runs" or not run_dir.is_dir():
        raise ValueError("Evaluation run must be an existing directory directly under this project's runs/")
    destination = run_dir / "evaluation"
    destination.mkdir(exist_ok=True)
    state = {"status": "running", "started_utc": utc_now(), "run_dir": str(run_dir)}
    write_json(destination / "status.json", state)
    results = []
    try:
        selections = select_checkpoints(run_dir)
        manifests = prepare(root)
        for selection in selections:
            checkpoint = Path(selection["checkpoint"])
            checkpoint_hash = sha256(checkpoint)
            for manifest in manifests:
                manifest = Path(manifest).resolve()
                predictions = destination / selection["role"] / (manifest.stem + ".jsonl")
                summary_path = predictions.with_suffix(".summary.json")
                request_path = predictions.with_suffix(".request.json")
                identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
                            "manifest": str(manifest), "manifest_sha256": sha256(manifest),
                            "evaluation_code_sha256": sha256(Path(__file__).with_name("evaluate.py")),
                            "batch_size": 16, "device": "auto"}
                previous = json.loads(request_path.read_text(encoding="utf-8")) if request_path.exists() else None
                if previous is not None and previous != identity:
                    raise ValueError(f"Evaluation inputs changed; preserve existing outputs before retrying: {predictions}")
                if previous is None and (predictions.exists() or summary_path.exists()):
                    raise FileExistsError(f"Evaluation outputs have no matching input audit: {predictions}")
                complete = predictions.is_file() and summary_path.is_file()
                if complete:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if summary.get("checkpoint") != str(checkpoint) or summary.get("manifest") != str(manifest):
                        raise ValueError(f"Evaluation summary inputs do not match: {summary_path}")
                else:
                    write_json(request_path, identity)
                    command = [sys.executable, "-u", "-B", str(Path(__file__).with_name("evaluate.py")),
                               "--checkpoint", str(checkpoint), "--manifest", str(manifest),
                               "--output", str(predictions), "--batch-size", "16"]
                    if predictions.exists() or summary_path.exists():
                        command.append("--overwrite")
                    print(json.dumps({"evaluation": manifest.stem, "checkpoint": selection["role"], "epoch": selection["epoch"]}), flush=True)
                    result = runner(command, cwd=root, check=False)
                    if result.returncode:
                        raise subprocess.CalledProcessError(result.returncode, command)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    if not predictions.is_file() or summary.get("checkpoint") != str(checkpoint) or summary.get("manifest") != str(manifest):
                        raise ValueError(f"Evaluation did not publish matching outputs: {summary_path}")
                results.append({**summary, "selected_as": "+".join(selection["selected_as"]),
                                "epoch": selection["epoch"], "condition": manifest.stem})
                write_results(destination, selections, results)
        state.update(status="complete", finished_utc=utc_now(), completed_evaluations=len(results))
        write_json(destination / "status.json", state)
        return destination / "summary.json"
    except Exception as error:
        state.update(status="failed", failed_utc=utc_now(), completed_evaluations=len(results),
                     error=f"{type(error).__name__}: {error}", returncode=getattr(error, "returncode", 1))
        write_json(destination / "status.json", state)
        raise


def run_training_stage(args, runner=None, evaluator=None):
    runner = runner or subprocess.run
    evaluator = evaluator or evaluate_run
    root = args.output_root.resolve()
    if not args.run_training:
        raise ValueError("Full training requires --run-training")
    before = set((root / "runs").glob(args.phase + "_*"))
    command = [sys.executable, "-u", "-B", str(Path(__file__).with_name("train.py")),
               "--phase", args.phase, "--run-training", "--output-root", str(root)]
    for option in ("init_from", "resume"):
        if getattr(args, option):
            command.extend(["--" + option.replace("_", "-"), str(getattr(args, option).resolve())])
    status_path = root / "logs" / ("stage_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    state = {"status": "training", "phase": args.phase, "started_utc": utc_now(), "command": command}
    write_json(status_path, state)
    try:
        result = runner(command, cwd=root, check=False)
        if result.returncode:
            raise subprocess.CalledProcessError(result.returncode, command)
        if args.resume:
            run_dir = resume_run_directory(args.resume, root)
        else:
            created = set((root / "runs").glob(args.phase + "_*")) - before
            if len(created) != 1:
                raise RuntimeError(f"Cannot identify the completed training run: {sorted(map(str, created))}")
            run_dir = created.pop()
        state.update(status="training_complete", run_dir=str(run_dir), training_finished_utc=utc_now(),
                     full_test_evaluation_requested=bool(getattr(args, "evaluate_after_training", False)))
        write_json(status_path, state)
        if not getattr(args, "evaluate_after_training", False):
            return status_path
        state.update(status="evaluating")
        write_json(status_path, state)
        summary = evaluator(root, run_dir)
        state.update(status="complete", finished_utc=utc_now(), evaluation_summary=str(summary))
        write_json(status_path, state)
        return summary
    except Exception as error:
        state.update(status="failed", failed_utc=utc_now(), error=f"{type(error).__name__}: {error}",
                     returncode=getattr(error, "returncode", 1))
        write_json(status_path, state)
        raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--phase", choices=("baseline", "mixed", "noise"), default="baseline")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--evaluate-after-training", action="store_true",
                        help="Opt in to full test evaluation after successful training")
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument("--init-from", type=Path)
    initialization.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-run", type=Path, help="Retry only final evaluation; do not train")
    args = parser.parse_args(argv)
    if args.evaluate_run:
        if args.run_training or args.init_from or args.resume or args.evaluate_after_training:
            parser.error("--evaluate-run cannot be combined with training options")
    elif not args.run_training:
        parser.error("Full training requires --run-training; evaluation-only requires --evaluate-run")
    return args


def main():
    args = parse_args()
    try:
        summary = evaluate_run(args.output_root, args.evaluate_run) if args.evaluate_run else run_training_stage(args)
        key = "evaluation_summary" if args.evaluate_run or args.evaluate_after_training else "training_status"
        print(json.dumps({key: str(summary)}), flush=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)


if __name__ == "__main__":
    main()
