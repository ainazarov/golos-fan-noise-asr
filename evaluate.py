"""Evaluate a trained hybrid ASR checkpoint without updating its parameters.

NeMo and PyTorch are imported only by the CLI, so metric tests need only Python's
standard library. Predictions and the summary are published atomically after a
successful evaluation. Existing output files require an explicit --overwrite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Iterator


CHUNK_SIZE = 64


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace; deliberately preserve punctuation."""
    return " ".join(text.lower().split())


def word_edit_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Return word-level Levenshtein counts, including empty-reference insertions.

    Equal-cost paths prefer substitution, then deletion, then insertion. The
    chosen decomposition is deterministic; the total edit distance is minimal.
    """
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()
    n, m = len(reference_words), len(hypothesis_words)
    # Operation codes: 0 = match, 1 = substitution, 2 = deletion, 3 = insertion.
    operations = [bytearray(m + 1) for _ in range(n + 1)]
    operations[0][1:] = bytes([3]) * m
    previous = list(range(m + 1))
    for i, reference_word in enumerate(reference_words, start=1):
        current = [i] + [0] * m
        operations[i][0] = 2
        for j, hypothesis_word in enumerate(hypothesis_words, start=1):
            if reference_word == hypothesis_word:
                current[j] = previous[j - 1]
                operations[i][j] = 0
            else:
                cost, operation = min(
                    (previous[j - 1] + 1, 1),
                    (previous[j] + 1, 2),
                    (current[j - 1] + 1, 3),
                )
                current[j] = cost
                operations[i][j] = operation
        previous = current

    counts = {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "reference_words": n,
        "hypothesis_words": m,
    }
    i, j = n, m
    while i or j:
        operation = operations[i][j]
        if operation == 0:
            i -= 1
            j -= 1
        elif operation == 1:
            counts["substitutions"] += 1
            i -= 1
            j -= 1
        elif operation == 2:
            counts["deletions"] += 1
            i -= 1
        else:
            counts["insertions"] += 1
            j -= 1
    return counts


class WordErrorAccumulator:
    """Accumulate corpus WER rather than averaging per-recording WER."""

    def __init__(self) -> None:
        self.counts = {
            "substitutions": 0,
            "deletions": 0,
            "insertions": 0,
            "reference_words": 0,
            "hypothesis_words": 0,
        }
        self.records = 0
        self.empty_reference_records = 0
        self.empty_reference_insertions = 0
        self.empty_reference_nonempty_hypotheses = 0

    def add(self, reference: str, hypothesis: str) -> dict[str, int]:
        counts = word_edit_counts(reference, hypothesis)
        self.records += 1
        for name, value in counts.items():
            self.counts[name] += value
        if counts["reference_words"] == 0:
            self.empty_reference_records += 1
            self.empty_reference_insertions += counts["insertions"]
            self.empty_reference_nonempty_hypotheses += int(
                counts["hypothesis_words"] > 0
            )
        return counts

    def summary(self) -> dict[str, Any]:
        errors = sum(self.counts[k] for k in ("substitutions", "deletions", "insertions"))
        denominator = self.counts["reference_words"]
        wer = errors / denominator if denominator else None
        return {
            "records": self.records,
            **self.counts,
            "word_errors": errors,
            "wer": wer,
            "wer_percent": 100.0 * wer if wer is not None else None,
            "empty_reference_records": self.empty_reference_records,
            "empty_reference_insertions": self.empty_reference_insertions,
            "empty_reference_nonempty_hypotheses": self.empty_reference_nonempty_hypotheses,
            "nonempty_reference_records": self.records - self.empty_reference_records,
        }


def iter_manifest_chunks(manifest: Path) -> Iterator[list[dict[str, Any]]]:
    """Read small groups of records, resolving relative paths against the manifest."""
    chunk: list[dict[str, Any]] = []
    with manifest.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {manifest}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {manifest}:{line_number}")
            if not isinstance(record.get("text"), str):
                raise ValueError(f"Expected string text at {manifest}:{line_number}")
            if not isinstance(record.get("audio_filepath"), str) or not record["audio_filepath"]:
                raise ValueError(f"Expected audio_filepath at {manifest}:{line_number}")
            audio_path = Path(record["audio_filepath"]).expanduser()
            if not audio_path.is_absolute():
                audio_path = manifest.parent / audio_path
            audio_path = audio_path.resolve()
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio at {manifest}:{line_number}: {audio_path}")
            record["audio_filepath"] = str(audio_path)
            chunk.append(record)
            if len(chunk) == CHUNK_SIZE:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def hypothesis_texts(result: Any, expected: int) -> list[str]:
    """Accept NeMo's Hypothesis/string list, selecting its first tuple output."""
    if isinstance(result, tuple):
        if not result:
            raise RuntimeError("NeMo returned an empty transcription tuple")
        result = result[0]
    if not isinstance(result, list) or len(result) != expected:
        count = len(result) if hasattr(result, "__len__") else "unknown"
        raise RuntimeError(f"Expected {expected} transcription hypotheses, received {count}")
    texts = []
    for item in result:
        text = item if isinstance(item, str) else getattr(item, "text", None)
        if not isinstance(text, str):
            raise RuntimeError(f"Unsupported transcription hypothesis: {type(item).__name__}")
        texts.append(text)
    return texts


def publish_file(temporary: Path, destination: Path, overwrite: bool) -> None:
    """Publish a completed same-filesystem file atomically, respecting no-clobber."""
    if overwrite:
        os.replace(temporary, destination)
    else:
        # Unlike an existence check followed by replace(), link() cannot clobber
        # a file created by another process while evaluation was running.
        os.link(temporary, destination)
        temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Predictions JSONL path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.checkpoint.suffix.lower() not in (".ckpt", ".nemo"):
        parser.error("--checkpoint must be a .ckpt or .nemo file")
    for path in (args.checkpoint, args.manifest):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")
    summary_path = args.output.with_suffix(".summary.json")
    for path in (args.output, summary_path):
        if path in (args.checkpoint, args.manifest):
            parser.error(f"Output must not replace an input: {path}")
        if path.exists() and (not args.overwrite or not path.is_file()):
            parser.error(f"Output already exists (use --overwrite for files): {path}")
    return args


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
    os.environ.setdefault("STRICT_NUMBA_COMPAT_CHECK", "0")
    # Registration must precede restoring a model whose training config refers
    # to this project's augmentation. It does not augment evaluation inputs.
    from audio_utils import register_fan_noise

    register_fan_noise()
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if args.checkpoint.suffix.lower() == ".nemo":
        model = EncDecHybridRNNTCTCBPEModel.restore_from(
            str(args.checkpoint), map_location=torch.device("cpu")
        )
    else:
        model = EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
            str(args.checkpoint), map_location="cpu"
        )
    model.to(device)
    model.eval()
    # None means reuse the checkpoint's decoding configuration, including its
    # TDT durations. 'rnnt' selects the transducer branch of the hybrid model.
    model.change_decoding_strategy(decoder_type="rnnt", verbose=False)
    model_load_seconds = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".summary.json")
    temporary_paths: list[Path] = []
    accumulator = WordErrorAccumulator()
    evaluation_started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=args.output.parent,
            prefix=f".{args.output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            predictions_temporary = Path(stream.name)
            temporary_paths.append(predictions_temporary)
            with torch.inference_mode():
                for records in iter_manifest_chunks(args.manifest):
                    result = model.transcribe(
                        audio=[record["audio_filepath"] for record in records],
                        batch_size=args.batch_size,
                        return_hypotheses=True,
                        num_workers=0,
                        verbose=False,
                    )
                    predictions = hypothesis_texts(result, expected=len(records))
                    for record, prediction in zip(records, predictions):
                        reference = record["text"]
                        counts = accumulator.add(reference, prediction)
                        output_record = {
                            "id": record.get("id"),
                            "audio_filepath": record["audio_filepath"],
                            "reference": reference,
                            "prediction": prediction,
                            "normalized_reference": normalize_text(reference),
                            "normalized_prediction": normalize_text(prediction),
                            **counts,
                        }
                        stream.write(json.dumps(output_record, ensure_ascii=False, allow_nan=False) + "\n")
            if not accumulator.records:
                raise ValueError("The evaluation manifest contains no records")
            stream.flush()
            os.fsync(stream.fileno())

        summary = {
            "checkpoint": str(args.checkpoint),
            "manifest": str(args.manifest),
            "predictions": str(args.output),
            "device": device,
            "batch_size": args.batch_size,
            "chunk_size": CHUNK_SIZE,
            "normalization": "lowercase and collapse whitespace; punctuation preserved",
            "decoder_branch": "transducer (checkpoint decoding configuration preserved)",
            "model_load_seconds": model_load_seconds,
            "evaluation_seconds": time.perf_counter() - evaluation_started,
            **accumulator.summary(),
        }
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=args.output.parent,
            prefix=f".{summary_path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            summary_temporary = Path(stream.name)
            temporary_paths.append(summary_temporary)
            json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        publish_file(predictions_temporary, args.output, args.overwrite)
        publish_file(summary_temporary, summary_path, args.overwrite)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
