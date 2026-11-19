"""Materialize deterministic noisy validation/test audio; never train a model."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_utils import mix_at_snr


def create_noisy(root, source_name, noise_name, snrs, overwrite=False):
    rows = [json.loads(x) for x in (root / "data" / source_name).read_text(encoding="utf-8").splitlines()]
    noises = [json.loads(x) for x in (root / "data" / noise_name).read_text(encoding="utf-8").splitlines()]
    generated = []
    for snr in snrs:
        stem = Path(source_name).stem.replace("_original", "") + f"_fan_{snr}db"
        manifest = root / "data" / (stem + ".jsonl")
        if manifest.exists() and not overwrite:
            raise FileExistsError(f"{manifest}: use --overwrite explicitly")
        audio_dir = root / "data" / "eval_audio" / stem
        audio_dir.mkdir(parents=True, exist_ok=True)
        output_rows = []
        for row in rows:
            key = f"112:{row['id']}:{noise_name}:{snr}"
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))
            n = noises[int(rng.integers(len(noises)))]
            speech, sr = sf.read(row["audio_filepath"], dtype="float32")
            noise, nsr = sf.read(n["audio_filepath"], dtype="float32")
            if sr != 16000 or nsr != sr:
                raise ValueError("Expected 16 kHz speech and fan noise")
            mixed, details = mix_at_snr(speech, noise, snr, int(rng.integers(len(noise))))
            filename = hashlib.sha256(row["id"].encode()).hexdigest() + ".wav"
            target = audio_dir / filename
            sf.write(target, mixed, sr, subtype="FLOAT")
            output_rows.append({**row, "audio_filepath": str(target),
                                "id": row["id"] + f"_fan_{snr}db",
                                "source_id": row["id"], "noise_filepath": n["audio_filepath"],
                                "mix": details})
        tmp = manifest.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in output_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(manifest)
        generated.append(manifest)
        print(json.dumps({"manifest": str(manifest), "recordings": len(output_rows)}, ensure_ascii=False), flush=True)
    return generated


def prepare_validation(root, speech_set="golos", overwrite=False):
    root = Path(root)
    combined = root / "data" / ("val_mixed_selection.jsonl" if speech_set == "mixed" else "val_selection.jsonl")
    if combined.exists() and not overwrite:
        raise FileExistsError(combined)
    if speech_set == "mixed":
        source_names = ["val_original.jsonl", "val_sova_original.jsonl"]
        original = root / "data" / "val_mixed_original.jsonl"
        if not original.is_file():
            raise FileNotFoundError(original)
        generated = []
        for source_name in source_names:
            noisy = root / "data" / (Path(source_name).stem.replace("_original", "") + "_fan_10db.jsonl")
            if not noisy.exists() or overwrite:
                create_noisy(root, source_name, "noise_val.jsonl", [10], overwrite)
            source_rows = [json.loads(line) for line in (root / "data" / source_name).read_text(encoding="utf-8").splitlines()]
            noisy_rows = [json.loads(line) for line in noisy.read_text(encoding="utf-8").splitlines()]
            if [(r["id"], r["text"]) for r in source_rows] != [(r.get("source_id"), r["text"]) for r in noisy_rows]:
                raise ValueError(f"Noisy validation manifest does not match its source: {noisy}")
            if not all(Path(r["audio_filepath"]).is_file() for r in noisy_rows):
                raise FileNotFoundError(f"Noisy validation audio is missing: {noisy}")
            generated.append(noisy)
        noisy_content = "".join(path.read_text(encoding="utf-8") for path in generated)
        original_content = original.read_text(encoding="utf-8")
        original_references = [(row["id"], row["text"]) for row in map(json.loads, original_content.splitlines())]
        noisy_references = [(row["source_id"], row["text"]) for row in map(json.loads, noisy_content.splitlines())]
        if original_references != noisy_references:
            raise ValueError("Mixed validation ordering does not match original/noisy sources")
        (root / "data" / "val_mixed_fan_10db.jsonl").write_text(noisy_content, encoding="utf-8")
        combined.write_text(original_content + noisy_content, encoding="utf-8")
    else:
        generated = create_noisy(root, "val_original.jsonl", "noise_val.jsonl", [10], overwrite)
        combined.write_text((root / "data" / "val_original.jsonl").read_text(encoding="utf-8") + generated[0].read_text(encoding="utf-8"), encoding="utf-8")
        originals = (root / "data" / "benchmark_val.jsonl").read_text(encoding="utf-8").splitlines()[:64]
        noisy = generated[0].read_text(encoding="utf-8").splitlines()[:64]
        interleaved = [row for pair in zip(originals, noisy) for row in pair]
        (root / "data" / "benchmark_val_selection.jsonl").write_text("\n".join(interleaved) + "\n", encoding="utf-8")
    return combined


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--split", choices=["validation", "test"], default="validation")
    ap.add_argument("--speech-set", choices=["golos", "mixed"], default="golos")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    root = args.output_root.resolve()
    if args.split == "validation":
        prepare_validation(root, args.speech_set, args.overwrite)
    else:
        domains = ["crowd", "farfield", "sova"] if args.speech_set == "mixed" else ["crowd", "farfield"]
        for domain in domains:
            create_noisy(root, f"test_{domain}_original.jsonl", "noise_test.jsonl", [5, 10, 20], args.overwrite)


if __name__ == "__main__":
    main()
