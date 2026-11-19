#!/usr/bin/env python3
"""Prepare real industrial-fan noise with a machine-disjoint split.

Only normal recordings from the source archive's fan/train directory are used.
The recordings already include factory ambience; they are not isolated HVAC
sounds. Source machine IDs are held apart, but independent background-recording
sessions cannot be established from the public metadata.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import time
import urllib.request
import wave
import zipfile
import zlib


SOURCE_PAGE = "https://zenodo.org/records/3678171"
DOWNLOAD_URL = SOURCE_PAGE + "/files/dev_data_fan.zip?download=1"
ARCHIVE_NAME = "dev_data_fan.zip"
EXPECTED_MD5 = "649bdfc06263ae7a838963f43b6641e6"
EXPECTED_BYTES = 1_354_774_772
EXPECTED_COUNTS = {"00": 911, "02": 916, "04": 933, "06": 915}
MACHINE_SPLIT = {"00": "train", "02": "train", "04": "val", "06": "test"}
MEMBER_PATTERN = re.compile(
    r"^fan/(train|test)/(normal|anomaly)_id_(\d{2})_\d+\.wav$"
)
CHUNK_BYTES = 1024 * 1024


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path) -> None:
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"Archive has {size:,} bytes; expected {EXPECTED_BYTES:,}: {path}")
    checksum = md5_file(path)
    if checksum != EXPECTED_MD5:
        raise ValueError(f"Archive MD5 mismatch: {checksum}; expected {EXPECTED_MD5}")


def obtain_archive(path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            verify_archive(path)
            print(f"Reusing verified archive: {path}", flush=True)
            return path
        except ValueError:
            if not overwrite:
                raise

    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > EXPECTED_BYTES:
        if not overwrite:
            raise ValueError(f"Partial download is larger than the archive: {partial}")
        offset = 0
    if offset == EXPECTED_BYTES:
        try:
            verify_archive(partial)
        except ValueError:
            if not overwrite:
                raise
            offset = 0
        else:
            os.replace(partial, path)
            return path

    headers = {"User-Agent": "sound-course-proj1/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(DOWNLOAD_URL, headers=headers)
    print(f"Downloading {ARCHIVE_NAME}; starting at {offset:,} bytes", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
        if offset and status == 206:
            actual_range = response.headers.get("Content-Range", "")
            if not actual_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"Unexpected resume response: {actual_range}")
            mode = "ab"
        elif status == 200:
            offset = 0
            mode = "wb"
        else:
            raise ValueError(f"Unexpected download status: {status}")

        received = offset
        last_report = time.monotonic()
        with partial.open(mode) as stream:
            while chunk := response.read(CHUNK_BYTES):
                stream.write(chunk)
                received += len(chunk)
                if received > EXPECTED_BYTES:
                    raise ValueError("Download exceeded the expected archive length")
                if time.monotonic() - last_report >= 5:
                    print(
                        f"Downloaded {received / EXPECTED_BYTES:.1%} "
                        f"({received / 1024**2:.1f} MiB)",
                        flush=True,
                    )
                    last_report = time.monotonic()
    verify_archive(partial)
    os.replace(partial, path)
    print("Download complete; official MD5 verified", flush=True)
    return path


def existing_member_matches(path: Path, member: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF == member.CRC


def extract_member(
    archive: zipfile.ZipFile, member: zipfile.ZipInfo, destination: Path, overwrite: bool
) -> None:
    if destination.exists() and not overwrite:
        if not existing_member_matches(destination, member):
            raise ValueError(f"Existing audio differs from the archive; use --overwrite: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with archive.open(member) as source, partial.open("wb") as output:
        # ZipFile checks the member CRC when its stream reaches EOF.
        shutil.copyfileobj(source, output, CHUNK_BYTES)
    os.replace(partial, destination)


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        observed = (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype())
        if observed != (16000, 1, 2, "NONE"):
            raise ValueError(f"Expected 16 kHz mono PCM16, got {observed}: {path}")
        if audio.getnframes() <= 0:
            raise ValueError(f"Empty audio: {path}")
        return audio.getnframes() / audio.getframerate()


def write_output(path: Path, text: str, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Output already exists with different contents; use --overwrite: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def prepare(output_root: Path, archive_path: Path, overwrite: bool) -> dict:
    verify_archive(archive_path)
    noise_root = (output_root / "assets" / "noise").resolve()
    rows: dict[str, list[dict]] = {name: [] for name in ("train", "val", "test")}
    by_machine: collections.Counter = collections.Counter()
    hours_by_machine: collections.Counter = collections.Counter()
    excluded: collections.Counter = collections.Counter()
    seen = set()

    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            match = MEMBER_PATTERN.fullmatch(member.filename)
            if not match:
                excluded["other_members"] += 1
                continue
            original_split, condition, machine_id = match.groups()
            if condition != "normal":
                excluded["anomalous_recordings"] += 1
                continue
            if original_split != "train":
                excluded["normal_recordings_from_source_test"] += 1
                continue
            if machine_id not in MACHINE_SPLIT:
                raise ValueError(f"Unexpected machine ID: {member.filename}")

            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in member.filename:
                raise ValueError(f"Unsafe archive path: {member.filename}")
            if member.filename in seen:
                raise ValueError(f"Duplicate archive path: {member.filename}")
            seen.add(member.filename)
            destination = (noise_root / Path(*relative.parts)).resolve()
            destination.relative_to(noise_root)  # Reject escapes, including existing symlinks.
            extract_member(archive, member, destination, overwrite)
            duration = audio_duration(destination)
            split = MACHINE_SPLIT[machine_id]
            rows[split].append(
                {
                    "audio_filepath": str(destination),
                    "duration": duration,
                    "text": "",
                    "machine_id": machine_id,
                    "source_file": member.filename,
                }
            )
            by_machine[machine_id] += 1
            hours_by_machine[machine_id] += duration / 3600
            if len(seen) % 500 == 0:
                print(f"Extracted and checked {len(seen):,} normal fan recordings", flush=True)

    if dict(by_machine) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected source counts: {dict(by_machine)}; expected {EXPECTED_COUNTS}")

    report = {
        "source": {
            "name": "DCASE 2020 Task 2 development data, MIMII fan subset",
            "page": SOURCE_PAGE,
            "download_url": DOWNLOAD_URL,
            "archive": str(archive_path.resolve()),
            "archive_bytes": EXPECTED_BYTES,
            "md5": EXPECTED_MD5,
            "license": "CC BY-NC-SA 4.0",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "citation_instructions": "Cite the three papers listed on the source record.",
        },
        "scope": "Normal industrial fan recordings with existing factory background ambience.",
        "selection": "Only fan/train/normal_id_*.wav from the original archive; no original test or anomaly recordings.",
        "split_rule": {"train": ["00", "02"], "val": ["04"], "test": ["06"]},
        "independence": "Machine-disjoint. Independent source sessions or factory-background recordings are not verifiable from the metadata.",
        "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM_16"},
        "counts_by_machine": dict(by_machine),
        "hours_by_machine": {key: round(value, 6) for key, value in sorted(hours_by_machine.items())},
        "excluded_members": dict(excluded),
        "splits": {},
    }
    for split, records in rows.items():
        manifest = output_root / "data" / f"noise_{split}.jsonl"
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
        write_output(manifest, text, overwrite)
        report["splits"][split] = {
            "manifest": str(manifest.resolve()),
            "recordings": len(records),
            "hours": round(sum(row["duration"] for row in records) / 3600, 6),
        }
    write_output(
        output_root / "reports" / "noise_audit.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        overwrite,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--archive", type=Path, help="Use an existing official ZIP instead of downloading it")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files if they already exist")
    args = parser.parse_args()
    root = args.output_root.resolve()
    archive = args.archive.resolve() if args.archive else obtain_archive(
        root / "assets" / "downloads" / ARCHIVE_NAME, args.overwrite
    )
    report = prepare(root, archive, args.overwrite)
    print(json.dumps(report["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
