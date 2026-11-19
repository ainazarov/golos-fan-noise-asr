"""Lightweight checkpoint identity checks, without importing the GPU runtime."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def configuration_sha256(config):
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def initialization_source_phase(phase):
    parents = {"mixed": "baseline", "noise": "mixed"}
    if phase not in parents:
        raise ValueError(f"Phase {phase!r} does not initialize from a previous stage")
    return parents[phase]


def validate_checkpoint_identity(identity, *, tokenizer_sha256, phase, config_sha256=None):
    if not isinstance(identity, dict):
        raise ValueError("Checkpoint has no project identity metadata")
    expected = {"tokenizer_sha256": tokenizer_sha256, "vocab_size": 112, "phase": phase}
    if config_sha256 is not None:
        expected["config_sha256"] = config_sha256
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match the requested experiment")


def resume_run_directory(checkpoint, root):
    checkpoint = Path(checkpoint).resolve()
    run_dir = checkpoint.parent.parent
    if checkpoint.parent.name != "checkpoints" or run_dir.parent != Path(root).resolve() / "runs":
        raise ValueError("Resume from this project's runs/<run>/checkpoints/*.ckpt directory")
    return run_dir
