"""Train on Golos, continue on Golos/Sova, then adapt with noise."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess

from run_stage import run_training_stage, select_checkpoints, utc_now, write_json


STAGE_EPOCHS = {"baseline": 10, "mixed": 25, "noise": 5}


def run_experiment(root, *, run_training=False, stage_runner=None, checkpoint_selector=None):
    if not run_training:
        raise ValueError("Full experiment training requires --run-training")
    root = Path(root).resolve()
    stage_runner = stage_runner or run_training_stage
    checkpoint_selector = checkpoint_selector or select_checkpoints
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    status_path = root / "logs" / f"experiment_{stamp}.json"
    state = {"status": "running", "started_utc": utc_now(), "phase": None,
             "phase_order": list(STAGE_EPOCHS), "stage_epochs": STAGE_EPOCHS,
             "completed_stages": {}, "continuation_checkpoint": None,
             "final_checkpoint": None, "full_test_evaluation": False}
    write_json(status_path, state)
    try:
        for phase in state["phase_order"]:
            state["phase"] = phase
            write_json(status_path, state)
            print(json.dumps({"experiment_status": str(status_path), "phase": phase,
                              "init_from": state["continuation_checkpoint"]}), flush=True)
            args = argparse.Namespace(output_root=root, phase=phase, run_training=True,
                                      init_from=Path(state["continuation_checkpoint"]) if phase != "baseline" else None,
                                      resume=None, evaluate_after_training=False)
            stage_status_path = Path(stage_runner(args))
            stage_status = json.loads(stage_status_path.read_text(encoding="utf-8"))
            if stage_status.get("status") != "training_complete":
                raise RuntimeError(f"Stage did not complete training: {stage_status_path}")
            run_dir = Path(stage_status["run_dir"]).resolve()
            if run_dir.parent != root / "runs" or not run_dir.is_dir():
                raise ValueError(f"Stage returned an invalid project run directory: {run_dir}")
            selected = checkpoint_selector(run_dir)
            final = next((record for record in selected if "final" in record["selected_as"]), None)
            if final is None:
                raise ValueError(f"No final {phase} checkpoint was selected")
            if final["epoch"] != STAGE_EPOCHS[phase]:
                raise ValueError(f"Stage {phase} completed {final['epoch']} epochs; expected {STAGE_EPOCHS[phase]}")
            if not isinstance(final.get("val_wer"), (int, float)) or not math.isfinite(final["val_wer"]):
                raise ValueError(f"Stage {phase} has no finite validation WER for its final checkpoint")
            checkpoint = str(Path(final["checkpoint"]).resolve())
            state["completed_stages"][phase] = {"run_dir": str(run_dir),
                                                "stage_status": str(stage_status_path),
                                                "final_checkpoint": checkpoint,
                                                "final_checkpoint_epoch": final["epoch"],
                                                "finished_utc": utc_now()}
            state["continuation_checkpoint"] = checkpoint
            write_json(status_path, state)
        state.update(status="complete", phase=None, final_checkpoint=state["continuation_checkpoint"], finished_utc=utc_now())
        write_json(status_path, state)
        return status_path
    except Exception as error:
        state.update(status="failed", failed_utc=utc_now(),
                     error=f"{type(error).__name__}: {error}",
                     returncode=getattr(error, "returncode", 1))
        write_json(status_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-training", action="store_true")
    args = parser.parse_args()
    if not args.run_training:
        parser.error("Full experiment training requires --run-training")
    try:
        status = run_experiment(args.output_root, run_training=True)
        print(json.dumps({"experiment_status": str(status)}), flush=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)


if __name__ == "__main__":
    main()
