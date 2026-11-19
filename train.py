"""Bounded benchmarks and explicitly requested training for Golos-112.

No arguments start a training job. Full phases require --run-training.
Benchmark weights are never saved or reused as an experimental baseline.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

# Reproduce the working course CUDA/Numba setup; actual compatibility is tested.
os.environ.setdefault("NUMBA_CUDA_USE_NVIDIA_BINDING", "1")
os.environ.setdefault("STRICT_NUMBA_COMPAT_CHECK", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf, open_dict
import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
from audio_utils import register_fan_noise
from checkpoint_utils import configuration_sha256, initialization_source_phase, resume_run_directory, validate_checkpoint_identity
from experiment_logging import EpochHistory, ProgressHistory, scalar_metrics


class BenchmarkChecks(Callback):
    def __init__(self, warmup):
        self.warmup = warmup
        self.timings = []
        self.losses = []
        self.gradients_checked = False
        self.parameters_changed = False
        self.previous = None

    def on_train_start(self, trainer, module):
        self.parameter = next(module.encoder.parameters())
        self.initial_parameter = self.parameter.detach().clone()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.previous = time.perf_counter()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        if not self.gradients_checked or trainer.global_step + 1 == trainer.max_steps:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError("No gradients produced")
            if not torch.stack([torch.isfinite(g).all() for g in gradients]).all().item():
                raise RuntimeError("Non-finite gradients")
            if not any(torch.count_nonzero(g).item() for g in gradients[:10]):
                raise RuntimeError("All sampled gradients are zero")
            self.gradients_checked = True

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        if loss is not None:
            value = float(loss.detach().cpu())
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss")
            self.losses.append(value)
        torch.cuda.synchronize()
        now = time.perf_counter()
        step = int(trainer.global_step)
        if step > self.warmup:
            self.timings.append(now - self.previous)
        if step == self.warmup:
            torch.cuda.reset_peak_memory_stats()
        self.previous = now
        if step % 20 == 0:
            print(json.dumps({"benchmark_step": step, "loss": self.losses[-1]}), flush=True)

    def on_train_end(self, trainer, module):
        self.parameters_changed = not torch.equal(self.initial_parameter, self.parameter.detach())
        if not self.gradients_checked or not self.parameters_changed:
            raise RuntimeError("Gradient/parameter-update check failed")
        if not torch.stack([torch.isfinite(p).all() for p in module.parameters()]).all().item():
            raise RuntimeError("Non-finite final parameters")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CheckpointIdentity(Callback):
    def __init__(self, tokenizer_sha256, phase, config_sha256):
        self.identity = {"tokenizer_sha256": tokenizer_sha256, "vocab_size": 112,
                         "phase": phase, "config_sha256": config_sha256}

    def on_save_checkpoint(self, trainer, module, checkpoint):
        checkpoint["proj1"] = dict(self.identity)


class AuditedModelCheckpoint(ModelCheckpoint):
    def __init__(self, *, run_dir, **kwargs):
        super().__init__(**kwargs)
        self.history = EpochHistory(run_dir)
        self.checkpoint_history = EpochHistory(run_dir, "checkpoint_history")
        self.validation_epoch = None
        self.validation_step = None
        self.validation_metrics = {}

    def on_train_start(self, trainer, module):
        super().on_train_start(trainer, module)
        if trainer.is_global_zero:
            self.history.resume_from_step(int(trainer.global_step))
            self.checkpoint_history.resume_from_step(int(trainer.global_step))

    def on_validation_end(self, trainer, module):
        if not trainer.sanity_checking:
            self.validation_epoch = int(trainer.current_epoch)
            self.validation_step = int(trainer.global_step)
            self.validation_metrics = scalar_metrics(trainer.callback_metrics, prefix="val")
        super().on_validation_end(trainer, module)

    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero and Path(filepath).name.startswith("epoch="):
            fresh = (self.validation_epoch == int(trainer.current_epoch) and
                     self.validation_step == int(trainer.global_step))
            values = dict(epoch_index=trainer.current_epoch, global_step=trainer.global_step,
                          checkpoint=filepath, metrics=self.validation_metrics if fresh else {},
                          validation_completed=fresh)
            self.checkpoint_history.record(**values)
            if fresh:
                self.history.record(**values)


class TrainingProgress(Callback):
    def __init__(self, run_dir, segment, interval=50, stdout_interval=500):
        self.history = ProgressHistory(run_dir, segment)
        self.interval = interval
        self.stdout_interval = stdout_interval
        self.last_step = None
        self.state = "starting"

    def on_train_start(self, trainer, module):
        self.started = self.previous = time.perf_counter()
        self.last_step = int(trainer.global_step)
        self.observed_epoch = int(trainer.current_epoch) + 1
        self.state = "running"
        if trainer.is_global_zero:
            self.history.status("running", global_step=self.last_step, epoch=int(trainer.current_epoch) + 1)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        self.observed_epoch = int(trainer.current_epoch) + 1
        step = int(trainer.global_step)
        if not trainer.is_global_zero or not step or step % self.interval or step == self.last_step:
            return
        now = time.perf_counter()
        count, seconds = step - self.last_step, now - self.previous
        metrics = scalar_metrics({name: value for name, value in trainer.callback_metrics.items()
                                  if "loss" in name and not name.startswith("val")})
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None:
            sampled = scalar_metrics({"train_loss": loss})
            if sampled.get("train_loss") is None:
                raise FloatingPointError(f"Non-finite training loss at optimizer step {step}")
            metrics.update(sampled)
        def first(*names):
            return next((metrics[name] for name in names if name in metrics), None)
        gpu = module.device.type == "cuda"
        batches = trainer.num_training_batches
        row = {"epoch": int(trainer.current_epoch) + 1, "global_step": step,
               "batch_in_epoch": int(batch_idx) + 1,
               "train_batches_per_epoch": int(batches) if math.isfinite(batches) else None,
               "segment_elapsed_seconds": now - self.started, "interval_steps": count,
               "interval_seconds": seconds, "seconds_per_step": seconds / count,
               "steps_per_second": count / seconds if seconds else None,
               "learning_rate": float(trainer.optimizers[0].param_groups[0]["lr"]),
               "train_loss": first("train_loss", "loss"),
               "rnnt_loss": first("train_rnnt_loss", "train_loss_rnnt", "rnnt_loss"),
               "ctc_loss": first("train_ctc_loss", "train_loss_ctc", "ctc_loss"),
               "gpu_allocated_gib": torch.cuda.memory_allocated(module.device) / 2**30 if gpu else None,
               "gpu_reserved_gib": torch.cuda.memory_reserved(module.device) / 2**30 if gpu else None,
               "train_metrics": metrics}
        self.history.append(row)
        if step % self.stdout_interval == 0:
            print(json.dumps({"progress": {key: row[key] for key in
                                          ["epoch", "global_step", "train_loss", "learning_rate", "seconds_per_step"]}}), flush=True)
        self.previous, self.last_step = now, step

    def on_train_end(self, trainer, module):
        if trainer.is_global_zero and self.state == "running":
            self.state = "finished"
            self.history.status("finished", global_step=int(trainer.global_step), epoch=self.observed_epoch,
                                segment_elapsed_seconds=time.perf_counter() - self.started)

    def on_exception(self, trainer, module, exception):
        self.state = "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
        if trainer.is_global_zero:
            self.history.status(self.state, global_step=int(trainer.global_step),
                                epoch=int(trainer.current_epoch) + 1, error_type=type(exception).__name__,
                                error=str(exception))


def runtime_metadata(root):
    versions = {"python": platform.python_version(), "torch": torch.__version__,
                "lightning": pl.__version__, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version()}
    for distribution in ["nemo_toolkit", "sentencepiece", "torchaudio"]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                  capture_output=True, text=True, timeout=5)
        git_commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        git_commit = None
    properties = torch.cuda.get_device_properties(0)
    return {"versions": versions, "git_commit": git_commit,
            "gpu": {"name": properties.name, "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor]},
            "cpu_count": os.cpu_count(), "torch_num_threads": torch.get_num_threads()}


def stress_manifest(root):
    """Read metadata only; create one batch of the longest training recordings."""
    import heapq
    rows = (json.loads(line) for line in (root / "data" / "train.jsonl").open())
    selected = heapq.nlargest(16, rows, key=lambda r: (r["duration"], len(r["text"]), r["id"]))
    path = root / "data" / "benchmark_stress.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected), encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True, choices=["benchmark", "baseline", "mixed", "noise"])
    ap.add_argument("--benchmark-kind", choices=["clean", "noise", "stress"], default="clean")
    ap.add_argument("--steps", type=int, default=110)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--run-training", action="store_true")
    initialization = ap.add_mutually_exclusive_group()
    initialization.add_argument("--init-from", type=Path, help="Previous stage's .ckpt; preserve weights and start this stage's optimizer")
    initialization.add_argument("--resume", type=Path, help="Same-phase .ckpt; restore model, optimizer, scheduler and epoch")
    args = ap.parse_args()
    root = args.output_root.resolve()
    benchmark = args.phase == "benchmark"
    if not benchmark and not args.run_training:
        ap.error("Full training is disabled unless --run-training is explicitly supplied")
    if benchmark and (args.steps > 500 or args.steps < 1 or not 0 <= args.warmup < args.steps):
        ap.error("Benchmark must have 1..500 steps and 0 <= warmup < steps")
    if args.phase in ["mixed", "noise"] and args.init_from is None and args.resume is None:
        ap.error("Continuation stages require --init-from; interrupted stages require --resume")
    if args.phase in ["benchmark", "baseline"] and args.init_from:
        ap.error("Baseline/benchmark initialize from scratch")
    if benchmark and args.resume:
        ap.error("Benchmark cannot resume a training checkpoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    register_fan_noise()
    config_phase = "noise" if benchmark and args.benchmark_kind == "noise" else ("baseline" if benchmark else args.phase)
    cfg = OmegaConf.load(root / "configs" / (config_phase + ".yaml"))
    pl.seed_everything(int(cfg.seed), workers=True)
    callback = BenchmarkChecks(args.warmup if args.benchmark_kind != "stress" else 0) if benchmark else None
    if benchmark:
        with open_dict(cfg):
            path = stress_manifest(root) if args.benchmark_kind == "stress" else root / "data" / "benchmark_train.jsonl"
            cfg.model.train_ds.manifest_filepath = str(path)
            cfg.model.train_ds.shuffle = args.benchmark_kind != "stress"
            cfg.model.validation_ds.manifest_filepath = str(root / "data" / "benchmark_val_selection.jsonl")
            if not Path(cfg.model.validation_ds.manifest_filepath).exists():
                cfg.model.validation_ds.manifest_filepath = str(root / "data" / "benchmark_val.jsonl")
            cfg.model.optim.lr = 0.0003
            cfg.model.optim.sched.warmup_steps = 10 if args.benchmark_kind != "stress" else 0
            cfg.trainer.max_epochs = 1
            cfg.trainer.max_steps = 1 if args.benchmark_kind == "stress" else args.steps
            cfg.trainer.limit_train_batches = cfg.trainer.max_steps
            cfg.trainer.limit_val_batches = 0
    tokenizer_sha256 = file_sha256(root / "tokenizer112/tokenizer.model")
    config_sha256 = configuration_sha256(OmegaConf.to_container(cfg, resolve=True))
    if args.resume:
        if args.resume.suffix != ".ckpt":
            raise ValueError("Resume requires a trusted project .ckpt")
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(state.get("proj1"), tokenizer_sha256=tokenizer_sha256,
                                     phase=args.phase, config_sha256=config_sha256)
        del state
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"benchmark_{args.benchmark_kind}_{stamp}" if benchmark else f"{args.phase}_{stamp}"
    if args.resume:
        run_dir = resume_run_directory(args.resume, root)
    else:
        run_dir = root / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    callbacks = [callback] if benchmark else [CheckpointIdentity(tokenizer_sha256, args.phase, config_sha256), TrainingProgress(run_dir, stamp), AuditedModelCheckpoint(
        run_dir=run_dir,
        dirpath=str(run_dir / "checkpoints"), monitor=None,
        save_top_k=-1, save_last=True, filename="epoch={epoch:02d}-step={step}",
        every_n_epochs=1, save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )]
    logger = TensorBoardLogger(save_dir=str(run_dir), name="tensorboard", version="")
    if not benchmark:
        logger = [logger, CSVLogger(save_dir=str(run_dir), name="csv", version=stamp,
                                   flush_logs_every_n_steps=50)]
    started = time.perf_counter()
    trainer = pl.Trainer(**trainer_kwargs, callbacks=callbacks, logger=logger,
                         enable_checkpointing=not benchmark, default_root_dir=str(run_dir))
    model = EncDecHybridRNNTCTCBPEModel(cfg=cfg.model, trainer=trainer)
    if model.tokenizer.vocab_size != 112:
        raise RuntimeError(f"Expected 112-token vocabulary, got {model.tokenizer.vocab_size}")
    if args.init_from:
        if args.init_from.suffix != ".ckpt":
            raise ValueError("Use a trusted .ckpt produced by this project's previous stage")
        state = torch.load(args.init_from, map_location="cpu", weights_only=False)
        validate_checkpoint_identity(state.get("proj1"), tokenizer_sha256=tokenizer_sha256,
                                     phase=initialization_source_phase(args.phase))
        model.load_state_dict(state["state_dict"], strict=True)
        del state
    metadata = {"phase": args.phase, "seed": int(cfg.seed), "tokenizer_sha256": tokenizer_sha256,
                "config_sha256": config_sha256,
                "init_checkpoint": str(args.init_from.resolve()) if args.init_from else None,
                "init_checkpoint_sha256": file_sha256(args.init_from) if args.init_from else None,
                "resume_checkpoint": str(args.resume.resolve()) if args.resume else None,
                "resume_checkpoint_sha256": file_sha256(args.resume) if args.resume else None,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": runtime_metadata(root),
                "optimizer_policy": "restore optimizer and scheduler state" if args.resume else
                                    "fresh optimizer and scheduler for this stage",
                "max_epochs": int(cfg.trainer.max_epochs),
                "mel_bins": int(cfg.model.preprocessor.features)}
    metadata_name = f"resume_metadata_{stamp}.json" if args.resume else "run_metadata.json"
    (run_dir / metadata_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    initialization_seconds = time.perf_counter() - started
    print(json.dumps({"run": str(run_dir), "phase": args.phase,
                      "steps_cap": cfg.trainer.get("max_steps"), "vocab_size": model.tokenizer.vocab_size,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "gpu": torch.cuda.get_device_name(0)}), flush=True)
    fit_started = time.perf_counter()
    trainer.fit(model, ckpt_path=str(args.resume.resolve()) if args.resume else None)
    fit_seconds = time.perf_counter() - fit_started
    if benchmark:
        if int(trainer.global_step) != int(cfg.trainer.max_steps):
            raise RuntimeError(f"Expected {cfg.trainer.max_steps} optimizer steps, got {trainer.global_step}")
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        trainer.limit_val_batches = 4
        torch.cuda.synchronize()
        val_started = time.perf_counter()
        validation = trainer.validate(model, verbose=False)
        torch.cuda.synchronize()
        validation_seconds = time.perf_counter() - val_started
        metrics = validation[0]
        for name in ["val_wer", "val_wer_ctc"]:
            if name not in metrics or not math.isfinite(float(metrics[name])):
                raise RuntimeError(f"Missing/non-finite validation metric: {name}")
        timings = sorted(callback.timings)
        report = {
            "status": "passed", "benchmark_kind": args.benchmark_kind,
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
            "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "precision": cfg.trainer.precision, "batch_size": cfg.model.train_ds.batch_size,
            "vocab_size": model.tokenizer.vocab_size, "optimizer_steps": int(trainer.global_step),
            "warmup_steps_excluded": callback.warmup, "measured_steps": len(timings),
            "mean_seconds_per_step": statistics.mean(timings),
            "median_seconds_per_step": statistics.median(timings),
            "p90_seconds_per_step": timings[min(len(timings)-1, math.ceil(.9 * len(timings))-1)],
            "measured_total_seconds": sum(timings), "initialization_seconds": initialization_seconds,
            "fit_seconds": fit_seconds, "peak_allocated_gib": peak_allocated / 2**30,
            "peak_reserved_gib": peak_reserved / 2**30, "first_loss": callback.losses[0],
            "last_loss": callback.losses[-1], "gradients_finite_nonzero": callback.gradients_checked,
            "encoder_parameters_changed": callback.parameters_changed,
            "validation_batches": 4, "validation_seconds": validation_seconds, "validation_metrics": metrics,
            "timing_scope": "CUDA-synchronized consecutive batch ends, including loader waiting; excludes first warmup steps",
            "scheduler_note": "Benchmark-only 10-step LR warmup tests peak LR; production warmup and horizon remain unchanged.",
            "accuracy_note": "Untrained-model smoke test; WER is not experiment quality.",
            "full_training_started": False, "benchmark_weights_saved": False,
        }
        out = root / "reports" / f"benchmark_{args.benchmark_kind}.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
