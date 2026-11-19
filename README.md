# Fan-Noise Robustness in Russian Speech Recognition

Russian speech recognition with FastConformer TDT/CTC: train from scratch on
Golos, continue on Golos + Sova, then adapt to fan noise.

Server project: `/tf/ai_nazarov/proj1`. No course checkpoint or external ASR
weights initialize the experiment.

## Experiment

One model uses tokenizer **112** and **128 mel bins** throughout three stages:

| Stage | Epochs | Training speech | Validation speech | Peak learning rate |
|---|---:|---|---|---:|
| `baseline` | 10 | Original Golos | Original Golos development set | 3e-4 |
| `mixed` | 25 | Original Golos + Sova | Original Golos + Sova development sets | 1e-4 |
| `noise` | 5 | Golos + Sova, with online fan augmentation | Original and fixed-10-dB mixed development sets | 3e-5 |

This is **40 epochs of computation**. The mixed stage initializes from the final
baseline checkpoint; the noise stage initializes from the final mixed checkpoint.
Weights continue between stages. Each new stage starts a fresh optimizer and
learning-rate schedule. There is no additional clean-control branch.

The model is the course FastConformer hybrid TDT/CTC: 17 encoder layers, width
512, 16 kHz audio, 128 mel bins, and auxiliary CTC loss weight 0.3. SpecAugment
stays enabled in every training stage. All stages use AdamW, batch 16, FP32,
one GPU, gradient clipping at 1.0, and cosine decay to 1e-6. Warmup is 5,000
optimizer steps for the baseline and 1,000 steps for each continuation stage.

The SentencePiece BPE tokenizer has **112 total entries including `<unk>`** and
maximum piece length 4. It was trained only on the Golos training transcripts
and remains fixed for all three stages. Other tokenizer folders contain separate
tokenizer artifacts; these are not substituted into the shared model.

## Speech data and splits

- Golos training recordings must have nonempty transcripts and durations from
  0.3 through 20 seconds. Text normalization collapses whitespace and preserves
  other characters.
- Golos development data contains approximately 2 hours: 1.8 hours crowd and
  0.2 hours farfield, reserved deterministically with seed 112 and SHA256 ordering.
  The remaining eligible recordings form the Golos training set.
- Official Golos crowd/farfield tests are untouched by training. Their positive
  durations and empty references are retained; predictions on empty references
  count as insertions in corpus WER.
- Sova comes from `/share/sova/sova3_man.json`. Its stale source paths are rebased
  to `/share/sova/`. The same duration and nonempty-text filters apply. Sova text
  also maps `ё` to `е` consistently across train/development/test, matching the
  existing tokenizer's character inventory. Approximately 2 hours each are
  reserved for Sova development and test data before mixing training corpora.
- A mixed epoch is one shuffled pass through the combined Golos and Sova training
  recordings. Neither corpus is oversampled. Development and test recordings
  are excluded from this manifest.
- Splits are recording-disjoint. Speaker IDs are unavailable, so speaker
  disjointness is not claimed. Preparation audits describe duplicate checks,
  file checks, exact counts, durations and exclusions.

Validation runs every **five completed epochs**, including the end of each
stage. It is independent of checkpoint saving. Baseline WER uses original Golos
development audio. Mixed-stage WER is the corpus WER of the concatenated Golos
and Sova development sets, weighted by their reference-word counts.

Noise-stage validation contains each mixed development recording twice: in its
original form and with fixed 10 dB fan noise. With identical references, its
aggregate WER equals the mean of the original and noisy mixed corpus WERs.
Separate corpus/noise-condition results can be obtained by evaluating saved
checkpoints against the corresponding individual manifests.

## Noise source and scope

Noise is applied online to approximately **50%** of training examples during
the final stage. SNR is sampled uniformly from **5–20 dB**, with a random noise
recording and offset on every presentation. Whole-clip RMS of the original
speech defines SNR. Short noise is wrapped; a common gain prevents clipping.
Online augmentation does not double the epoch size. Original recordings may
already contain environmental noise.

The source is real **industrial fan recordings with existing factory ambience**
from the [DCASE 2020/MIMII development dataset](https://zenodo.org/records/3678171).
License: **CC BY-NC-SA 4.0**. Retain attribution when sharing artifacts.

Only `fan/train/normal_*.wav` is used. The official archive size and MD5 are
verified before extraction. Splits use physical machine IDs:

| Our split | Source machine IDs | Clips |
|---|---|---:|
| Train | 00, 02 | 1,827 |
| Development | 04 | 933 |
| Test | 06 | 915 |

The same fan machine is not reused across these splits. Public metadata does
not establish independent background/session recordings. Results concern this
industrial-fan family, not arbitrary noise or all household HVAC conditions.

## Preparation

Run commands from `/tf/ai_nazarov/proj1` in the existing course Python environment.
Do not silently upgrade the shared environment.

For a new workspace:

```bash
python -B prepare_data.py
python -B prepare_sova.py
python -B prepare_noise.py
python -B configure.py
python -B prepare_eval.py --split validation --speech-set mixed
python -B -m unittest discover -s tests -v
```

Preparation scripts protect their primary outputs; `--overwrite` is explicit.
The populated project already contains tokenizer artifacts and configurations,
so do not rerun `configure.py` over them during training. Speech manifests
reference shared WAVs without copying them. Noise and fixed evaluation mixtures
are stored under the project and excluded from Git.

## Training and checkpoints

Run the complete fresh experiment:

```bash
python -B run_experiment.py --run-training
```

The order is **baseline 10 → mixed 25 → noise 5**. The pipeline stops if a stage
fails and does not automatically evaluate official test sets. The fresh command
does not resume an interrupted experiment.

Individual stages can also be launched explicitly:

```bash
python -B run_stage.py --phase baseline --run-training
python -B run_stage.py --phase mixed --run-training --init-from /absolute/path/to/baseline-final.ckpt
python -B run_stage.py --phase noise --run-training --init-from /absolute/path/to/mixed-final.ckpt
```

Full training requires `--run-training`. Preparation and bounded benchmarks do
not launch the full experiment. Run logs record actual launch and progress
independently of this README.

Every completed epoch is permanently saved, and each stage also maintains
`checkpoints/last.ckpt`. The 40-epoch experiment therefore retains 40 epoch
checkpoints plus one latest checkpoint per stage. These contain model, optimizer,
scheduler and training progress. Plan roughly **55 GiB** for checkpoints; actual
sizes are recorded in checkpoint history. Checkpoints and logs stay on the
server and are excluded from Git.

Each run under `runs/` contains:

- `tensorboard/`: training loss, learning rate and available validation metrics.
- `csv/`: training/validation metrics, with a fresh segment on resume.
- `progress/`: CSV/JSONL segments containing losses, learning rate, throughput,
  elapsed time and GPU memory every 50 optimizer steps.
- `progress_status.json`: the latest recorded progress snapshot.
- `checkpoint_history.csv` and `.jsonl`: every saved epoch's filename, size and
  global step, including epochs without validation.
- `validation_history.csv` and `.jsonl`: only evaluated checkpoints and their
  TDT/CTC WER. Earlier validation metrics are not reused for later epochs.
- `evaluation/`: test results, created only by explicit evaluation.

Histories use one-based completed epochs; checkpoint filenames use Lightning's
zero-based epoch index. Logging adds no extra model inference or explicit GPU
synchronization, though its I/O cost is not zero. Sampled training-batch WER is
separate from full development-set validation. Raw transcripts are not printed
during training.

Stopping midway through an epoch loses work after the last saved checkpoint.
Before the first epoch completes, there is no resumable epoch checkpoint.
To resume the same stage:

```bash
python -B run_stage.py --phase mixed --run-training --resume /tf/ai_nazarov/proj1/runs/mixed_TIMESTAMP/checkpoints/last.ckpt
```

Use the matching phase name and its own checkpoint. Resume restores model,
optimizer, scheduler and epoch state in the original run directory, and checks
the configuration and tokenizer identities. `--resume` and `--init-from` are
mutually exclusive. The latter starts a new continuation stage with fresh
optimizer state. An old 80-mel checkpoint is not compatible with the 128-mel
frontend.

## Evaluation after training

No full test sweep runs automatically. All saved checkpoints remain available
for later evaluation.

```bash
python -B run_stage.py --evaluate-run /tf/ai_nazarov/proj1/runs/noise_TIMESTAMP
```

This evaluates the lowest recorded development-WER checkpoint and the final
checkpoint, deduplicating them if they are the same file. It covers original
Golos crowd/farfield audio and fixed fan noise at 5, 10 and 20 dB. When the Sova
test manifest exists, it also evaluates Sova under those four conditions, for
12 conditions in total. Missing fixed-noise test manifests are generated on
demand. Test results do not select checkpoints.

To prepare all fixed test conditions separately:

```bash
python -B prepare_eval.py --split test --speech-set mixed
```

`evaluate.py --checkpoint ... --manifest ... --output ...` supports individual
checkpoint/condition evaluation. It saves predictions and corpus WER with
substitution, deletion and insertion counts. Compare pre-noise and post-noise
checkpoints on identical manifests and decoding settings; report corpus and
SNR conditions separately.

## Verification and runtime

The Golos training split has **1,091,741 recordings / 1,212.876 hours**, with
**1,831 recordings / 2.001 hours** reserved for development. Sova has 203,080
eligible recordings / 274.712 hours before its development/test reservations.
Exact mixed-training totals are recorded by Sova preparation.

The 128-mel model was benchmarked on a Tesla V100 32 GB with batch 16 and FP32.
Each condition ran 500 optimizer steps, with 50 warmup steps excluded and
**450 timed steps**. The 8,192-recording samples preserve corpus/domain and
duration proportions. The mixed and noisy runs use the same speech sample.

| Condition | Mean seconds/step | Full-epoch estimate before Sova holdouts |
|---|---:|---:|
| Golos | 0.233796 | 4.431 hours |
| Golos + Sova | 0.241790 | 5.435 hours |
| Noisy Golos + Sova | 0.243098 | 5.465 hours |

All three checks passed finite-loss/gradient and encoder-update checks. A
separate batch of the longest recordings also passed, peaking at **12.47 GiB**
allocated GPU memory. These checks establish execution and throughput, not
recognition accuracy. Benchmark weights are not saved or reused.

Before Sova holdout reservations, **10 + 25 + 5 epochs** implied approximately
**207.52 training hours**, or **238.65 hours / 9.94 days** with a 15% allowance
for validation, saving and throughput variation. The current forecast uses
the final prepared split sizes instead of those preliminary totals.

`reports/benchmark128.json` stores the current timing measurements. Recalculate
the forecast for a specified start time using the final Golos/Sova data audits
and the three stage configurations:

```bash
python -B forecast.py --start 2026-11-19T16:00:00+03:00 --allowance 0.15
```

Omitting `--start` uses the current time; an explicit start must include its UTC
offset. The command writes `reports/runtime_forecast.json` with stage order,
epoch sizes, stage durations, and cumulative training-only and planning finish
times in Moscow. This is a conditional forecast, not a record that training has
started. It assumes uninterrupted GPU access and excludes comprehensive final
test evaluation. Its allowance is not a statistical guarantee.

The older `reports/benchmark_clean.json`, `benchmark_noise.json` and
`benchmark_stress.json` describe the previous 80-mel setup. They should not be
used to schedule the current experiment. The earlier checkpoint I/O probe also
used that model and measured approximately 1.264 GiB per full checkpoint.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Golos filtering, splits, audits and benchmark samples |
| `prepare_sova.py` | Sova splits and combined Golos/Sova manifests |
| `prepare_noise.py` | Verified fan download and machine-ID splits |
| `configure.py` | Training-only tokenizer and three stage configurations |
| `audio_utils.py` | SNR mixing and NeMo online augmentation |
| `prepare_eval.py` | Reproducible fixed-noise development/test sets |
| `train.py` | Guarded training phases and bounded benchmarks |
| `run_stage.py` | Run/resume one stage; explicitly evaluate saved checkpoints |
| `run_experiment.py` | Sequential 10 + 25 + 5 fresh experiment |
| `experiment_logging.py` | Progress, validation and checkpoint histories |
| `checkpoint_utils.py` | Configuration, phase and tokenizer identity checks |
| `evaluate.py` | Chunked transcription and corpus WER |
| `forecast.py` | Current 128-mel stage forecast using final prepared split sizes |
| `configs/` | Baseline, mixed and noise configurations |
| `reports/` | Preparation audits and verification reports |
| `tests/` | Split, augmentation, evaluation, pipeline and resume checks |
