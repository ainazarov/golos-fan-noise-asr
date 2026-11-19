"""Identical, explicit SNR mixing for online training and fixed evaluation."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def mix_at_snr(speech, noise, snr_db, offset=0):
    """Mix mono signals, wrapping short noise; SNR uses whole-clip RMS.

    Golos originals already contain background noise, so SNR is relative to
    the original waveform, not an unavailable isolated clean-speech signal.
    A common gain on speech and noise prevents clipping without changing SNR.
    """
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    if speech.ndim != 1 or noise.ndim != 1 or not len(speech) or not len(noise):
        raise ValueError("Expected nonempty mono speech and noise")
    if not np.isfinite(speech).all() or not np.isfinite(noise).all():
        raise ValueError("Non-finite audio")
    segment = noise[(np.arange(len(speech)) + int(offset)) % len(noise)]
    speech_rms = float(np.sqrt(np.mean(speech ** 2)))
    noise_rms = float(np.sqrt(np.mean(segment ** 2)))
    if noise_rms < 1e-10:
        raise ValueError("Noise recording is silent")
    if speech_rms < 1e-10:
        return speech.astype(np.float32), {"skipped_silent_original": True}
    noise_gain = speech_rms / (noise_rms * 10 ** (float(snr_db) / 20))
    mixed = speech + segment * noise_gain
    common_gain = min(1.0, 0.99 / max(float(np.max(np.abs(mixed))), 1e-10))
    mixed *= common_gain
    return mixed.astype(np.float32), {
        "requested_snr_db": float(snr_db), "noise_gain": noise_gain,
        "common_gain": common_gain, "noise_offset_samples": int(offset),
        "skipped_silent_original": False,
    }


def register_fan_noise():
    """Register once in each training process; worker RNGs are Lightning-seeded."""
    from nemo.collections.asr.parts.preprocessing.perturb import (
        Perturbation, perturbation_types, register_perturbation,
    )
    if "proj1_fan_noise" in perturbation_types:
        return

    class FanNoise(Perturbation):
        def __init__(self, manifest_path, min_snr_db=5.0, max_snr_db=20.0):
            self.files = [json.loads(x)["audio_filepath"]
                          for x in Path(manifest_path).read_text().splitlines() if x.strip()]
            if not self.files:
                raise ValueError("Empty noise manifest")
            self.min_snr_db, self.max_snr_db = min_snr_db, max_snr_db

        def perturb(self, data):
            noise, sr = sf.read(random.choice(self.files), dtype="float32")
            if sr != data.sample_rate or noise.ndim != 1:
                raise ValueError("Fan recordings must match mono speech sampling rate")
            data._samples, _ = mix_at_snr(
                data._samples, noise, random.uniform(self.min_snr_db, self.max_snr_db),
                random.randrange(len(noise)),
            )

        def max_augmentation_length(self, length):
            return length

    register_perturbation("proj1_fan_noise", FanNoise)
