"""Tuning: how far a recording sits from the A440 semitone grid, three ways.

- ``tuning_offset_cents``: from the audio itself (synctoolbox filterbank); strand A's chroma.
- ``estimate_tuning_offset``: from an f0 curve (histogram of cents mod 100); note-ify.
- ``octave_and_tuning``: from f0 measured inside known notes vs their written pitch; also
  gives the written → sounding shift; strand A's alignment check.
"""
from __future__ import annotations

import numpy as np
from synctoolbox.feature.utils import estimate_tuning

from src.features.chroma import FEATURE_SAMPLE_RATE


def tuning_offset_cents(audio_22k: np.ndarray) -> int:
    """Global tuning deviation from A440 equal temperament, in cents (−50..50)."""
    return int(estimate_tuning(audio_22k, FEATURE_SAMPLE_RATE))


def hz_to_cents_from_a440(f0_hz: np.ndarray) -> np.ndarray:
    """Pitch in cents relative to A4 = 440 Hz."""
    return 1200 * np.log2(np.maximum(f0_hz, 1e-3) / 440.0)


def estimate_tuning_offset(cents: np.ndarray, voiced: np.ndarray, config) -> float:
    """The recording's offset from the A440 semitone grid, in cents, in [-50, 50).

    Takes stable voiced frames, folds their pitch onto one semitone (cents mod 100), and
    returns the peak of the circularly smoothed histogram — the grid the player sits on.
    ``config``: the `configs/features/tuning.yaml` settings.
    """
    slope = np.abs(np.gradient(cents))
    stable = voiced & (slope < config.stable_slope_cents_per_frame)
    if stable.sum() < 10:
        return 0.0
    folded = np.mod(cents[stable], 100.0)
    histogram = np.bincount(np.floor(folded).astype(int) % 100, minlength=100).astype(float)
    kernel_offsets = np.arange(-50, 50)
    kernel = np.exp(-0.5 * (kernel_offsets / config.smoothing_cents) ** 2)
    # circular convolution: the fold wraps, so 99 cents sits next to 0 cents
    smoothed = np.real(np.fft.ifft(np.fft.fft(histogram) * np.fft.fft(np.roll(kernel, -50))))
    peak = float(np.argmax(smoothed)) + 0.5
    return peak - 100.0 if peak >= 50.0 else peak


def octave_and_tuning(written_pitch: np.ndarray, measured_midi: np.ndarray) -> tuple[int, float]:
    """Whole-semitone offset (sounding − written, usually a multiple of 12) and tuning in cents.

    Only notes that are close to a whole-semitone offset from the mode vote on the tuning.
    """
    deviation = measured_midi - written_pitch
    deviation = deviation[~np.isnan(deviation)]
    if len(deviation) == 0:
        return 0, 0.0
    semitones = np.round(deviation).astype(int)
    values, counts = np.unique(semitones, return_counts=True)
    shift = int(values[np.argmax(counts)])
    near = deviation[np.abs(deviation - shift) < 0.5]
    return shift, float(np.median(near - shift) * 100)
