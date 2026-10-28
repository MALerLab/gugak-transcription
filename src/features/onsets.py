"""Onset detection: where notes start, from the audio or from its energy curve.

- ``spectral_flux_onsets``: from the waveform; strand A's alignment check.
- ``onset_frames``: from a frame energy curve (rises in dB); note-ify splits repeated notes there.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def spectral_flux_onsets(audio: np.ndarray, sample_rate: int, config) -> np.ndarray:
    """Onset times (s): peaks of half-wave-rectified log-magnitude flux above a moving median.

    ``config``: the ``spectral_flux`` section of `configs/features/onsets.yaml`.
    """
    hop = int(config.hop_seconds * sample_rate)
    window = config.window
    if len(audio) < window:
        return np.array([])
    frames = np.lib.stride_tricks.sliding_window_view(audio, window)[::hop] * np.hanning(window)
    magnitude = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(np.diff(magnitude, axis=0), 0).sum(axis=1)
    flux = flux / max(flux.max(), 1e-12)
    threshold = median_filter(flux, size=max(3, int(config.median_seconds / config.hop_seconds))) + config.delta
    is_peak = (flux > threshold) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    peaks = np.nonzero(is_peak)[0]
    min_gap = int(config.min_gap_seconds / config.hop_seconds)
    kept = [p for i, p in enumerate(peaks) if i == 0 or p - peaks[i - 1] >= min_gap]
    return (np.array(kept) + 1) * hop / sample_rate + window / (2 * sample_rate)


def onset_frames(rms_db: np.ndarray, hop_seconds: float, config) -> np.ndarray:
    """Frames where the energy rises by ``rise_db`` within a short window (local peaks only).

    ``config``: the ``energy`` section of `configs/features/onsets.yaml`.
    """
    window = max(1, int(round(config.rise_window_seconds / hop_seconds)))
    floor = np.maximum(rms_db, -120.0)
    previous_min = np.array([floor[max(0, i - window):i].min() if i > 0 else floor[0] for i in range(len(floor))])
    rise = floor - previous_min
    min_gap = max(1, int(round(config.min_inter_onset_seconds / hop_seconds)))
    candidates = np.flatnonzero(rise >= config.rise_db)
    onsets = []
    for frame in candidates:
        if onsets and frame - onsets[-1] < min_gap:
            if rise[frame] > rise[onsets[-1]]:
                onsets[-1] = frame
            continue
        onsets.append(frame)
    # place the onset at the start of the rise, not its top
    return np.array([max(0, f - window // 2) for f in onsets], dtype=int)
