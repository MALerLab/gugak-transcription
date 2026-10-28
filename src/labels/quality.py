"""Alignment-quality numbers: does each aligned note line up with what the stem plays?

- **pitch hit rate**: the stem's f0 (YIN) in the middle of a note is within half a semitone
  of the note's pitch. Monophonic stems only make sense here (대금, 피리, 해금, 아쟁);
  plucked 가야금/거문고 are reported too, with that caveat.
- **onset F-measure**: note onsets vs spectral-flux onsets detected in the stem.

Every number is reported next to the same notes shifted in time (chance level), and next to
the linear (unaligned) baseline, so "0.6" can be read as better or worse than nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter


def yin_f0(audio: np.ndarray, sample_rate: int, config) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise f0 in Hz (NaN when unvoiced) and frame times, YIN with FFT correlation.

    Args:
        audio: mono signal.
        sample_rate: its rate.
        config: quality config (window, hop, f0 range, threshold).
    """
    window = int(config.yin_window_seconds * sample_rate)
    hop = int(config.yin_hop_seconds * sample_rate)
    tau_min = int(sample_rate / config.yin_fmax_hz)
    tau_max = int(sample_rate / config.yin_fmin_hz)
    frame_length = window + tau_max
    num_frames = max(0, (len(audio) - frame_length) // hop + 1)
    f0 = np.full(num_frames, np.nan)
    fft_size = 1 << int(np.ceil(np.log2(frame_length + window)))

    for chunk_start in range(0, num_frames, 2000):  # chunks keep memory flat
        idx = np.arange(chunk_start, min(num_frames, chunk_start + 2000))
        frames = np.lib.stride_tricks.sliding_window_view(audio, frame_length)[idx * hop]
        frames = frames - frames.mean(axis=1, keepdims=True)
        # cross term r(τ) = Σ_j x_j x_{j+τ}, j < window
        spectrum_full = np.fft.rfft(frames, fft_size)
        spectrum_head = np.fft.rfft(frames[:, :window], fft_size)
        cross = np.fft.irfft(np.conj(spectrum_head) * spectrum_full, fft_size)[:, :tau_max + 1]
        squares = np.concatenate([np.zeros((len(idx), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
        energy_head = squares[:, window:window + 1]
        tau = np.arange(tau_max + 1)
        energy_shifted = squares[:, tau + window] - squares[:, tau]
        difference = np.maximum(energy_head + energy_shifted - 2 * cross, 0.0)
        # cumulative-mean-normalised difference
        cmnd = np.ones_like(difference)
        cumulative = np.cumsum(difference[:, 1:], axis=1)
        cmnd[:, 1:] = difference[:, 1:] * tau[1:] / np.maximum(cumulative, 1e-12)
        cmnd[:, :tau_min] = 1.0
        # first dip below threshold, then walk to its local minimum
        below = cmnd < config.yin_threshold
        has_dip = below.any(axis=1)
        first = np.argmax(below, axis=1)
        for row in np.nonzero(has_dip)[0]:
            t = first[row]
            while t + 1 <= tau_max and cmnd[row, t + 1] < cmnd[row, t]:
                t += 1
            if 1 <= t < tau_max:  # parabolic interpolation around the dip
                a, b, c = cmnd[row, t - 1], cmnd[row, t], cmnd[row, t + 1]
                denominator = a - 2 * b + c
                t = t + 0.5 * (a - c) / denominator if denominator > 0 else t
            f0[idx[row]] = sample_rate / t
    times = (np.arange(num_frames) * hop + window / 2) / sample_rate
    return f0, times


def note_f0_midi(timed: pd.DataFrame, f0: np.ndarray, times: np.ndarray, config) -> np.ndarray:
    """Median f0 (as fractional MIDI) in the core of each note; NaN where too short/unvoiced."""
    f0_midi = 69 + 12 * np.log2(f0 / 440.0)
    low, high = config.note_core_fraction
    out = np.full(len(timed), np.nan)
    for i, (start, duration) in enumerate(zip(timed["start"].to_numpy(), timed["duration"].to_numpy())):
        if duration < config.min_note_seconds:
            continue
        lo, hi = np.searchsorted(times, [start + low * duration, start + high * duration])
        values = f0_midi[lo:hi]
        values = values[~np.isnan(values)]
        if len(values) >= config.min_voiced_frames:
            out[i] = np.median(values)
    return out


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


def pitch_hit_rate(written_pitch: np.ndarray, measured_midi: np.ndarray, shift: int,
                   tuning_cents: float, tolerance: float) -> tuple[float, int]:
    """Share of checkable notes whose f0 is within ``tolerance`` semitones; and how many were checkable."""
    deviation = measured_midi - (written_pitch + shift + tuning_cents / 100)
    checkable = ~np.isnan(deviation)
    if checkable.sum() == 0:
        return np.nan, 0
    return float((np.abs(deviation[checkable]) < tolerance).mean()), int(checkable.sum())


def spectral_flux_onsets(audio: np.ndarray, sample_rate: int, config) -> np.ndarray:
    """Onset times (s): peaks of half-wave-rectified log-magnitude flux above a moving median."""
    hop = int(config.onset_hop_seconds * sample_rate)
    window = config.onset_window
    if len(audio) < window:
        return np.array([])
    frames = np.lib.stride_tricks.sliding_window_view(audio, window)[::hop] * np.hanning(window)
    magnitude = np.log1p(100 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.maximum(np.diff(magnitude, axis=0), 0).sum(axis=1)
    flux = flux / max(flux.max(), 1e-12)
    threshold = median_filter(flux, size=max(3, int(config.onset_median_seconds / config.onset_hop_seconds))) + config.onset_delta
    is_peak = (flux > threshold) & (flux >= np.roll(flux, 1)) & (flux >= np.roll(flux, -1))
    peaks = np.nonzero(is_peak)[0]
    min_gap = int(config.onset_min_gap_seconds / config.onset_hop_seconds)
    kept = [p for i, p in enumerate(peaks) if i == 0 or p - peaks[i - 1] >= min_gap]
    return (np.array(kept) + 1) * hop / sample_rate + window / (2 * sample_rate)


def onset_f_measure(reference: np.ndarray, estimated: np.ndarray, tolerance: float) -> float:
    """Greedy one-to-one onset matching within ``tolerance``; F-measure."""
    if len(reference) == 0 or len(estimated) == 0:
        return np.nan
    reference, estimated = np.sort(reference), np.sort(estimated)
    used = np.zeros(len(estimated), dtype=bool)
    hits = 0
    for onset in reference:
        lo, hi = np.searchsorted(estimated, [onset - tolerance, onset + tolerance])
        free = [j for j in range(lo, hi) if not used[j]]
        if free:
            used[min(free, key=lambda j: abs(estimated[j] - onset))] = True
            hits += 1
    precision, recall = hits / len(estimated), hits / len(reference)
    return 2 * precision * recall / (precision + recall) if hits else 0.0
