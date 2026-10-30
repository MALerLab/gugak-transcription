"""YIN f0 tracker (de Cheveigné & Kawahara 2002): an algorithm, no model. Strand A's alignment check."""
from __future__ import annotations

import numpy as np


def yin_track(mono: np.ndarray, sample_rate: int, config) -> dict[str, np.ndarray]:
    """YIN on mono audio, as an f0 track.

    Args:
        mono: mono float audio.
        sample_rate: its rate.
        config: the `configs/features/f0/yin.yaml` settings.

    Returns:
        dict with ``time`` (s), ``f0_hz`` and ``confidence``, one value per hop. YIN has no
        confidence: its ``f0_hz`` is NaN where unvoiced, and ``confidence`` is 1 where voiced.
    """
    f0_hz, time = yin_f0(mono, sample_rate, config)
    return {"time": time, "f0_hz": f0_hz, "confidence": np.isfinite(f0_hz).astype(np.float32)}


def yin_f0(audio: np.ndarray, sample_rate: int, config) -> tuple[np.ndarray, np.ndarray]:
    """Frame-wise f0 in Hz (NaN when unvoiced) and frame times, YIN with FFT correlation.

    Args:
        audio: mono signal.
        sample_rate: its rate.
        config: the `configs/features/f0/yin.yaml` settings.
    """
    window = int(config.window_seconds * sample_rate)
    hop = int(config.hop_seconds * sample_rate)
    tau_min = int(sample_rate / config.fmax_hz)
    tau_max = int(sample_rate / config.fmin_hz)
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
        below = cmnd < config.threshold
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
