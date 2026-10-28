"""Alignment features for audio stems and timed note tables (synctoolbox wrappers).

Both sides get the same three features, so they can be compared frame by frame:
quantised chroma (what pitch classes sound), DLNCO (where notes start, per pitch class)
and a coarse CENS (smoothed chroma, for locating a section inside a whole score).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma, quantized_chroma_to_CENS
from synctoolbox.feature.csv_tools import df_to_pitch_features, df_to_pitch_onset_features
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning

from src.data.audio import read_audio, to_mono

FEATURE_SAMPLE_RATE = 22050  # synctoolbox's filterbank is designed for 22.05 kHz


def load_stem_mix(stem_paths: list[Path], start_seconds: float = 0.0,
                  duration_seconds: float | None = None) -> tuple[np.ndarray, int]:
    """Sum mono versions of several stems; returns (audio, sample rate) at the file rate.

    Stems are trimmed to the shortest one (never padded, per the dataset gotchas).
    """
    signals, sample_rate = [], None
    for path in stem_paths:
        audio, sample_rate = read_audio(path)
        start = int(start_seconds * sample_rate)
        stop = None if duration_seconds is None else start + int(duration_seconds * sample_rate)
        signals.append(to_mono(audio[start:stop]))
    length = min(len(s) for s in signals)
    return np.sum([s[:length] for s in signals], axis=0), sample_rate


def to_feature_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample to 22.05 kHz for synctoolbox (44.1 kHz → exact 1:2)."""
    if sample_rate == FEATURE_SAMPLE_RATE:
        return audio
    gcd = np.gcd(sample_rate, FEATURE_SAMPLE_RATE)
    return resample_poly(audio, FEATURE_SAMPLE_RATE // gcd, sample_rate // gcd)


def audio_features(audio_22k: np.ndarray, feature_rate: int, tuning_cents: int,
                   with_onsets: bool = True) -> dict[str, np.ndarray]:
    """Chroma (+ DLNCO) of a 22.05 kHz mono signal.

    Args:
        audio_22k: mono audio at 22.05 kHz.
        feature_rate: frames per second.
        tuning_cents: tuning deviation, shifts the filterbank.
        with_onsets: also compute DLNCO (slow-ish; not needed for locating).
    """
    pitch = audio_to_pitch_features(f_audio=audio_22k, Fs=FEATURE_SAMPLE_RATE,
                                    tuning_offset=tuning_cents, feature_rate=feature_rate)
    chroma = quantize_chroma(pitch_to_chroma(f_pitch=pitch))
    features = {"pitch": pitch, "chroma": chroma}
    if with_onsets:
        peaks = audio_to_pitch_onset_features(f_audio=audio_22k, Fs=FEATURE_SAMPLE_RATE,
                                              tuning_offset=tuning_cents)
        features["dlnco"] = pitch_onset_features_to_DLNCO(
            f_peaks=peaks, feature_rate=feature_rate, feature_sequence_length=chroma.shape[1])
    return features


def score_features(timed: pd.DataFrame, feature_rate: int, num_frames: int | None = None,
                   with_onsets: bool = True) -> dict[str, np.ndarray]:
    """Chroma (+ DLNCO) of a timed note table (``start``/``duration`` in seconds).

    Args:
        timed: note table with seconds.
        feature_rate: frames per second.
        num_frames: pad/crop to this many frames (defaults to the table's own length).
        with_onsets: also compute DLNCO.
    """
    table = pd.DataFrame({"start": timed["start"].to_numpy(), "duration": timed["duration"].to_numpy(),
                          "pitch": timed["pitch"].to_numpy(), "velocity": 80, "instrument": "pitched"})
    pitch = df_to_pitch_features(table, feature_rate=feature_rate)
    if num_frames is not None:
        pitch = _fit_frames(pitch, num_frames)
    chroma = quantize_chroma(pitch_to_chroma(f_pitch=pitch))
    features = {"pitch": pitch, "chroma": chroma}
    if with_onsets:
        peaks = df_to_pitch_onset_features(table)
        features["dlnco"] = pitch_onset_features_to_DLNCO(
            f_peaks=peaks, feature_rate=feature_rate, feature_sequence_length=chroma.shape[1])
    return features


def coarse_cens(chroma: np.ndarray, feature_rate: int, smooth_frames: int, downsample: int) -> np.ndarray:
    """Smoothed, downsampled chroma (CENS) for coarse matching; columns are unit-norm."""
    cens, _ = quantized_chroma_to_CENS(chroma, smooth_frames, downsample, feature_rate)
    return cens


def tuning_offset_cents(audio_22k: np.ndarray) -> int:
    """Global tuning deviation from A440 equal temperament, in cents (−50..50)."""
    return int(estimate_tuning(audio_22k, FEATURE_SAMPLE_RATE))


def _fit_frames(matrix: np.ndarray, num_frames: int) -> np.ndarray:
    """Crop or zero-pad the time axis of a (bins, frames) matrix."""
    if matrix.shape[1] >= num_frames:
        return matrix[:, :num_frames]
    return np.pad(matrix, ((0, 0), (0, num_frames - matrix.shape[1])))
