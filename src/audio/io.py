"""Audio reading: WAV → float arrays, mono downmix, resampling, summing stems.

``read_audio`` and ``to_mono`` are copied unchanged from the gugak stem separation repo
(`src/data/audio.py`), so stems are read the same way in both repos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def read_audio(path, start: int = 0, frames: int = -1, dtype: str = "float32"):
    """Read a WAV as ``(frames, channels)`` float array + sample rate (always 2-D)."""
    audio, sr = sf.read(str(path), start=start, frames=frames, dtype=dtype, always_2d=True)
    return audio, sr


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix ``(frames, channels)`` -> mono ``(frames,)`` by channel average."""
    return audio.mean(axis=1)


def load_mono_resampled(audio_path: str | Path, target_sample_rate: int,
                        start_seconds: float = 0.0, duration_seconds: float | None = None) -> np.ndarray:
    """Read a WAV (or an excerpt of it), downmix to mono and resample (torchaudio).

    Args:
        audio_path: WAV file.
        target_sample_rate: output sample rate.
        start_seconds: excerpt start.
        duration_seconds: excerpt length; None reads to the end.
    """
    import torch  # imported here so reading audio elsewhere doesn't pull in torch
    import torchaudio
    source_sample_rate = sf.info(str(audio_path)).samplerate
    start_frame = int(round(start_seconds * source_sample_rate))
    num_frames = -1 if duration_seconds is None else int(round(duration_seconds * source_sample_rate))
    audio, source_sample_rate = read_audio(audio_path, start=start_frame, frames=num_frames)
    mono = torch.from_numpy(to_mono(audio)).float()
    if source_sample_rate != target_sample_rate:
        mono = torchaudio.functional.resample(mono, source_sample_rate, target_sample_rate)
    return mono.numpy()


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
