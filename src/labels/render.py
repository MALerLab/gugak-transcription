"""Stage 4: render MIDI with General MIDI stand-ins and build listening clips.

Each clip is three WAVs of the same 30 s: the real stem (or matched-stem mix), the render
of the MIDI, and both layered (render loudness-matched to the real audio).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from src.data.audio import rms


def render_midi(midi_path: Path, wav_path: Path, config) -> None:
    """MIDI → stereo WAV with fluidsynth (-n no MIDI input, -i no shell, -q quiet, -F file out)."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["fluidsynth", "-niq", "-g", str(config.fluidsynth_gain), "-r", str(config.sample_rate),
                    "-F", str(wav_path), os.path.expanduser(config.soundfont), str(midi_path)],
                   check=True, stdout=subprocess.DEVNULL)


def excerpt(audio: np.ndarray, sample_rate: int, start_seconds: float, seconds: float) -> np.ndarray:
    """``seconds`` of stereo audio from ``start_seconds``, zero-padded if the source is shorter."""
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    start, length = int(start_seconds * sample_rate), int(seconds * sample_rate)
    piece = audio[start:start + length]
    return np.pad(piece, ((0, length - len(piece)), (0, 0)))


def write_clip_set(real: np.ndarray, rendered: np.ndarray, sample_rate: int, out_stem: Path,
                   config) -> dict[str, Path]:
    """Write ``<out_stem>_real.wav``, ``_render.wav``, ``_layered.wav``; returns the paths."""
    rendered = rendered * (config.layer_render_to_real_rms * rms(real) / max(rms(rendered), 1e-9))
    layered = real + rendered
    peak = max(np.abs(real).max(), np.abs(rendered).max(), np.abs(layered).max(), 1e-9)
    # Same gain on all three so levels stay comparable: real audio to the target loudness,
    # unless that would clip
    gain = min(config.target_rms / max(rms(real), 1e-9), config.peak_ceiling / peak)
    paths = {}
    for name, audio in (("real", real), ("render", rendered), ("layered", layered)):
        path = out_stem.parent / f"{out_stem.name}_{name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), (audio * gain).astype(np.float32), sample_rate, subtype="PCM_16")
        paths[name] = path
    return paths
