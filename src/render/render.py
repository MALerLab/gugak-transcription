"""Render MIDI with General MIDI stand-ins (fluidsynth) and build listening files.

Two ways to put a render next to the real audio:
- ``write_clip_set`` (strand A): real / render / layered WAVs, render loudness-matched and
  summed in the centre, one shared gain on all three.
- ``write_listening_triplet`` (strand B): render + layered WAVs, stem panned left and
  render right at equal audible loudness.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.io import read_audio
from src.audio.measure import rms


def render_midi(midi_path: str | Path, wav_path: str | Path, config) -> np.ndarray:
    """Render a MIDI file to a stereo WAV with fluidsynth; returns the audio, ``(frames, 2)``."""
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["fluidsynth", "-niq",                 # -n no MIDI input driver, -i no interactive shell, -q quiet
         "-g", str(config.fluidsynth_gain),    # master gain
         "-r", str(config.sample_rate),        # output sample rate
         "-F", str(wav_path),                  # render to this file instead of an audio device
         os.path.expanduser(config.soundfont), str(midi_path)],
        check=True, capture_output=True)
    audio, _ = read_audio(wav_path)
    return audio


# --- strand A: clip sets ----------------------------------------------------

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


# --- strand B: panned triplets ----------------------------------------------

def audible_rms(audio: np.ndarray, sample_rate: int, floor_db: float = -50.0) -> float:
    """RMS over 50 ms blocks louder than ``floor_db`` below the loudest block (so rests don't count)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    block = int(0.05 * sample_rate)
    blocks = mono[: len(mono) // block * block].reshape(-1, block)
    block_rms = np.sqrt((blocks.astype(np.float64) ** 2).mean(axis=1))
    if block_rms.max() <= 0:
        return 0.0
    loud = block_rms > block_rms.max() * 10 ** (floor_db / 20)
    return float(np.sqrt((block_rms[loud] ** 2).mean()))


def layer(stem: np.ndarray, render: np.ndarray, sample_rate: int, config) -> np.ndarray:
    """Stem panned left + render panned right, at equal audible loudness; ``(frames, 2)``."""
    length = len(stem)
    render = np.pad(render, ((0, max(0, length - len(render))), (0, 0)))[:length]
    stem_mono, render_mono = stem.mean(axis=1), render.mean(axis=1)
    render_mono = render_mono * (audible_rms(stem_mono, sample_rate) / max(audible_rms(render_mono, sample_rate), 1e-9))
    left_gain, right_gain = (1 + config.layer_pan) / 2, (1 - config.layer_pan) / 2
    mixed = np.stack([left_gain * stem_mono + right_gain * render_mono,
                      right_gain * stem_mono + left_gain * render_mono], axis=1)
    peak = np.abs(mixed).max()
    return mixed * (config.layer_peak_ceiling / peak) if peak > config.layer_peak_ceiling else mixed


def write_listening_triplet(stem_excerpt: np.ndarray, midi_path: Path, out_dir: Path, suffix: str, config) -> None:
    """Write render<suffix>.wav and layered<suffix>.wav for a MIDI file next to the stem excerpt."""
    render = render_midi(midi_path, out_dir / f"render{suffix}.wav", config)
    sf.write(out_dir / f"layered{suffix}.wav", layer(stem_excerpt, render, config.sample_rate, config),
             config.sample_rate, subtype="PCM_16")
