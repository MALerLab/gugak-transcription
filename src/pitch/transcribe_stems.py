"""Note-ify every tracked stem into a full-length MIDI file, plus per-stem diagnostics.

Usage: python -m src.pitch.transcribe_stems <picked_stems.parquet> <f0_dir> <notes_config.yaml> <out_dir>
Writes <out_dir>/midi/<stem_id>.mid and <out_dir>/stems_summary.parquet.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.pitch.notes import Note, notes_from_f0, notes_to_midi
from src.pitch.track import load_mono_resampled
from src.pitch.track_batch import f0_path

TRACK_CONFIG = "configs/pitch/track.yaml"
RENDER_CONFIG = "configs/render/render.yaml"


def note_roll(notes: list[Note], num_frames: int) -> np.ndarray:
    """Per-frame snapped pitch of the note covering it, NaN where no note sounds."""
    roll = np.full(num_frames, np.nan)
    for note in notes:
        roll[note.start_frame:note.end_frame] = note.pitch
    return roll


def stem_diagnostics(result: dict, confidence: np.ndarray, hop_seconds: float) -> dict:
    """Reference-free numbers that hint at how well a stem was note-ified.

    - ``off_note_fraction``: share of in-note frames whose pitch sits ≥ 30 cents off the note's
      snapped pitch — time spent bending, sliding or in wide vibrato (a 시김새 proxy).
    - ``far_off_note_fraction``: same at ≥ 50 cents — where the MIDI note is plainly not the sounding pitch.
    - ``uncovered_fraction``: voiced frames that ended up in no note (dropped as too short).
    """
    notes, semitones, voiced = result["notes"], result["semitones"], result["voiced"]
    roll = note_roll(notes, len(semitones))
    in_note = ~np.isnan(roll)
    deviation = np.abs(semitones[in_note] - roll[in_note])
    duration = len(semitones) * hop_seconds
    return {
        "tuning_offset_cents": result["tuning_offset_cents"],
        "note_count": len(notes),
        "notes_per_second": len(notes) / max(voiced.sum() * hop_seconds, 1e-6),
        "voiced_fraction": float(voiced.mean()),
        "mean_confidence_voiced": float(confidence[voiced].mean()) if voiced.any() else 0.0,
        "off_note_fraction": float((deviation >= 0.3).mean()) if in_note.any() else 0.0,
        "far_off_note_fraction": float((deviation >= 0.5).mean()) if in_note.any() else 0.0,
        "uncovered_fraction": float((voiced & ~in_note).sum() / max(voiced.sum(), 1)),
        "duration_seconds": duration,
    }


def transcribe_stem(stem: dict, f0_dir: str, notes_config_dict: dict, out_dir: str) -> dict:
    """Note-ify one stem, write its MIDI, return its diagnostics row."""
    track_config, render_config = OmegaConf.load(TRACK_CONFIG), OmegaConf.load(RENDER_CONFIG)
    notes_config = OmegaConf.create(notes_config_dict)
    track = dict(np.load(f0_path(f0_dir, stem["stem_id"])))
    mono = load_mono_resampled(stem["stem_path"], track_config.model_sample_rate)
    result = notes_from_f0(track, mono, track_config.model_sample_rate, track_config.hop_seconds, notes_config)
    midi = notes_to_midi(result, track_config.hop_seconds, render_config.gm_programs[stem["instrument"]], notes_config)
    midi.write(str(Path(out_dir) / "midi" / f"{stem['stem_id']}.mid"))
    return {"stem_id": stem["stem_id"], "song_id": stem["song_id"], "instrument": stem["instrument"],
            "genreSub": stem["genreSub"], "genreDetail": stem["genreDetail"], "title": stem["title"], "split": stem["split"],
            **stem_diagnostics(result, track["confidence"], track_config.hop_seconds)}


def _transcribe_job(arguments):
    return transcribe_stem(*arguments)


def transcribe_stems(stems: pd.DataFrame, f0_dir: str, notes_config, out_dir: str, workers: int = 12) -> pd.DataFrame:
    """Transcribe all stems in parallel processes; returns the diagnostics table."""
    (Path(out_dir) / "midi").mkdir(parents=True, exist_ok=True)
    jobs = [(stem, f0_dir, OmegaConf.to_container(notes_config), out_dir) for stem in stems.to_dict("records")]
    with ProcessPoolExecutor(workers) as pool:
        return pd.DataFrame(list(pool.map(_transcribe_job, jobs)))


if __name__ == "__main__":
    stems_path, f0_dir, notes_config_path, out_dir = sys.argv[1:5]
    summary = transcribe_stems(pd.read_parquet(stems_path), f0_dir, OmegaConf.load(notes_config_path), out_dir)
    summary.to_parquet(Path(out_dir) / "stems_summary.parquet")
    print(summary.groupby(["genreSub", "instrument"]).mean(numeric_only=True).round(3).to_string())
