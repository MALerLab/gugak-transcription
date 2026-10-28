"""Command: note-ify every tracked stem into a full-length MIDI file, plus per-stem diagnostics.

Writes <out_dir>/midi/<stem_id>.mid and <out_dir>/stems_summary.parquet.

Usage: python -m src.pipelines.transcribe_stems --config experiments/<exp>/configs/transcribe_stems.yaml
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.io import load_mono_resampled
from src.evaluate.transcription import stem_diagnostics
from src.notes.midi import note_table_to_midi
from src.pipelines.common import active_f0_tracker, f0_path, load_run_config
from src.transcribe.note_ify import notes_from_f0


def transcribe_stem(stem: dict, config) -> dict:
    """Note-ify one stem, write its MIDI, return its diagnostics row."""
    run, tracker = config.run, active_f0_tracker(config.features.f0)
    track = dict(np.load(f0_path(run.f0_dir, stem["stem_id"])))
    mono = load_mono_resampled(stem["stem_path"], tracker.model_sample_rate)
    result = notes_from_f0(track, mono, tracker.model_sample_rate, tracker.hop_seconds, config.transcribe.note_ify,
                           config.features.tuning, config.features.onsets)
    notes = result["notes"].assign(part=stem["instrument"])
    note_table_to_midi(notes, Path(run.out_dir) / "midi" / f"{stem['stem_id']}.mid",
                       dict(config.notes.midi.gm_programs))
    return {"stem_id": stem["stem_id"], "song_id": stem["song_id"], "instrument": stem["instrument"],
            "genreSub": stem["genreSub"], "genreDetail": stem["genreDetail"], "title": stem["title"], "split": stem["split"],
            **stem_diagnostics(result, track["confidence"], tracker.hop_seconds)}


def _transcribe_job(arguments):
    return transcribe_stem(*arguments)


def transcribe_stems(config) -> pd.DataFrame:
    """Transcribe all stems of ``run.stems_table`` in parallel processes; returns the diagnostics table."""
    run = config.run
    (Path(run.out_dir) / "midi").mkdir(parents=True, exist_ok=True)
    jobs = [(stem, config) for stem in pd.read_parquet(run.stems_table).to_dict("records")]
    with ProcessPoolExecutor(run.num_workers) as pool:
        return pd.DataFrame(list(pool.map(_transcribe_job, jobs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/transcribe_stems.yaml")
    config = load_run_config(parser.parse_args().config)
    summary = transcribe_stems(config)
    summary.to_parquet(Path(config.run.out_dir) / "stems_summary.parquet")
    print(summary.groupby(["genreSub", "instrument"]).mean(numeric_only=True).round(3).to_string())
