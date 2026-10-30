"""Command: run note-ify on solo-dataset clips and score each against the clip's own MIDI.

Writes <out_prefix>_clips.parquet (one row per clip) and <out_prefix>_sigimsae.parquet.

Usage: python -m src.pipelines.score_solo --config experiments/<exp>/configs/score_solo.yaml
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.io import load_mono_resampled
from src.evaluate.transcription import (estimated_notes, note_scores, oracle_segmentation_pitch_accuracy,
                                        reference_notes, reference_transposition, sigimsae_breakdown)
from src.pipelines.common import active_f0_tracker, f0_path, load_run_config
from src.transcribe.note_ify import notes_from_f0


def score_clip(clip: dict, config, mono=None) -> tuple[dict, list[dict]]:
    """Note-ify one clip and score it. Returns (clip row, 시김새 rows)."""
    tracker, evaluate_config = active_f0_tracker(config), config.evaluate.transcription
    track = dict(np.load(f0_path(config.run.f0_dir, clip["clip_id"])))
    if mono is None:
        mono = load_mono_resampled(clip["clip_path"], tracker.model_sample_rate)
    result = notes_from_f0(track, mono, tracker.model_sample_rate, tracker.hop_seconds, config.transcribe.note_ify,
                           config.features.tuning, config.features.onsets)
    reference_intervals, reference_pitches = reference_notes(clip["midi_path"])
    estimate_intervals, estimate_pitches = estimated_notes(result["notes"])
    transposition = reference_transposition(reference_intervals, reference_pitches, result["semitones"],
                                            result["voiced"], tracker.hop_seconds)
    row = {"clip_id": clip["clip_id"], "instrument": clip["instrument"], "piece": clip["piece"],
           "partition": clip["partition"], "tuning_offset_cents": result["tuning_offset_cents"],
           "reference_note_count": len(reference_pitches), "estimated_note_count": len(estimate_pitches),
           "reference_transposition": transposition,
           "oracle_segmentation_pitch_accuracy": oracle_segmentation_pitch_accuracy(
               reference_intervals, reference_pitches, result["semitones"], result["voiced"], tracker.hop_seconds),
           **note_scores(reference_intervals, reference_pitches, estimate_intervals, estimate_pitches, evaluate_config)}
    # Grid-aligned: a whole-clip ±1 semitone gap is the quarter-tone naming ambiguity (a note 40 cents
    # above G is a sharp G or a flat G#), not a transcription error, so it is shifted away
    grid_shift = -transposition if abs(transposition) == 1 else 0
    aligned_pitches = estimate_pitches + grid_shift
    row.update({f"aligned_{k}": v for k, v in note_scores(reference_intervals, reference_pitches, estimate_intervals,
                                                          aligned_pitches, evaluate_config).items()})
    sigimsae_rows = [{"clip_id": clip["clip_id"], "instrument": clip["instrument"], "reference_transposition": transposition, **r}
                     for r in sigimsae_breakdown(clip["sigimsae"], reference_intervals, reference_pitches,
                                                 estimate_intervals, aligned_pitches, evaluate_config)]
    return row, sigimsae_rows


def _score_clip_job(arguments):
    return score_clip(*arguments)


def score_clips(clips: pd.DataFrame, config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every clip in parallel processes."""
    with ProcessPoolExecutor(config.run.num_workers) as pool:
        results = list(pool.map(_score_clip_job, [(clip, config) for clip in clips.to_dict("records")], chunksize=4))
    clip_rows = pd.DataFrame([r for r, _ in results])
    sigimsae_rows = pd.DataFrame([s for _, rows in results for s in rows])
    return clip_rows, sigimsae_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/score_solo.yaml")
    config = load_run_config(parser.parse_args().config)
    run = config.run
    clips = pd.read_parquet(run.clips_table)
    clips = clips[clips["partition"] == run.partition]
    clip_rows, sigimsae_rows = score_clips(clips, config)
    Path(run.out_prefix).parent.mkdir(parents=True, exist_ok=True)
    clip_rows.to_parquet(f"{run.out_prefix}_clips.parquet")
    sigimsae_rows.to_parquet(f"{run.out_prefix}_sigimsae.parquet")
    print(clip_rows.drop(columns=["clip_id", "piece", "partition"]).groupby("instrument").mean(numeric_only=True).round(3))
