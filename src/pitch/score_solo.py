"""Run note-ify on solo-dataset clips and score each against the clip's own MIDI.

Usage: python -m src.pitch.score_solo <solo_clips.parquet> <f0_dir> <partition> <out_prefix>
Writes <out_prefix>_clips.parquet (one row per clip) and <out_prefix>_sigimsae.parquet.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.pitch.evaluate import (estimated_notes, note_scores, oracle_segmentation_pitch_accuracy, reference_notes,
                                reference_transposition, sigimsae_breakdown)
from src.pitch.notes import notes_from_f0
from src.pitch.track import load_mono_resampled
from src.pitch.track_batch import f0_path

TRACK_CONFIG = "configs/pitch/track.yaml"
NOTES_CONFIG = "configs/pitch/notes.yaml"
EVALUATE_CONFIG = "configs/pitch/evaluate.yaml"


def score_clip(clip: dict, f0_dir: str, track_config, notes_config, evaluate_config, mono=None) -> tuple[dict, list[dict]]:
    """Note-ify one clip and score it. Returns (clip row, 시김새 rows)."""
    track = dict(np.load(f0_path(f0_dir, clip["clip_id"])))
    if mono is None:
        mono = load_mono_resampled(clip["clip_path"], track_config.model_sample_rate)
    result = notes_from_f0(track, mono, track_config.model_sample_rate, track_config.hop_seconds, notes_config)
    reference_intervals, reference_pitches = reference_notes(clip["midi_path"])
    estimate_intervals, estimate_pitches = estimated_notes(result["notes"], track_config.hop_seconds)
    transposition = reference_transposition(reference_intervals, reference_pitches, result["semitones"],
                                            result["voiced"], track_config.hop_seconds)
    row = {"clip_id": clip["clip_id"], "instrument": clip["instrument"], "piece": clip["piece"],
           "partition": clip["partition"], "tuning_offset_cents": result["tuning_offset_cents"],
           "reference_note_count": len(reference_pitches), "estimated_note_count": len(estimate_pitches),
           "reference_transposition": transposition,
           "oracle_segmentation_pitch_accuracy": oracle_segmentation_pitch_accuracy(
               reference_intervals, reference_pitches, result["semitones"], result["voiced"], track_config.hop_seconds),
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
    clip, f0_dir, configs = arguments
    return score_clip(clip, f0_dir, *[OmegaConf.create(c) for c in configs])


def score_clips(clips: pd.DataFrame, f0_dir: str, notes_config, workers: int = 16) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every clip in parallel processes with the given note-ify settings."""
    configs = [OmegaConf.to_container(OmegaConf.load(TRACK_CONFIG)), OmegaConf.to_container(notes_config),
               OmegaConf.to_container(OmegaConf.load(EVALUATE_CONFIG))]
    jobs = [(clip, f0_dir, configs) for clip in clips.to_dict("records")]
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_score_clip_job, jobs, chunksize=4))
    clip_rows = pd.DataFrame([r for r, _ in results])
    sigimsae_rows = pd.DataFrame([s for _, rows in results for s in rows])
    return clip_rows, sigimsae_rows


if __name__ == "__main__":
    clips_path, f0_dir, partition, out_prefix = sys.argv[1:5]
    clips = pd.read_parquet(clips_path)
    clips = clips[clips["partition"] == partition]
    clip_rows, sigimsae_rows = score_clips(clips, f0_dir, OmegaConf.load(NOTES_CONFIG))
    clip_rows.to_parquet(f"{out_prefix}_clips.parquet")
    sigimsae_rows.to_parquet(f"{out_prefix}_sigimsae.parquet")
    print(clip_rows.drop(columns=["clip_id", "piece", "partition"]).groupby("instrument").mean(numeric_only=True).round(3))
