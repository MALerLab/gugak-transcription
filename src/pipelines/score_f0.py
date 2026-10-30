"""Command: track f0 with several trackers on one answer key's audio, and score each against the reference notes.

Answer keys (src/datasets/answer_keys.py): six dragon = exp001's per-stem aligned MIDI on single
stems; solo = solo clips against their own MIDI. One run scores one answer key, so either can be
rerun alone. Writes <f0_dir>/<tracker>/<item_id>.npz (files already there are skipped, so an
interrupted run resumes), <tables_dir>/scores.parquet (one row per item × tracker) and
<tables_dir>/summary_by_instrument.parquet.

Usage: python -m src.pipelines.score_f0 --config experiments/<exp>/configs/score_f0_<answer key>.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled
from src.datasets.answer_keys import six_dragon_stem_pairs, solo_clip_pairs
from src.evaluate.f0 import f0_scores
from src.evaluate.transcription import reference_notes
from src.features.f0 import make_f0_tracker
from src.pipelines.common import f0_path, load_run_config

ACCURACY_COLUMNS = ["raw_pitch_accuracy", "raw_chroma_accuracy", "best_shift_raw_pitch_accuracy"]


def list_items(config) -> pd.DataFrame:
    """Every (audio, reference MIDI) pair of the answer key ``run.answer_key`` names."""
    run = config.run
    if run.answer_key == "six_dragon":
        six_dragon = run.six_dragon
        return six_dragon_stem_pairs(six_dragon.quality_table, six_dragon.match_table, six_dragon.midi_dir,
                                     list(six_dragon.splits), config.datasets.separation, config.datasets.instruments)
    if run.answer_key == "solo":
        solo = run.solo
        return solo_clip_pairs(solo.clips_table, solo.partition, solo.reference_check_table,
                               solo.max_reference_transposition)
    raise ValueError(f"unknown answer key {run.answer_key!r}, expected six_dragon or solo")


def track_items(items: pd.DataFrame, tracker: str, config) -> None:
    """Track f0 on every item's audio with one tracker, skipping items already tracked."""
    tracker_config = config.features.f0[tracker]
    out_dir = Path(config.run.f0_dir) / tracker
    out_dir.mkdir(parents=True, exist_ok=True)
    remaining = [item for item in items.to_dict("records") if not f0_path(out_dir, item["item_id"]).exists()]
    if not remaining:
        return
    track_f0 = make_f0_tracker(tracker, tracker_config)
    for item in tqdm(remaining, desc=tracker, mininterval=30):
        mono = load_mono_resampled(item["audio_path"], tracker_config.model_sample_rate)
        np.savez_compressed(f0_path(out_dir, item["item_id"]), **track_f0(mono, tracker_config.model_sample_rate))


def score_items(items: pd.DataFrame, trackers: list[str], config) -> pd.DataFrame:
    """Score every item's f0 track, for every tracker, against the item's reference notes."""
    rows = []
    for item in tqdm(items.to_dict("records"), desc="score", mininterval=30):
        # A missing part can come back as NaN instead of None after a pandas round trip
        part = item["reference_part"] if isinstance(item["reference_part"], str) else None
        reference_intervals, reference_pitches = reference_notes(item["reference_midi_path"], part)
        for tracker in trackers:
            track = dict(np.load(f0_path(Path(config.run.f0_dir) / tracker, item["item_id"])))
            rows.append({**item, "tracker": tracker, "reference_note_count": len(reference_pitches),
                         **f0_scores(reference_intervals, reference_pitches, track, config.evaluate.f0)})
    return pd.DataFrame(rows)


def summarise_by_instrument(scores: pd.DataFrame) -> pd.DataFrame:
    """Per tracker × instrument: item means of the accuracies. Row ``all`` = mean of the instrument means."""
    per_instrument = scores.groupby(["tracker", "instrument"]).agg(
        items=("item_id", "size"), scored_hours=("scored_seconds", lambda seconds: seconds.sum() / 3600),
        **{column: (column, "mean") for column in ACCURACY_COLUMNS}).reset_index()
    overall = per_instrument.groupby("tracker").agg(
        items=("items", "sum"), scored_hours=("scored_hours", "sum"),
        **{column: (column, "mean") for column in ACCURACY_COLUMNS}).reset_index().assign(instrument="all")
    return pd.concat([per_instrument, overall], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/score_f0_<answer key>.yaml")
    config = load_run_config(parser.parse_args().config)
    run = config.run
    items = list_items(config)
    if run.limit_items:
        items = items.head(run.limit_items)
    print(items.groupby("instrument").size())
    for tracker in run.trackers:
        track_items(items, tracker, config)
    scores = score_items(items, list(run.trackers), config)
    summary = summarise_by_instrument(scores)
    Path(run.tables_dir).mkdir(parents=True, exist_ok=True)
    scores.to_parquet(Path(run.tables_dir) / "scores.parquet")
    summary.to_parquet(Path(run.tables_dir) / "summary_by_instrument.parquet")
    print(summary.round(3).to_string(index=False))
