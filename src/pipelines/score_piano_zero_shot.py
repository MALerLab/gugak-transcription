"""Command: transcribe gugak audio with the piano-pretrained model as-is, and score it against both answer keys.

Answer keys (src/datasets/answer_keys.py): six dragon = exp001's per-stem aligned MIDI on single
stems; solo = solo clips against their own MIDI. Writes one MIDI file + one note table per item,
and <tables_dir>/scores.parquet (one row per item).

Usage: python -m src.pipelines.score_piano_zero_shot --config experiments/<exp>/configs/score_piano_zero_shot.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled
from src.datasets.answer_keys import six_dragon_stem_pairs, solo_clip_pairs
from src.evaluate.transcription import estimated_notes, note_scores, onset_only_scores, reference_notes
from src.notes.midi import note_table_to_midi
from src.pipelines.common import load_run_config
from src.transcribe.piano_transcription import MODEL_SAMPLE_RATE, load_piano_transcription, transcribe_audio


def list_items(config) -> pd.DataFrame:
    """Every (audio, reference MIDI) pair of both answer keys."""
    six_dragon, solo = config.run.six_dragon, config.run.solo
    return pd.concat([
        six_dragon_stem_pairs(six_dragon.quality_table, six_dragon.match_table, six_dragon.midi_dir,
                              list(six_dragon.splits), config.datasets.separation, config.datasets.instruments),
        solo_clip_pairs(solo.clips_table, solo.partition, solo.reference_check_table, solo.max_reference_transposition),
    ], ignore_index=True)


def score_item(item: dict, notes: pd.DataFrame, evaluate_config) -> dict:
    """One item's scores: onset + pitch (strict and lenient), onset + pitch + offset, onset only."""
    # A missing part can come back as NaN instead of None after a pandas round trip
    part = item["reference_part"] if isinstance(item["reference_part"], str) else None
    reference_intervals, reference_pitches = reference_notes(item["reference_midi_path"], part)
    estimate_intervals, estimate_pitches = estimated_notes(notes)
    return {"reference_note_count": len(reference_pitches), "estimated_note_count": len(estimate_pitches),
            **note_scores(reference_intervals, reference_pitches, estimate_intervals, estimate_pitches, evaluate_config),
            **onset_only_scores(reference_intervals, estimate_intervals, evaluate_config)}


def transcribe_and_score(items: pd.DataFrame, config) -> pd.DataFrame:
    """Transcribe every item's audio, write its MIDI + note table, and score it."""
    run = config.run
    transcriptor = load_piano_transcription(config.transcribe.piano_transcription, run.device)
    gm_programs = dict(config.notes.midi.gm_programs)
    rows = []
    for item in tqdm(items.to_dict("records"), desc="transcribe", mininterval=30):
        audio = load_mono_resampled(item["audio_path"], MODEL_SAMPLE_RATE)
        notes = transcribe_audio(transcriptor, audio, item["instrument"])
        notes.to_parquet(Path(run.note_tables_dir) / item["answer_key"] / f"{item['item_id']}.parquet")
        note_table_to_midi(notes, Path(run.midi_dir) / item["answer_key"] / f"{item['item_id']}.mid", gm_programs)
        rows.append({**item, "duration_seconds": len(audio) / MODEL_SAMPLE_RATE,
                     **score_item(item, notes, config.evaluate.transcription)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/score_piano_zero_shot.yaml")
    config = load_run_config(parser.parse_args().config)
    run = config.run
    items = list_items(config)
    if run.limit_items_per_answer_key:
        items = items.groupby("answer_key", group_keys=False).head(run.limit_items_per_answer_key)
    for answer_key in items["answer_key"].unique():
        (Path(run.note_tables_dir) / answer_key).mkdir(parents=True, exist_ok=True)
    print(items.groupby(["answer_key", "instrument"]).size())
    scores = transcribe_and_score(items, config)
    Path(run.scores_table).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(run.scores_table)
    print(scores.groupby(["answer_key", "instrument"])[["onset_f1", "onset_f1_lenient", "onset_only_f1"]].mean().round(3))
