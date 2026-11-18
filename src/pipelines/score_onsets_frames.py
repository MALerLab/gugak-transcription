"""Command: decode stored Onsets and Frames outputs into notes and score them against the answer keys.

Reads <predictions_dir>/items.parquet + one ``.npz`` per item (written by predict_onsets_frames),
decodes with the recipe's thresholds, writes one note table + MIDI per item, and
<scores_table> (one row per item). Six dragon items are scored twice: over every note, and with
the grace-note windows left out (``*_masked``, the main score). Solo items have no windows.

Optional extras, off unless the run config sets them:
- ``run.tuning_gate_f0_dirs`` (answer key → stock PESTO f0 dir, ``<item_id>.npz``): the tuning-gated onset
  F1 next to today's (``*_tuning_gated``, src/evaluate/transcription.py), from each item's tuning offset.
- ``run.mark_unreliable_solo_clips``: a ``drop_reason`` column on solo items whose MIDI the solo MIDI
  audit flagged (src/datasets/solo.py); None = kept.

Usage: python -m src.pipelines.score_onsets_frames --config experiments/<exp>/configs/score_onsets_frames.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.datasets.solo import unreliable_solo_clips
from src.evaluate.f0 import recording_tuning_offset
from src.evaluate.transcription import estimated_notes, masked_and_unmasked_scores, tuning_gated_scores
from src.notes.midi import note_table_to_midi
from src.pipelines.common import f0_path, item_grace_windows, item_reference_notes, load_run_config
from src.transcribe.onsets_frames import decode_notes


def decoding_config(transcribe_config, onset_thresholds: dict[str, float], instrument: str):
    """The recipe's decoding settings, with the instrument's tuned onset threshold if there is one."""
    if instrument not in onset_thresholds:
        return transcribe_config
    return OmegaConf.merge(transcribe_config, {"onset_threshold": onset_thresholds[instrument]})


def decode_item(item: dict, transcribe_config) -> pd.DataFrame:
    """One item's note table from its stored model outputs."""
    outputs = np.load(item["prediction_path"])
    return decode_notes(outputs["onset"].astype(np.float32), outputs["frame"].astype(np.float32),
                        outputs["velocity"].astype(np.float32), item["instrument"], transcribe_config)


def item_reference_intervals_and_pitches(item: dict) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches as MIDI numbers) of one item's reference notes."""
    reference = item_reference_notes(item)
    start = reference["start"].to_numpy()
    reference_intervals = np.stack([start, start + reference["duration"].clip(lower=1e-3).to_numpy()], axis=1).reshape(-1, 2)
    return reference_intervals, reference["pitch"].to_numpy(float)


def item_tuning_offset(item: dict, reference_intervals: np.ndarray, reference_pitches: np.ndarray, config) -> dict:
    """How far one item's audio sits from its reference notes (cents), from its cached f0 track."""
    track = dict(np.load(f0_path(config.run.tuning_gate_f0_dirs[item["answer_key"]], item["item_id"])))
    return recording_tuning_offset(reference_intervals, reference_pitches, track, config.evaluate.f0)


def score_item(item: dict, notes: pd.DataFrame, config) -> dict:
    """One item's scores, every note and grace windows left out; plus the tuning-gated scores if the run asks."""
    reference_intervals, reference_pitches = item_reference_intervals_and_pitches(item)
    windows = item_grace_windows(item, config)
    scores = masked_and_unmasked_scores(reference_intervals, reference_pitches, *estimated_notes(notes), windows,
                                        config.evaluate.transcription)
    if not config.run.get("tuning_gate_f0_dirs"):
        return scores
    offset = item_tuning_offset(item, reference_intervals, reference_pitches, config)
    gated = tuning_gated_scores(reference_intervals, reference_pitches, *estimated_notes(notes), windows,
                                offset["tuning_offset_cents"], scores, config.evaluate.transcription)
    return {**scores, **offset, **gated}


def summarise_scores(scores: pd.DataFrame) -> None:
    """Print the mean scores per answer key, split and instrument, then the mean of instrument means."""
    columns = [c for c in ("onset_f1_masked", "onset_f1_masked_tuning_gated") if c in scores]
    test_sets = {f"{key} {split}": group for (key, split), group in scores.groupby(["answer_key", "split"])}
    if "drop_reason" in scores:
        solo_test = scores[(scores["answer_key"] == "solo") & (scores["split"] == "test")]
        test_sets["solo test, unreliable MIDI dropped"] = solo_test[solo_test["drop_reason"].isna()]
    for name, group in test_sets.items():
        by_instrument = group.groupby("instrument")[columns].mean()
        print(f"--- {name} ({len(group)} items)\n{by_instrument.round(3)}\nall {by_instrument.mean().round(4).to_dict()}")


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run = config.run
    items = pd.read_parquet(Path(run.predictions_dir) / "items.parquet")
    if run.get("mark_unreliable_solo_clips"):
        reasons = unreliable_solo_clips(config.datasets.solo).set_index("clip_id")["drop_reason"]
        items["drop_reason"] = [reasons.get(item_id) if key == "solo" else None
                                for item_id, key in zip(items["item_id"], items["answer_key"])]
    if run.get("limit_items_per_instrument"):
        items = items.groupby(["answer_key", "split", "instrument"], group_keys=False).head(run.limit_items_per_instrument)
    gm_programs = dict(config.notes.midi.gm_programs)
    onset_thresholds = {}
    if run.get("onset_thresholds_table"):
        thresholds = pd.read_parquet(run.onset_thresholds_table)
        onset_thresholds = dict(zip(thresholds["instrument"], thresholds["onset_threshold"].astype(float)))
        print(f"onset thresholds per instrument: {onset_thresholds}", flush=True)
    rows = []
    for item in tqdm(items.to_dict("records"), desc="score", mininterval=30):
        notes = decode_item(item, decoding_config(config.transcribe.onsets_frames, onset_thresholds, item["instrument"]))
        note_table_dir = Path(run.note_tables_dir) / item["answer_key"]
        note_table_dir.mkdir(parents=True, exist_ok=True)
        notes.to_parquet(note_table_dir / f"{item['item_id']}.parquet")
        note_table_to_midi(notes, Path(run.midi_dir) / item["answer_key"] / f"{item['item_id']}.mid", gm_programs)
        rows.append({**item, **score_item(item, notes, config)})
    scores = pd.DataFrame(rows)
    Path(run.scores_table).parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(run.scores_table)
    summarise_scores(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/score_onsets_frames.yaml")
    main(parser.parse_args().config)
