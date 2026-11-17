"""Command: decode stored Onsets and Frames outputs into notes and score them against the answer keys.

Reads <predictions_dir>/items.parquet + one ``.npz`` per item (written by predict_onsets_frames),
decodes with the recipe's thresholds, writes one note table + MIDI per item, and
<scores_table> (one row per item). Six dragon items are scored twice: over every note, and with
the grace-note windows left out (``*_masked``, the main score). Solo items have no windows.

Usage: python -m src.pipelines.score_onsets_frames --config experiments/<exp>/configs/score_onsets_frames.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.evaluate.transcription import estimated_notes, masked_and_unmasked_scores
from src.notes.midi import note_table_to_midi
from src.pipelines.common import item_grace_windows, item_reference_notes, load_run_config
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


def score_item(item: dict, notes: pd.DataFrame, config) -> dict:
    """One item's scores, every note and grace windows left out."""
    reference = item_reference_notes(item)
    start = reference["start"].to_numpy()
    reference_intervals = np.stack([start, start + reference["duration"].clip(lower=1e-3).to_numpy()], axis=1).reshape(-1, 2)
    return masked_and_unmasked_scores(reference_intervals, reference["pitch"].to_numpy(float), *estimated_notes(notes),
                                      item_grace_windows(item, config), config.evaluate.transcription)


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run = config.run
    items = pd.read_parquet(Path(run.predictions_dir) / "items.parquet")
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
    summary = scores.groupby(["answer_key", "split", "instrument"])[["onset_f1_masked", "onset_f1"]].mean()
    print(summary.round(3))
    print(summary.groupby(["answer_key", "split"]).mean().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/score_onsets_frames.yaml")
    main(parser.parse_args().config)
