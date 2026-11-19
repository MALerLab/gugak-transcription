"""Command: pick each instrument's onset threshold for Onsets and Frames decoding on val items.

Decodes the stored model outputs (predict_onsets_frames) of the val items of ``run.tuning_datasets``
(six dragon stems and / or solo clips) at every threshold in ``run.onset_threshold_grid`` and keeps, per
instrument, the one with the best mean onset F1 with the grace-note windows left out (with several
datasets: the mean over datasets of the instrument's mean in each). The frame threshold stays at the recipe's value (it only sets
how long a note lasts, never where one starts). Test is never looked at.

Writes <thresholds_table> (instrument, onset_threshold, val score) and <tables_dir>/threshold_sweep.parquet.

Usage: python -m src.pipelines.tune_onsets_frames_thresholds --config experiments/<exp>/configs/tune_onsets_frames_thresholds.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from src.evaluate.transcription import estimated_notes, masked_and_unmasked_scores
from src.pipelines.common import item_grace_windows, item_reference_notes, load_run_config
from src.transcribe.onsets_frames import decode_notes


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run = config.run
    items = pd.read_parquet(Path(run.predictions_dir) / "items.parquet")
    # exp008–exp009's run configs predate tuning_datasets: six dragon only
    tuning_datasets = list(run.get("tuning_datasets", ["six_dragon"]))
    items = items[items["answer_key"].isin(tuning_datasets) & (items["split"] == run.tuning_split)]
    print(f"tuning on {run.tuning_split} items {items['answer_key'].value_counts().to_dict()}", flush=True)
    rows = []
    for item in tqdm(items.to_dict("records"), desc="sweep", mininterval=30):
        outputs = {name: array.astype(np.float32) for name, array in np.load(item["prediction_path"]).items()}
        reference = item_reference_notes(item)
        start = reference["start"].to_numpy()
        reference_intervals = np.stack([start, start + reference["duration"].clip(lower=1e-3).to_numpy()], 1).reshape(-1, 2)
        windows = item_grace_windows(item, config)
        for onset_threshold in run.onset_threshold_grid:
            decoding = OmegaConf.merge(config.transcribe.onsets_frames, {"onset_threshold": float(onset_threshold)})
            notes = decode_notes(outputs["onset"], outputs["frame"], outputs["velocity"], item["instrument"], decoding)
            scores = masked_and_unmasked_scores(reference_intervals, reference["pitch"].to_numpy(float),
                                                *estimated_notes(notes), windows, config.evaluate.transcription)
            rows.append({"item_id": item["item_id"], "answer_key": item["answer_key"], "instrument": item["instrument"],
                         "onset_threshold": float(onset_threshold), **scores})
    sweep = pd.DataFrame(rows)
    tables_dir = Path(run.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    sweep.to_parquet(tables_dir / "threshold_sweep.parquet")
    by_dataset = sweep.groupby(["answer_key", "instrument", "onset_threshold"])[run.selection_metric].mean()
    by_threshold = by_dataset.groupby(["instrument", "onset_threshold"]).mean().reset_index()
    chosen = by_threshold.loc[by_threshold.groupby("instrument")[run.selection_metric].idxmax()].reset_index(drop=True)
    chosen.to_parquet(run.thresholds_table)
    print(by_threshold.pivot(index="onset_threshold", columns="instrument", values=run.selection_metric).round(3))
    print(chosen.round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/tune_onsets_frames_thresholds.yaml")
    main(parser.parse_args().config)
