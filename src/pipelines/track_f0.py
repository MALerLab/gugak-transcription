"""Command: track f0 for every audio file in a table; one .npz (time, f0_hz, confidence) per file.

The tracker is ``run.f0_tracker`` in the run config. Files already tracked are skipped,
so an interrupted run resumes.

Usage: python -m src.pipelines.track_f0 --config experiments/<exp>/configs/track_f0.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.io import load_mono_resampled
from src.features.f0 import make_f0_tracker
from src.pipelines.common import active_f0_tracker, f0_path, load_run_config


def track_table(config) -> None:
    """Run the tracker over every row of ``run.table``, skipping files already done."""
    run, tracker_config = config.run, active_f0_tracker(config)
    sample_rate = tracker_config.model_sample_rate
    track_f0 = make_f0_tracker(run.f0_tracker, tracker_config)
    table = pd.read_parquet(run.table)
    Path(run.out_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()
    for row_index, row in enumerate(table.itertuples()):
        out_path = f0_path(run.out_dir, getattr(row, run.id_column))
        if out_path.exists():
            continue
        mono = load_mono_resampled(getattr(row, run.path_column), sample_rate)
        track = track_f0(mono, sample_rate)
        np.savez_compressed(out_path, **track)
        print(f"[{row_index + 1}/{len(table)}] {out_path.name} {len(mono) / sample_rate:.0f}s audio,"
              f" {time.time() - started:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/track_f0.yaml")
    track_table(load_run_config(parser.parse_args().config))
