"""Track f0 for every audio file in a table; one .npz (time, f0_hz, confidence) per file.

Usage: python -m src.pitch.track_batch <table.parquet> <path_column> <id_column> <out_dir>
Files already tracked are skipped, so an interrupted run resumes.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.pitch.track import load_mono_resampled, track_f0

TRACK_CONFIG = "configs/pitch/track.yaml"


def f0_path(out_dir: str | Path, file_id: str) -> Path:
    """Where one file's f0 track is stored."""
    return Path(out_dir) / f"{file_id}.npz"


def track_table(table_path: str, path_column: str, id_column: str, out_dir: str) -> None:
    """Run the tracker over every row of a table, skipping files already done."""
    config = OmegaConf.load(TRACK_CONFIG)
    table = pd.read_parquet(table_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    started = time.time()
    for row_index, row in enumerate(table.itertuples()):
        out_path = f0_path(out_dir, getattr(row, id_column))
        if out_path.exists():
            continue
        mono = load_mono_resampled(getattr(row, path_column), config.model_sample_rate)
        track = track_f0(mono, config)
        np.savez_compressed(out_path, **track)
        print(f"[{row_index + 1}/{len(table)}] {out_path.name} {len(mono) / config.model_sample_rate:.0f}s audio,"
              f" {time.time() - started:.0f}s elapsed", flush=True)


if __name__ == "__main__":
    track_table(*sys.argv[1:5])
