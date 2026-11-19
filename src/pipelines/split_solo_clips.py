"""Command: split the solo dataset's clips into train / val / test for training (src/datasets/solo.py).

Writes ``run.out_table``: one row per clip, with ``split`` and ``drop_reason`` (None = used).

Usage: python -m src.pipelines.split_solo_clips --config experiments/<exp>/configs/split_solo_clips.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.solo import split_solo_clips
from src.pipelines.common import load_run_config


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    clips = split_solo_clips(config.datasets.solo, config.datasets.instruments)
    out_table = Path(config.run.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    clips.to_parquet(out_table)
    used = clips["drop_reason"].isna()
    summary = clips.assign(used=used, hours=clips["duration_seconds"] / 3600).groupby(["split", "used"], dropna=False)
    print(f"{len(clips)} clips → {out_table}\n{summary.agg(clips=('clip_id', 'size'), hours=('hours', 'sum'), pieces=('piece', 'nunique')).round(2)}")
    print(clips.loc[used].groupby(["split", "instrument"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/split_solo_clips.yaml")
    main(parser.parse_args().config)
