"""Command: pick the solo-dataset clips strand B is scored against, with their dev/test partition (src/datasets/solo.py).

Writes ``run.out_table``: one row per clip.

Usage: python -m src.pipelines.pick_solo_clips --config experiments/<exp>/configs/pick_solo_clips.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.solo import pick_solo_clips
from src.pipelines.common import load_run_config


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    clips = pick_solo_clips(config.datasets.solo, config.datasets.instruments)
    out_table = Path(config.run.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    clips.to_parquet(out_table)
    print(f"{len(clips)} clips → {out_table}", clips["partition"].value_counts().to_dict())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/pick_solo_clips.yaml")
    main(parser.parse_args().config)
