"""Command: re-place the run's 장 groups jointly (src/align/relocate.py).

Optional step after match_scores. Rewrites ``match_table.parquet``; the one-at-a-time version
is kept as ``match_table_stage1.parquet``.

Usage: python -m src.pipelines.relocate_songs --config experiments/<exp>/configs/relocate_songs.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.align.relocate import relocate_group
from src.datasets.six_dragon import list_musicxml_pieces
from src.pipelines.common import load_run_config


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    run, separation, six_dragon = config.run, config.datasets.separation, config.datasets.six_dragon
    tables_dir = Path(run.tables_dir)
    table_path = tables_dir / "match_table.parquet"
    stage1_copy = tables_dir / "match_table_stage1.parquet"
    if not stage1_copy.exists():  # keep the one-at-a-time spans for the report
        shutil.copy(table_path, stage1_copy)
    matches = pd.read_parquet(stage1_copy)
    pieces = list_musicxml_pieces(Path(six_dragon.musicxml_dir), list(six_dragon.part_names))

    matches["relocated"] = False
    for group in run.groups:
        for new in relocate_group(group, matches, pieces, separation, config.datasets.instruments, config.align.warp):
            index = matches.index[matches["song_id"] == new["song_id"]][0]
            print(new["song_id"], f"{matches.at[index, 'start_quarter']:.0f}-{matches.at[index, 'end_quarter']:.0f}",
                  "→", f"{new['start_quarter']:.0f}-{new['end_quarter']:.0f}", f"qps {new['quarters_per_second']:.2f}")
            for key in ("start_quarter", "end_quarter", "quarters_per_second"):
                matches.at[index, key] = new[key]
            matches.at[index, "relocated"] = True
            if matches.at[index, "confidence"] == "rejected":  # alone it lost; placed with its group
                matches.at[index, "confidence"] = "low"
    for song_id in run.get("readmit", []):
        matches.loc[matches["song_id"] == song_id, "confidence"] = "low"
    matches.to_parquet(table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/relocate_songs.yaml")
    main(parser.parse_args().config)
