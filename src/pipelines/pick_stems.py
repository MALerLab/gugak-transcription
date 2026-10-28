"""Command: pick strand B's stems from the separation dataset (src/datasets/separation.py).

Writes ``run.out_table``: one row per stem, with ``stem_id`` = the stem's filename (no extension).

Usage: python -m src.pipelines.pick_stems --config experiments/<exp>/configs/pick_stems.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.datasets.separation import pick_stems
from src.pipelines.common import load_run_config


def main(config_path: str) -> None:
    config = load_run_config(config_path)
    stems = pick_stems(config.datasets.separation, config.datasets.instruments)
    stems["stem_id"] = [Path(path).stem for path in stems["stem_path"]]
    out_table = Path(config.run.out_table)
    out_table.parent.mkdir(parents=True, exist_ok=True)
    stems.to_parquet(out_table)
    print(f"{len(stems)} stems → {out_table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/pick_stems.yaml")
    main(parser.parse_args().config)
