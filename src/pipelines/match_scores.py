"""Command: match six dragon MusicXML pieces to the separation dataset's 정악 songs (src/align/match.py).

Writes ``match_costs.parquet`` (song × piece) and ``match_table.parquet`` (per song) to ``tables_dir``.

Usage: python -m src.pipelines.match_scores --config experiments/<exp>/configs/match_scores.yaml [--decide-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.align.match import audio_cens_for_song, compute_costs, decide_matches, rule_candidates
from src.datasets.separation import list_jeongak_songs
from src.datasets.six_dragon import list_musicxml_pieces
from src.pipelines.common import load_run_config


def main(config_path: str, decide_only: bool) -> None:
    config = load_run_config(config_path)
    run, separation, six_dragon, match_config = (config.run, config.datasets.separation,
                                                 config.datasets.six_dragon, config.align.match)
    tables_dir = Path(run.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Both catalogues + title rules
    pieces = list_musicxml_pieces(Path(six_dragon.musicxml_dir), list(six_dragon.part_names))
    songs = list_jeongak_songs(separation)
    found = [rule_candidates(t, pieces, match_config) for t in songs["title"]]
    songs["candidates"] = [c for c, _ in found]
    songs["rule"] = [r for _, r in found]
    print(f"{len(songs)} 정악 songs with audio, {len(pieces)} pieces, "
          f"{(songs['candidates'].str.len() > 0).sum()} songs with a title candidate")

    if decide_only:  # reuse saved costs; tunings come from the feature cache
        costs = pd.read_parquet(tables_dir / "match_costs.parquet")
        songs = songs[songs["song_id"].isin(costs["song_id"])].copy()
        songs["tuning_cents"] = [audio_cens_for_song(s, [], match_config, run.cache_dir)[1] for s in songs["song_id"]]
        audio = dict.fromkeys(songs["song_id"])
    else:
        costs, audio = compute_costs(songs, pieces, match_config, separation, config.datasets.instruments,
                                     run.cache_dir, run.num_workers)
        costs.to_parquet(tables_dir / "match_costs.parquet")

    # Decide + write
    matches = decide_matches(songs[songs["song_id"].isin(audio)], pieces, costs, match_config,
                             dict(config.datasets.instruments.english_names))
    matches = matches.merge(songs[["song_id", "tuning_cents", "stems"]], on="song_id")
    matches.to_parquet(tables_dir / "match_table.parquet")
    print(matches["confidence"].value_counts().to_string())
    for tier in (["high"], ["high", "medium"], ["high", "medium", "low"]):
        kept = matches[matches["confidence"].isin(tier)]
        print(f"{'+'.join(tier)}: {len(kept)} songs, {kept['duration_seconds'].sum() / 3600:.2f} h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/match_scores.yaml")
    parser.add_argument("--decide-only", action="store_true", help="reuse match_costs.parquet")
    arguments = parser.parse_args()
    main(arguments.config, arguments.decide_only)
