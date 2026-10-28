"""Command: refine, align (per song and per stem) and measure every accepted match.

Per song it writes the aligned note tables (written pitch, three timings: linear, per_song,
per_stem) to the cache, and one quality row per (song, part, method) to the experiment folder.

Usage: python -m src.pipelines.align_songs --config experiments/<exp>/configs/align_songs.yaml [--songs ... [--merge]]
"""
from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.align import warp
from src.align.stems import load_part_stems
from src.datasets.six_dragon import list_musicxml_pieces
from src.evaluate.alignment import measure_part
from src.features import chroma
from src.notes.musicxml import musicxml_to_note_table
from src.pipelines.common import load_run_config


def process_song(row: dict, config) -> tuple[list[dict], dict]:
    """Refine + align + measure one matched song. Returns (quality rows, song summary)."""
    six_dragon, warp_config = config.datasets.six_dragon, config.align.warp
    pieces = list_musicxml_pieces(Path(six_dragon.musicxml_dir), list(six_dragon.part_names))
    piece = pieces.set_index("piece").loc[row["piece"]]
    table = musicxml_to_note_table(Path(piece["path"]), list(piece["parts"]))
    stems = load_part_stems(row["song_id"], list(row["stems"]), list(piece["parts"]), config.datasets.separation,
                            config.datasets.instruments)
    parts = list(stems)
    mix = np.sum(list(stems.values()), axis=0)
    audio_seconds = len(mix) / chroma.FEATURE_SAMPLE_RATE
    tuning = int(row["tuning_cents"])

    # 1. Refine the section boundaries
    chroma_10hz = chroma.audio_features(mix, warp_config.refine_feature_rate, tuning, with_onsets=False)["chroma"]
    section, refine_cost = warp.refine_section(chroma_10hz, table, row["start_quarter"], row["end_quarter"],
                                               row["quarters_per_second"], row["chroma_shift"],
                                               audio_seconds, warp_config)
    section_table = warp.crop_section(table[table["part"].isin(parts)], section.start_quarter, section.end_quarter)
    grace = config.notes.table.grace_seconds

    # 2. Three timings: linear baseline, one path per song, one path per stem
    timed = {"linear": warp.warp_notes(section_table, section.linear_map, grace)}
    song_map, song_path = warp.align_section(mix, tuning, section_table, section, row["chroma_shift"], warp_config)
    timed["per_song"] = warp.warp_notes(section_table, song_map, grace)
    per_stem = []
    for part in parts:
        part_table = section_table[section_table["part"] == part]
        stem_map, _ = warp.align_section(stems[part], tuning, part_table, section, row["chroma_shift"], warp_config)
        per_stem.append(warp.warp_notes(part_table, stem_map, grace))
    timed["per_stem"] = pd.concat(per_stem, ignore_index=True)

    out_dir = Path(config.run.cache_dir) / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, notes in timed.items():
        notes.assign(song_id=row["song_id"], piece=row["piece"]).to_parquet(out_dir / f"{row['song_id']}_{method}.parquet")
    np.save(out_dir / f"{row['song_id']}_per_song_path.npy", song_path)

    # 3. Quality per stem
    quality_rows = []
    for part in parts:
        for q_row in measure_part(part, stems[part], chroma.FEATURE_SAMPLE_RATE, timed, config.evaluate.alignment,
                                  config.features.f0, config.features.onsets):
            quality_rows.append({"song_id": row["song_id"], "piece": row["piece"], **q_row})
    summary = {"song_id": row["song_id"], "piece": row["piece"], "audio_seconds": audio_seconds,
               "section_start_quarter": section.start_quarter, "section_end_quarter": section.end_quarter,
               "section_quarters_per_second": section.quarters_per_second, "refine_cost": refine_cost,
               "parts_aligned": parts, "notes": int((~section_table["is_grace"]).sum())}
    return quality_rows, summary


def _job(args):
    row, config = args
    try:
        return process_song(row, config)
    except Exception:
        return [], {"song_id": row["song_id"], "piece": row["piece"], "error": traceback.format_exc()}


def main(config_path: str, song_ids: list[str] | None, merge: bool) -> None:
    config = load_run_config(config_path)
    tables_dir = Path(config.run.tables_dir)
    matches = pd.read_parquet(tables_dir / "match_table.parquet")
    accepted = matches[matches["confidence"].isin(["high", "medium", "low"])]
    if song_ids:
        accepted = accepted[accepted["song_id"].isin(song_ids)]
    jobs = [(r, config) for r in accepted.to_dict("records")]
    print(f"aligning {len(jobs)} songs")

    quality_rows, summaries = [], []
    with ProcessPoolExecutor(config.run.num_workers) as pool:
        for rows, summary in pool.map(_job, jobs):
            quality_rows.extend(rows)
            summaries.append(summary)
            print(summary["song_id"], "ERROR" if "error" in summary else "ok", flush=True)
            if "error" in summary:
                print(summary["error"], flush=True)
    new_quality, new_sections = pd.DataFrame(quality_rows), pd.DataFrame(summaries)
    if song_ids and merge:  # replace these songs' rows in the full tables
        for name, new in (("alignment_quality", new_quality), ("alignment_sections", new_sections)):
            full = pd.read_parquet(tables_dir / f"{name}.parquet")
            full = pd.concat([full[~full["song_id"].isin(song_ids)], new], ignore_index=True)
            full.to_parquet(tables_dir / f"{name}.parquet")
        return
    suffix = "" if not song_ids else "_subset"
    new_quality.to_parquet(tables_dir / f"alignment_quality{suffix}.parquet")
    new_sections.to_parquet(tables_dir / f"alignment_sections{suffix}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/align_songs.yaml")
    parser.add_argument("--songs", nargs="*", help="limit to these song_ids")
    parser.add_argument("--merge", action="store_true", help="with --songs: update the full tables")
    arguments = parser.parse_args()
    main(arguments.config, arguments.songs, arguments.merge)
