"""Stages 3 + quality for every accepted match: refine, align (per song and per stem), measure.

Per song it writes the aligned note tables (written pitch, three timings: linear, per_song,
per_stem) to the cache, and one quality row per (song, part, method) to the experiment folder.

Usage: python -m src.labels.pipeline --configs configs/labels
"""
from __future__ import annotations

import argparse
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.labels import align, convert, features, match, quality


def load_configs(config_dir: str):
    """All stage configs as one namespace: config.match, config.convert, …"""
    names = ["match", "convert", "align", "quality", "render"]
    return OmegaConf.create({n: OmegaConf.load(Path(config_dir) / f"{n}.yaml") for n in names})


def cache_dir(config) -> Path:
    return Path(os.path.expanduser(config.match.cache_dir))


def load_part_stems(song_id: str, stems: list[str], parts: list[str], config) -> dict[str, np.ndarray]:
    """Part → mono 22.05 kHz stem audio, all trimmed to the shortest stem."""
    paths = match.song_stem_paths(song_id, stems, config.match)
    audio = {}
    for part in parts:
        if part in paths:
            signal, sample_rate = features.load_stem_mix([paths[part]])
            audio[part] = features.to_feature_rate(signal, sample_rate)
    length = min(len(a) for a in audio.values())
    return {part: a[:length] for part, a in audio.items()}


def measure_part(part: str, stem_22k: np.ndarray, timed_by_method: dict[str, pd.DataFrame], config) -> list[dict]:
    """Pitch hit rate + onset F for one stem, for every timing method and the chance level."""
    q = config.quality
    sample_rate = features.FEATURE_SAMPLE_RATE
    f0, times = quality.yin_f0(stem_22k, sample_rate, q)
    detected_onsets = quality.spectral_flux_onsets(stem_22k, sample_rate, q)

    # Octave/tuning from the per-song alignment, then used for every method alike
    reference = timed_by_method["per_song"]
    reference = reference[(reference["part"] == part) & ~reference["is_grace"]]
    shift, tuning = quality.octave_and_tuning(reference["pitch"].to_numpy(),
                                              quality.note_f0_midi(reference, f0, times, q))
    rows = []

    def score(name: str, timed: pd.DataFrame, offset: float = 0.0):
        notes = timed[(timed["part"] == part) & ~timed["is_grace"]].copy()
        notes["start"] = notes["start"] + offset
        measured = quality.note_f0_midi(notes, f0, times, q)
        hit_rate, checked = quality.pitch_hit_rate(notes["pitch"].to_numpy(), measured, shift, tuning,
                                                   q.pitch_hit_semitones)
        onset_f = quality.onset_f_measure(notes["start"].to_numpy(), detected_onsets, q.onset_tolerance_seconds)
        return {"method": name, "pitch_hit_rate": hit_rate, "notes_checked": checked,
                "notes": len(notes), "onset_f": onset_f}

    for method, timed in timed_by_method.items():
        rows.append(score(method, timed))
    chance = [score("chance", timed_by_method["per_song"], offset) for offset in q.chance_offsets_seconds]
    rows.append({"method": "chance", "pitch_hit_rate": np.nanmean([c["pitch_hit_rate"] for c in chance]),
                 "notes_checked": chance[0]["notes_checked"], "notes": chance[0]["notes"],
                 "onset_f": np.nanmean([c["onset_f"] for c in chance])})
    for row in rows:
        row.update({"part": part, "sounding_shift": shift, "tuning_cents_f0": tuning})
    return rows


def process_song(row: dict, config) -> tuple[list[dict], dict]:
    """Refine + align + measure one matched song. Returns (quality rows, song summary)."""
    pieces = convert.list_musicxml_pieces(Path(config.convert.musicxml_dir), list(config.convert.part_names))
    piece = pieces.set_index("piece").loc[row["piece"]]
    table = convert.musicxml_to_note_table(Path(piece["path"]), list(piece["parts"]))
    stems = load_part_stems(row["song_id"], list(row["stems"]), list(piece["parts"]), config)
    parts = list(stems)
    mix = np.sum(list(stems.values()), axis=0)
    audio_seconds = len(mix) / features.FEATURE_SAMPLE_RATE
    tuning = int(row["tuning_cents"])

    # 1. Refine the section boundaries
    chroma_10hz = features.audio_features(mix, config.align.refine_feature_rate, tuning, with_onsets=False)["chroma"]
    section, refine_cost = align.refine_section(chroma_10hz, table, row["start_quarter"], row["end_quarter"],
                                                row["quarters_per_second"], row["chroma_shift"],
                                                audio_seconds, config.align)
    section_table = align.crop_section(table[table["part"].isin(parts)], section.start_quarter, section.end_quarter)
    grace = config.convert.grace_seconds

    # 2. Three timings: linear baseline, one path per song, one path per stem
    timed = {"linear": align.warp_notes(section_table, section.linear_map, grace)}
    song_map, song_path = align.align_section(mix, tuning, section_table, section, row["chroma_shift"], config.align)
    timed["per_song"] = align.warp_notes(section_table, song_map, grace)
    per_stem = []
    for part in parts:
        part_table = section_table[section_table["part"] == part]
        stem_map, _ = align.align_section(stems[part], tuning, part_table, section, row["chroma_shift"], config.align)
        per_stem.append(align.warp_notes(part_table, stem_map, grace))
    timed["per_stem"] = pd.concat(per_stem, ignore_index=True)

    out_dir = cache_dir(config) / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, notes in timed.items():
        notes.assign(song_id=row["song_id"], piece=row["piece"]).to_parquet(out_dir / f"{row['song_id']}_{method}.parquet")
    np.save(out_dir / f"{row['song_id']}_per_song_path.npy", song_path)

    # 3. Quality per stem
    quality_rows = []
    for part in parts:
        for q_row in measure_part(part, stems[part], timed, config):
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


def main(config_dir: str, song_ids: list[str] | None, merge: bool) -> None:
    config = load_configs(config_dir)
    output_dir = Path(config.match.output_dir)
    matches = pd.read_parquet(output_dir / "match_table.parquet")
    accepted = matches[matches["confidence"].isin(["high", "medium", "low"])]
    if song_ids:
        accepted = accepted[accepted["song_id"].isin(song_ids)]
    jobs = [(r, config) for r in accepted.to_dict("records")]
    print(f"aligning {len(jobs)} songs")

    quality_rows, summaries = [], []
    with ProcessPoolExecutor(config.match.num_workers) as pool:
        for rows, summary in pool.map(_job, jobs):
            quality_rows.extend(rows)
            summaries.append(summary)
            print(summary["song_id"], "ERROR" if "error" in summary else "ok", flush=True)
            if "error" in summary:
                print(summary["error"], flush=True)
    new_quality, new_sections = pd.DataFrame(quality_rows), pd.DataFrame(summaries)
    if song_ids and merge:  # replace these songs' rows in the full tables
        for name, new in (("alignment_quality", new_quality), ("alignment_sections", new_sections)):
            full = pd.read_parquet(output_dir / f"{name}.parquet")
            full = pd.concat([full[~full["song_id"].isin(song_ids)], new], ignore_index=True)
            full.to_parquet(output_dir / f"{name}.parquet")
        return
    suffix = "" if not song_ids else "_subset"
    new_quality.to_parquet(output_dir / f"alignment_quality{suffix}.parquet")
    new_sections.to_parquet(output_dir / f"alignment_sections{suffix}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs/labels")
    parser.add_argument("--songs", nargs="*", help="limit to these song_ids")
    parser.add_argument("--merge", action="store_true", help="with --songs: update the full tables")
    arguments = parser.parse_args()
    main(arguments.configs, arguments.songs, arguments.merge)
