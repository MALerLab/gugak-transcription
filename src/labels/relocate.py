"""Stage 1b: re-place consecutive 장 of one movement jointly (fixes double-time spans).

Found in exp001: in slow movements (관악/평조 상령산, 중령산) the one-recording-at-a-time
locate sometimes matches a 장 to twice as much score as it really covers, so the spans of
consecutive 장 overlap. Groups listed in ``configs/labels/relocate.yaml`` are placed again
with one path through their concatenated recordings (``locate.locate_consecutive_sections``).

Usage: python -m src.labels.relocate --configs configs/labels
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.labels import convert, features, locate
from src.labels.pipeline import load_configs, load_part_stems


def relocate_group(group, matches: pd.DataFrame, pieces: pd.DataFrame, configs) -> list[dict]:
    """New spans for one group of consecutive recordings of the same piece."""
    match_config, align_config = configs.match, configs.align
    rows = matches.set_index("song_id").loc[list(group.songs)]
    pieces_used = rows["piece"].unique()
    if len(pieces_used) != 1:
        raise ValueError(f"group mixes pieces: {pieces_used}")
    piece = pieces.set_index("piece").loc[pieces_used[0]]
    # Fine CENS (10 frames/s, as in align's refine step) of each recording's matched stems
    feature_rate, smooth, downsample = (align_config.refine_feature_rate, align_config.refine_cens_smooth_frames,
                                        align_config.refine_cens_downsample)
    audio_cens = []
    for song_id in group.songs:
        stems = load_part_stems(song_id, list(rows.at[song_id, "stems"]), list(piece["parts"]), configs)
        mix = np.sum(list(stems.values()), axis=0)
        chroma = features.audio_features(mix, feature_rate, int(rows.at[song_id, "tuning_cents"]), with_onsets=False)["chroma"]
        audio_cens.append(features.coarse_cens(chroma, feature_rate, smooth, downsample))
    frame_rate = feature_rate / downsample

    # Score tempo: the whole movement over the whole recording when both ends are anchored
    table = convert.musicxml_to_note_table(Path(piece["path"]), list(piece["parts"]))
    table = table[~table["is_grace"]]
    total_quarters = float((table["onset_quarter"] + table["duration_quarter"]).max())
    total_seconds = sum(a.shape[1] for a in audio_cens) / frame_rate
    quarters_per_second = total_quarters / total_seconds if group.anchor == "both" else float(group.quarters_per_second)
    timed = convert.quarters_to_seconds(table, lambda q: q / quarters_per_second, 0.0)
    chroma = features.score_features(timed, feature_rate, with_onsets=False)["chroma"]
    score_cens = features.coarse_cens(chroma, feature_rate, smooth, downsample)
    shift = int(rows["chroma_shift"].iloc[0])
    spans = locate.locate_consecutive_sections(audio_cens, score_cens, frame_rate, quarters_per_second,
                                               shift, anchor_end=group.anchor == "both")
    out = []
    for song_id, (start, end), cens in zip(group.songs, spans, audio_cens):
        out.append({"song_id": song_id, "start_quarter": start, "end_quarter": end,
                    "quarters_per_second": (end - start) / (cens.shape[1] / frame_rate)})
    return out


def main(config_dir: str) -> None:
    configs = load_configs(config_dir)
    match_config, convert_config = configs.match, configs.convert
    relocate_config = OmegaConf.load(Path(config_dir) / "relocate.yaml")
    tables_dir = Path(match_config.tables_dir)
    table_path = tables_dir / "match_table.parquet"
    stage1_copy = tables_dir / "match_table_stage1.parquet"
    if not stage1_copy.exists():  # keep the one-at-a-time spans for the report
        shutil.copy(table_path, stage1_copy)
    matches = pd.read_parquet(stage1_copy)
    pieces = convert.list_musicxml_pieces(Path(convert_config.musicxml_dir), list(convert_config.part_names))

    matches["relocated"] = False
    for group in relocate_config.groups:
        for new in relocate_group(group, matches, pieces, configs):
            index = matches.index[matches["song_id"] == new["song_id"]][0]
            print(new["song_id"], f"{matches.at[index, 'start_quarter']:.0f}-{matches.at[index, 'end_quarter']:.0f}",
                  "→", f"{new['start_quarter']:.0f}-{new['end_quarter']:.0f}", f"qps {new['quarters_per_second']:.2f}")
            for key in ("start_quarter", "end_quarter", "quarters_per_second"):
                matches.at[index, key] = new[key]
            matches.at[index, "relocated"] = True
            if matches.at[index, "confidence"] == "rejected":  # alone it lost; placed with its group
                matches.at[index, "confidence"] = "low"
    for song_id in relocate_config.get("readmit", []):
        matches.loc[matches["song_id"] == song_id, "confidence"] = "low"
    matches.to_parquet(table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs/labels")
    main(parser.parse_args().configs)
