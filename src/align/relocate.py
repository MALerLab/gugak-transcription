"""Re-place consecutive 장 of one movement jointly (fixes double-time spans); optional match step.

Found in exp001: in slow movements (관악/평조 상령산, 중령산) the one-recording-at-a-time
locate sometimes matches a 장 to twice as much score as it really covers, so the spans of
consecutive 장 overlap. Groups listed in the run config (``run.groups``) are placed again
with one path through their concatenated recordings (``locate.locate_consecutive_sections``).

Command: ``python -m src.pipelines.relocate_songs``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.align import locate
from src.align.stems import load_part_stems
from src.features import chroma
from src.notes.musicxml import musicxml_to_note_table
from src.notes.table import quarters_to_seconds


def relocate_group(group, matches: pd.DataFrame, pieces: pd.DataFrame, separation_config, instruments_config,
                   warp_config) -> list[dict]:
    """New spans for one group of consecutive recordings of the same piece.

    Args:
        group: one entry of the run's ``groups`` (songs in order, anchor).
        separation_config: the `configs/datasets/separation.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings.
        warp_config: the `configs/align/warp.yaml` settings (its refine CENS settings are reused).
    """
    rows = matches.set_index("song_id").loc[list(group.songs)]
    pieces_used = rows["piece"].unique()
    if len(pieces_used) != 1:
        raise ValueError(f"group mixes pieces: {pieces_used}")
    piece = pieces.set_index("piece").loc[pieces_used[0]]
    # Fine CENS (10 frames/s, as in align's refine step) of each recording's matched stems
    feature_rate, smooth, downsample = (warp_config.refine_feature_rate, warp_config.refine_cens_smooth_frames,
                                        warp_config.refine_cens_downsample)
    audio_cens = []
    for song_id in group.songs:
        stems = load_part_stems(song_id, list(rows.at[song_id, "stems"]), list(piece["parts"]), separation_config,
                                 instruments_config)
        mix = np.sum(list(stems.values()), axis=0)
        audio_chroma = chroma.audio_features(mix, feature_rate, int(rows.at[song_id, "tuning_cents"]), with_onsets=False)["chroma"]
        audio_cens.append(chroma.coarse_cens(audio_chroma, feature_rate, smooth, downsample))
    frame_rate = feature_rate / downsample

    # Score tempo: the whole movement over the whole recording when both ends are anchored
    table = musicxml_to_note_table(Path(piece["path"]), list(piece["parts"]))
    table = table[~table["is_grace"]]
    total_quarters = float((table["onset_quarter"] + table["duration_quarter"]).max())
    total_seconds = sum(a.shape[1] for a in audio_cens) / frame_rate
    quarters_per_second = total_quarters / total_seconds if group.anchor == "both" else float(group.quarters_per_second)
    timed = quarters_to_seconds(table, lambda q: q / quarters_per_second, 0.0)
    score_chroma = chroma.score_features(timed, feature_rate, with_onsets=False)["chroma"]
    score_cens = chroma.coarse_cens(score_chroma, feature_rate, smooth, downsample)
    shift = int(rows["chroma_shift"].iloc[0])
    spans = locate.locate_consecutive_sections(audio_cens, score_cens, frame_rate, quarters_per_second,
                                               shift, anchor_end=group.anchor == "both")
    out = []
    for song_id, (start, end), cens in zip(group.songs, spans, audio_cens):
        out.append({"song_id": song_id, "start_quarter": start, "end_quarter": end,
                    "quarters_per_second": (end - start) / (cens.shape[1] / frame_rate)})
    return out
