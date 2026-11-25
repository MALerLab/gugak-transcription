"""Answer keys for scoring transcriptions: which audio goes with which reference MIDI.

- six dragon: exp001's per-stem aligned MIDI (MusicXML labels at sounding pitch), one part per
  single stem of the separation dataset, usable songs only
- solo: the solo dataset's clips (exp002's pick, or a training split table), each against its own MIDI

One row per scored item: item_id, answer_key, source_id (song or clip), instrument, split,
audio_path, reference_midi_path, reference_part (the MIDI track to score against; None = all).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.separation import nfc, song_stem_paths


def six_dragon_stem_pairs(quality_table: str, match_table: str, midi_dir: str, splits: list[str],
                          separation_config, instruments_config) -> pd.DataFrame:
    """Every (stem, part) of exp001's usable songs in the given splits.

    Args:
        quality_table: exp001's ``alignment_quality_by_song.parquet`` (``usable``, ``split`` per song).
        match_table: exp001's ``match_table.parquet`` (``stems_with_part``: stems that got labels).
        midi_dir: exp001's ``aligned_midi/per_stem/`` (``<song>.mid``, one track per part).
        splits: frozen-split names to keep, e.g. ``[val, test]``.
        separation_config: the `configs/datasets/separation.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings.
    """
    quality = pd.read_parquet(quality_table)
    songs = quality[quality["usable"] & quality["split"].isin(list(splits))]
    stems_with_part = pd.read_parquet(match_table).set_index("song_id")["stems_with_part"]
    rows = []
    for song_id, song in songs.iterrows():
        stems = [nfc(s) for s in stems_with_part.loc[song_id]]
        for instrument, stem_path in song_stem_paths(song_id, stems, separation_config, instruments_config).items():
            rows.append({"item_id": f"{song_id}_{instrument}", "answer_key": "six_dragon", "source_id": song_id,
                         "instrument": instrument, "split": song["split"], "audio_path": str(stem_path),
                         "reference_midi_path": str(Path(midi_dir) / f"{song_id}.mid"), "reference_part": instrument})
    return pd.DataFrame(rows)


def six_dragon_grace_windows(aligned_cache_dir: str, song_id: str, part: str, seconds_before: float,
                             seconds_after: float) -> np.ndarray:
    """``(n, 2)`` time windows in seconds around the grace notes of one song's part, from exp001's per-stem note table.

    exp001 wrote each grace note (시김새 acciaccatura) as a made-up fixed-length note and started its main
    note that much later (src/notes/table.py), so neither timing comes from the audio. Training and
    scoring leave these windows out: what sounds there is neither taught nor graded.

    Args:
        aligned_cache_dir: exp001's ``cache/aligned/`` (``<song>_per_stem.parquet``, with ``is_grace``).
        song_id: separation dataset song.
        part: instrument (= MusicXML part name).
        seconds_before: window start before the grace note's start.
        seconds_after: window end after the grace note's start (covers the grace note and the main note's start).
    """
    table = pd.read_parquet(Path(aligned_cache_dir) / f"{song_id}_per_stem.parquet", columns=["part", "is_grace", "start"])
    starts = np.unique(table.loc[table["is_grace"] & (table["part"] == part), "start"].to_numpy())
    return np.stack([starts - seconds_before, starts + seconds_after], axis=1).reshape(-1, 2)


def solo_clip_pairs(clips_table: str, partition: str, reference_check_table: str,
                    max_reference_transposition: int) -> pd.DataFrame:
    """The solo clips of one partition, minus references exp002 found broken.

    Args:
        clips_table: exp002's ``solo_clips.parquet``.
        partition: ``dev`` or ``test`` (exp002's split by piece).
        reference_check_table: exp002's scores for that partition, with ``reference_transposition``
            (whole-semitone gap between the reference MIDI and the tracked pitch).
        max_reference_transposition: larger gaps (octave errors and worse) mark a broken reference.
    """
    clips = pd.read_parquet(clips_table)
    clips = clips[clips["partition"] == partition]
    check = pd.read_parquet(reference_check_table).set_index("clip_id")["reference_transposition"]
    broken = check.index[check.abs() > max_reference_transposition]
    clips = clips[~clips["clip_id"].isin(broken)]
    return pd.DataFrame({"item_id": clips["clip_id"], "answer_key": "solo", "source_id": clips["clip_id"],
                         "instrument": clips["instrument"], "split": partition, "audio_path": clips["clip_path"],
                         "reference_midi_path": clips["midi_path"], "reference_part": None}).reset_index(drop=True)


def solo_split_pairs(split_table: str, splits: list[str]) -> pd.DataFrame:
    """The solo clips of the given splits of a training split table, minus the ones with a ``drop_reason``.

    Args:
        split_table: a split_solo_clips table (``split``, ``drop_reason`` per clip; src/datasets/solo.py).
        splits: split names to keep, e.g. ``[train]``.
    """
    clips = pd.read_parquet(split_table)
    clips = clips[clips["split"].isin(list(splits)) & clips["drop_reason"].isna()]
    # genre_family: tables from before the stratified split (exp012) don't carry it
    genre_family = clips["genre_family"] if "genre_family" in clips else None
    return pd.DataFrame({"item_id": clips["clip_id"], "answer_key": "solo", "source_id": clips["clip_id"],
                         "instrument": clips["instrument"], "split": clips["split"], "audio_path": clips["clip_path"],
                         "reference_midi_path": clips["midi_path"], "reference_part": None,
                         "genre_family": genre_family}).reset_index(drop=True)
