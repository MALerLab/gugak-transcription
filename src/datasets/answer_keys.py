"""Answer keys for scoring transcriptions: which audio goes with which reference MIDI.

- six dragon: exp001's per-stem aligned MIDI (MusicXML labels at sounding pitch), one part per
  single stem of the separation dataset, usable songs only
- solo: the solo dataset's clips (exp002's pick), each against its own MIDI

One row per scored item: item_id, answer_key, source_id (song or clip), instrument, split,
audio_path, reference_midi_path, reference_part (the MIDI track to score against; None = all).
"""
from __future__ import annotations

from pathlib import Path

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
