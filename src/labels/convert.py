"""MusicXML → note table → MIDI.

A note table is one row per note: ``part``, ``onset_quarter``, ``duration_quarter``, ``pitch``
(MIDI number as written), ``is_grace``. Score time is in quarter notes; the six dragon
MusicXML has no tempo marks, so seconds only exist once a tempo or an alignment is applied.
"""
from __future__ import annotations

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import partitura as pt
import pretty_midi


def parse_musicxml_filename(path: Path, part_names: list[str]) -> tuple[str, list[str]]:
    """Split ``<piece>_<part>_<part>.musicxml`` into the piece title and its part list.

    Args:
        path: MusicXML file path.
        part_names: every known part name, used to find where the part suffix starts.
    """
    stem = unicodedata.normalize("NFC", path.stem)
    part_pattern = "|".join(part_names)
    match = re.match(rf"(.+?)_((?:{part_pattern})(?:_(?:{part_pattern}))*)$", stem)
    if match is None:
        raise ValueError(f"unexpected MusicXML filename: {path.name}")
    return match.group(1), match.group(2).split("_")


def list_musicxml_pieces(musicxml_dir: Path, part_names: list[str]) -> pd.DataFrame:
    """One row per MusicXML file: ``piece``, ``suite``, ``movement``, ``parts``, ``path``.

    ``suite`` is the first word of a two-word title (e.g. 영산회상 in "영산회상 상령산"),
    empty for single-word titles like 보허자.
    """
    rows = []
    for path in sorted(Path(musicxml_dir).glob("*.musicxml")):
        piece, parts = parse_musicxml_filename(path, part_names)
        suite, _, movement = piece.rpartition(" ")
        rows.append({"piece": piece, "suite": suite, "movement": movement,
                     "parts": parts, "path": str(path)})
    return pd.DataFrame(rows)


def musicxml_to_note_table(path: Path, part_names_in_file: list[str]) -> pd.DataFrame:
    """Read every part of a MusicXML file into one note table (ties merged).

    Args:
        path: MusicXML file.
        part_names_in_file: part names in score order, from the filename.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # partitura warns about every tuplet/cadenza guess
        score = pt.load_musicxml(str(path))
    if len(score.parts) != len(part_names_in_file):
        raise ValueError(f"{path.name}: {len(score.parts)} parts vs {len(part_names_in_file)} in name")

    rows = []
    for part, part_name in zip(score.parts, part_names_in_file):
        quarter_of = part.quarter_map  # score divisions → quarter notes
        for note in part.notes_tied:  # notes_tied: one object per tied chain
            is_grace = isinstance(note, pt.score.GraceNote)
            onset = float(quarter_of(note.start.t))
            duration = 0.0 if is_grace else float(quarter_of(note.end_tied.t)) - onset
            rows.append({"part": part_name, "onset_quarter": onset,
                         "duration_quarter": duration, "pitch": int(note.midi_pitch),
                         "is_grace": is_grace})
    table = pd.DataFrame(rows)
    return table.sort_values(["part", "onset_quarter", "is_grace"],
                             ascending=[True, True, False]).reset_index(drop=True)


def quarters_to_seconds(table: pd.DataFrame, quarter_to_second, grace_seconds: float) -> pd.DataFrame:
    """Add ``start`` and ``duration`` in seconds through a score-time → audio-time map.

    Grace notes take the first ``grace_seconds`` of their main note, which starts that much later.

    Args:
        table: note table.
        quarter_to_second: vectorised function, quarter-note positions → seconds.
        grace_seconds: fixed grace-note length.
    """
    timed = table.copy()
    start = quarter_to_second(timed["onset_quarter"].to_numpy())
    end = quarter_to_second((timed["onset_quarter"] + timed["duration_quarter"]).to_numpy())
    grace = timed["is_grace"].to_numpy()
    end = np.where(grace, start + grace_seconds, end)

    # Delay each main note that has a grace note at the same onset in the same part
    keys = pd.MultiIndex.from_arrays([timed["part"], timed["onset_quarter"].round(6)])
    grace_keys = set(keys[grace])
    delayed = np.array([(k in grace_keys) for k in keys]) & ~grace
    start = np.where(delayed, np.minimum(start + grace_seconds, end - 1e-3), start)

    timed["start"] = start
    timed["duration"] = np.maximum(end - start, 1e-3)
    return timed


def note_table_to_midi(timed: pd.DataFrame, path: Path, gm_programs: dict[str, int],
                       velocity: int, pitch_shift: dict[str, int] | None = None) -> None:
    """Write a timed note table as a MIDI file, one track per part.

    Args:
        timed: note table with ``start``/``duration`` in seconds.
        path: output ``.mid`` path.
        gm_programs: part → General MIDI program (0-based).
        velocity: velocity for every note.
        pitch_shift: part → semitones added to the written pitch (sounding-pitch correction).
    """
    pitch_shift = pitch_shift or {}
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for part_name, notes in timed.groupby("part", sort=False):
        track = pretty_midi.Instrument(program=gm_programs[part_name], name=part_name)
        shift = pitch_shift.get(part_name, 0)
        for row in notes.itertuples():
            track.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(row.pitch + shift),
                                                start=float(row.start),
                                                end=float(row.start + row.duration)))
        midi.instruments.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
