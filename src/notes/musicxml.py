"""MusicXML → note table (score side, time in quarter-note beats).

The six dragon MusicXML has no tempo marks, so seconds only exist once a tempo or an
alignment is applied (``src/notes/table.py``).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import partitura as pt


def musicxml_to_note_table(path: Path, part_names_in_file: list[str]) -> pd.DataFrame:
    """Read every part of a MusicXML file into one note table (ties merged).

    Columns: ``part``, ``onset_quarter``, ``duration_quarter``, ``pitch`` (MIDI number as
    written), ``is_grace``.

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
