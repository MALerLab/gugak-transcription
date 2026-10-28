"""The note table: one row per note, the most complete record of the notes (PROJECT.md → vocabulary).

Columns in use:
- ``pitch``: MIDI number (as written on the score side, on the tuned grid from note-ify)
- ``start``, ``duration``: seconds
- ``part``: MusicXML part name or stem instrument; one MIDI track per part
- ``velocity``: from note-ify (the score side gets a constant from config when written)
- ``is_grace``: score side only
- ``onset_quarter``, ``duration_quarter``: score side only, quarter-note beats before alignment
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def note_table_to_frames(table: pd.DataFrame, hop_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """(start frame, end frame exclusive) of each note on a frame grid of ``hop_seconds``."""
    start_frames = np.round(table["start"].to_numpy() / hop_seconds).astype(int)
    end_frames = np.round((table["start"] + table["duration"]).to_numpy() / hop_seconds).astype(int)
    return start_frames, end_frames
