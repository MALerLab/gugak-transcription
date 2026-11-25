"""Note table → frame-level label rolls (the format Onsets and Frames trains on), and time windows → frame masks.

Rolls follow Jong Wook Kim's Onsets and Frames repo (``onsets_and_frames/dataset.py``): per key, the
onset roll is on for the first frames of a note, the frame roll from its onset to its end, and the
offset roll for the frames right after its end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def note_table_to_label_rolls(table: pd.DataFrame, num_frames: int, hop_seconds: float, min_pitch: int,
                              num_keys: int, onset_frames: int, offset_frames: int,
                              onset_shape: str = "hard", onset_half_width: float = 1.0) -> dict[str, np.ndarray]:
    """Onset, offset and frame rolls of a note table, each ``(num_frames, num_keys)`` uint8.

    ``onset_shape``: ``hard`` (the repo's: 1 for ``onset_frames`` frames from the nearest frame) or ``triangle``:
    the onset roll is float16, 1 − |frame − exact start| / ``onset_half_width`` (in frames) where positive, so the
    targets carry the start's position between frames; half-width 1 splits each start between its two neighbouring
    frames, wider ones also spread it to match how uncertain the label's timing is.

    Args:
        table: note table (``pitch``, ``start``, ``duration`` in seconds).
        num_frames: roll length.
        hop_seconds: frame hop.
        min_pitch: MIDI number of key 0.
        num_keys: number of keys; pitches outside ``min_pitch .. min_pitch + num_keys - 1`` are skipped.
        onset_frames: frames the onset roll stays on from each note start.
        offset_frames: frames the offset roll stays on after each note end.
    """
    rolls = {name: np.zeros((num_frames, num_keys), np.uint8) for name in ("onset", "offset", "frame")}
    start_frames = np.round(table["start"].to_numpy() / hop_seconds).astype(int)
    end_frames = np.round((table["start"] + table["duration"]).to_numpy() / hop_seconds).astype(int)
    keys = table["pitch"].to_numpy().astype(int) - min_pitch
    if onset_shape not in ("hard", "triangle"):
        raise ValueError(f"unknown onset_shape {onset_shape!r}")
    if onset_shape == "triangle":
        rolls["onset"] = np.zeros((num_frames, num_keys), np.float16)
    exact_starts = table["start"].to_numpy() / hop_seconds
    for start, end, key, exact_start in zip(start_frames, end_frames, keys, exact_starts):
        if not 0 <= key < num_keys or start >= num_frames:
            continue
        onset_end = min(start + onset_frames, num_frames)
        end = min(max(end, onset_end), num_frames)
        rolls["frame"][start:end, key] = 1
        if onset_shape == "hard":
            rolls["onset"][start:onset_end, key] = 1
        else:
            first = int(np.floor(exact_start - onset_half_width)) + 1
            for frame in range(max(first, 0), min(int(np.ceil(exact_start + onset_half_width)), num_frames)):
                weight = 1 - abs(frame - exact_start) / onset_half_width
                if weight > 0:
                    rolls["onset"][frame, key] = max(rolls["onset"][frame, key], weight)
        rolls["offset"][end:min(end + offset_frames, num_frames), key] = 1
    return rolls


def windows_to_frame_mask(windows: np.ndarray, num_frames: int, hop_seconds: float) -> np.ndarray:
    """``(num_frames,)`` bool: False inside any ``(start, end)`` window in seconds, True elsewhere."""
    mask = np.ones(num_frames, bool)
    for start, end in np.asarray(windows).reshape(-1, 2):
        mask[max(int(np.floor(start / hop_seconds)), 0):max(int(np.ceil(end / hop_seconds)), 0)] = False
    return mask
