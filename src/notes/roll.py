"""Note table → frame-level label rolls (the format Onsets and Frames trains on), and time windows → frame masks.

Rolls follow Jong Wook Kim's Onsets and Frames repo (``onsets_and_frames/dataset.py``): per key, the
onset roll is on for the first frames of a note, the frame roll from its onset to its end, and the
offset roll for the frames right after its end.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def note_table_to_label_rolls(table: pd.DataFrame, num_frames: int, hop_seconds: float, min_pitch: int,
                              num_keys: int, onset_frames: int, offset_frames: int) -> dict[str, np.ndarray]:
    """Onset, offset and frame rolls of a note table, each ``(num_frames, num_keys)`` uint8.

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
    for start, end, key in zip(start_frames, end_frames, keys):
        if not 0 <= key < num_keys or start >= num_frames:
            continue
        onset_end = min(start + onset_frames, num_frames)
        end = min(max(end, onset_end), num_frames)
        rolls["frame"][start:end, key] = 1
        rolls["onset"][start:onset_end, key] = 1
        rolls["offset"][end:min(end + offset_frames, num_frames), key] = 1
    return rolls


def windows_to_frame_mask(windows: np.ndarray, num_frames: int, hop_seconds: float) -> np.ndarray:
    """``(num_frames,)`` bool: False inside any ``(start, end)`` window in seconds, True elsewhere."""
    mask = np.ones(num_frames, bool)
    for start, end in np.asarray(windows).reshape(-1, 2):
        mask[max(int(np.floor(start / hop_seconds)), 0):max(int(np.ceil(end / hop_seconds)), 0)] = False
    return mask
