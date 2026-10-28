"""Note-ify: an f0 curve (+ the stem's energy) → a note table.

Steps: frame energy → voicing → tuning offset from A440 → segmentation into notes → note table.
Two segmentation methods: ``naive`` (snap every frame to a semitone, merge equal runs) and
``stability`` (hysteresis on a smoothed pitch curve, so vibrato and short bends stay inside
one note, plus energy onsets to split repeated same-pitch notes).

Segmentation works on f0 frames (``FrameNote``); only the output is in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from src.features.onsets import onset_frames
from src.features.tuning import estimate_tuning_offset, hz_to_cents_from_a440


@dataclass
class FrameNote:
    """A note while segmentation works on it, in f0 frames."""
    start_frame: int
    end_frame: int  # exclusive
    pitch: int      # MIDI note number on the tuning-corrected grid


# --- frame features ---------------------------------------------------------

def frame_rms_db(mono: np.ndarray, sample_rate: int, num_frames: int, hop_seconds: float, config) -> np.ndarray:
    """RMS in dBFS of a window centred on each f0 frame; silence (below the floor) → -240 dB."""
    hop = int(round(hop_seconds * sample_rate))
    half_window = int(round(config.energy_window_seconds * sample_rate)) // 2
    padded = np.pad(mono.astype(np.float64), half_window)
    prefix = np.concatenate(([0.0], np.cumsum(padded * padded)))
    centres = np.arange(num_frames) * hop
    starts = np.clip(centres, 0, len(padded) - 1)
    ends = np.clip(centres + 2 * half_window, 0, len(padded))
    rms = np.sqrt(np.maximum(prefix[ends] - prefix[starts], 0) / np.maximum(ends - starts, 1))
    return np.where(rms > config.silence_floor, 20 * np.log10(np.maximum(rms, 1e-12)), -240.0)


def voiced_mask(confidence: np.ndarray, rms_db: np.ndarray, hop_seconds: float, config) -> np.ndarray:
    """Frames that carry a note: confident f0 and loud enough, with short gaps bridged."""
    audible = rms_db > -240.0
    loud_level = np.percentile(rms_db[audible], 95) if audible.any() else 0.0
    voiced = (confidence >= config.confidence_threshold) & (rms_db >= loud_level - config.energy_gate_db)
    return bridge_short_gaps(voiced, int(round(config.gap_fill_seconds / hop_seconds)))


def bridge_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Set False runs of at most ``max_gap_frames`` that sit between True frames to True."""
    mask = mask.copy()
    runs = runs_of_true(~mask)
    for start, end in runs:
        if start > 0 and end < len(mask) and end - start <= max_gap_frames:
            mask[start:end] = True
    return mask


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each contiguous True run, end exclusive."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


# --- segmentation -----------------------------------------------------------

def segment_naive(semitones: np.ndarray, voiced: np.ndarray, min_note_frames: int) -> list[FrameNote]:
    """Snap every voiced frame to the nearest semitone and merge equal neighbours."""
    notes = []
    snapped = np.round(semitones).astype(int)
    for region_start, region_end in runs_of_true(voiced):
        start = region_start
        for frame in range(region_start + 1, region_end + 1):
            if frame == region_end or snapped[frame] != snapped[start]:
                notes.append(FrameNote(start, frame, int(snapped[start])))
                start = frame
    return [n for n in notes if n.end_frame - n.start_frame >= min_note_frames]


def segment_region_by_hysteresis(curve: np.ndarray, offset: int, hysteresis: float, min_switch_frames: int) -> list[FrameNote]:
    """Split one voiced region where the pitch leaves the current note far enough, for long enough.

    The new note is back-dated to the frame where the departure began.
    """
    notes = []
    current = int(np.round(curve[0]))
    note_start = 0
    candidate, departure_start = None, None
    for frame, value in enumerate(curve):
        if abs(value - current) <= hysteresis:
            candidate = None
            continue
        rounded = int(np.round(value))
        if rounded != candidate:
            candidate, departure_start = rounded, frame
        if frame - departure_start + 1 >= min_switch_frames:
            notes.append(FrameNote(offset + note_start, offset + departure_start, current))
            current, note_start, candidate = candidate, departure_start, None
    notes.append(FrameNote(offset + note_start, offset + len(curve), current))
    return notes


def split_at_onsets(notes: list[FrameNote], onsets: np.ndarray, min_note_frames: int) -> list[FrameNote]:
    """Cut notes at energy onsets that fall inside them (not too close to either end)."""
    result = []
    for note in notes:
        inside = onsets[(onsets >= note.start_frame + min_note_frames) & (onsets <= note.end_frame - min_note_frames)]
        bounds = [note.start_frame, *inside.tolist(), note.end_frame]
        result += [FrameNote(a, b, note.pitch) for a, b in zip(bounds[:-1], bounds[1:])]
    return result


def merge_short_notes(notes: list[FrameNote], min_note_frames: int) -> list[FrameNote]:
    """Fold notes shorter than the minimum into a touching neighbour; drop isolated ones."""
    notes = list(notes)
    changed = True
    while changed:
        changed = False
        for index, note in enumerate(notes):
            if note.end_frame - note.start_frame >= min_note_frames:
                continue
            previous = notes[index - 1] if index > 0 and notes[index - 1].end_frame == note.start_frame else None
            following = notes[index + 1] if index + 1 < len(notes) and notes[index + 1].start_frame == note.end_frame else None
            if previous is not None and (following is None or
                                         abs(previous.pitch - note.pitch) <= abs(following.pitch - note.pitch)):
                previous.end_frame = note.end_frame
            elif following is not None:
                following.start_frame = note.start_frame
            del notes[index]
            changed = True
            break
    return notes


def repitch_and_merge(notes: list[FrameNote], semitones: np.ndarray) -> list[FrameNote]:
    """Give each note the median pitch of its frames, then merge touching notes that now share a pitch."""
    for note in notes:
        note.pitch = int(np.round(np.median(semitones[note.start_frame:note.end_frame])))
    merged = []
    for note in notes:
        if merged and merged[-1].end_frame == note.start_frame and merged[-1].pitch == note.pitch:
            merged[-1].end_frame = note.end_frame
        else:
            merged.append(note)
    return merged


def segment_stability(semitones: np.ndarray, voiced: np.ndarray, rms_db: np.ndarray, hop_seconds: float, config,
                      onsets_config) -> list[FrameNote]:
    """Hysteresis segmentation of each voiced region, then onset splits and short-note merging."""
    median_frames = max(1, int(round(config.median_filter_seconds / hop_seconds)))
    min_switch_frames = max(1, int(round(config.min_switch_seconds / hop_seconds)))
    min_note_frames = max(1, int(round(config.min_note_seconds / hop_seconds)))
    notes = []
    for region_start, region_end in runs_of_true(voiced):
        curve = median_filter(semitones[region_start:region_end], size=median_frames, mode="nearest")
        notes += segment_region_by_hysteresis(curve, region_start, config.hysteresis_semitones, min_switch_frames)
    notes = repitch_and_merge(notes, semitones)
    if config.onset_split:
        notes = split_at_onsets(notes, onset_frames(rms_db, hop_seconds, onsets_config.energy), min_note_frames)
    notes = merge_short_notes(notes, min_note_frames)
    # merging keeps the survivor's pitch; re-pitch over the grown span but keep onset splits
    for note in notes:
        note.pitch = int(np.round(np.median(semitones[note.start_frame:note.end_frame])))
    return notes


# --- top level --------------------------------------------------------------

def note_velocities(notes: list[FrameNote], rms_db: np.ndarray, velocity_range) -> list[int]:
    """Map each note's peak RMS (dB) linearly onto the velocity range, per recording."""
    if not notes:
        return []
    peaks = np.array([rms_db[n.start_frame:n.end_frame].max() for n in notes])
    low, high = np.percentile(peaks, 5), np.percentile(peaks, 95)
    scaled = np.clip((peaks - low) / max(high - low, 1e-6), 0, 1)
    return [int(round(velocity_range[0] + s * (velocity_range[1] - velocity_range[0]))) for s in scaled]


def frame_notes_to_note_table(notes: list[FrameNote], velocities: list[int], hop_seconds: float) -> pd.DataFrame:
    """Frame notes → note table: ``start``, ``duration`` (s), ``pitch``, ``velocity``."""
    start = np.array([n.start_frame * hop_seconds for n in notes], dtype=float)
    end = np.array([n.end_frame * hop_seconds for n in notes], dtype=float)
    return pd.DataFrame({"start": start, "duration": end - start,
                         "pitch": np.array([n.pitch for n in notes], dtype=int),
                         "velocity": np.array(velocities, dtype=int)})


def notes_from_f0(track: dict, mono: np.ndarray, sample_rate: int, hop_seconds: float, config,
                  tuning_config, onsets_config) -> dict:
    """Run the whole note-ify step on one f0 track.

    Args:
        track: ``time``, ``f0_hz``, ``confidence`` arrays from src/features/f0.py.
        mono: the same audio the f0 was tracked on (for energy).
        sample_rate: its sample rate.
        hop_seconds: the f0 hop.
        config: the `configs/transcribe/note_ify.yaml` settings.
        tuning_config: the `configs/features/tuning.yaml` settings.
        onsets_config: the `configs/features/onsets.yaml` settings.

    Returns:
        dict with ``notes`` (note table, one part), ``tuning_offset_cents``, ``semitones`` (tuned,
        fractional MIDI numbers per frame), ``voiced`` and ``rms_db`` per frame.
    """
    num_frames = len(track["f0_hz"])
    rms_db = frame_rms_db(mono, sample_rate, num_frames, hop_seconds, config)
    voiced = voiced_mask(track["confidence"], rms_db, hop_seconds, config)
    cents = hz_to_cents_from_a440(track["f0_hz"])
    tuning_offset_cents = estimate_tuning_offset(cents, voiced, tuning_config)
    snap_offset_cents = tuning_offset_cents if config.tuning_correction else 0.0
    semitones = 69 + (cents - snap_offset_cents) / 100.0
    if config.method == "naive":
        notes = segment_naive(semitones, voiced, max(1, int(round(config.min_note_seconds / hop_seconds))))
    else:
        notes = segment_stability(semitones, voiced, rms_db, hop_seconds, config, onsets_config)
    table = frame_notes_to_note_table(notes, note_velocities(notes, rms_db, config.velocity_range), hop_seconds)
    return {"notes": table, "tuning_offset_cents": tuning_offset_cents, "semitones": semitones,
            "voiced": voiced, "rms_db": rms_db}
