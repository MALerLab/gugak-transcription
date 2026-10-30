"""Transcription scores: note tables vs a reference MIDI (the solo dataset's), overall and inside
시김새 regions; plus reference-free diagnostics for stems that have no reference."""
from __future__ import annotations

import json

import mir_eval
import numpy as np
import pandas as pd

from src.notes.midi import midi_to_note_table
from src.notes.table import note_table_to_frames


def reference_notes(midi_path: str, part: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches as MIDI numbers) of every note in a MIDI file, sorted by onset.

    ``part`` keeps only that track (by name); None keeps every track.
    """
    table = midi_to_note_table(midi_path)
    if part is not None:
        table = table[table["part"] == part]
    start = table["start"].to_numpy()
    intervals = np.stack([start, np.maximum(start + table["duration"].to_numpy(), start + 1e-3)], axis=1).reshape(-1, 2)
    return intervals, table["pitch"].to_numpy(dtype=float)


def estimated_notes(notes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches as MIDI numbers) of a note table."""
    start = notes["start"].to_numpy(dtype=float)
    intervals = np.stack([start, start + notes["duration"].to_numpy(dtype=float)], axis=1).reshape(-1, 2)
    return intervals, notes["pitch"].to_numpy(dtype=float)


def reference_transposition(reference_intervals: np.ndarray, reference_pitches: np.ndarray,
                            semitones: np.ndarray, voiced: np.ndarray, hop_seconds: float) -> int:
    """Median whole-semitone gap between the tracked pitch and the reference MIDI where both sound.

    Non-zero means the reference MIDI is written transposed against the audio (seen in a few clips).
    """
    reference_roll = np.full(len(semitones), np.nan)
    for (start, end), pitch in zip(reference_intervals, reference_pitches):
        reference_roll[int(round(start / hop_seconds)):int(round(end / hop_seconds))] = pitch
    both = voiced & ~np.isnan(reference_roll)
    if both.sum() < 20:
        return 0
    return int(np.median(np.round(semitones[both] - reference_roll[both])))


def oracle_segmentation_pitch_accuracy(reference_intervals, reference_pitches, semitones: np.ndarray,
                                       voiced: np.ndarray, hop_seconds: float) -> float:
    """Share of reference notes whose pitch the tracker gets right when handed the reference note boundaries.

    Each reference note gets the rounded median of the tracked (tuned) pitch over its voiced frames.
    Separates pitch-tracking error from segmentation error: a high value with a low note F1 means
    the f0 is fine and the note cutting is what fails.
    """
    correct, counted = 0, 0
    for (start, end), pitch in zip(reference_intervals, reference_pitches):
        frames = slice(int(round(start / hop_seconds)), max(int(round(end / hop_seconds)), int(round(start / hop_seconds)) + 1))
        mask = voiced[frames]
        counted += 1
        if mask.any():
            correct += int(np.round(np.median(semitones[frames][mask]))) == int(pitch)
    return correct / counted if counted else 0.0


def note_scores(reference_intervals, reference_pitches, estimate_intervals, estimate_pitches, config) -> dict:
    """Note-level precision / recall / F1: onset + pitch, and onset + pitch + offset."""
    to_hz = lambda p: 440.0 * 2 ** ((p - 69) / 12)
    scores = {}
    if len(reference_pitches) == 0 or len(estimate_pitches) == 0:
        return {**{f"{k}_{m}": 0.0 for k in ("onset", "onset_offset") for m in ("precision", "recall", "f1")},
                "onset_f1_lenient": 0.0}
    for name, offset_ratio in (("onset", None), ("onset_offset", config.offset_ratio)):
        precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            reference_intervals, to_hz(reference_pitches), estimate_intervals, to_hz(estimate_pitches),
            onset_tolerance=config.onset_tolerance_seconds, pitch_tolerance=config.pitch_tolerance_cents,
            offset_ratio=offset_ratio)
        scores.update({f"{name}_precision": precision, f"{name}_recall": recall, f"{name}_f1": f1})
    _, _, scores["onset_f1_lenient"], _ = mir_eval.transcription.precision_recall_f1_overlap(
        reference_intervals, to_hz(reference_pitches), estimate_intervals, to_hz(estimate_pitches),
        onset_tolerance=config.lenient_onset_tolerance_seconds, pitch_tolerance=config.pitch_tolerance_cents,
        offset_ratio=None)
    return scores


def onset_only_scores(reference_intervals, estimate_intervals, config) -> dict:
    """Note onsets matched by time alone, pitch ignored: does the model hear that a note starts at all."""
    if len(reference_intervals) == 0 or len(estimate_intervals) == 0:
        return {"onset_only_precision": 0.0, "onset_only_recall": 0.0, "onset_only_f1": 0.0}
    f1, precision, recall = mir_eval.onset.f_measure(np.sort(reference_intervals[:, 0]), np.sort(estimate_intervals[:, 0]),
                                                     window=config.onset_tolerance_seconds)
    return {"onset_only_precision": precision, "onset_only_recall": recall, "onset_only_f1": f1}


def sigimsae_breakdown(sigimsae_json: str, reference_intervals, reference_pitches,
                       estimate_intervals, estimate_pitches, config) -> list[dict]:
    """Per 시김새 region type, and for notes outside every region: how notes are cut there.

    Counts reference notes, estimated notes and matched notes (onset + pitch) whose onsets fall
    inside each region type. estimated / reference > 1 means the region is chopped into extra
    notes; < 1 means notes are merged or missed.
    """
    to_hz = lambda p: 440.0 * 2 ** ((p - 69) / 12)
    matches = []
    if len(reference_pitches) and len(estimate_pitches):
        matches = mir_eval.transcription.match_notes(
            reference_intervals, to_hz(reference_pitches), estimate_intervals, to_hz(estimate_pitches),
            onset_tolerance=config.onset_tolerance_seconds, pitch_tolerance=config.pitch_tolerance_cents,
            offset_ratio=None)
    matched_reference = {r for r, _ in matches}
    regions = json.loads(sigimsae_json)

    def inside(onsets: np.ndarray, start: float, end: float) -> np.ndarray:
        return (onsets >= start - config.onset_tolerance_seconds) & (onsets < end)

    rows = []
    reference_onsets, estimate_onsets = reference_intervals[:, 0], estimate_intervals[:, 0]
    in_any_reference = np.zeros(len(reference_onsets), bool)
    in_any_estimate = np.zeros(len(estimate_onsets), bool)
    for kind in sorted({r[0] for r in regions}):
        reference_in = np.zeros(len(reference_onsets), bool)
        estimate_in = np.zeros(len(estimate_onsets), bool)
        for name, start, end in regions:
            if name == kind:
                reference_in |= inside(reference_onsets, start, end)
                estimate_in |= inside(estimate_onsets, start, end)
        in_any_reference |= reference_in
        in_any_estimate |= estimate_in
        rows.append({"region": kind, "reference_notes": int(reference_in.sum()), "estimated_notes": int(estimate_in.sum()),
                     "matched_notes": int(sum(1 for r in np.flatnonzero(reference_in) if r in matched_reference))})
    outside_reference = np.flatnonzero(~in_any_reference)
    rows.append({"region": "outside", "reference_notes": int(len(outside_reference)),
                 "estimated_notes": int((~in_any_estimate).sum()),
                 "matched_notes": int(sum(1 for r in outside_reference if r in matched_reference))})
    return rows


def note_roll(notes: pd.DataFrame, num_frames: int, hop_seconds: float) -> np.ndarray:
    """Per-frame pitch of the note covering it, NaN where no note sounds."""
    roll = np.full(num_frames, np.nan)
    start_frames, end_frames = note_table_to_frames(notes, hop_seconds)
    for start_frame, end_frame, pitch in zip(start_frames, end_frames, notes["pitch"].to_numpy()):
        roll[start_frame:end_frame] = pitch
    return roll


def stem_diagnostics(result: dict, confidence: np.ndarray, hop_seconds: float) -> dict:
    """Reference-free numbers that hint at how well a stem was note-ified.

    - ``off_note_fraction``: share of in-note frames whose pitch sits ≥ 30 cents off the note's
      snapped pitch — time spent bending, sliding or in wide vibrato (a 시김새 proxy).
    - ``far_off_note_fraction``: same at ≥ 50 cents — where the MIDI note is plainly not the sounding pitch.
    - ``uncovered_fraction``: voiced frames that ended up in no note (dropped as too short).
    """
    notes, semitones, voiced = result["notes"], result["semitones"], result["voiced"]
    roll = note_roll(notes, len(semitones), hop_seconds)
    in_note = ~np.isnan(roll)
    deviation = np.abs(semitones[in_note] - roll[in_note])
    duration = len(semitones) * hop_seconds
    return {
        "tuning_offset_cents": result["tuning_offset_cents"],
        "note_count": len(notes),
        "notes_per_second": len(notes) / max(voiced.sum() * hop_seconds, 1e-6),
        "voiced_fraction": float(voiced.mean()),
        "mean_confidence_voiced": float(confidence[voiced].mean()) if voiced.any() else 0.0,
        "off_note_fraction": float((deviation >= 0.3).mean()) if in_note.any() else 0.0,
        "far_off_note_fraction": float((deviation >= 0.5).mean()) if in_note.any() else 0.0,
        "uncovered_fraction": float((voiced & ~in_note).sum() / max(voiced.sum(), 1)),
        "duration_seconds": duration,
    }
