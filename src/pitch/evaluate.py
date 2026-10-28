"""Score note events against a reference MIDI (the solo dataset's), overall and inside 시김새 regions."""
from __future__ import annotations

import json

import mir_eval
import numpy as np
import pretty_midi

from src.pitch.notes import Note


def reference_notes(midi_path: str) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches as MIDI numbers) of every note in a MIDI file, sorted by onset."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = sorted((n for inst in midi.instruments if not inst.is_drum for n in inst.notes), key=lambda n: n.start)
    intervals = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in notes]).reshape(-1, 2)
    return intervals, np.array([n.pitch for n in notes], dtype=float)


def estimated_notes(notes: list[Note], hop_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """(intervals in s, pitches as MIDI numbers) of note-ified output."""
    intervals = np.array([[n.start_frame * hop_seconds, n.end_frame * hop_seconds] for n in notes]).reshape(-1, 2)
    return intervals, np.array([n.pitch for n in notes], dtype=float)


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
