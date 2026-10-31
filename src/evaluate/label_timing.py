"""Label timing: do a reference's notes sit early or late against the audio, by a steady amount or a wandering one?

The reference notes are shifted in time against an independent judge of the audio, and the shift
where they fit best is read off. Shift convention: negative = the notes moved earlier, so a best
shift of −60 ms means the labels run 60 ms late.

Two judges:
- **pitch**: a tracked f0 curve. Score per shift = share of in-note frames whose tracked pitch is
  within the tolerance of the note's pitch. Whole notes are scored (not just their middles), since
  the timing evidence sits where the pitch changes.
- **onsets**: note starts detected from the audio (spectral flux), matched to the reference note
  starts. Only meaningful for plucked instruments, whose attacks the detector sees.
"""
from __future__ import annotations

import mir_eval
import numpy as np

from src.evaluate.f0 import reference_pitch_frames


def shifts_in_frames(config, hop_seconds: float) -> np.ndarray:
    """The tested shifts as whole tracker frames, from ``shift_range_seconds`` (both ends included)."""
    low, high = config.shift_range_seconds
    return np.arange(int(round(low / hop_seconds)), int(round(high / hop_seconds)) + 1)


def tracked_midi(f0_hz: np.ndarray) -> np.ndarray:
    """f0 in Hz → fractional MIDI number; NaN where unvoiced (0 or NaN)."""
    f0_hz = np.nan_to_num(f0_hz, nan=0.0)
    midi = np.full(len(f0_hz), np.nan)
    voiced = f0_hz > 0
    midi[voiced] = 69 + 12 * np.log2(f0_hz[voiced] / 440.0)
    return midi


def pitch_hits_by_shift(reference_midi: np.ndarray, measured_midi: np.ndarray, shifts: np.ndarray,
                        tolerance_cents: float) -> tuple[np.ndarray, np.ndarray]:
    """Per shift and per frame: is the frame scored, and is it a hit, with the reference moved by ``shift`` frames.

    Args:
        reference_midi: the reference pitch per tracker frame (NaN = no single note sounding).
        measured_midi: the tracked pitch per frame (NaN = unvoiced, counts as a miss).
        shifts: whole-frame shifts of the reference; negative = earlier.
        tolerance_cents: how far off still counts as a hit.

    Returns:
        (scored, hit): boolean arrays of shape (shifts, frames).
    """
    frame_count = len(reference_midi)
    scored = np.zeros((len(shifts), frame_count), bool)
    hit = np.zeros((len(shifts), frame_count), bool)
    for row, shift in enumerate(shifts):
        # The reference value that lands on frame t after the shift came from frame t − shift
        shifted = np.full(frame_count, np.nan)
        if shift >= 0:
            shifted[shift:] = reference_midi[:frame_count - shift]
        else:
            shifted[:shift] = reference_midi[-shift:]
        scored[row] = np.isfinite(shifted)
        hit[row] = scored[row] & (np.abs(np.nan_to_num(measured_midi, nan=-1e3) - shifted) * 100 < tolerance_cents)
    return scored, hit


def pitch_scores_by_shift(reference_intervals: np.ndarray, reference_pitches: np.ndarray, times: np.ndarray,
                          f0_hz: np.ndarray, shifts: np.ndarray, config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame scored / hit arrays for every shift, plus where the reference pitch changes.

    Returns:
        (scored, hit, pitch_change): the first two as in ``pitch_hits_by_shift``; ``pitch_change``
        marks frames where the unshifted reference pitch differs from the frame before (the timing evidence).
    """
    reference = reference_pitch_frames(reference_intervals, reference_pitches, times, config.pitch_reference)
    scored, hit = pitch_hits_by_shift(reference, tracked_midi(f0_hz), shifts, config.pitch_tolerance_cents)
    pitch_change = np.zeros(len(reference), bool)
    both = np.isfinite(reference[1:]) & np.isfinite(reference[:-1])
    pitch_change[1:] = both & (reference[1:] != reference[:-1])
    return scored, hit, pitch_change


def onset_scores_by_shift(reference_onsets: np.ndarray, detected_onsets: np.ndarray, shift_seconds: np.ndarray,
                          tolerance_seconds: float) -> np.ndarray:
    """Onset F-measure (time only, pitch ignored) of the reference note starts moved by each shift."""
    if len(reference_onsets) == 0 or len(detected_onsets) == 0:
        return np.zeros(len(shift_seconds))
    reference_onsets, detected_onsets = np.sort(reference_onsets), np.sort(detected_onsets)
    return np.array([mir_eval.onset.f_measure(reference_onsets + shift, detected_onsets, window=tolerance_seconds)[0]
                     for shift in shift_seconds])


def best_shift(shift_seconds: np.ndarray, scores: np.ndarray) -> float:
    """The shift with the highest score; ties go to the shift closest to zero."""
    best = np.flatnonzero(scores == np.nanmax(scores))
    return float(shift_seconds[best[np.argmin(np.abs(shift_seconds[best]))]])


def window_best_shifts(scored: np.ndarray, hit: np.ndarray, pitch_change: np.ndarray, times: np.ndarray,
                       shift_seconds: np.ndarray, config) -> list[dict]:
    """The best pitch shift inside each ``window_seconds`` stretch of one item.

    A window counts only with enough scored audio and enough pitch changes to carry timing evidence;
    the rest are returned with ``best_shift_seconds`` NaN.
    """
    hop_seconds = float(np.median(np.diff(times)))
    window_frames = int(round(config.window_seconds / hop_seconds))
    zero = int(np.flatnonzero(shift_seconds == 0)[0])
    rows = []
    for start in range(0, len(times), window_frames):
        frames = slice(start, start + window_frames)
        scored_seconds = scored[zero, frames].sum() * hop_seconds
        changes = int(pitch_change[frames].sum())
        accuracy = hit[:, frames].sum(axis=1) / np.maximum(scored[:, frames].sum(axis=1), 1)
        usable = scored_seconds >= config.min_window_scored_seconds and changes >= config.min_window_pitch_changes
        rows.append({"window_start_seconds": float(times[start]), "scored_seconds": float(scored_seconds),
                     "pitch_changes": changes,
                     "best_shift_seconds": best_shift(shift_seconds, accuracy) if usable else np.nan,
                     "accuracy_at_zero": float(accuracy[zero]), "accuracy_at_best": float(accuracy.max())})
    return rows


def drift_summary(windows: list[dict]) -> dict:
    """Steady vs wandering, from one item's usable windows.

    - ``window_shift_iqr_seconds``: spread of the window best shifts (middle half); small = steady
    - ``window_shift_slope_seconds_per_minute``: straight-line trend of the best shift over the item; a slow drift
    """
    usable = [w for w in windows if np.isfinite(w["best_shift_seconds"])]
    if len(usable) < 2:
        return {"usable_windows": len(usable), "window_shift_median_seconds": np.nan,
                "window_shift_iqr_seconds": np.nan, "window_shift_slope_seconds_per_minute": np.nan}
    shifts = np.array([w["best_shift_seconds"] for w in usable])
    starts = np.array([w["window_start_seconds"] for w in usable])
    q25, q75 = np.percentile(shifts, [25, 75])
    slope = np.polyfit(starts / 60.0, shifts, 1)[0] if np.ptp(starts) > 0 else np.nan
    return {"usable_windows": len(usable), "window_shift_median_seconds": float(np.median(shifts)),
            "window_shift_iqr_seconds": float(q75 - q25), "window_shift_slope_seconds_per_minute": float(slope)}
