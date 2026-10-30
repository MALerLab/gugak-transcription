"""f0 accuracy against reference notes: is the tracked pitch the note's pitch, frame by frame?

The reference notes become one pitch per tracker frame. Only frames in the middle of a note
(``note_core_fraction``) where no other note sounds are scored, so the alignment's timing error at
note edges and ringing overlaps stay out. Voicing is ignored: every tracker here gives a pitch on
every frame, and they mark silence differently, so these are pitch numbers, not voicing numbers.

- **raw pitch accuracy**: share of scored frames within ``pitch_tolerance_cents`` of the reference
- **raw chroma accuracy**: the same, with octave errors forgiven
- **best-shift raw pitch accuracy**: the reference moved by whichever whole-semitone shift in
  ``shift_semitones`` suits the tracker best (the solo MIDI's whole-clip semitone gaps)
"""
from __future__ import annotations

import mir_eval
import numpy as np


def reference_pitch_frames(reference_intervals: np.ndarray, reference_pitches: np.ndarray,
                           times: np.ndarray, config) -> np.ndarray:
    """The reference pitch (MIDI number) on each frame; NaN where the frame isn't scored.

    Args:
        reference_intervals: (start, end) in seconds per note.
        reference_pitches: MIDI number per note.
        times: the tracker's frame times in seconds.
        config: the `configs/evaluate/f0.yaml` settings.
    """
    starts, ends = reference_intervals[:, 0], reference_intervals[:, 1]
    durations = ends - starts

    # How many notes sound on each frame, over their full length
    sounding = np.zeros(len(times) + 1, dtype=int)
    np.add.at(sounding, np.searchsorted(times, starts), 1)
    np.add.at(sounding, np.searchsorted(times, ends), -1)
    sounding = np.cumsum(sounding)[:-1]

    # Each long-enough note's pitch on the frames in its core
    low, high = config.note_core_fraction
    core_firsts = np.searchsorted(times, starts + low * durations)
    core_lasts = np.searchsorted(times, starts + high * durations)
    reference = np.full(len(times), np.nan)
    for note in np.nonzero(durations >= config.min_note_seconds)[0]:
        reference[core_firsts[note]:core_lasts[note]] = reference_pitches[note]
    reference[sounding != 1] = np.nan
    return reference


def raw_pitch_and_chroma_accuracy(reference_midi: np.ndarray, f0_hz: np.ndarray,
                                  tolerance_cents: float) -> tuple[float, float]:
    """mir_eval's raw pitch and raw chroma accuracy over the frames with a reference pitch.

    Args:
        reference_midi: reference pitch per frame (NaN = not scored).
        f0_hz: tracked f0 per frame; NaN or 0 (unvoiced) counts as wrong.
        tolerance_cents: how far off still counts as right.
    """
    scored = np.isfinite(reference_midi)
    reference_cents = mir_eval.melody.hz2cents(mir_eval.util.midi_to_hz(reference_midi[scored]))
    estimate_cents = mir_eval.melody.hz2cents(np.nan_to_num(f0_hz[scored], nan=0.0))
    voicing = np.ones(scored.sum())  # every scored frame counts; the tracker's voicing is ignored
    return (mir_eval.melody.raw_pitch_accuracy(voicing, reference_cents, voicing, estimate_cents, tolerance_cents),
            mir_eval.melody.raw_chroma_accuracy(voicing, reference_cents, voicing, estimate_cents, tolerance_cents))


def f0_scores(reference_intervals: np.ndarray, reference_pitches: np.ndarray, track: dict, config) -> dict:
    """One item's f0 scores against its reference notes.

    Args:
        reference_intervals: (start, end) in seconds per note.
        reference_pitches: MIDI number per note.
        track: ``time`` and ``f0_hz`` arrays from src/features/f0/.
        config: the `configs/evaluate/f0.yaml` settings.
    """
    reference = reference_pitch_frames(reference_intervals, reference_pitches, track["time"], config)
    scored_frames = int(np.isfinite(reference).sum())
    hop_seconds = float(np.median(np.diff(track["time"])))
    if scored_frames == 0:
        return {"scored_frames": 0, "scored_seconds": 0.0}
    raw_pitch, raw_chroma = raw_pitch_and_chroma_accuracy(reference, track["f0_hz"], config.pitch_tolerance_cents)

    # Best whole-semitone shift of the reference, picked per item and per tracker
    shifted = {shift: raw_pitch_and_chroma_accuracy(reference + shift, track["f0_hz"], config.pitch_tolerance_cents)[0]
               for shift in config.shift_semitones}
    best_shift = max(shifted, key=lambda shift: (shifted[shift], -abs(shift)))  # ties go to the smaller shift
    return {"scored_frames": scored_frames, "scored_seconds": scored_frames * hop_seconds,
            "raw_pitch_accuracy": raw_pitch, "raw_chroma_accuracy": raw_chroma,
            "best_shift_raw_pitch_accuracy": shifted[best_shift], "best_shift_semitones": best_shift}
