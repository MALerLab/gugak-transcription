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


def reference_pitch_offset(reference_midi: np.ndarray, f0_hz: np.ndarray, tolerance_cents: float) -> dict:
    """How a reference's written pitch sits against the tracked pitch, over the frames with a reference pitch.

    - ``semitone_gap``: the most common whole-semitone gap, tracked − written (0 fine, ±1 a key-naming
      clash or wrong notes, ±12 an octave error)
    - ``residual_cents``: mean of what's left once that gap is taken out, over frames within half a
      semitone of it: how far the recording sits off the reference's keys. A mean, not a median:
      stock PESTO's output clusters on its 1/3-semitone bins, so a median lands on 0 or ±33 cents
    - ``agreement`` / ``agreement_after_gap``: share of frames within ``tolerance_cents``, as written /
      with the reference moved by ``semitone_gap``

    Args:
        reference_midi: reference pitch per frame (NaN = not scored), from ``reference_pitch_frames``.
        f0_hz: tracked f0 per frame; NaN or 0 (unvoiced) count as misses, and don't vote on the gap.
        tolerance_cents: how far off still counts as agreeing.
    """
    scored = np.isfinite(reference_midi)
    f0_hz = np.nan_to_num(f0_hz, nan=0.0)
    deviation = np.full(len(reference_midi), np.nan)
    voiced = scored & (f0_hz > 0)
    deviation[voiced] = 69 + 12 * np.log2(f0_hz[voiced] / 440.0) - reference_midi[voiced]
    voted = deviation[np.isfinite(deviation)]
    if len(voted) == 0:
        return {"scored_frames": int(scored.sum()), "semitone_gap": np.nan, "residual_cents": np.nan,
                "agreement": np.nan, "agreement_after_gap": np.nan}

    # The gap = most common whole-semitone deviation; the residual = how far off the gap the recording sits
    gaps, counts = np.unique(np.round(voted).astype(int), return_counts=True)
    semitone_gap = int(gaps[np.argmax(counts)])
    near_gap = voted[np.abs(voted - semitone_gap) < 0.5]
    tolerance = tolerance_cents / 100.0
    return {"scored_frames": int(scored.sum()), "semitone_gap": semitone_gap,
            "residual_cents": float(np.mean(near_gap - semitone_gap) * 100),
            "agreement": float((np.abs(deviation[scored]) < tolerance).sum() / scored.sum()),
            "agreement_after_gap": float((np.abs(deviation[scored] - semitone_gap) < tolerance).sum() / scored.sum())}


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
