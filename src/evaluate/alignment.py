"""Alignment-quality numbers: does each aligned note line up with what the stem plays?

- **pitch hit rate**: the stem's f0 (YIN in exp001, ``run.f0_tracker`` in config) in the middle of a note is within half a semitone
  of the note's pitch. Monophonic stems only make sense here (대금, 피리, 해금, 아쟁);
  plucked 가야금/거문고 are reported too, with that caveat.
- **onset F-measure**: note onsets vs spectral-flux onsets detected in the stem.

Every number is reported next to the same notes shifted in time (chance level), and next to
the linear (unaligned) baseline, so "0.6" can be read as better or worse than nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.f0 import track_f0
from src.features.onsets import spectral_flux_onsets
from src.features.tuning import octave_and_tuning


def note_f0_midi(timed: pd.DataFrame, f0: np.ndarray, times: np.ndarray, config) -> np.ndarray:
    """Median f0 (as fractional MIDI) in the core of each note; NaN where too short/unvoiced."""
    f0_midi = 69 + 12 * np.log2(f0 / 440.0)
    low, high = config.note_core_fraction
    out = np.full(len(timed), np.nan)
    for i, (start, duration) in enumerate(zip(timed["start"].to_numpy(), timed["duration"].to_numpy())):
        if duration < config.min_note_seconds:
            continue
        lo, hi = np.searchsorted(times, [start + low * duration, start + high * duration])
        values = f0_midi[lo:hi]
        values = values[~np.isnan(values)]
        if len(values) >= config.min_voiced_frames:
            out[i] = np.median(values)
    return out


def pitch_hit_rate(written_pitch: np.ndarray, measured_midi: np.ndarray, shift: int,
                   tuning_cents: float, tolerance: float) -> tuple[float, int]:
    """Share of checkable notes whose f0 is within ``tolerance`` semitones; and how many were checkable."""
    deviation = measured_midi - (written_pitch + shift + tuning_cents / 100)
    checkable = ~np.isnan(deviation)
    if checkable.sum() == 0:
        return np.nan, 0
    return float((np.abs(deviation[checkable]) < tolerance).mean()), int(checkable.sum())


def onset_f_measure(reference: np.ndarray, estimated: np.ndarray, tolerance: float) -> float:
    """Greedy one-to-one onset matching within ``tolerance``; F-measure."""
    if len(reference) == 0 or len(estimated) == 0:
        return np.nan
    reference, estimated = np.sort(reference), np.sort(estimated)
    used = np.zeros(len(estimated), dtype=bool)
    hits = 0
    for onset in reference:
        lo, hi = np.searchsorted(estimated, [onset - tolerance, onset + tolerance])
        free = [j for j in range(lo, hi) if not used[j]]
        if free:
            used[min(free, key=lambda j: abs(estimated[j] - onset))] = True
            hits += 1
    precision, recall = hits / len(estimated), hits / len(reference)
    return 2 * precision * recall / (precision + recall) if hits else 0.0


def measure_part(part: str, stem: np.ndarray, sample_rate: int, timed_by_method: dict[str, pd.DataFrame],
                 config, f0_tracker: str, f0_config, onsets_config) -> list[dict]:
    """Pitch hit rate + onset F for one stem, for every timing method and the chance level.

    Args:
        part: the part whose notes are checked.
        stem: that part's mono stem audio.
        sample_rate: its rate.
        timed_by_method: method name → aligned note table (must include ``per_song``).
        config: the `configs/evaluate/alignment.yaml` settings.
        f0_tracker: the tracker to run (``yin`` in exp001).
        f0_config: that tracker's `configs/features/f0/<tracker>.yaml` settings.
        onsets_config: the `configs/features/onsets.yaml` settings.
    """
    tracked = track_f0(stem, sample_rate, f0_tracker, f0_config)
    f0, times = tracked["f0_hz"], tracked["time"]
    detected_onsets = spectral_flux_onsets(stem, sample_rate, onsets_config.spectral_flux)

    # Octave/tuning from the per-song alignment, then used for every method alike
    reference = timed_by_method["per_song"]
    reference = reference[(reference["part"] == part) & ~reference["is_grace"]]
    shift, tuning = octave_and_tuning(reference["pitch"].to_numpy(),
                                      note_f0_midi(reference, f0, times, config))
    rows = []

    def score(name: str, timed: pd.DataFrame, offset: float = 0.0):
        notes = timed[(timed["part"] == part) & ~timed["is_grace"]].copy()
        notes["start"] = notes["start"] + offset
        measured = note_f0_midi(notes, f0, times, config)
        hit_rate, checked = pitch_hit_rate(notes["pitch"].to_numpy(), measured, shift, tuning,
                                           config.pitch_hit_semitones)
        onset_f = onset_f_measure(notes["start"].to_numpy(), detected_onsets, config.onset_tolerance_seconds)
        return {"method": name, "pitch_hit_rate": hit_rate, "notes_checked": checked,
                "notes": len(notes), "onset_f": onset_f}

    for method, timed in timed_by_method.items():
        rows.append(score(method, timed))
    chance = [score("chance", timed_by_method["per_song"], offset) for offset in config.chance_offsets_seconds]
    rows.append({"method": "chance", "pitch_hit_rate": np.nanmean([c["pitch_hit_rate"] for c in chance]),
                 "notes_checked": chance[0]["notes_checked"], "notes": chance[0]["notes"],
                 "onset_f": np.nanmean([c["onset_f"] for c in chance])})
    for row in rows:
        row.update({"part": part, "sounding_shift": shift, "tuning_cents_f0": tuning})
    return rows
