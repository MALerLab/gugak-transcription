"""Command: does each solo clip's MIDI match its own audio, in timing and in pitch?

Every solo clip (AI Hub 71470) of the project's instruments, every genre, against a tracked f0 curve
of the clip itself. Read-only on the dataset: it only measures.

- **pitch**: written vs tracked pitch in the middle of each note (src/evaluate/f0.py): the whole-semitone
  gap, how far off that gap the recording sits (cents), and how often the two agree.
- **timing**: the clip's notes shifted earlier and later against the audio (src/evaluate/label_timing.py),
  after taking out the pitch gap. Two judges: the tracked pitch (every instrument) and detected onsets
  (plucked instruments). Negative best shift = the MIDI runs late.

Writes <f0_dir>/<clip_id>.npz (tracked once; files already there are skipped, so an interrupted run
resumes) and, in <tables_dir>: clips.parquet (the clip list), items.parquet (one row per clip × judge,
timing + pitch), summary_by_instrument.parquet.

Usage: python -m src.pipelines.audit_solo_midi --config experiments/<exp>/configs/audit_solo_midi.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled
from src.datasets.solo import pick_solo_clips
from src.evaluate.f0 import reference_pitch_frames, reference_pitch_offset
from src.evaluate.label_timing import best_shift, onset_scores_by_shift, pitch_scores_by_shift, shifts_in_frames
from src.evaluate.transcription import reference_notes
from src.features.f0 import make_f0_tracker
from src.features.onsets import spectral_flux_onsets
from src.pipelines.common import active_f0_tracker, f0_path, load_run_config


def track_clips(clips: pd.DataFrame, config) -> None:
    """Track f0 once per clip, skipping clips already tracked."""
    tracker_config = active_f0_tracker(config)
    remaining = [clip for clip in clips.to_dict("records") if not f0_path(config.run.f0_dir, clip["clip_id"]).exists()]
    if not remaining:
        return
    track_f0 = make_f0_tracker(config.run.f0_tracker, tracker_config)
    Path(config.run.f0_dir).mkdir(parents=True, exist_ok=True)
    for clip in tqdm(remaining, desc="track f0", mininterval=30):
        mono = load_mono_resampled(clip["clip_path"], tracker_config.model_sample_rate)
        np.savez_compressed(f0_path(config.run.f0_dir, clip["clip_id"]), **track_f0(mono, tracker_config.model_sample_rate))


def check_clip_pitch(reference_intervals: np.ndarray, reference_pitches: np.ndarray, track: dict, config) -> dict:
    """Written vs tracked pitch over the middles of the clip's notes (``evaluate.f0``'s scored frames)."""
    f0_config = config.evaluate.f0
    reference = reference_pitch_frames(reference_intervals, reference_pitches, track["time"], f0_config)
    return reference_pitch_offset(reference, track["f0_hz"], f0_config.pitch_tolerance_cents)


def check_clip_timing(reference_intervals: np.ndarray, reference_pitches: np.ndarray, track: dict,
                      detected_onsets: np.ndarray | None, config) -> list[dict]:
    """Best shift of the clip's notes per judge: tracked pitch, and detected onsets if given.

    ``peak_at_range_edge`` = the best shift sits at the end of the tested range: no real peak, the judge
    can't time this clip.
    """
    timing_config = config.evaluate.label_timing
    hop_seconds = float(np.median(np.diff(track["time"])))
    shifts = shifts_in_frames(timing_config, hop_seconds)
    shift_seconds = np.round(shifts * hop_seconds, 6)
    zero = int(np.flatnonzero(shifts == 0)[0])
    edges = (shift_seconds[0], shift_seconds[-1])

    # Pitch judge: share of in-note frames on the written pitch, per shift
    scored, hit, _ = pitch_scores_by_shift(reference_intervals, reference_pitches, track["time"], track["f0_hz"],
                                           shifts, timing_config)
    accuracy = hit.sum(axis=1) / np.maximum(scored.sum(axis=1), 1)
    shift = best_shift(shift_seconds, accuracy)
    rows = [{"judge": "pitch", "best_shift_seconds": shift, "score_at_zero": float(accuracy[zero]),
             "score_at_best": float(accuracy.max()), "peak_at_range_edge": shift in edges}]

    # Onset judge: note starts vs detected plucks
    if detected_onsets is not None:
        f_measure = onset_scores_by_shift(reference_intervals[:, 0], detected_onsets, shift_seconds,
                                          timing_config.onset_tolerance_seconds)
        shift = best_shift(shift_seconds, f_measure)
        rows.append({"judge": "onsets", "best_shift_seconds": shift, "score_at_zero": float(f_measure[zero]),
                     "score_at_best": float(f_measure.max()), "peak_at_range_edge": shift in edges,
                     "detected_onset_count": len(detected_onsets)})
    return rows


def audit_clip(clip: dict, config) -> list[dict]:
    """One clip: its pitch check, then its timing per judge (pitch gap taken out first); one row per judge."""
    reference_intervals, reference_pitches = reference_notes(clip["midi_path"])
    track = dict(np.load(f0_path(config.run.f0_dir, clip["clip_id"])))
    hop_seconds = float(np.median(np.diff(track["time"])))
    keys = {key: clip[key] for key in ("clip_id", "instrument", "genre_code", "piece", "partition")}
    facts = {"reference_note_count": len(reference_pitches), "audio_seconds": float(len(track["time"]) * hop_seconds),
             "midi_end_seconds": float(reference_intervals[:, 1].max()) if len(reference_pitches) else np.nan}
    if len(reference_pitches) == 0:
        return [{**keys, **facts, "judge": "none"}]

    pitch = check_clip_pitch(reference_intervals, reference_pitches, track, config)
    gap = 0 if np.isnan(pitch["semitone_gap"]) else pitch["semitone_gap"]
    detected_onsets = None
    if clip["instrument"] in set(config.evaluate.label_timing.onset_instruments):
        audio = load_mono_resampled(clip["clip_path"], config.run.onset_sample_rate)
        detected_onsets = spectral_flux_onsets(audio, config.run.onset_sample_rate, config.features.onsets.spectral_flux)
    timing = check_clip_timing(reference_intervals, reference_pitches + gap, track, detected_onsets, config)
    return [{**keys, **facts, **pitch, **row} for row in timing]


def summarise(items: pd.DataFrame, config) -> pd.DataFrame:
    """Per instrument × judge: clip counts, pitch gap counts, median timing and agreement."""
    checked = items[items["judge"] != "none"]
    late_limit = config.run.summary_timing_limit_seconds
    return checked.groupby(["instrument", "judge"]).agg(
        clips=("clip_id", "size"),
        gap_0=("semitone_gap", lambda gap: int((gap == 0).sum())),
        gap_plus_minus_1=("semitone_gap", lambda gap: int((gap.abs() == 1).sum())),
        gap_octave_or_more=("semitone_gap", lambda gap: int((gap.abs() >= 12).sum())),
        gap_other=("semitone_gap", lambda gap: int(((gap.abs() > 1) & (gap.abs() < 12)).sum())),
        median_residual_cents=("residual_cents", "median"),
        median_agreement_after_gap=("agreement_after_gap", "median"),
        median_best_shift_seconds=("best_shift_seconds", "median"),
        clips_off_beyond_limit=("best_shift_seconds", lambda shift: int((shift.abs() > late_limit).sum())),
        clips_peak_at_range_edge=("peak_at_range_edge", "sum"),
    ).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/audit_solo_midi.yaml")
    config = load_run_config(parser.parse_args().config)
    run = config.run
    tables_dir = Path(run.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Clip list: the project's instruments, every genre
    clips = pick_solo_clips(config.datasets.solo, config.datasets.instruments)
    if run.limit_clips_per_instrument:
        clips = clips.groupby("instrument", group_keys=False).head(run.limit_clips_per_instrument)
    clips.to_parquet(tables_dir / "clips.parquet")
    print(clips.groupby("instrument").size())

    # Track, then measure every clip
    track_clips(clips, config)
    rows = []
    for clip in tqdm(clips.to_dict("records"), desc="audit", mininterval=30):
        rows += audit_clip(clip, config)
    items = pd.DataFrame(rows)
    items.to_parquet(tables_dir / "items.parquet")
    summary = summarise(items, config)
    summary.to_parquet(tables_dir / "summary_by_instrument.parquet")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
