"""Command: do the six dragon labels run early or late against their stems, steadily or wandering?

Shifts each reference's notes against two judges of the audio (src/evaluate/label_timing.py): a
tracked f0 curve (every instrument) and detected onsets (plucked instruments). Six dragon = exp001's
aligned MIDI on single stems, every split, both label sets (per-stem and per-song alignment). The
solo answer key (clips against their own MIDI, timed independently of exp001) is the control: a
judge that is itself early or late shows up there too.

Writes <f0_dir>/<answer key>/<item_id>.npz (tracked once per audio; files already there are skipped,
so an interrupted run resumes) and, in <tables_dir>: items.parquet (best shift per item × label set
× judge), curves.parquet (score per shift), windows.parquet (best pitch shift per window).

Usage: python -m src.pipelines.audit_label_timing --config experiments/<exp>/configs/audit_label_timing.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.audio.io import load_mono_resampled
from src.datasets.answer_keys import six_dragon_stem_pairs, solo_clip_pairs
from src.evaluate.label_timing import (best_shift, drift_summary, onset_scores_by_shift, pitch_scores_by_shift,
                                       shifts_in_frames, window_best_shifts)
from src.evaluate.transcription import reference_notes
from src.features.f0 import make_f0_tracker
from src.features.onsets import spectral_flux_onsets
from src.pipelines.common import active_f0_tracker, f0_path, load_run_config


def list_items(config) -> pd.DataFrame:
    """Every (audio, reference MIDI) pair: six dragon once per label set, plus the solo control."""
    run = config.run
    six_dragon = run.six_dragon
    tables = [six_dragon_stem_pairs(six_dragon.quality_table, six_dragon.match_table, midi_dir, list(six_dragon.splits),
                                    config.datasets.separation, config.datasets.instruments).assign(label_set=label_set)
              for label_set, midi_dir in six_dragon.label_sets.items()]
    solo = run.solo
    tables.append(solo_clip_pairs(solo.clips_table, solo.partition, solo.reference_check_table,
                                  solo.max_reference_transposition).assign(label_set="solo_midi"))
    return pd.concat(tables, ignore_index=True)


def track_items(audio_items: pd.DataFrame, config) -> None:
    """Track f0 once per audio file, skipping files already tracked."""
    tracker_config = active_f0_tracker(config)
    remaining = [item for item in audio_items.to_dict("records")
                 if not f0_path(Path(config.run.f0_dir) / item["answer_key"], item["item_id"]).exists()]
    if not remaining:
        return
    track_f0 = make_f0_tracker(config.run.f0_tracker, tracker_config)
    for item in tqdm(remaining, desc="track f0", mininterval=30):
        out_path = f0_path(Path(config.run.f0_dir) / item["answer_key"], item["item_id"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mono = load_mono_resampled(item["audio_path"], tracker_config.model_sample_rate)
        np.savez_compressed(out_path, **track_f0(mono, tracker_config.model_sample_rate))


def audit_item(item: dict, detected_onsets: np.ndarray | None, config) -> tuple[list[dict], list[dict], list[dict]]:
    """One item × label set: (item rows, curve rows, window rows) for the pitch judge and, if given, the onset judge.

    ``peak_at_range_edge`` = the best shift sits at the end of the tested range: no real peak, the
    judge can't time this item (e.g. the pitch judge on plucked notes, whose ringing pitch fits any shift).
    """
    timing_config = config.evaluate.label_timing
    # A missing part can come back as NaN instead of None after a pandas round trip
    part = item["reference_part"] if isinstance(item["reference_part"], str) else None
    reference_intervals, reference_pitches = reference_notes(item["reference_midi_path"], part)
    track = dict(np.load(f0_path(Path(config.run.f0_dir) / item["answer_key"], item["item_id"])))
    hop_seconds = float(np.median(np.diff(track["time"])))
    shifts = shifts_in_frames(timing_config, hop_seconds)
    shift_seconds = np.round(shifts * hop_seconds, 6)
    keys = {key: item[key] for key in ("answer_key", "label_set", "item_id", "source_id", "instrument", "split")}

    # Pitch judge: whole item, then per window
    scored, hit, pitch_change = pitch_scores_by_shift(reference_intervals, reference_pitches, track["time"],
                                                      track["f0_hz"], shifts, timing_config)
    hits, scored_frames = hit.sum(axis=1), scored.sum(axis=1)
    accuracy = hits / np.maximum(scored_frames, 1)
    zero = int(np.flatnonzero(shifts == 0)[0])
    windows = [{**keys, **window} for window in
               window_best_shifts(scored, hit, pitch_change, track["time"], shift_seconds, timing_config)]
    item_rows = [{**keys, "judge": "pitch", "reference_note_count": len(reference_pitches),
                  "scored_seconds": float(scored_frames[zero] * hop_seconds), "pitch_changes": int(pitch_change.sum()),
                  "best_shift_seconds": best_shift(shift_seconds, accuracy),
                  "score_at_zero": float(accuracy[zero]), "score_at_best": float(accuracy.max()),
                  **drift_summary(windows)}]
    item_rows[0]["peak_at_range_edge"] = item_rows[0]["best_shift_seconds"] in (shift_seconds[0], shift_seconds[-1])
    curve_rows = [{**keys, "judge": "pitch", "shift_seconds": shift, "hits": int(h), "scored": int(s), "score": float(a)}
                  for shift, h, s, a in zip(shift_seconds, hits, scored_frames, accuracy)]

    # Onset judge: plucked instruments only
    if detected_onsets is not None:
        f_measure = onset_scores_by_shift(reference_intervals[:, 0], detected_onsets, shift_seconds,
                                          timing_config.onset_tolerance_seconds)
        item_rows.append({**keys, "judge": "onsets", "reference_note_count": len(reference_pitches),
                          "detected_onset_count": len(detected_onsets),
                          "best_shift_seconds": best_shift(shift_seconds, f_measure),
                          "score_at_zero": float(f_measure[zero]), "score_at_best": float(f_measure.max())})
        item_rows[-1]["peak_at_range_edge"] = item_rows[-1]["best_shift_seconds"] in (shift_seconds[0], shift_seconds[-1])
        curve_rows += [{**keys, "judge": "onsets", "shift_seconds": shift, "score": float(f)}
                       for shift, f in zip(shift_seconds, f_measure)]
    return item_rows, curve_rows, windows


def audit_items(items: pd.DataFrame, config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Every item × label set through ``audit_item``; onsets are detected once per audio file."""
    onsets_config = config.features.onsets.spectral_flux
    onset_instruments = set(config.evaluate.label_timing.onset_instruments)
    item_rows, curve_rows, window_rows = [], [], []
    for _, same_audio in tqdm(items.groupby(["answer_key", "item_id"], sort=False), desc="audit", mininterval=30):
        first = same_audio.iloc[0]
        detected_onsets = None
        if first["instrument"] in onset_instruments:
            audio = load_mono_resampled(first["audio_path"], config.run.onset_sample_rate)
            detected_onsets = spectral_flux_onsets(audio, config.run.onset_sample_rate, onsets_config)
        for item in same_audio.to_dict("records"):
            rows = audit_item(item, detected_onsets, config)
            item_rows += rows[0]
            curve_rows += rows[1]
            window_rows += rows[2]
    return pd.DataFrame(item_rows), pd.DataFrame(curve_rows), pd.DataFrame(window_rows)


def summarise(item_table: pd.DataFrame, curve_table: pd.DataFrame) -> pd.DataFrame:
    """Per answer key × label set × judge × instrument: the best shift of the pooled curve and the item medians."""
    pitch = curve_table[curve_table["judge"] == "pitch"]
    pooled_pitch = (pitch.groupby(["answer_key", "label_set", "judge", "instrument", "shift_seconds"])[["hits", "scored"]]
                    .sum().assign(score=lambda t: t["hits"] / t["scored"].clip(lower=1))["score"])
    pooled_onsets = (curve_table[curve_table["judge"] == "onsets"]
                     .groupby(["answer_key", "label_set", "judge", "instrument", "shift_seconds"])["score"].mean())
    pooled = pd.concat([pooled_pitch, pooled_onsets]).reset_index()
    pooled_best = (pooled.groupby(["answer_key", "label_set", "judge", "instrument"])
                   .apply(lambda t: best_shift(t["shift_seconds"].to_numpy(), t["score"].to_numpy()), include_groups=False)
                   .rename("pooled_best_shift_seconds"))
    medians = item_table.groupby(["answer_key", "label_set", "judge", "instrument"]).agg(
        items=("item_id", "size"), items_peak_at_range_edge=("peak_at_range_edge", "sum"),
        median_item_best_shift_seconds=("best_shift_seconds", "median"),
        median_window_shift_iqr_seconds=("window_shift_iqr_seconds", "median"),
        median_window_shift_slope_seconds_per_minute=("window_shift_slope_seconds_per_minute", "median"))
    return medians.join(pooled_best).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="experiments/<exp>/configs/audit_label_timing.yaml")
    config = load_run_config(parser.parse_args().config)
    run = config.run
    items = list_items(config)
    if run.limit_items:
        items = items.groupby("label_set", group_keys=False).head(run.limit_items)
    print(items.groupby(["answer_key", "label_set", "instrument"]).size())
    track_items(items.drop_duplicates(["answer_key", "item_id"]), config)
    item_table, curve_table, window_table = audit_items(items, config)
    tables_dir = Path(run.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    item_table.to_parquet(tables_dir / "items.parquet")
    curve_table.to_parquet(tables_dir / "curves.parquet")
    window_table.to_parquet(tables_dir / "windows.parquet")
    summary = summarise(item_table, curve_table)
    summary.to_parquet(tables_dir / "summary_by_instrument.parquet")
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
