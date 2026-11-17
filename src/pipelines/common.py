"""Helpers shared by the pipeline commands: run configs and where artifacts live. Not a command."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.datasets.answer_keys import six_dragon_grace_windows, six_dragon_stem_pairs, solo_clip_pairs
from src.notes.midi import midi_to_note_table


def load_run_config(config_path: str | Path) -> DictConfig:
    """One run's full config, as written by src/launch/make_run_config.py into experiments/<exp>/configs/."""
    return OmegaConf.load(config_path)


def list_answer_key_items(config, six_dragon_splits: list[str], with_solo: bool) -> pd.DataFrame:
    """(audio, reference MIDI) pairs: six dragon stems of the given splits, plus the solo clips if asked."""
    six_dragon = config.run.six_dragon
    tables = [six_dragon_stem_pairs(six_dragon.quality_table, six_dragon.match_table, six_dragon.midi_dir,
                                    list(six_dragon_splits), config.datasets.separation, config.datasets.instruments)]
    if with_solo:
        solo = config.run.solo
        tables.append(solo_clip_pairs(solo.clips_table, solo.partition, solo.reference_check_table,
                                      solo.max_reference_transposition))
    return pd.concat(tables, ignore_index=True)


def item_grace_windows(item: dict, config) -> np.ndarray:
    """``(n, 2)`` seconds left out of training and scoring: the six dragon grace-note windows; none for solo."""
    if item["answer_key"] != "six_dragon":
        return np.zeros((0, 2))
    answer_keys = config.datasets.answer_keys
    return six_dragon_grace_windows(config.run.six_dragon.aligned_cache_dir, item["source_id"], item["instrument"],
                                    answer_keys.grace_window_seconds_before, answer_keys.grace_window_seconds_after)


def item_reference_notes(item: dict) -> pd.DataFrame:
    """The reference note table of one item: its part of the reference MIDI, or every track if it has no part."""
    notes = midi_to_note_table(item["reference_midi_path"])
    # A missing part can come back as NaN instead of None after a pandas round trip
    part = item["reference_part"] if isinstance(item["reference_part"], str) else None
    return notes if part is None else notes[notes["part"] == part].reset_index(drop=True)


def cached_audio_path(config, item: dict) -> Path:
    """Where one item's mono audio at the model's sample rate is cached."""
    return Path(config.run.audio_cache_dir) / item["answer_key"] / f"{item['item_id']}.npy"


def active_f0_tracker(config) -> DictConfig:
    """The recipe of the tracker ``run.f0_tracker`` names (hop, sample rate, …), from ``features.f0``."""
    return config.features.f0[config.run.f0_tracker]


def f0_path(out_dir: str | Path, file_id: str) -> Path:
    """Where one file's f0 track is stored."""
    return Path(out_dir) / f"{file_id}.npz"
