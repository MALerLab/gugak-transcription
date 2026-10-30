"""Helpers shared by the pipeline commands: run configs and where artifacts live. Not a command."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_run_config(config_path: str | Path) -> DictConfig:
    """One run's full config, as written by src/launch/make_run_config.py into experiments/<exp>/configs/."""
    return OmegaConf.load(config_path)


def active_f0_tracker(config) -> DictConfig:
    """The recipe of the tracker ``run.f0_tracker`` names (hop, sample rate, …), from ``features.f0``."""
    return config.features.f0[config.run.f0_tracker]


def f0_path(out_dir: str | Path, file_id: str) -> Path:
    """Where one file's f0 track is stored."""
    return Path(out_dir) / f"{file_id}.npz"
