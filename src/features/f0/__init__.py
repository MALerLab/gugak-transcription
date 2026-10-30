"""f0 tracking: mono audio → pitch curve + per-frame confidence, one module per tracker.

Each tracker has its own recipe, `configs/features/f0/<tracker>.yaml`. A run config holds the
recipes its command ``uses`` under ``features.f0.<tracker>``; ``run.f0_tracker`` picks one when a
command runs a single tracker.
- ``yin``: strand A's alignment check (an algorithm, no model)
- ``crepe``: strand B since exp002
- ``pesto``: stock PESTO, trained on singing voice
- ``custom_pesto``: MALer Lab's gugak-trained PESTO (`externals/Custom_PESTO`)

Every tracker returns a dict with ``time`` (s), ``f0_hz`` and ``confidence``, one value per hop.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

F0_TRACKERS = ("yin", "crepe", "pesto", "custom_pesto")


def make_f0_tracker(tracker: str, tracker_config) -> Callable[[np.ndarray, int], dict[str, np.ndarray]]:
    """A function (mono audio, sample rate) → f0 track, with the tracker's model loaded once up front.

    Args:
        tracker: one of ``F0_TRACKERS``.
        tracker_config: that tracker's `configs/features/f0/<tracker>.yaml` settings.
    """
    if tracker == "yin":
        from src.features.f0.yin import yin_track
        return lambda mono, sample_rate: yin_track(mono, sample_rate, tracker_config)
    if tracker == "crepe":
        from src.features.f0.crepe import crepe_track
        return lambda mono, sample_rate: crepe_track(mono, sample_rate, tracker_config)
    if tracker == "pesto":
        from src.features.f0.pesto import load_pesto, pesto_track
        model = load_pesto(tracker_config)
        return lambda mono, sample_rate: pesto_track(model, mono, sample_rate, tracker_config)
    if tracker == "custom_pesto":
        from src.features.f0.custom_pesto import custom_pesto_track, load_custom_pesto
        model = load_custom_pesto(tracker_config)
        return lambda mono, sample_rate: custom_pesto_track(model, mono, sample_rate, tracker_config)
    raise ValueError(f"unknown f0 tracker {tracker!r}, expected one of {F0_TRACKERS}")


def track_f0(mono: np.ndarray, sample_rate: int, tracker: str, tracker_config) -> dict[str, np.ndarray]:
    """One file's f0 track. Loads the model on every call; for many files use ``make_f0_tracker``."""
    return make_f0_tracker(tracker, tracker_config)(mono, sample_rate)
