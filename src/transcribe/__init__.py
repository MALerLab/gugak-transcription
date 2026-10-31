"""Transcription: audio → note table.

Zero-shot piano models (``run.transcriber`` in the command's recipe picks one; each has its own recipe
in `configs/transcribe/`):
- ``piano_transcription``: Kong et al. 2021 (exp003)
- ``onsets_frames``: Onsets and Frames, Hawthorne et al. 2018 (exp004)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

TRANSCRIBERS = ("piano_transcription", "onsets_frames")


def make_transcriber(transcriber: str, transcriber_config, device: str) -> tuple[int, Callable[[np.ndarray, str], pd.DataFrame]]:
    """(the sample rate it expects, a function (mono audio, part) → note table), with the model loaded once.

    Args:
        transcriber: one of ``TRANSCRIBERS``.
        transcriber_config: that model's `configs/transcribe/<transcriber>.yaml` settings.
        device: ``cuda`` or ``cpu``.
    """
    if transcriber == "piano_transcription":
        from src.transcribe.piano_transcription import MODEL_SAMPLE_RATE, load_piano_transcription, transcribe_audio
        model = load_piano_transcription(transcriber_config, device)
        return MODEL_SAMPLE_RATE, lambda audio, part: transcribe_audio(model, audio, part)
    if transcriber == "onsets_frames":
        from src.transcribe.onsets_frames import OnsetsFramesTranscriber
        return transcriber_config.sample_rate, OnsetsFramesTranscriber(transcriber_config, device)
    raise ValueError(f"unknown transcriber {transcriber!r}, expected one of {TRANSCRIBERS}")
