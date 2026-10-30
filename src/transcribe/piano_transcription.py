"""Piano transcription model run as-is on any audio → note table (zero-shot: no fine-tuning).

The model: Kong et al. 2021, "High-resolution piano transcription with pedals by regressing onset
and offset times", via the authors' ``piano_transcription_inference`` package. Trained on MAESTRO
piano only. Its pedal output is dropped: only notes are kept.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import torch
from piano_transcription_inference import PianoTranscription, models
from piano_transcription_inference import config as model_config

MODEL_SAMPLE_RATE = model_config.sample_rate  # 16 kHz, fixed by the trained model


def check_checkpoint_keys(checkpoint_path: str, model_type: str) -> None:
    """Load the weights once with strict key matching.

    The package loads with ``strict=False``, so a checkpoint that doesn't fit the model would load
    silently with random layers; this raises instead.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = getattr(models, model_type)(frames_per_second=model_config.frames_per_second,
                                        classes_num=model_config.classes_num)
    model.load_state_dict(checkpoint["model"], strict=True)


def load_piano_transcription(config, device: str) -> PianoTranscription:
    """The pretrained model with the thresholds from config.

    Args:
        config: the `configs/transcribe/piano_transcription.yaml` settings.
        device: ``cuda`` or ``cpu``.
    """
    check_checkpoint_keys(config.checkpoint_path, config.model_type)
    transcriptor = PianoTranscription(model_type=config.model_type, checkpoint_path=config.checkpoint_path,
                                      segment_samples=int(config.segment_seconds * MODEL_SAMPLE_RATE),
                                      device=torch.device(device))
    transcriptor.onset_threshold = config.onset_threshold
    transcriptor.offset_threshod = config.offset_threshold  # sic: the package's attribute name
    transcriptor.frame_threshold = config.frame_threshold
    transcriptor.pedal_offset_threshold = config.pedal_offset_threshold
    return transcriptor


def transcribe_audio(transcriptor: PianoTranscription, audio: np.ndarray, part: str) -> pd.DataFrame:
    """Note table of mono audio at ``MODEL_SAMPLE_RATE`` (columns part, pitch, velocity, start, duration).

    Args:
        transcriptor: from ``load_piano_transcription``.
        audio: mono samples at 16 kHz.
        part: written into the ``part`` column (the stem's instrument).
    """
    # The package prints one line per 10 s segment; keep the run log readable
    with contextlib.redirect_stdout(io.StringIO()):
        events = transcriptor.transcribe(audio.astype(np.float32), None)["est_note_events"]
    table = pd.DataFrame({"part": part,
                          "pitch": [int(e["midi_note"]) for e in events],
                          "velocity": [int(e["velocity"]) for e in events],
                          "start": [float(e["onset_time"]) for e in events],
                          "duration": [float(e["offset_time"] - e["onset_time"]) for e in events]},
                         columns=["part", "pitch", "velocity", "start", "duration"])
    return table.sort_values("start", kind="stable").reset_index(drop=True)
