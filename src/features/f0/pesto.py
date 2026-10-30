"""Stock PESTO f0 tracker (Riou et al. 2023), through the ``pesto-pitch`` package.

PESTO learns pitch without labels: pitch-shift the input, and its output has to move by exactly
that much. The bundled weights were trained on MIR-1K (singing voice).
"""
from __future__ import annotations

import numpy as np


def load_pesto(config):
    """The stock PESTO model named by ``config.checkpoint``, on ``config.device``.

    Args:
        config: the `configs/features/f0/pesto.yaml` settings.
    """
    import pesto
    model = pesto.load_model(config.checkpoint, step_size=config.hop_seconds * 1000,
                             sampling_rate=config.model_sample_rate)
    model.reduction = config.reduction
    return model.to(config.device)


def pesto_track(model, mono: np.ndarray, sample_rate: int, config) -> dict[str, np.ndarray]:
    """Run a loaded PESTO model on mono audio at ``config.model_sample_rate``.

    Args:
        model: from ``load_pesto``.
        mono: mono float audio.
        sample_rate: its rate (must be the model's).
        config: the `configs/features/f0/pesto.yaml` settings.

    Returns:
        dict with ``time`` (s), ``f0_hz`` and ``confidence`` (PESTO's voicing head, 0–1), one value per hop.
    """
    import torch
    if sample_rate != config.model_sample_rate:
        raise ValueError(f"PESTO is set up for {config.model_sample_rate} Hz audio, got {sample_rate} Hz")
    audio = torch.from_numpy(mono).float().to(config.device)
    with torch.inference_mode():
        f0_hz, confidence, _volume = model(audio, sr=sample_rate, convert_to_freq=True, return_activations=False)
    f0_hz = f0_hz.cpu().numpy().astype(np.float32)
    confidence = confidence.cpu().numpy().astype(np.float32)
    time = np.arange(len(f0_hz), dtype=np.float32) * config.hop_seconds
    return {"time": time, "f0_hz": f0_hz, "confidence": confidence}
