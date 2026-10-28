"""f0 tracking: one stem's audio → pitch curve + per-frame confidence, via torchcrepe."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchcrepe

from src.data.audio import read_audio, to_mono

DECODERS = {
    "viterbi": torchcrepe.decode.viterbi,
    "weighted_argmax": torchcrepe.decode.weighted_argmax,
    "argmax": torchcrepe.decode.argmax,
}


def load_mono_resampled(audio_path: str | Path, target_sample_rate: int,
                        start_seconds: float = 0.0, duration_seconds: float | None = None) -> np.ndarray:
    """Read a WAV (or an excerpt of it), downmix to mono and resample.

    Args:
        audio_path: WAV file.
        target_sample_rate: output sample rate.
        start_seconds: excerpt start.
        duration_seconds: excerpt length; None reads to the end.
    """
    import soundfile as sf
    source_sample_rate = sf.info(str(audio_path)).samplerate
    start_frame = int(round(start_seconds * source_sample_rate))
    num_frames = -1 if duration_seconds is None else int(round(duration_seconds * source_sample_rate))
    audio, source_sample_rate = read_audio(audio_path, start=start_frame, frames=num_frames)
    mono = torch.from_numpy(to_mono(audio)).float()
    if source_sample_rate != target_sample_rate:
        mono = torchaudio.functional.resample(mono, source_sample_rate, target_sample_rate)
    return mono.numpy()


def track_f0(mono: np.ndarray, config) -> dict[str, np.ndarray]:
    """Run torchcrepe on mono audio already at ``config.model_sample_rate``.

    Args:
        mono: mono float audio.
        config: the `configs/pitch/track.yaml` settings.

    Returns:
        dict with ``time`` (s), ``f0_hz`` and ``confidence`` (CREPE periodicity, 0–1), one value per hop.
    """
    hop_length = int(round(config.hop_seconds * config.model_sample_rate))
    audio = torch.from_numpy(mono).float()[None]
    f0_hz, confidence = torchcrepe.predict(
        audio, config.model_sample_rate, hop_length,
        fmin=config.fmin_hz, fmax=config.fmax_hz, model=config.model,
        decoder=DECODERS[config.decoder], return_periodicity=True,
        batch_size=config.batch_size, device=config.device,
    )
    f0_hz = f0_hz[0].cpu().numpy().astype(np.float32)
    confidence = confidence[0].cpu().numpy().astype(np.float32)
    time = np.arange(len(f0_hz), dtype=np.float32) * config.hop_seconds
    return {"time": time, "f0_hz": f0_hz, "confidence": confidence}
