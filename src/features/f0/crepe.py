"""CREPE f0 tracker (Kim et al. 2018), through torchcrepe. Strand B's tracker since exp002."""
from __future__ import annotations

import numpy as np


def crepe_track(mono: np.ndarray, sample_rate: int, config) -> dict[str, np.ndarray]:
    """Run torchcrepe on mono audio already at ``config.model_sample_rate``.

    Args:
        mono: mono float audio.
        sample_rate: its rate (must be the model's).
        config: the `configs/features/f0/crepe.yaml` settings.

    Returns:
        dict with ``time`` (s), ``f0_hz`` and ``confidence`` (CREPE periodicity, 0–1), one value per hop.
    """
    import torch  # imported here so YIN users don't pull in torch
    import torchcrepe
    if sample_rate != config.model_sample_rate:
        raise ValueError(f"CREPE needs {config.model_sample_rate} Hz audio, got {sample_rate} Hz")
    # torchcrepe dithers its pitch output with scipy noise (numpy's global RNG); seeding per call
    # makes each file's f0 reproducible, whatever order the files run in
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    hop_length = int(round(config.hop_seconds * config.model_sample_rate))
    audio = torch.from_numpy(mono).float()[None]
    f0_hz, confidence = torchcrepe.predict(
        audio, config.model_sample_rate, hop_length,
        fmin=config.fmin_hz, fmax=config.fmax_hz, model=config.model,
        decoder=getattr(torchcrepe.decode, config.decoder),  # viterbi | weighted_argmax | argmax
        return_periodicity=True, batch_size=config.batch_size, device=config.device,
    )
    f0_hz = f0_hz[0].cpu().numpy().astype(np.float32)
    confidence = confidence[0].cpu().numpy().astype(np.float32)
    time = np.arange(len(f0_hz), dtype=np.float32) * config.hop_seconds
    return {"time": time, "f0_hz": f0_hz, "confidence": confidence}
