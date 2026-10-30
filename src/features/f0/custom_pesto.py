"""MALer Lab's custom PESTO f0 tracker: PESTO retrained on gugak audio (MALerLab/Custom_PESTO, private).

The lab's repo is cloned, untracked, into ``externals/`` and its code runs from there as-is: its
model loader and its CREPE-style Viterbi smoothing. Nothing of it is copied into src/, so the
public repo never carries it. The recipe pins the clone's commit and the checkpoint's md5.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np


def import_custom_pesto(config) -> ModuleType:
    """The lab repo's ``utils`` package, after checking the clone is at the pinned commit.

    Imported under its own name, so it can't clash with any other ``utils`` package.

    Args:
        config: the `configs/features/f0/custom_pesto.yaml` settings.
    """
    repo_dir = Path(config.repo_dir)
    commit = subprocess.run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if commit != config.repo_commit:
        raise RuntimeError(f"{repo_dir} is at {commit}, the recipe pins {config.repo_commit}")
    module_name = "custom_pesto_utils"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, repo_dir / "utils" / "__init__.py",
                                                  submodule_search_locations=[str(repo_dir / "utils")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module  # registered first, so the package's relative imports resolve
    spec.loader.exec_module(module)
    return module


def load_custom_pesto(config):
    """The custom PESTO model from ``config.checkpoint``, on ``config.device``.

    Args:
        config: the `configs/features/f0/custom_pesto.yaml` settings.
    """
    checkpoint_md5 = hashlib.md5(Path(config.checkpoint).read_bytes()).hexdigest()
    if checkpoint_md5 != config.checkpoint_md5:
        raise RuntimeError(f"{config.checkpoint} has md5 {checkpoint_md5}, the recipe pins {config.checkpoint_md5}")
    custom_pesto = import_custom_pesto(config)
    model = custom_pesto.load_custom_pesto(config.checkpoint, config.hop_seconds * 1000)
    return model.to(config.device)


def custom_pesto_track(model, mono: np.ndarray, sample_rate: int, config) -> dict[str, np.ndarray]:
    """Run a loaded custom PESTO model on mono audio at ``config.model_sample_rate``.

    Args:
        model: from ``load_custom_pesto``.
        mono: mono float audio.
        sample_rate: its rate (must be the model's).
        config: the `configs/features/f0/custom_pesto.yaml` settings.

    Returns:
        dict with ``time`` (s), ``f0_hz`` and ``confidence``, one value per hop. The model has no
        voicing head; ``confidence`` is the peak output activation, as the lab's Viterbi uses it.
    """
    import torch
    if sample_rate != config.model_sample_rate:
        raise ValueError(f"custom PESTO is set up for {config.model_sample_rate} Hz audio, got {sample_rate} Hz")
    custom_pesto = import_custom_pesto(config)
    audio = torch.from_numpy(mono).float()[None].to(config.device)
    f0_hz, _volume, activations = custom_pesto.custom_predict(
        audio, sr=sample_rate, model=model, num_chunks=1, max_chunks=config.max_chunks,
        convert_to_freq=True, use_viterbi=config.use_viterbi)
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    confidence = activations.max(axis=-1).astype(np.float32)
    time = np.arange(len(f0_hz), dtype=np.float32) * config.hop_seconds
    return {"time": time, "f0_hz": f0_hz, "confidence": confidence}
