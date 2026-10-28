"""Alignment input: a song's stems per MusicXML part, mono at the chroma feature rate."""
from __future__ import annotations

import numpy as np

from src.audio.io import load_stem_mix
from src.datasets.separation import song_stem_paths
from src.features.chroma import to_feature_rate


def load_part_stems(song_id: str, stems: list[str], parts: list[str], separation_config,
                    instruments_config) -> dict[str, np.ndarray]:
    """Part → mono 22.05 kHz stem audio, all trimmed to the shortest stem.

    Args:
        song_id: separation-dataset song.
        stems: the song's stem names (Korean, as on disk).
        parts: MusicXML parts wanted; parts without a stem are skipped.
        separation_config: the `configs/datasets/separation.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings (Korean stem → part name).
    """
    paths = song_stem_paths(song_id, stems, separation_config, instruments_config)
    audio = {}
    for part in parts:
        if part in paths:
            signal, sample_rate = load_stem_mix([paths[part]])
            audio[part] = to_feature_rate(signal, sample_rate)
    length = min(len(a) for a in audio.values())
    return {part: a[:length] for part, a in audio.items()}
