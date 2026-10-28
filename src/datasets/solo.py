"""Solo dataset (AI Hub 71470): clips with their MIDI and 시김새 labels, picked as strand B's answer key."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

from src.datasets.instruments import to_english_names


def piece_partition(piece_name: str, dev_fraction: float, seed: int) -> str:
    """Deterministic dev/test assignment of a piece from a hash of its NFC name."""
    digest = hashlib.sha1(f"{seed}:{unicodedata.normalize('NFC', piece_name.strip())}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) / 0xFFFFFFFF < dev_fraction else "test"


def read_clip_label(label_path: Path) -> dict:
    """The fields strand B uses from one clip's label JSON, incl. its 시김새 regions."""
    label = json.loads(label_path.read_text())
    annotations = label["annotation_data_info"]
    sigimsae = [(s["annotation_name"], float(s["start_time"]), float(s["end_time"]))
                for s in annotations["single_tonguing_cd"]]
    return {
        "clip_id": label["music_source_info"]["music_src_nm"],
        "piece": label["music_source_info"]["music_nm_kor"].strip(),
        "genre_code": label["music_type_info"]["music_genre_cd"],
        "mode_code": annotations["mode_cd"],
        "sigimsae": json.dumps(sigimsae, ensure_ascii=False),
    }


def pick_solo_clips(config, instruments_config) -> pd.DataFrame:
    """Solo clips of the configured genres and instruments, with a dev/test partition by piece.

    Args:
        config: the `configs/datasets/solo.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings.
    """
    manifest = pd.read_parquet(config.source_manifest)
    clips = manifest[manifest["dataset"].astype(str) == "71470"][["clip_id", "instrument_canonical", "out_path", "out_duration"]]
    clips = clips.rename(columns={"instrument_canonical": "instrument", "out_path": "clip_path", "out_duration": "duration_seconds"})
    clips["instrument"] = to_english_names(clips["instrument"].map(lambda name: unicodedata.normalize("NFC", name)),
                                           instruments_config)
    clips = clips[clips["instrument"].isin(list(config.instruments))]

    labels = pd.DataFrame([read_clip_label(Path(config.labels_dir) / f"{clip_id}.json") for clip_id in clips["clip_id"]])
    clips = clips.merge(labels, on="clip_id", validate="one_to_one")
    clips = clips[clips["genre_code"].isin(list(config.genre_codes))].copy()
    clips["midi_path"] = [str(Path(config.midi_dir) / f"{clip_id}.mid") for clip_id in clips["clip_id"]]
    clips["partition"] = [piece_partition(p, config.dev_fraction, config.seed) for p in clips["piece"]]
    return clips.sort_values("clip_id").reset_index(drop=True)
