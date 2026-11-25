"""Solo dataset (AI Hub 71470): clips with their MIDI and 시김새 labels, picked as an answer key and split for training."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

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
        # The dataset's own genre family, from its source folder: ".../F_민속악" → "민속악"
        "genre_family": unicodedata.normalize("NFC", Path(label["dataset_info"]["dataset_src_path"]).name.split("_", 1)[1]),
        "mode_code": annotations["mode_cd"],
        "sigimsae": json.dumps(sigimsae, ensure_ascii=False),
    }


def pick_solo_clips(config, instruments_config) -> pd.DataFrame:
    """Solo clips of the configured genres (null = every genre) and instruments, with a dev/test partition by piece.

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
    # genre_codes null = every genre
    if config.genre_codes is not None:
        clips = clips[clips["genre_code"].isin(list(config.genre_codes))]
    clips = clips.copy()
    clips["midi_path"] = [str(Path(config.midi_dir) / f"{clip_id}.mid") for clip_id in clips["clip_id"]]
    clips["partition"] = [piece_partition(p, config.dev_fraction, config.seed) for p in clips["piece"]]
    return clips.sort_values("clip_id").reset_index(drop=True)


def unreliable_solo_clips(config) -> pd.DataFrame:
    """Clips whose MIDI looks unreliable against their own audio, from the solo MIDI audit (exp010): clip_id + drop_reason.

    Dropped: any whole-semitone gap beyond ``max_abs_semitone_gap`` (``semitone_off``, ``octave_or_more_off``,
    ``other_gap``), or on pitch but ``agreement_after_gap`` below ``min_agreement_after_gap`` (``low_agreement``).
    Clips with no checkable notes (NaN gap) are not listed, so they stay in.

    Args:
        config: the `configs/datasets/solo.yaml` settings (its ``unreliable_midi`` block).
    """
    rules = config.unreliable_midi
    audit = pd.read_parquet(rules.audit_table)
    audit = audit[audit["judge"] == "pitch"].set_index("clip_id")
    gap = audit["semitone_gap"].abs()
    reason = pd.Series(None, index=audit.index, dtype=object)
    reason[audit["agreement_after_gap"] < rules.min_agreement_after_gap] = "low_agreement"
    # A gap overrules low agreement: it names what is wrong
    gap_kind = pd.Series("other_gap", index=audit.index)
    gap_kind[gap == 1] = "semitone_off"
    gap_kind[gap >= 12] = "octave_or_more_off"
    too_far = gap > rules.max_abs_semitone_gap
    reason[too_far] = gap_kind[too_far]
    return reason.dropna().rename("drop_reason").reset_index()


def drop_unreliable_solo_clips(clips: pd.DataFrame, config) -> pd.DataFrame:
    """``clips`` (with a ``clip_id`` column) minus the ones ``unreliable_solo_clips`` lists."""
    unreliable = unreliable_solo_clips(config)
    return clips[~clips["clip_id"].isin(unreliable["clip_id"])].reset_index(drop=True)


def split_solo_clips(config, instruments_config) -> pd.DataFrame:
    """Every solo clip of the configured instruments, any genre, split into train / val / test by piece, stratified
    by genre family and instrument.

    Each genre family × instrument cell aims for ``test_fraction`` and ``val_fraction`` of its kept audio time in
    test and val. Pieces go in a seeded order (one piece can span several instruments, and sometimes families):
    a piece joins test, else val, if it helps a cell still under that split's share and pushes none past
    ``overshoot`` × the share; the rest is train. No piece is in two splits. A cell with a single piece can't
    split, so that piece lands whole in one split.
    Clips ``unreliable_solo_clips`` lists keep their split but carry their ``drop_reason``, so readers skip them.

    Args:
        config: the `configs/datasets/solo.yaml` settings (its ``training_split`` and ``unreliable_midi`` blocks).
        instruments_config: the `configs/datasets/instruments.yaml` settings.
    """
    rules = config.training_split
    clips = pick_solo_clips(OmegaConf.merge(config, {"genre_codes": None}), instruments_config)
    clips = clips.drop(columns=["partition", "sigimsae"])
    reasons = unreliable_solo_clips(config).set_index("clip_id")["drop_reason"]
    clips["drop_reason"] = clips["clip_id"].map(reasons).astype(object)
    clips.loc[clips["drop_reason"].isna(), "drop_reason"] = None

    kept = clips[clips["drop_reason"].isna()]
    hours = kept.groupby(["piece", "genre_family", "instrument"])["duration_seconds"].sum().div(3600)
    per_piece = hours.unstack(["genre_family", "instrument"], fill_value=0.0)
    per_piece = per_piece.reindex(sorted(clips["piece"].unique()), fill_value=0.0)
    fractions = {"test": rules.test_fraction, "val": rules.val_fraction}
    targets = {name: fraction * per_piece.sum() for name, fraction in fractions.items()}
    filled = {name: pd.Series(0.0, index=per_piece.columns) for name in fractions}
    assigned = {}
    for piece in np.random.default_rng(rules.seed).permutation(per_piece.index.to_numpy()):
        amounts = per_piece.loc[piece]
        present = amounts > 0
        for name in fractions:
            helps = (present & (filled[name] < targets[name])).any()
            overfills = (present & (filled[name] + amounts > rules.overshoot * targets[name])).any()
            if helps and not overfills:
                assigned[piece] = name
                filled[name] += amounts
                break
    clips["split"] = clips["piece"].map(assigned).fillna("train")
    return clips.reset_index(drop=True)
