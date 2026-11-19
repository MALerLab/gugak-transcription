"""Solo dataset (AI Hub 71470): clips with their MIDI and 시김새 labels, picked as strand B's answer key and split for training."""
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
    """Every solo clip of the configured instruments, any genre, with a train / val / test split for training.

    - test = the clips of ``training_split.test_clips_table``'s ``test_partition`` (exp002's 588), unchanged;
    - a non-test clip of a test piece is left out (``drop_reason`` = ``test_piece``, split None);
    - every other clip goes with its piece (one piece can span several instruments): pieces shuffled with
      ``seed``, then each one taken into val if one of its instruments is still under ``val_fraction`` of
      that instrument's kept clips and none goes past ``val_overshoot`` × that share; the rest is train.
      Filling per instrument keeps val usable for a mean of instrument means (a plain 10 % of pieces left
      아쟁 with 4 val clips).
    Clips ``unreliable_solo_clips`` lists keep their split but carry their ``drop_reason``, so readers skip
    them in training and the test set stays whole (scoring marks them instead).

    Args:
        config: the `configs/datasets/solo.yaml` settings (its ``training_split`` and ``unreliable_midi`` blocks).
        instruments_config: the `configs/datasets/instruments.yaml` settings.
    """
    rules = config.training_split
    clips = pick_solo_clips(OmegaConf.merge(config, {"genre_codes": rules.genre_codes}), instruments_config)
    clips = clips.drop(columns=["partition", "sigimsae"])
    test_table = pd.read_parquet(rules.test_clips_table)
    test_ids = set(test_table.loc[test_table["partition"] == rules.test_partition, "clip_id"])
    missing = test_ids - set(clips["clip_id"])
    if missing:
        raise ValueError(f"{len(missing)} test clips are not in the solo pick, e.g. {sorted(missing)[:3]}")
    is_test = clips["clip_id"].isin(test_ids)
    test_pieces = set(clips.loc[is_test, "piece"])
    clips["split"] = pd.Series(None, index=clips.index, dtype=object)
    clips.loc[is_test, "split"] = "test"
    reasons = unreliable_solo_clips(config).set_index("clip_id")["drop_reason"]
    clips["drop_reason"] = clips["clip_id"].map(reasons).astype(object)
    clips.loc[clips["drop_reason"].isna(), "drop_reason"] = None
    in_test_piece = ~is_test & clips["piece"].isin(test_pieces)
    clips.loc[in_test_piece, "drop_reason"] = "test_piece"

    # Pieces in a seeded order; a piece joins val while it helps an under-filled instrument and overfills none
    pool = clips[~is_test & ~in_test_piece]
    kept = pool[pool["drop_reason"].isna()]
    per_piece = kept.groupby(["piece", "instrument"]).size().unstack(fill_value=0)
    per_piece = per_piece.reindex(sorted(pool["piece"].unique()), fill_value=0)
    target = rules.val_fraction * per_piece.sum()
    in_val = pd.Series(0, index=per_piece.columns)
    val_pieces = set()
    for piece in np.random.default_rng(rules.seed).permutation(per_piece.index.to_numpy()):
        counts = per_piece.loc[piece]
        helps = ((counts > 0) & (in_val < target)).any()
        overfills = ((counts > 0) & (in_val + counts > rules.val_overshoot * target)).any()
        if helps and not overfills:
            val_pieces.add(piece)
            in_val += counts
    clips.loc[pool.index, "split"] = np.where(pool["piece"].isin(val_pieces), "val", "train")
    return clips.reset_index(drop=True)
