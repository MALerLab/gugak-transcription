"""Pick: the separation dataset's 민속악 stems that strand B tracks, one row per stem."""
from __future__ import annotations

import unicodedata

import pandas as pd


def nfc(text: str) -> str:
    """Korean names need NFC before any join (PROJECT.md → gotchas)."""
    return unicodedata.normalize("NFC", text)


def read_song_metadata(metadata_csv: str) -> pd.DataFrame:
    """Per-song table (`metadata.csv` is authoritative); keyed by NFC song_id."""
    metadata = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    metadata["song_id"] = metadata["project_name"].map(nfc)
    return metadata[["song_id", "title", "genreMajor", "genreSub", "genreDetail", "tempoBpm", "western_key", "악기구성"]]


def pick_stems(config) -> pd.DataFrame:
    """Melodic stems of the configured 민속악 genres, with song metadata and the frozen split.

    Args:
        config: the `configs/data/pick.yaml` settings.

    Returns:
        One row per stem: song_id, instrument, stem_path, split, title, genreDetail, ...
    """
    manifest = pd.read_parquet(config.source_manifest)
    stems = manifest[(manifest["dataset"].astype(str) == "71955") & (manifest["role"] == "stem")].copy()
    stems["song_id"] = stems["song_id"].map(nfc)
    stems["instrument"] = stems["instrument_canonical"].map(nfc)
    stems = stems[stems["genre_sub"].isin(list(config.genre_subs)) & stems["instrument"].isin(list(config.instruments))]

    metadata = read_song_metadata(config.metadata_csv)
    picked = stems.merge(metadata, on="song_id", how="left", validate="many_to_one")
    picked = picked[picked["genreMajor"] == config.genre_major]
    picked = picked.rename(columns={"out_path": "stem_path", "out_duration": "duration_seconds"})
    columns = ["song_id", "instrument", "instrument_raw", "stem_path", "split", "duration_seconds", "active_frac",
               "title", "genreSub", "genreDetail", "tempoBpm", "western_key"]
    return picked[columns].sort_values(["genreSub", "instrument", "song_id"]).reset_index(drop=True)
