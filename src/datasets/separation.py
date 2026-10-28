"""Separation dataset (AI Hub 71955): songs, stems, metadata, and the picks the strands use.

- strand A: 정악 songs with audio (``list_jeongak_songs``) and their stems per MusicXML part
- strand B: 민속악 melodic stems (``pick_stems``)
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import soundfile as sf

from src.datasets.instruments import to_english_names


def nfc(text):
    """NFC-normalise Korean text; names need it before any join (PROJECT.md → gotchas). Non-strings pass through."""
    return unicodedata.normalize("NFC", text) if isinstance(text, str) else text


def read_song_metadata(metadata_csv: str) -> pd.DataFrame:
    """Per-song table (`metadata.csv` is authoritative); keyed by NFC song_id."""
    metadata = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    metadata["song_id"] = metadata["project_name"].map(nfc)
    return metadata[["song_id", "title", "genreMajor", "genreSub", "genreDetail", "tempoBpm", "western_key", "악기구성"]]


def pick_stems(config, instruments_config) -> pd.DataFrame:
    """Melodic stems of the configured 민속악 genres, with song metadata and the frozen split.

    Args:
        config: the `configs/datasets/separation.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings.

    Returns:
        One row per stem: song_id, instrument, stem_path, split, title, genreDetail, ...
    """
    manifest = pd.read_parquet(config.source_manifest)
    stems = manifest[(manifest["dataset"].astype(str) == "71955") & (manifest["role"] == "stem")].copy()
    stems["song_id"] = stems["song_id"].map(nfc)
    stems["instrument"] = to_english_names(stems["instrument_canonical"].map(nfc), instruments_config)
    pick = config.pick_stems
    stems = stems[stems["genre_sub"].isin(list(pick.genre_subs)) & stems["instrument"].isin(list(pick.instruments))]

    metadata = read_song_metadata(config.metadata_csv)
    picked = stems.merge(metadata, on="song_id", how="left", validate="many_to_one")
    picked = picked[picked["genreMajor"] == pick.genre_major]
    picked = picked.rename(columns={"out_path": "stem_path", "out_duration": "duration_seconds"})
    columns = ["song_id", "instrument", "instrument_raw", "stem_path", "split", "duration_seconds", "active_frac",
               "title", "genreSub", "genreDetail", "tempoBpm", "western_key"]
    return picked[columns].sort_values(["genreSub", "instrument", "song_id"]).reset_index(drop=True)


def list_jeongak_songs(config) -> pd.DataFrame:
    """정악 songs with audio on disk: title, split, stems, duration (from the master WAV).

    Args:
        config: the `configs/datasets/separation.yaml` settings.
    """
    metadata = pd.read_csv(config.metadata_csv, encoding="utf-8-sig").map(nfc)
    songs = metadata[metadata["genreMajor"] == config.jeongak_songs.genre_major].rename(
        columns={"project_name": "song_id", "악기구성": "instrument_list"})
    split = pd.read_parquet(config.eval_manifest)[["song_id", "split"]].map(nfc)
    songs = songs.merge(split, on="song_id", how="inner")  # inner: only songs with audio

    rows = []
    for song in songs.itertuples():
        song_dir = Path(config.ingest_dir) / song.song_id
        stems = sorted(nfc(p.name)[len(song.song_id) + 1:-4] for p in song_dir.glob("*.wav"))
        stems = [s for s in stems if s != "master"]
        duration = sf.info(str(song_dir / f"{song.song_id}_master.wav")).duration
        rows.append({"song_id": song.song_id, "title": song.title, "genre_sub": song.genreSub,
                     "genre_detail": song.genreDetail, "split": song.split,
                     "duration_seconds": duration, "stems": stems})
    return pd.DataFrame(rows)


def song_stem_paths(song_id: str, stems: list[str], config, instruments_config) -> dict[str, Path]:
    """English instrument (= MusicXML part) name → stem WAV, for stems with an English name.

    Args:
        song_id: the song.
        stems: its stem names as on disk (Korean).
        config: the `configs/datasets/separation.yaml` settings.
        instruments_config: the `configs/datasets/instruments.yaml` settings.
    """
    song_dir = Path(config.ingest_dir) / song_id
    english = dict(instruments_config.english_names)
    return {english[s]: song_dir / f"{song_id}_{s}.wav" for s in stems if s in english}
