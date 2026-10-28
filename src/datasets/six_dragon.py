"""Six dragon dataset: the catalogue of MusicXML scores, one file per piece."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


def parse_musicxml_filename(path: Path, part_names: list[str]) -> tuple[str, list[str]]:
    """Split ``<piece>_<part>_<part>.musicxml`` into the piece title and its part list.

    Args:
        path: MusicXML file path.
        part_names: every known part name, used to find where the part suffix starts.
    """
    stem = unicodedata.normalize("NFC", path.stem)
    part_pattern = "|".join(part_names)
    match = re.match(rf"(.+?)_((?:{part_pattern})(?:_(?:{part_pattern}))*)$", stem)
    if match is None:
        raise ValueError(f"unexpected MusicXML filename: {path.name}")
    return match.group(1), match.group(2).split("_")


def list_musicxml_pieces(musicxml_dir: Path, part_names: list[str]) -> pd.DataFrame:
    """One row per MusicXML file: ``piece``, ``suite``, ``movement``, ``parts``, ``path``.

    ``suite`` is the first word of a two-word title (e.g. 영산회상 in "영산회상 상령산"),
    empty for single-word titles like 보허자.
    """
    rows = []
    for path in sorted(Path(musicxml_dir).glob("*.musicxml")):
        piece, parts = parse_musicxml_filename(path, part_names)
        suite, _, movement = piece.rpartition(" ")
        rows.append({"piece": piece, "suite": suite, "movement": movement,
                     "parts": parts, "path": str(path)})
    return pd.DataFrame(rows)
