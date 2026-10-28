"""Stage 4 driver: aligned MIDI for every song, plus the 30 s listening clips.

Sounding pitch = written pitch + the per-(piece, part) shift measured from the stems' f0
(median over that piece's songs), so transposed notation (e.g. 보허자) comes out right.

Usage: python -m src.labels.listening --configs configs/labels
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.audio import read_audio
from src.labels import convert, match, render
from src.labels.pipeline import cache_dir, load_configs


def sounding_shifts(quality_table: pd.DataFrame) -> dict[tuple[str, str], int]:
    """(piece, part) → semitones from written to sounding pitch (median over songs)."""
    rows = quality_table[quality_table["method"] == "per_song"]
    return rows.groupby(["piece", "part"])["sounding_shift"].median().round().astype(int).to_dict()


def write_song_midi(song_id: str, piece: str, method: str, shifts: dict, out_path: Path, config) -> pd.DataFrame:
    """Aligned note table of one song/method → MIDI at sounding pitch; returns the table."""
    timed = pd.read_parquet(cache_dir(config) / "aligned" / f"{song_id}_{method}.parquet")
    part_shift = {part: shifts.get((piece, part), 0) for part in timed["part"].unique()}
    convert.note_table_to_midi(timed, out_path, dict(config.convert.gm_programs),
                               config.convert.velocity, part_shift)
    return timed


def busiest_window(timed: pd.DataFrame, parts: list[str], seconds: float, audio_seconds: float) -> float:
    """Start time of the ``seconds`` window holding the most note onsets of ``parts``."""
    onsets = np.sort(timed.loc[timed["part"].isin(parts), "start"].to_numpy())
    starts = np.arange(0, max(audio_seconds - seconds, 0) + 1e-9, 1.0)
    counts = np.searchsorted(onsets, starts + seconds) - np.searchsorted(onsets, starts)
    return float(starts[int(np.argmax(counts))])


def build_clip(clip, matches: pd.DataFrame, shifts: dict, config) -> list[dict]:
    """Write the three WAVs (+ MIDI) of one clip, and of its unaligned version if asked."""
    row = matches.set_index("song_id").loc[clip.song_id]
    folder = Path(config.match.output_dir) / "listening" / clip.name
    stem_paths = match.song_stem_paths(clip.song_id, list(row["stems"]), config.match)
    parts = list(clip.parts) if clip.parts != "all" else [p for p in stem_paths if p in row["parts_in_musicxml"]]
    methods = [clip.method] + (["linear"] if clip.get("with_unaligned", False) else [])

    # The real audio: one stem or the sum of the clip's stems, stereo as on disk
    real, sample_rate = None, None
    for part in parts:
        audio, sample_rate = read_audio(stem_paths[part])
        real = audio if real is None else real[:len(audio)] + audio[:len(real)]
    audio_seconds = len(real) / sample_rate

    written = []
    start = clip.get("start_seconds", None)
    for method in methods:
        midi_path = folder / f"{clip.name}_{'aligned' if method != 'linear' else 'unaligned'}.mid"
        timed = write_song_midi(clip.song_id, row["piece"], method, shifts, midi_path, config)
        # Render only the clip's parts, so the render is directly comparable with the real audio
        parts_midi = folder / f"_{method}_parts.mid"
        convert.note_table_to_midi(timed[timed["part"].isin(parts)], parts_midi, dict(config.convert.gm_programs),
                                   config.convert.velocity, {p: shifts.get((row["piece"], p), 0) for p in parts})
        full_render = folder / f"_{method}_render_full.wav"
        render.render_midi(parts_midi, full_render, config.render)
        rendered, _ = read_audio(full_render)
        if start is None:  # picked once, from the clip's main method, so all versions line up
            start = busiest_window(timed, parts, config.render.clip_seconds, audio_seconds)
        tag = "" if method != "linear" else "_unaligned"
        paths = render.write_clip_set(render.excerpt(real, sample_rate, start, config.render.clip_seconds),
                                      render.excerpt(rendered, sample_rate, start, config.render.clip_seconds),
                                      sample_rate, folder / f"{clip.name}{tag}", config.render)
        parts_midi.unlink()
        full_render.unlink()
        written.append({"clip": clip.name, "song_id": clip.song_id, "piece": row["piece"], "parts": parts,
                        "method": method, "start_seconds": start, **{k: str(v) for k, v in paths.items()},
                        "midi": str(midi_path)})
    return written


def main(config_dir: str) -> None:
    config = load_configs(config_dir)
    output_dir = Path(config.match.output_dir)
    matches = pd.read_parquet(output_dir / "match_table.parquet")
    quality_table = pd.read_parquet(output_dir / "alignment_quality.parquet")
    shifts = sounding_shifts(quality_table)

    # Aligned MIDI for every aligned song (labels; kept on storage, not in the repo)
    midi_dir = Path(os.path.expanduser(config.render.aligned_midi_dir))
    sections = pd.read_parquet(output_dir / "alignment_sections.parquet")
    aligned = sections[sections["error"].isna()] if "error" in sections.columns else sections
    for song in aligned.itertuples():
        for method in ("per_song", "per_stem"):
            write_song_midi(song.song_id, song.piece, method, shifts, midi_dir / method / f"{song.song_id}.mid", config)

    # Listening clips
    rows = []
    for clip in config.render.clips:
        rows.extend(build_clip(clip, matches, shifts, config))
        print("clip", clip.name, flush=True)
    pd.DataFrame(rows).to_parquet(output_dir / "listening_clips.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="configs/labels")
    main(parser.parse_args().configs)
