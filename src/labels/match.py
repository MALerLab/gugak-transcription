"""Stage 1: match six dragon MusicXML pieces to the separation dataset's 정악 songs.

Two steps:
1. Title rules (``configs/labels/match.yaml``) propose candidate pieces per song.
2. Audio check: each song is located inside every piece (``locate.py``). The candidate's cost
   is compared with the best *unrelated* piece (different movement name) — the null.

Usage: python -m src.labels.match --config configs/labels/match.yaml
"""
from __future__ import annotations

import argparse
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from omegaconf import OmegaConf

from src.labels import convert, features, locate


def nfc(text):
    """NFC-normalise Korean text (filenames and metadata disagree otherwise)."""
    return unicodedata.normalize("NFC", text) if isinstance(text, str) else text


def list_jeongak_songs(config) -> pd.DataFrame:
    """정악 songs with audio on disk: title, split, stems, duration (from the master WAV)."""
    metadata = pd.read_csv(config.metadata_csv, encoding="utf-8-sig").map(nfc)
    songs = metadata[metadata["genreMajor"] == config.genre_major].rename(
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


def rule_candidates(title: str, pieces: pd.DataFrame, config) -> tuple[list[str], str]:
    """Candidate pieces for one of our titles, plus how the rule found them.

    Returns:
        (candidate piece names, rule) where rule is one of ``suite+movement``,
        ``suite+movement alias``, ``piece alias``, ``가곡 form``, ``none``.
    """
    known = set(pieces["piece"])
    # Suites: 관악영산회상 / 평조회상 / 현악영산회상 / 천년만세 + a movement name
    for alias, suite in config.suite_aliases.items():
        if alias in title:
            for movement in config.movements:
                if movement in title:
                    target = config.movement_aliases.get(movement, movement)
                    rule = "suite+movement alias" if target != movement else "suite+movement"
                    piece = f"{suite} {target}"
                    return ([piece] if piece in known else []), rule
    # Whole pieces named by a title prefix (longest alias first: 절화(길군악) before 길군악)
    for alias in sorted(config.piece_aliases, key=len, reverse=True):
        if title.startswith(alias):
            return [p for p in config.piece_aliases[alias] if p in known], "piece alias"
    # 가곡: the form name is the title before "(" (optionally after 평조/계면조)
    words = title.split("(")[0].split()
    form = words[1] if words[0] in ("평조", "계면조") and len(words) > 1 else words[0]
    form = next((v for k, v in config.gagok_form_aliases.items() if form.startswith(k)), form)
    gagok = [f"{suite} {form}" for suite in config.gagok_suites if f"{suite} {form}" in known]
    if gagok:
        return gagok, "가곡 form"
    return [], "none"


def song_stem_paths(song_id: str, stems: list[str], config) -> dict[str, Path]:
    """Part name → stem WAV, for the stems that have a MusicXML part."""
    song_dir = Path(config.ingest_dir) / song_id
    return {config.stem_to_part[s]: song_dir / f"{song_id}_{s}.wav"
            for s in stems if s in config.stem_to_part}


def audio_cens_for_song(song_id: str, stem_paths: list[Path], config) -> tuple[np.ndarray, int]:
    """Coarse CENS of the mix of a song's part-matched stems (cached), plus its tuning."""
    cache = Path(os.path.expanduser(config.cache_dir)) / "audio_cens" / f"{song_id}.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["cens"], int(stored["tuning"])
    audio, sample_rate = features.load_stem_mix(stem_paths)
    audio_22k = features.to_feature_rate(audio, sample_rate)
    tuning = features.tuning_offset_cents(audio_22k)
    chroma = features.audio_features(audio_22k, config.feature_rate, tuning, with_onsets=False)["chroma"]
    cens = features.coarse_cens(chroma, config.feature_rate, config.cens_smooth_frames, config.cens_downsample)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, cens=cens, tuning=tuning)
    return cens, tuning


def score_cens_for_piece(piece: str, path: str, parts: list[str], config) -> dict[float, np.ndarray]:
    """Coarse CENS of a whole score at every tempo of the grid (cached)."""
    cache = Path(os.path.expanduser(config.cache_dir)) / "score_cens" / f"{piece}.npz"
    if cache.exists():
        stored = np.load(cache)
        return {float(k): stored[k] for k in stored.files}
    table = convert.musicxml_to_note_table(Path(path), parts)
    table = table[~table["is_grace"]]  # ornaments barely move coarse chroma
    by_tempo = {}
    for quarters_per_second in config.quarters_per_second_grid:
        timed = convert.quarters_to_seconds(table, lambda q: q / quarters_per_second, 0.0)
        chroma = features.score_features(timed, config.feature_rate, with_onsets=False)["chroma"]
        by_tempo[quarters_per_second] = features.coarse_cens(
            chroma, config.feature_rate, config.cens_smooth_frames, config.cens_downsample)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{str(k): v for k, v in by_tempo.items()})
    return by_tempo


def _audio_job(args):
    song_id, stem_paths, config = args
    return song_id, audio_cens_for_song(song_id, stem_paths, config)


def _score_job(args):
    piece, path, parts, config = args
    return piece, score_cens_for_piece(piece, path, parts, config)


def _locate_job(args):
    """Locate one song in every piece; full chroma-shift search only for its candidates."""
    song_id, audio_cens, score_cens, candidates, config = args
    frame_rate = config.feature_rate / config.cens_downsample
    rows = []
    for piece, by_tempo in score_cens.items():
        shifts = config.chroma_shifts_candidates if piece in candidates else config.chroma_shifts_null
        result = locate.locate_in_score(audio_cens, by_tempo, frame_rate, list(shifts))
        rows.append({"song_id": song_id, "piece": piece, "cost": result.cost,
                     "chroma_shift": result.chroma_shift,
                     "quarters_per_second": result.quarters_per_second,
                     "start_quarter": result.start_quarter, "end_quarter": result.end_quarter})
    return rows


def decide_matches(songs: pd.DataFrame, pieces: pd.DataFrame, costs: pd.DataFrame, config) -> pd.DataFrame:
    """One row per song: chosen piece, null margin, confidence, located span, part/stem overlap."""
    movement_of = dict(zip(pieces["piece"], pieces["movement"]))
    parts_of = dict(zip(pieces["piece"], pieces["parts"]))
    rows = []
    for song in songs.itertuples():
        song_costs = costs[costs["song_id"] == song.song_id].set_index("piece")
        overall_best = song_costs["cost"].idxmin()
        row = {"song_id": song.song_id, "title": song.title, "split": song.split,
               "duration_seconds": song.duration_seconds, "rule": song.rule,
               "rule_candidates": list(song.candidates), "overall_best_piece": overall_best}
        if not song.candidates:
            rows.append({**row, "piece": None, "confidence": "no candidate"})
            continue
        candidate_costs = song_costs.loc[list(song.candidates)].sort_values("cost")
        piece = candidate_costs.index[0]
        best = candidate_costs.iloc[0]
        unrelated = song_costs[[movement_of[p] != movement_of[piece] for p in song_costs.index]]
        null_cost = float(unrelated["cost"].min())
        null_margin = null_cost - float(best["cost"])
        runner_up_margin = (float(candidate_costs["cost"].iloc[1]) - float(best["cost"])
                            if len(candidate_costs) > 1 else np.nan)
        parts = parts_of[piece]
        stems_with_part = [s for s in song.stems if config.stem_to_part.get(s) in parts]
        # Identity: the candidate (or a same-movement version of it) must beat all 91 pieces
        identity_ok = movement_of[overall_best] == movement_of[piece]
        candidates_resolved = len(candidate_costs) == 1 or runner_up_margin >= config.min_runner_up_margin
        if not identity_ok:
            confidence = "rejected"
        elif overall_best == piece and null_margin >= config.min_null_margin_high and candidates_resolved:
            confidence = "high"
        elif null_margin >= config.min_null_margin_medium:
            confidence = "medium"
        else:
            confidence = "low"
        rows.append({**row, "piece": piece, "confidence": confidence,
                     "cost": float(best["cost"]), "null_cost": null_cost, "null_margin": null_margin,
                     "null_piece": unrelated["cost"].idxmin(), "runner_up_margin": runner_up_margin,
                     "chroma_shift": int(best["chroma_shift"]),
                     "quarters_per_second": float(best["quarters_per_second"]),
                     "start_quarter": float(best["start_quarter"]), "end_quarter": float(best["end_quarter"]),
                     "parts_in_musicxml": list(parts), "stems_with_part": stems_with_part,
                     "parts_without_stem": [p for p in parts if p not in
                                            {config.stem_to_part[s] for s in stems_with_part}],
                     "stems_without_part": [s for s in song.stems if s not in stems_with_part]})
    return pd.DataFrame(rows)


def main(config_path: str, decide_only: bool) -> None:
    config = OmegaConf.load(config_path)
    convert_config = OmegaConf.load(Path(config_path).with_name("convert.yaml"))
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Both catalogues + title rules
    pieces = convert.list_musicxml_pieces(Path(convert_config.musicxml_dir), list(convert_config.part_names))
    songs = list_jeongak_songs(config)
    found = [rule_candidates(t, pieces, config) for t in songs["title"]]
    songs["candidates"] = [c for c, _ in found]
    songs["rule"] = [r for _, r in found]
    print(f"{len(songs)} 정악 songs with audio, {len(pieces)} pieces, "
          f"{(songs['candidates'].str.len() > 0).sum()} songs with a title candidate")

    if decide_only:  # reuse saved costs; tunings come from the feature cache
        costs = pd.read_parquet(output_dir / "match_costs.parquet")
        songs = songs[songs["song_id"].isin(costs["song_id"])].copy()
        songs["tuning_cents"] = [audio_cens_for_song(s, [], config)[1] for s in songs["song_id"]]
        audio = dict.fromkeys(songs["song_id"])
    else:
        costs, audio = compute_costs(songs, pieces, config)
        costs.to_parquet(output_dir / "match_costs.parquet")

    # Decide + write
    matches = decide_matches(songs[songs["song_id"].isin(audio)], pieces, costs, config)
    matches = matches.merge(songs[["song_id", "tuning_cents", "stems"]], on="song_id")
    matches.to_parquet(output_dir / "match_table.parquet")
    print(matches["confidence"].value_counts().to_string())
    for tier in (["high"], ["high", "medium"], ["high", "medium", "low"]):
        kept = matches[matches["confidence"].isin(tier)]
        print(f"{'+'.join(tier)}: {len(kept)} songs, {kept['duration_seconds'].sum() / 3600:.2f} h")


def compute_costs(songs: pd.DataFrame, pieces: pd.DataFrame, config) -> tuple[pd.DataFrame, dict]:
    """Features (cached) for every song and piece, then locate every song in every piece."""
    with ProcessPoolExecutor(config.num_workers) as pool:
        score_cens = dict(pool.map(_score_job, [(p.piece, p.path, p.parts, config) for p in pieces.itertuples()]))
        audio_jobs = []
        for song in songs.itertuples():
            stem_paths = list(song_stem_paths(song.song_id, song.stems, config).values())
            if stem_paths:
                audio_jobs.append((song.song_id, stem_paths, config))
        audio = dict(pool.map(_audio_job, audio_jobs))
        songs["tuning_cents"] = songs["song_id"].map(lambda s: audio[s][1] if s in audio else np.nan)
        print("features done", flush=True)

        # Locate every song in every piece
        locate_jobs = [(s.song_id, audio[s.song_id][0], score_cens, set(s.candidates), config)
                       for s in songs.itertuples() if s.song_id in audio]
        costs = pd.DataFrame([row for rows in pool.map(_locate_job, locate_jobs) for row in rows])
    return costs, audio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/labels/match.yaml")
    parser.add_argument("--decide-only", action="store_true", help="reuse match_costs.parquet")
    arguments = parser.parse_args()
    main(arguments.config, arguments.decide_only)
