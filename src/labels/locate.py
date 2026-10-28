"""Locate a recorded section (e.g. one 장) inside a whole-movement score.

The recording is the query, the score the database: subsequence DTW over coarse chroma
(CENS) finds the stretch of score that best explains the audio. The score is rendered at
several global tempi so the DTW only has to absorb local tempo changes (factor 0.5–2).
The mean cost per audio frame doubles as the match check: a wrong piece costs more.
"""
from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np


@dataclass
class LocateResult:
    """Best placement of the audio inside the score."""
    cost: float                # mean cosine distance per audio frame (0 = identical)
    chroma_shift: int          # semitones the score chroma was rolled by
    quarters_per_second: float  # score tempo the best path was found at
    start_quarter: float       # score position of the first audio frame
    end_quarter: float         # score position of the last audio frame
    path: np.ndarray           # (audio frame, score frame) pairs


@numba.njit(cache=True)
def _subsequence_dtw_21(cost: np.ndarray) -> np.ndarray:
    """Accumulated cost, steps (1,1), (2,1), (1,2), each step weighted by the audio frames it covers.

    Row = audio frame, column = score frame. Any score column may start the path.
    """
    num_audio, num_score = cost.shape
    accumulated = np.full((num_audio, num_score), np.inf)
    accumulated[0, :] = cost[0, :]
    for n in range(1, num_audio):
        for m in range(1, num_score):
            best = accumulated[n - 1, m - 1] + cost[n, m]                     # (1,1)
            if n >= 2:
                best = min(best, accumulated[n - 2, m - 1] + 2.0 * cost[n, m])  # (2,1): audio twice as slow
            if m >= 2:
                best = min(best, accumulated[n - 1, m - 2] + cost[n, m])        # (1,2): audio twice as fast
            accumulated[n, m] = best
    return accumulated


@numba.njit(cache=True)
def _backtrack_21(accumulated: np.ndarray, cost: np.ndarray, end_column: int) -> np.ndarray:
    """Walk back from (last audio frame, end_column) along the cheapest predecessors."""
    n, m = accumulated.shape[0] - 1, end_column
    path = [(n, m)]
    while n > 0 and m > 0:
        here = accumulated[n, m]
        if abs(accumulated[n - 1, m - 1] + cost[n, m] - here) < 1e-9:
            n, m = n - 1, m - 1
        elif n >= 2 and abs(accumulated[n - 2, m - 1] + 2.0 * cost[n, m] - here) < 1e-9:
            n, m = n - 2, m - 1
        else:
            n, m = n - 1, m - 2
        path.append((n, m))
    out = np.empty((len(path), 2), dtype=np.int64)
    for i in range(len(path)):
        out[len(path) - 1 - i, 0] = path[i][0]
        out[len(path) - 1 - i, 1] = path[i][1]
    return out


@numba.njit(cache=True)
def _anchored_dtw_21(cost: np.ndarray) -> np.ndarray:
    """Like ``_subsequence_dtw_21`` but the path must start at score frame 0."""
    num_audio, num_score = cost.shape
    accumulated = np.full((num_audio, num_score), np.inf)
    accumulated[0, 0] = cost[0, 0]
    for n in range(1, num_audio):
        for m in range(1, num_score):
            best = accumulated[n - 1, m - 1] + cost[n, m]
            if n >= 2:
                best = min(best, accumulated[n - 2, m - 1] + 2.0 * cost[n, m])
            if m >= 2:
                best = min(best, accumulated[n - 1, m - 2] + cost[n, m])
            accumulated[n, m] = best
    return accumulated


def locate_consecutive_sections(audio_cens_list: list[np.ndarray], score_cens: np.ndarray,
                                frame_rate: float, quarters_per_second: float, chroma_shift: int,
                                anchor_end: bool) -> list[tuple[float, float]]:
    """Place several consecutive recordings (e.g. 1장, 2장, 3장…) in one score at once.

    The recordings are concatenated in order and aligned with one path that starts at the
    score's beginning (and, with ``anchor_end``, finishes at its end), so sections can
    neither overlap nor jump to a double-time reading of the score.

    Args:
        audio_cens_list: (12, N_i) CENS per recording, in performance order.
        score_cens: (12, M) whole-score CENS at ``quarters_per_second``.
        frame_rate: CENS frames per second.
        quarters_per_second: tempo the score CENS was rendered at.
        chroma_shift: semitone roll for the score chroma.
        anchor_end: force the path to end at the score's last frame.

    Returns:
        (start_quarter, end_quarter) per recording.
    """
    audio_cens = np.concatenate(audio_cens_list, axis=1)
    cost = cosine_cost(audio_cens, np.roll(score_cens, chroma_shift, axis=0))
    accumulated = _anchored_dtw_21(cost)
    end_column = cost.shape[1] - 1 if anchor_end else int(np.argmin(accumulated[-1]))
    path = _backtrack_21(accumulated, cost, end_column)
    to_quarter = quarters_per_second / frame_rate
    joins = np.cumsum([0] + [a.shape[1] for a in audio_cens_list])
    spans = []
    for first_frame, end_frame in zip(joins[:-1], joins[1:]):
        inside = path[(path[:, 0] >= first_frame) & (path[:, 0] < end_frame)]
        spans.append((float(inside[0, 1] * to_quarter), float((inside[-1, 1] + 1) * to_quarter)))
    return spans


def cosine_cost(audio_cens: np.ndarray, score_cens: np.ndarray) -> np.ndarray:
    """1 − cosine similarity between unit-norm columns: (audio frames, score frames)."""
    return np.clip(1.0 - audio_cens.T @ score_cens, 0.0, 2.0)


def locate_in_score(audio_cens: np.ndarray, score_cens_by_tempo: dict[float, np.ndarray],
                    frame_rate: float, chroma_shifts: list[int]) -> LocateResult:
    """Find the best (tempo, chroma shift, score span) for the audio.

    Args:
        audio_cens: (12, N) audio CENS at ``frame_rate``.
        score_cens_by_tempo: quarters-per-second → (12, M) score CENS at ``frame_rate``.
        frame_rate: CENS frames per second, both sides.
        chroma_shifts: semitone rolls of the score chroma to try.
    """
    best = None
    for quarters_per_second, score_cens in score_cens_by_tempo.items():
        for shift in chroma_shifts:
            cost = cosine_cost(audio_cens, np.roll(score_cens, shift, axis=0))
            accumulated = _subsequence_dtw_21(cost)
            end_column = int(np.argmin(accumulated[-1]))
            mean_cost = float(accumulated[-1, end_column]) / cost.shape[0]
            if best is None or mean_cost < best[0]:
                best = (mean_cost, shift, quarters_per_second, cost, accumulated, end_column)
    mean_cost, shift, quarters_per_second, cost, accumulated, end_column = best
    path = _backtrack_21(accumulated, cost, end_column)
    to_quarter = quarters_per_second / frame_rate  # score frame → quarter notes
    return LocateResult(cost=mean_cost, chroma_shift=int(shift),
                        quarters_per_second=float(quarters_per_second),
                        start_quarter=float(path[0, 1] * to_quarter),
                        end_quarter=float((path[-1, 1] + 1) * to_quarter), path=path)
