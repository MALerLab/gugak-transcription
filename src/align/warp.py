"""Fit a MusicXML section's timing to the recorded stems.

1. Refine the section boundaries found in stage 1 (subsequence DTW at 10 frames/s).
2. MrMsDTW (chroma + DLNCO, synctoolbox) between the audio and the score section,
   rendered at the section's mean tempo.
3. Warp every note's onset and offset through the path.

The *linear* baseline plays the same refined section at constant tempo: the right notes,
the right average speed, no timing fit — "the MusicXML's robot timing".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.dtw.utils import make_path_strictly_monotonic
from synctoolbox.feature.utils import shift_chroma_vectors

from src.align import locate
from src.features import chroma
from src.notes.table import quarters_to_seconds


@dataclass
class Section:
    """Where the recording sits in the score, and how fast it goes on average."""
    start_quarter: float
    end_quarter: float
    audio_seconds: float

    @property
    def quarters_per_second(self) -> float:
        return (self.end_quarter - self.start_quarter) / self.audio_seconds

    def linear_map(self, quarters: np.ndarray) -> np.ndarray:
        """Score quarters → audio seconds at constant tempo."""
        return (np.asarray(quarters) - self.start_quarter) / self.quarters_per_second


def crop_section(table: pd.DataFrame, start_quarter: float, end_quarter: float) -> pd.DataFrame:
    """Notes starting inside [start, end); durations clipped at the end."""
    inside = (table["onset_quarter"] >= start_quarter) & (table["onset_quarter"] < end_quarter)
    cropped = table[inside].copy()
    note_end = np.minimum(cropped["onset_quarter"] + cropped["duration_quarter"], end_quarter)
    cropped["duration_quarter"] = np.where(cropped["is_grace"], 0.0, note_end - cropped["onset_quarter"])
    return cropped


def refine_section(audio_chroma_10hz: np.ndarray, table: pd.DataFrame, coarse_start: float,
                   coarse_end: float, coarse_quarters_per_second: float, chroma_shift: int,
                   audio_seconds: float, config) -> tuple[Section, float]:
    """Re-locate the section at 10 frames/s, searching a padded window around stage 1's span.

    Returns:
        (refined section, mean cost per audio frame).
    """
    frame_rate = config.refine_feature_rate / config.refine_cens_downsample
    audio_cens = chroma.coarse_cens(audio_chroma_10hz, config.refine_feature_rate,
                                      config.refine_cens_smooth_frames, config.refine_cens_downsample)
    window_start = max(0.0, coarse_start - config.refine_pad_quarters)
    window = crop_section(table[~table["is_grace"]], window_start, coarse_end + config.refine_pad_quarters)
    by_tempo = {}
    for factor in config.refine_tempo_factors:
        quarters_per_second = coarse_quarters_per_second * factor
        timed = quarters_to_seconds(window, lambda q: (q - window_start) / quarters_per_second, 0.0)
        score_chroma = chroma.score_features(timed, config.refine_feature_rate, with_onsets=False)["chroma"]
        by_tempo[quarters_per_second] = chroma.coarse_cens(
            score_chroma, config.refine_feature_rate, config.refine_cens_smooth_frames, config.refine_cens_downsample)
    result = locate.locate_in_score(audio_cens, by_tempo, frame_rate, [chroma_shift])
    section = Section(start_quarter=window_start + result.start_quarter,
                      end_quarter=window_start + result.end_quarter, audio_seconds=audio_seconds)
    return section, result.cost


def mrmsdtw_path(audio_feats: dict, score_feats: dict, chroma_shift: int, config) -> np.ndarray:
    """MrMsDTW path as (2, L) frame indices: row 0 audio, row 1 score."""
    score_chroma = shift_chroma_vectors(score_feats["chroma"], chroma_shift)
    score_dlnco = shift_chroma_vectors(score_feats["dlnco"], chroma_shift)
    path = sync_via_mrmsdtw(f_chroma1=audio_feats["chroma"], f_onset1=audio_feats["dlnco"],
                            f_chroma2=score_chroma, f_onset2=score_dlnco,
                            input_feature_rate=config.feature_rate,
                            step_weights=np.array(config.step_weights, dtype=np.float64),
                            threshold_rec=config.threshold_rec, alpha=config.alpha)
    return make_path_strictly_monotonic(path)


def path_to_time_map(path: np.ndarray, feature_rate: int):
    """Score seconds (nominal tempo) → audio seconds, by linear interpolation along the path."""
    audio_seconds = path[0] / feature_rate
    score_seconds = path[1] / feature_rate
    return lambda t: np.interp(t, score_seconds, audio_seconds)


def align_section(audio_22k: np.ndarray, tuning_cents: int, section_table: pd.DataFrame,
                  section: Section, chroma_shift: int, config):
    """Fit one score section to one audio signal.

    Returns:
        (quarter → audio-seconds function, MrMsDTW path)
    """
    nominal = quarters_to_seconds(section_table[~section_table["is_grace"]],
                                          section.linear_map, 0.0)
    audio_feats = chroma.audio_features(audio_22k, config.feature_rate, tuning_cents)
    score_feats = chroma.score_features(nominal, config.feature_rate,
                                          num_frames=audio_feats["chroma"].shape[1])
    path = mrmsdtw_path(audio_feats, score_feats, chroma_shift, config)
    score_to_audio = path_to_time_map(path, config.feature_rate)
    return (lambda q: score_to_audio(section.linear_map(q))), path


def warp_notes(section_table: pd.DataFrame, quarter_to_second, grace_seconds: float) -> pd.DataFrame:
    """Timed note table through a quarter → seconds map (grace notes get fixed lengths)."""
    return quarters_to_seconds(section_table, quarter_to_second, grace_seconds)
