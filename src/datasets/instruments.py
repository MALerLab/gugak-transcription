"""Instrument names: the datasets' Korean names → the English names the code uses."""
from __future__ import annotations

import pandas as pd


def to_english_names(korean_names: pd.Series, config) -> pd.Series:
    """English name per Korean instrument name; NaN for instruments without one.

    Args:
        korean_names: instrument names as the dataset writes them (NFC).
        config: the `configs/datasets/instruments.yaml` settings.
    """
    return korean_names.map(dict(config.english_names))
