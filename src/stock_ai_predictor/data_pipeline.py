"""Data loading and feature engineering for financial time series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from CSV with strict schema checks.

    Args:
        csv_path: Absolute or relative CSV path containing OHLCV data.

    Returns:
        Sorted DataFrame with parsed datetime index.

    Raises:
        FileNotFoundError: If the CSV path does not exist.
        ValueError: If required columns are missing.
    """

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    frame = pd.read_csv(path)
    frame.columns = [column.lower() for column in frame.columns]

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {sorted(missing)}")

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values("date").set_index("date")

    return frame


def build_features(frame: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Build a compact feature set suitable for RL state construction.

    Args:
        frame: Input OHLCV DataFrame.
        window: Rolling window length used for moving statistics.

    Returns:
        DataFrame containing engineered features.
    """

    features = frame.copy()
    features["return_1"] = features["close"].pct_change()
    features["ma_short"] = features["close"].rolling(window).mean()
    features["ma_long"] = features["close"].rolling(window * 2).mean()
    features["momentum"] = features["close"] - features["close"].shift(window)

    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features
