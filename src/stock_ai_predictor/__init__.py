"""Reproducible research primitives for stock prediction experiments."""

from .backtest import BacktestResult, backtest_long_flat
from .data import OHLCVBar, load_ohlcv
from .strategy import moving_average_signal

__all__ = [
    "BacktestResult",
    "OHLCVBar",
    "backtest_long_flat",
    "load_ohlcv",
    "moving_average_signal",
]
