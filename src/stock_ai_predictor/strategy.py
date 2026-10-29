"""Deterministic baseline signals."""

from .data import OHLCVBar


def moving_average_signal(
    bars: list[OHLCVBar],
    short_window: int = 5,
    long_window: int = 20,
) -> list[int]:
    """Return long/flat signals without using future prices."""
    if short_window < 1 or long_window <= short_window:
        raise ValueError("Require 1 <= short_window < long_window")
    closes = [bar.close for bar in bars]
    signals = [0] * len(bars)
    for index in range(long_window - 1, len(bars)):
        short_average = sum(closes[index - short_window + 1:index + 1]) / short_window
        long_average = sum(closes[index - long_window + 1:index + 1]) / long_window
        signals[index] = int(short_average > long_average)
    return signals
