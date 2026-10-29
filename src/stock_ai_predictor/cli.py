"""Command-line entry point for the deterministic baseline."""

import argparse

from .backtest import backtest_long_flat
from .data import load_ohlcv
from .strategy import moving_average_signal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stock baseline backtest.")
    parser.add_argument("csv", help="Chronological OHLCV CSV path")
    parser.add_argument("--short-window", type=int, default=5)
    parser.add_argument("--long-window", type=int, default=20)
    args = parser.parse_args()

    bars = load_ohlcv(args.csv)
    signals = moving_average_signal(bars, args.short_window, args.long_window)
    result = backtest_long_flat(bars, signals)
    print(f"total_return={result.total_return:.4f}")
    print(f"max_drawdown={result.max_drawdown:.4f}")
    print(f"annualized_volatility={result.volatility:.4f}")
    print(f"trade_count={result.trade_count}")
