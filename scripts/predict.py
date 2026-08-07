"""Run a lightweight prediction and simulation pipeline from CSV data."""

from __future__ import annotations

import argparse

from stock_ai_predictor.data_pipeline import build_features, load_price_data
from stock_ai_predictor.portfolio import simulate_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock prediction workflow")
    parser.add_argument("--csv-path", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--window", type=int, default=5, help="Feature window size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_price_data(args.csv_path)
    features = build_features(frame, window=args.window)

    # Baseline signal: trend-following proxy for demonstration.
    actions = [1 if row.ma_short > row.ma_long else 2 for row in features.itertuples()]
    result = simulate_portfolio(features["close"], actions)

    print(f"Rows processed: {len(features)}")
    print(f"Cumulative return: {result.cumulative_return:.2%}")


if __name__ == "__main__":
    main()
