"""Focused tests for data pipeline and portfolio simulation behavior."""

from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from stock_ai_predictor.data_pipeline import build_features, load_price_data
from stock_ai_predictor.portfolio import simulate_portfolio


class PipelinePortfolioTests(unittest.TestCase):
    def test_load_and_features(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2025-01-01", periods=12, tz="UTC"),
                "open": [100 + i for i in range(12)],
                "high": [101 + i for i in range(12)],
                "low": [99 + i for i in range(12)],
                "close": [100 + i for i in range(12)],
                "volume": [1_000_000 for _ in range(12)],
            }
        )

        with tempfile.NamedTemporaryFile(suffix=".csv") as temp:
            frame.to_csv(temp.name, index=False)
            loaded = load_price_data(temp.name)
            features = build_features(loaded, window=3)

        self.assertGreater(len(features), 0)
        self.assertIn("momentum", features.columns)

    def test_portfolio_length_validation(self) -> None:
        prices = pd.Series([100.0, 101.0])
        with self.assertRaises(ValueError):
            simulate_portfolio(prices, [1])


if __name__ == "__main__":
    unittest.main()
