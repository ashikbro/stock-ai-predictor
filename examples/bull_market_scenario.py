"""Example scenario: bullish trend simulation."""

from __future__ import annotations

import pandas as pd

from stock_ai_predictor.portfolio import simulate_portfolio


prices = pd.Series([100, 102, 104, 106, 108], index=pd.date_range("2025-01-01", periods=5, tz="UTC"))
actions = [1, 0, 0, 0, 2]
result = simulate_portfolio(prices, actions)
print(f"Bull scenario return: {result.cumulative_return:.2%}")
