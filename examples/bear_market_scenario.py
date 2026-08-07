"""Example scenario: bearish trend simulation."""

from __future__ import annotations

import pandas as pd

from stock_ai_predictor.portfolio import simulate_portfolio


prices = pd.Series([108, 106, 104, 102, 100], index=pd.date_range("2025-01-01", periods=5, tz="UTC"))
actions = [2, 0, 0, 0, 1]
result = simulate_portfolio(prices, actions)
print(f"Bear scenario return: {result.cumulative_return:.2%}")
