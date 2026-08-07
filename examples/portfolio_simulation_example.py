"""Example scenario: mixed regime portfolio simulation."""

from __future__ import annotations

import pandas as pd

from stock_ai_predictor.metrics import compute_max_drawdown
from stock_ai_predictor.portfolio import simulate_portfolio


prices = pd.Series([100, 101, 99, 102, 98, 103], index=pd.date_range("2025-01-01", periods=6, tz="UTC"))
actions = [1, 0, 2, 1, 2, 0]
result = simulate_portfolio(prices, actions)
dd = compute_max_drawdown(result.equity_curve)
print(f"Mixed scenario return: {result.cumulative_return:.2%}")
print(f"Max drawdown: {dd:.2%}")
