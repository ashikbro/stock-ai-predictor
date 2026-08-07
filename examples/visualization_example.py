"""Visualization example for equity curve tracking."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from stock_ai_predictor.portfolio import simulate_portfolio


prices = pd.Series([100, 102, 101, 103, 105], index=pd.date_range("2025-01-01", periods=5, tz="UTC"))
actions = [1, 0, 0, 0, 2]
result = simulate_portfolio(prices, actions)

plt.figure(figsize=(8, 4))
result.equity_curve.plot(title="Sample Equity Curve")
plt.ylabel("Portfolio Value")
plt.tight_layout()
plt.show()
