"""Evaluation metrics for prediction and trading performance."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Compute maximum drawdown from an equity curve."""
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max) - 1.0
    return float(drawdown.min())


def compute_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio assuming daily returns."""
    excess = returns - (risk_free_rate / 252)
    std = float(excess.std())
    if std == 0.0:
        return 0.0
    return float(np.sqrt(252) * excess.mean() / std)
