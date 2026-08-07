"""Portfolio simulation utilities for RL strategy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class PortfolioResult:
    """Container for portfolio simulation outputs."""

    equity_curve: pd.Series
    cumulative_return: float


def simulate_portfolio(
    prices: pd.Series,
    actions: Iterable[int],
    transaction_cost_bps: float = 10.0,
    initial_cash: float = 10_000.0,
) -> PortfolioResult:
    """Simulate a single-asset strategy from action sequence.

    Args:
        prices: Price series aligned to actions.
        actions: Sequence of actions (`0` hold, `1` buy, `2` sell).
        transaction_cost_bps: Cost per trade in basis points.
        initial_cash: Initial portfolio value.

    Returns:
        PortfolioResult containing equity curve and cumulative return.

    Raises:
        ValueError: If lengths of prices and actions differ.
    """

    action_list = list(actions)
    if len(prices) != len(action_list):
        raise ValueError("prices and actions must have equal length")

    cash = initial_cash
    position = 0
    equity_values: list[float] = []

    for price, action in zip(prices, action_list):
        if action == 1 and position == 0:
            fee = price * (transaction_cost_bps / 10_000)
            cash -= price + fee
            position = 1
        elif action == 2 and position == 1:
            fee = price * (transaction_cost_bps / 10_000)
            cash += price - fee
            position = 0

        equity_values.append(cash + position * price)

    equity_curve = pd.Series(equity_values, index=prices.index)
    cumulative_return = (equity_curve.iloc[-1] / initial_cash) - 1.0
    return PortfolioResult(equity_curve=equity_curve, cumulative_return=float(cumulative_return))
