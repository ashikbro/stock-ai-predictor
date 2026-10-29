"""Simple long/flat backtesting with explicit trading costs."""

from dataclasses import dataclass
from math import sqrt

from .data import OHLCVBar


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    total_return: float
    max_drawdown: float
    volatility: float
    trade_count: int


def backtest_long_flat(
    bars: list[OHLCVBar],
    signals: list[int],
    *,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
) -> BacktestResult:
    """Backtest positions entered at each bar close, with fees and slippage."""
    if len(bars) != len(signals) or not bars:
        raise ValueError("bars and signals must be non-empty and have equal length")
    if initial_cash <= 0 or fee_rate < 0 or slippage_rate < 0:
        raise ValueError("initial_cash must be positive and costs cannot be negative")

    cash = initial_cash
    shares = 0.0
    equity_curve: list[float] = []
    trade_count = 0
    previous_signal = 0
    for bar, signal in zip(bars, signals):
        if signal not in (0, 1):
            raise ValueError("signals must contain only 0 (flat) or 1 (long)")
        if signal != previous_signal:
            if signal:
                execution_price = bar.close * (1 + slippage_rate)
                shares = cash / (execution_price * (1 + fee_rate))
                cash -= shares * execution_price * (1 + fee_rate)
            else:
                execution_price = bar.close * (1 - slippage_rate)
                cash += shares * execution_price * (1 - fee_rate)
                shares = 0.0
            trade_count += 1
            previous_signal = signal
        equity_curve.append(cash + shares * bar.close)

    if shares:
        final_price = bars[-1].close * (1 - slippage_rate)
        cash += shares * final_price * (1 - fee_rate)
        equity_curve[-1] = cash
        trade_count += 1

    peak = equity_curve[0]
    max_drawdown = 0.0
    returns: list[float] = []
    for value, previous in zip(equity_curve[1:], equity_curve):
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, (peak - value) / peak)
        returns.append(value / previous - 1)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns) if returns else 0.0
    return BacktestResult(
        equity_curve=equity_curve,
        total_return=equity_curve[-1] / initial_cash - 1,
        max_drawdown=max_drawdown,
        volatility=sqrt(variance) * sqrt(252),
        trade_count=trade_count,
    )
