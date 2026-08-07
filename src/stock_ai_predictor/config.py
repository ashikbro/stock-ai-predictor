"""Runtime configuration models for the stock prediction pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration values used by training and prediction scripts."""

    symbol: str = "AAPL"
    window: int = 5
    transaction_cost_bps: float = 10.0
    max_position: int = 1
