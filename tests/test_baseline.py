from datetime import date, timedelta

import pytest

from stock_ai_predictor.backtest import backtest_long_flat
from stock_ai_predictor.data import OHLCVBar, load_ohlcv
from stock_ai_predictor.strategy import moving_average_signal


def make_bars(closes):
    return [
        OHLCVBar(date(2024, 1, 1) + timedelta(days=index), close, close, close, close, 100)
        for index, close in enumerate(closes)
    ]


def test_signal_does_not_use_future_prices():
    bars = make_bars([10, 10, 10, 10, 20, 20])
    signals = moving_average_signal(bars, short_window=2, long_window=4)
    assert signals[:3] == [0, 0, 0]
    assert signals[-1] == 1


def test_backtest_applies_costs():
    bars = make_bars([10, 12, 14, 16, 18])
    result = backtest_long_flat(bars, [0, 1, 1, 1, 0], fee_rate=0.01, slippage_rate=0.01)
    assert result.total_return > 0
    assert result.trade_count == 2


def test_loader_rejects_unsorted_rows(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text(
        "date,open,high,low,close,volume\n"
        "2024-01-02,10,11,9,10,100\n"
        "2024-01-01,10,11,9,10,100\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="chronologically"):
        load_ohlcv(path)
