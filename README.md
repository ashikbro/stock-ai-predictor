# Stock AI Predictor

An experimental research tool for evaluating market-prediction features,
backtesting strategies, and simulating portfolios using historical financial
data.

## Status

**Baseline stage — deterministic research scaffold available.**

The repository includes validated OHLCV loading, a moving-average long/flat
baseline, cost-aware backtesting, and regression tests. It does not provide
trading advice or a production trading system.

## Important disclaimer

This project is for research and education only. It must not provide
personalized financial advice, guarantee returns, or be used as the sole basis
for trading decisions. Historical performance does not predict future results.

## First-release scope

- Import historical OHLCV data from versioned local files
- Create time-aware train, validation, and test splits
- Compare a naïve baseline against classical ML models
- Backtest with transaction costs, slippage, and position limits
- Simulate portfolios with transparent risk metrics
- Export predictions, trades, and evaluation results

Reinforcement learning should be added only after the deterministic baseline
and backtesting engine are validated. Every experiment should record its
dataset version, date range, features, random seed, model configuration, and
metrics.

## Recommended architecture

```text
data/          Versioned schemas and local, ignored datasets
src/           Feature engineering, models, and backtesting
experiments/   Reproducible configuration and result manifests
reports/       Generated evaluation reports and charts
tests/         Unit and leakage/regression tests
```

## Required safeguards

- Prevent look-ahead and survivorship bias in data preparation.
- Keep training and evaluation periods strictly chronological.
- Include fees, spread, slippage, and position sizing in backtests.
- Report drawdown, volatility, Sharpe ratio, turnover, and benchmark returns.
- Never commit API keys, broker credentials, or downloaded private datasets.
- Clearly label simulated results and synthetic/demo data.

## Roadmap

1. Scaffold the data schema and deterministic baseline.
2. Add leakage-tested feature engineering and walk-forward evaluation.
3. Implement a cost-aware backtesting and portfolio simulator.
4. Add dashboards and reproducible experiment reports.
5. Evaluate reinforcement learning against the validated baseline.

## Baseline usage

Use an ISO-date CSV with these columns:

```text
date,open,high,low,close,volume
```

Run the baseline locally:

```sh
python -m venv .venv
.venv/bin/pip install pytest
PYTHONPATH=src .venv/bin/pytest
PYTHONPATH=src .venv/bin/python -m stock_ai_predictor.cli data/example.csv
```

The backtest includes configurable transaction fees and slippage and reports
total return, maximum drawdown, annualized volatility, and trade count.

## Development

Run the regression tests with `PYTHONPATH=src pytest`.
