# Stock AI Predictor

Stock AI Predictor is a reinforcement-learning-first research project for forecasting market behavior and simulating data-driven trading decisions.

## Problem Statement

Stock prediction is difficult because markets are non-stationary, noisy, and influenced by macro events, sentiment shifts, and liquidity dynamics. Any realistic prediction system must handle uncertainty, regime changes, and risk-aware decision making rather than relying on single-point forecasts.

## Solution Overview

This project combines market data engineering, reinforcement learning (RL), baseline forecasters, and portfolio simulation to support:

- probabilistic price movement prediction,
- policy-driven trading decisions,
- repeatable backtesting with realistic constraints,
- and transparent metric/visual reporting.

## Key Features

- RL-based price prediction workflow
- Reinforcement learning trading agent (DQN-style baseline)
- Financial dataset integration and feature engineering
- Portfolio simulation with risk controls
- Performance visualization utilities
- Multiple prediction model interfaces (RL + baseline)
- Backtesting methodology and evaluation templates

## Tech Stack

- **ML/RL**: Python, NumPy, pandas, scikit-learn, TensorFlow (optional), PyTorch (optional)
- **Data**: yfinance, Alpha Vantage (via API key), Twelve Data (optional)
- **Visualization**: matplotlib, seaborn
- **Experimentation**: Jupyter notebooks

## Repository Layout

```text
src/stock_ai_predictor/   # Typed core modules (data, RL, portfolio, metrics)
scripts/                  # Standalone train/predict scripts
examples/                 # Market scenario demos and visualization examples
docs/                     # Technical documentation
data/                     # Sample datasets for local testing
notebooks/                # Tutorial notebooks
```

## Installation & Setup

### Option A: pip

```bash
git clone https://github.com/ashikbro/stock-ai-predictor.git
cd stock-ai-predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Option B: Conda

```bash
conda env create -f environment.yml
conda activate stock-ai-predictor
```

### Environment Variables

Copy and update environment settings:

```bash
cp .env.example .env
```

## Quick Start

### 1) Run Prediction Script

```bash
PYTHONPATH=src python scripts/predict.py \
  --csv-path data/sample_prices.csv \
  --window 5
```

### 2) Run Training Script

```bash
PYTHONPATH=src python scripts/train_agent.py \
  --csv-path data/sample_prices.csv \
  --episodes 10
```

### 3) Run Scenario Examples

```bash
PYTHONPATH=src python examples/bull_market_scenario.py
PYTHONPATH=src python examples/bear_market_scenario.py
PYTHONPATH=src python examples/portfolio_simulation_example.py
PYTHONPATH=src python examples/visualization_example.py
```

## Model Architecture & RL Algorithm

- Detailed architecture: [`docs/MODEL_ARCHITECTURE.md`](docs/MODEL_ARCHITECTURE.md)
- RL algorithm details and training loop behavior: [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)

## Data Sources & Collection

- Data providers and API setup: [`docs/API_SOURCES.md`](docs/API_SOURCES.md)
- Pipeline details and feature processing: [`docs/DATA_PIPELINE.md`](docs/DATA_PIPELINE.md)

## Usage Guides

- Training workflow: [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)
- Portfolio simulation: [`docs/BACKTESTING.md`](docs/BACKTESTING.md)
- Metrics and benchmarks: [`docs/RESULTS.md`](docs/RESULTS.md)

## Performance Metrics Explained

Common metrics used in this project:

- Regression: MAE, RMSE, MAPE
- Trading: cumulative return, volatility, Sharpe ratio, max drawdown, win rate
- Policy diagnostics: average episode reward, action distribution, turnover

## Backtesting Methodology

Backtesting is performed with:

1. rolling train/validation/test windows,
2. transaction cost and slippage assumptions,
3. position constraints,
4. benchmark comparison (buy-and-hold),
5. and regime-specific stress testing.

See full methodology in [`docs/BACKTESTING.md`](docs/BACKTESTING.md).

## Results and Benchmarks

Reference benchmark templates and example outputs are documented in [`docs/RESULTS.md`](docs/RESULTS.md).

## Financial Data Requirements

- OHLCV bars (daily or intraday)
- Corporate-action adjusted prices
- Optional fundamentals and macro indicators
- UTC-normalized timestamp index

## Risk Disclaimer

This project is for research and educational use only. It does **not** provide financial advice. Trading involves substantial risk, including potential loss of principal. Validate all assumptions before any live deployment.

## Roadmap

- [ ] Add PPO/SAC policy implementations
- [ ] Add walk-forward retraining orchestration
- [ ] Integrate probabilistic forecasting head
- [ ] Add multi-asset portfolio optimizer
- [ ] Add live paper-trading connector

## Contributing

Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for contribution workflow, coding standards, and PR expectations.
