# Data Pipeline

## Data Collection

Market bars can be ingested from:
- CSV exports,
- yfinance downloads,
- premium feeds (Alpha Vantage/Twelve Data).

## Processing Steps

1. Parse timestamps and sort ascending.
2. Validate required OHLCV columns.
3. Forward-fill sparse gaps conservatively.
4. Compute engineered features (returns, moving averages, momentum).
5. Normalize feature columns for stable RL training.

## Data Quality Checks

- Missing-value thresholds
- Duplicate timestamp detection
- Outlier sanity checks for price jumps
- Symbol/timeframe metadata capture
