# API Sources

## Primary Sources

- **Yahoo Finance (yfinance)**
  - Good free baseline for OHLCV prototyping
- **Alpha Vantage**
  - Time series + technical indicators
- **Twelve Data**
  - Flexible endpoints and interval coverage

## Data Selection Recommendations

- Prefer adjusted prices when backtesting equities.
- Store raw responses and processed snapshots separately.
- Record timezone and endpoint metadata for reproducibility.

## Credential Management

Keep all API keys in `.env` (never commit secrets). Use `.env.example` as a template for local configuration.
