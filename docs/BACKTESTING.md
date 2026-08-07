# Backtesting Methodology

## Core Method

1. Split data into sequential train/test windows.
2. Train policy on historical segment only.
3. Simulate trades on forward segment without leakage.
4. Apply transaction cost and slippage assumptions.
5. Aggregate trade log and risk metrics.

## Practical Assumptions

- Fixed cost in basis points per trade
- No look-ahead features
- Daily close execution in baseline mode
- Position sizing cap to bound risk

## Validation Guidance

- Repeat with different random seeds.
- Stress test on volatile intervals.
- Evaluate sensitivity to costs and spread.
