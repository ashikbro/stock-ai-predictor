# Training Guide

## Step-by-Step

1. Prepare dataset with OHLCV columns.
2. Configure environment variables (`.env`).
3. Run baseline training script:

```bash
python scripts/train_agent.py --csv-path data/sample_prices.csv --episodes 10
```

4. Track episode rewards and evaluation metrics.
5. Save checkpoints and compare against benchmark.

## RL Algorithm Notes

The baseline agent follows epsilon-greedy policy improvement:
- Start with high exploration (`epsilon_start`).
- Decay epsilon each episode until `epsilon_end`.
- Update Q-value table with temporal-difference target.

This baseline is intentionally lightweight for reproducibility and can be replaced by neural DQN/PPO agents.
