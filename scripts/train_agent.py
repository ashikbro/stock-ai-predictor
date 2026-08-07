"""Train a tabular Q-learning trading baseline on engineered features."""

from __future__ import annotations

import argparse

from stock_ai_predictor.data_pipeline import build_features, load_price_data
from stock_ai_predictor.rl_agent import QLearningTradingAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL trading baseline")
    parser.add_argument("--csv-path", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--episodes", type=int, default=10, help="Episode count")
    return parser.parse_args()


def discretize_state(momentum: float, threshold: float = 0.0) -> int:
    """Map momentum feature to a simple discrete state space."""
    return 1 if momentum > threshold else 0


def main() -> None:
    args = parse_args()
    frame = load_price_data(args.csv_path)
    features = build_features(frame)

    agent = QLearningTradingAgent()
    close = features["close"].to_numpy()
    momentum = features["momentum"].to_numpy()

    for _ in range(args.episodes):
        for i in range(len(features) - 1):
            state = discretize_state(float(momentum[i]))
            next_state = discretize_state(float(momentum[i + 1]))
            action = agent.select_action(state)

            # Reward approximates one-step return with directional alignment.
            price_return = (close[i + 1] - close[i]) / close[i]
            reward = float(price_return if action == 1 else -price_return if action == 2 else 0.0)

            agent.update(state, action, reward, next_state)

        agent.decay_epsilon()

    print(f"Training complete. Learned states: {len(agent.q_table)}")
    print(f"Final epsilon: {agent.epsilon:.4f}")


if __name__ == "__main__":
    main()
