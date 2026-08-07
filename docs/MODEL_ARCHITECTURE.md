# Model Architecture

## RL-Centric Design

The baseline implementation uses a DQN-style policy with a compact state vector:

- rolling returns window,
- short/long moving averages,
- normalized momentum,
- current position indicator.

The action space is discrete:
- `0`: hold
- `1`: buy
- `2`: sell

Reward function includes portfolio return minus trading costs and mild risk penalty.

## Multi-Model Strategy

The project supports hybrid experimentation:
- RL policy for decision making,
- baseline forecasters for directional priors,
- and ensemble-style gating in future roadmap iterations.

## Safety Constraints

- Position bounds (`max_position`) to avoid unlimited leverage assumptions.
- Transaction-cost modeling in every step.
- Deterministic simulation mode for reproducible experiments.
