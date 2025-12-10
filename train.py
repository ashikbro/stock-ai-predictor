"""
Training script for DQN agent on stock trading.
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.data.data_fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.rl_agent.environment import StockTradingEnv
from src.rl_agent.dqn_agent import DQNAgent


def train_agent(
    symbol: str = 'AAPL',
    num_episodes: int = None,
    save_frequency: int = 50,
    model_name: str = None
):
    """
    Train DQN agent on stock trading.
    
    Args:
        symbol: Stock symbol to train on
        num_episodes: Number of training episodes
        save_frequency: Save model every N episodes
        model_name: Name for saved model
    """
    if num_episodes is None:
        num_episodes = config.NUM_EPISODES
    
    if model_name is None:
        model_name = f"dqn_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"Training DQN Agent on {symbol}")
    print(f"Episodes: {num_episodes}")
    print("-" * 60)
    
    # Load and preprocess data
    print("Loading data...")
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    
    df = fetcher.load_data(symbol)
    if df.empty:
        print(f"No data found for {symbol}. Fetching...")
        data = fetcher.fetch_historical_data(symbol, period='2y')
        fetcher.save_data({symbol: data})
        df = data
    
    df = preprocessor.prepare_training_data(df)
    print(f"Data shape: {df.shape}")
    
    # Split into train and validation
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Create environment
    env = StockTradingEnv(train_df)
    val_env = StockTradingEnv(val_df)
    
    # Create agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    print(f"Device: {agent.device}")
    print("-" * 60)
    
    # Training metrics
    episode_rewards = []
    episode_returns = []
    val_returns = []
    losses = []
    
    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
        done = False
        while not done:
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            
            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state
        
        # Update target network
        if episode % config.TARGET_UPDATE == 0:
            agent.update_target_network()
        
        # Update epsilon
        agent.update_epsilon()
        
        # Get performance metrics
        metrics = env.get_performance_metrics()
        episode_rewards.append(episode_reward)
        episode_returns.append(metrics['total_return'])
        
        # Only track losses when training actually happened
        if steps > 0 and len(agent.memory) >= agent.batch_size:
            losses.append(episode_loss / steps)
        else:
            losses.append(0.0)
        
        # Validation
        if episode % 10 == 0:
            val_return = evaluate_agent(agent, val_env)
            val_returns.append(val_return)
            
            print(f"Episode {episode}/{num_episodes}")
            print(f"  Train Return: {metrics['total_return']:.4f}")
            print(f"  Val Return: {val_return:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Avg Loss: {episode_loss / steps:.6f}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")
            print(f"  Trades: {metrics['num_trades']}")
        
        # Save model
        if episode % save_frequency == 0 and episode > 0:
            model_path = os.path.join(config.MODEL_DIR, f"{model_name}_ep{episode}.pth")
            agent.save(model_path)
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, f"{model_name}_final.pth")
    agent.save(final_model_path)
    
    # Plot training curves
    plot_training_results(episode_rewards, episode_returns, val_returns, losses, model_name)
    
    print("\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    
    return agent, env


def evaluate_agent(agent: DQNAgent, env: StockTradingEnv) -> float:
    """
    Evaluate agent performance.
    
    Args:
        agent: Trained agent
        env: Evaluation environment
        
    Returns:
        Total return
    """
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, done, info = env.step(action)
    
    metrics = env.get_performance_metrics()
    return metrics['total_return']


def plot_training_results(rewards, returns, val_returns, losses, model_name):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].grid(True)
    
    # Returns
    axes[0, 1].plot(returns, label='Train')
    if val_returns and len(val_returns) > 0:
        val_episodes = np.arange(0, len(returns), len(returns) // len(val_returns))[:len(val_returns)]
        axes[0, 1].plot(val_episodes, val_returns, label='Validation')
    axes[0, 1].set_title('Portfolio Returns')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Losses
    axes[1, 0].plot(losses)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Moving average of returns
    window = 20
    if len(returns) >= window:
        ma_returns = pd.Series(returns).rolling(window=window).mean()
        axes[1, 1].plot(ma_returns)
        axes[1, 1].set_title(f'Moving Average Returns (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('MA Return')
        axes[1, 1].grid(True)
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data for MA', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plot_path = os.path.join(config.LOGS_DIR, f"{model_name}_training.png")
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Training plots saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DQN agent for stock trading')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--episodes', type=int, default=200, help='Number of episodes')
    parser.add_argument('--model-name', type=str, default=None, help='Model name')
    
    args = parser.parse_args()
    
    train_agent(
        symbol=args.symbol,
        num_episodes=args.episodes,
        model_name=args.model_name
    )
