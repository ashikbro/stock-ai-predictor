"""
Basic usage example for Stock AI Predictor.

This script demonstrates:
1. Fetching stock data
2. Preprocessing data
3. Creating trading environment
4. Training a simple agent
5. Evaluating performance
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data.data_fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.rl_agent.environment import StockTradingEnv
from src.rl_agent.dqn_agent import DQNAgent


def main():
    """Run basic example."""
    print("=" * 60)
    print("STOCK AI PREDICTOR - BASIC EXAMPLE")
    print("=" * 60)
    print()
    
    # Configuration
    SYMBOL = 'AAPL'
    EPISODES = 50  # Small number for quick demo
    
    print(f"Training on: {SYMBOL}")
    print(f"Episodes: {EPISODES}")
    print()
    
    # Step 1: Fetch Data
    print("Step 1: Fetching stock data...")
    print("-" * 60)
    
    fetcher = DataFetcher()
    df = fetcher.load_data(SYMBOL)
    
    if df.empty:
        print(f"No cached data found. Fetching {SYMBOL}...")
        df = fetcher.fetch_historical_data(SYMBOL, period='1y')
        fetcher.save_data({SYMBOL: df})
    
    print(f"✓ Loaded {len(df)} days of data")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()
    
    # Step 2: Preprocess Data
    print("Step 2: Preprocessing data...")
    print("-" * 60)
    
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.prepare_training_data(df)
    
    print(f"✓ Added technical indicators")
    print(f"  Columns: {len(df_processed.columns)}")
    print(f"  Features: {', '.join(df_processed.columns[:10].tolist())}...")
    print()
    
    # Step 3: Create Environment
    print("Step 3: Creating trading environment...")
    print("-" * 60)
    
    env = StockTradingEnv(df_processed)
    
    print(f"✓ Environment created")
    print(f"  State size: {env.observation_space.shape[0]}")
    print(f"  Action space: {env.action_space.n} (Buy/Hold/Sell)")
    print(f"  Initial balance: ${env.initial_balance:,.2f}")
    print()
    
    # Step 4: Create and Train Agent
    print("Step 4: Training DQN agent...")
    print("-" * 60)
    
    state_size = env.observation_space.shape[0]
    agent = DQNAgent(state_size, action_size=3)
    
    print(f"✓ Agent initialized")
    print(f"  Device: {agent.device}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}")
    print()
    
    # Training loop
    best_return = float('-inf')
    
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Train
            agent.train_step()
            
            episode_reward += reward
            state = next_state
        
        # Update networks
        if episode % 10 == 0:
            agent.update_target_network()
        
        agent.update_epsilon()
        
        # Get metrics
        metrics = env.get_performance_metrics()
        
        # Track best model
        if metrics['total_return'] > best_return:
            best_return = metrics['total_return']
        
        # Print progress
        if episode % 10 == 0 or episode == EPISODES - 1:
            print(f"Episode {episode:3d}/{EPISODES}: "
                  f"Return={metrics['total_return']:+.2%}, "
                  f"Sharpe={metrics['sharpe_ratio']:.2f}, "
                  f"Trades={metrics['num_trades']}, "
                  f"ε={agent.epsilon:.3f}")
    
    print()
    print("✓ Training completed!")
    print()
    
    # Step 5: Final Evaluation
    print("Step 5: Evaluating trained agent...")
    print("-" * 60)
    
    # Reset environment for final evaluation
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, done, info = env.step(action)
    
    final_metrics = env.get_performance_metrics()
    
    print("Final Performance:")
    print(f"  Total Return: {final_metrics['total_return']:+.2%}")
    print(f"  Final Value: ${final_metrics['final_value']:,.2f}")
    print(f"  Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {final_metrics['max_drawdown']:.2%}")
    print(f"  Number of Trades: {final_metrics['num_trades']}")
    print()
    
    # Compare to buy-and-hold
    initial_price = df_processed.iloc[config.WINDOW_SIZE]['Close']
    final_price = df_processed.iloc[-1]['Close']
    buy_hold_return = (final_price - initial_price) / initial_price
    
    print("Comparison to Buy-and-Hold:")
    print(f"  Buy-and-Hold Return: {buy_hold_return:+.2%}")
    print(f"  RL Agent Return: {final_metrics['total_return']:+.2%}")
    print(f"  Difference: {(final_metrics['total_return'] - buy_hold_return):+.2%}")
    print()
    
    # Save model
    model_path = os.path.join(config.MODEL_DIR, f"example_{SYMBOL}.pth")
    agent.save(model_path)
    
    print("=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Try with more episodes: python examples/basic_usage.py")
    print("2. Train on different stocks")
    print("3. Launch dashboard: streamlit run app.py")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
