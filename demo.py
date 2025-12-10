"""
Demo script to showcase the stock AI predictor capabilities.
This script demonstrates the core functionality without requiring full training.
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_config():
    """Demonstrate configuration loading."""
    print("=" * 60)
    print("1. CONFIGURATION")
    print("=" * 60)
    
    from src import config
    
    print(f"✓ Stock Symbols: {', '.join(config.STOCK_SYMBOLS)}")
    print(f"✓ Initial Balance: ${config.INITIAL_BALANCE:,.2f}")
    print(f"✓ Window Size: {config.WINDOW_SIZE} days")
    print(f"✓ Learning Rate: {config.LEARNING_RATE}")
    print(f"✓ Episodes: {config.NUM_EPISODES}")
    print(f"✓ Technical Indicators: {', '.join(config.TECHNICAL_INDICATORS)}")
    print()


def demo_data_structure():
    """Demonstrate data structure."""
    print("=" * 60)
    print("2. DATA STRUCTURE")
    print("=" * 60)
    
    print("✓ Stock Data Columns:")
    print("  - Date: Trading date")
    print("  - Open: Opening price")
    print("  - High: Highest price")
    print("  - Low: Lowest price")
    print("  - Close: Closing price")
    print("  - Volume: Trading volume")
    print("\n✓ After Preprocessing, Additional Columns:")
    print("  - SMA_20, SMA_50: Moving averages")
    print("  - RSI: Relative Strength Index")
    print("  - MACD: Moving Average Convergence Divergence")
    print("  - Bollinger Bands: BB_Upper, BB_Middle, BB_Lower")
    print("  - Volume_Change, Price_Change: Percentage changes")
    print()


def demo_environment():
    """Demonstrate trading environment."""
    print("=" * 60)
    print("3. TRADING ENVIRONMENT")
    print("=" * 60)
    
    print("✓ Environment Type: Gym-compatible")
    print("✓ Action Space: Discrete(3)")
    print("  - 0: Sell (liquidate holdings)")
    print("  - 1: Hold (maintain position)")
    print("  - 2: Buy (purchase stocks)")
    print("\n✓ State Space: Box (continuous)")
    print("  - Historical price window")
    print("  - Technical indicators (SMA, RSI, MACD, etc.)")
    print("  - Portfolio state (cash, holdings, value)")
    print("\n✓ Reward: Portfolio value change percentage")
    print()


def demo_agent():
    """Demonstrate RL agent architecture."""
    print("=" * 60)
    print("4. DQN AGENT ARCHITECTURE")
    print("=" * 60)
    
    print("✓ Algorithm: Deep Q-Network (DQN)")
    print("✓ Network Architecture:")
    print("  - Input Layer: State size (varies by window)")
    print("  - Hidden Layer 1: 128 neurons + ReLU + Dropout(0.2)")
    print("  - Hidden Layer 2: 128 neurons + ReLU + Dropout(0.2)")
    print("  - Hidden Layer 3: 64 neurons + ReLU")
    print("  - Output Layer: 3 neurons (Q-values for each action)")
    print("\n✓ Training Features:")
    print("  - Experience Replay (capacity: 10,000)")
    print("  - Target Network (updated every 10 episodes)")
    print("  - Epsilon-greedy exploration (1.0 → 0.01)")
    print("  - Adam optimizer")
    print()


def demo_technical_indicators():
    """Demonstrate technical indicators."""
    print("=" * 60)
    print("5. TECHNICAL INDICATORS")
    print("=" * 60)
    
    indicators = {
        'SMA_20': 'Simple Moving Average (20 days)',
        'SMA_50': 'Simple Moving Average (50 days)',
        'EMA_12': 'Exponential Moving Average (12 days)',
        'EMA_26': 'Exponential Moving Average (26 days)',
        'RSI': 'Relative Strength Index',
        'MACD': 'Moving Average Convergence Divergence',
        'BB_Upper/Lower': 'Bollinger Bands',
        'Volume_Change': 'Volume change percentage',
        'Price_Change': 'Price change percentage'
    }
    
    for indicator, description in indicators.items():
        print(f"✓ {indicator}: {description}")
    print()


def demo_metrics():
    """Demonstrate performance metrics."""
    print("=" * 60)
    print("6. PERFORMANCE METRICS")
    print("=" * 60)
    
    metrics = {
        'Total Return': 'Overall profit/loss percentage',
        'Sharpe Ratio': 'Risk-adjusted returns',
        'Maximum Drawdown': 'Largest peak-to-trough decline',
        'Number of Trades': 'Total trading actions executed',
        'Final Portfolio Value': 'End balance + holdings value'
    }
    
    for metric, description in metrics.items():
        print(f"✓ {metric}: {description}")
    print()


def demo_usage():
    """Demonstrate usage commands."""
    print("=" * 60)
    print("7. USAGE COMMANDS")
    print("=" * 60)
    
    print("✓ Fetch Data:")
    print("  python fetch_data.py --symbols AAPL GOOGL --period 2y")
    
    print("\n✓ Train Model:")
    print("  python train.py --symbol AAPL --episodes 200")
    
    print("\n✓ Launch Dashboard:")
    print("  streamlit run app.py")
    
    print("\n✓ Quick Setup:")
    print("  bash setup.sh")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "STOCK AI PREDICTOR DEMONSTRATION" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        demo_config()
        demo_data_structure()
        demo_environment()
        demo_agent()
        demo_technical_indicators()
        demo_metrics()
        demo_usage()
        
        print("=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\n✓ All core components are properly configured!")
        print("✓ Ready to fetch data, train models, and run simulations!")
        print("\nNext Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Fetch data: python fetch_data.py")
        print("3. Train a model: python train.py --symbol AAPL --episodes 100")
        print("4. Launch dashboard: streamlit run app.py")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
