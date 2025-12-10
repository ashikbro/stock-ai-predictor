# Getting Started Guide

## Overview

The Stock AI Predictor uses Deep Q-Network (DQN) reinforcement learning to learn optimal trading strategies from historical stock data. This guide will walk you through the entire process from setup to running predictions.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Understanding the Components](#understanding-the-components)
4. [Training Your First Model](#training-your-first-model)
5. [Using the Dashboard](#using-the-dashboard)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU optional (faster training)

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ashikbro/stock-ai-predictor.git
cd stock-ai-predictor

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Run verification tests
python test_setup.py

# Run demo
python demo.py
```

You should see all tests pass except for module imports (which require full dependency installation).

## Quick Start

### Option 1: Automated Setup

```bash
bash setup.sh
```

This script will:
- Create a virtual environment
- Install all dependencies
- Fetch initial stock data
- Set up directory structure

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file (optional - for Alpaca API)
cp .env.example .env
# Edit .env with your credentials

# 3. Fetch stock data
python fetch_data.py --symbols AAPL GOOGL MSFT --period 2y

# 4. Train a model
python train.py --symbol AAPL --episodes 100

# 5. Launch dashboard
streamlit run app.py
```

## Understanding the Components

### 1. Data Layer (`src/data/`)

**data_fetcher.py**: Fetches historical stock data
- Uses Yahoo Finance API by default
- Supports Alpaca API (configure in .env)
- Caches data locally in `data/` directory

**preprocessor.py**: Processes raw data
- Adds technical indicators (SMA, RSI, MACD, etc.)
- Normalizes features
- Creates time series sequences

### 2. RL Agent (`src/rl_agent/`)

**environment.py**: Trading environment
- OpenAI Gym-compatible interface
- Actions: Buy, Hold, Sell
- Rewards based on portfolio value changes
- Tracks cash, holdings, and performance

**dqn_agent.py**: DQN implementation
- Neural network with 3 hidden layers
- Experience replay buffer (10,000 transitions)
- Target network for stability
- Epsilon-greedy exploration

### 3. User Interface

**app.py**: Streamlit dashboard
- Interactive stock charts
- Real-time simulation
- Performance metrics visualization
- Model selection and configuration

**train.py**: Training script
- Command-line interface
- Progress tracking
- Validation during training
- Model checkpointing

## Training Your First Model

### Basic Training

```bash
python train.py --symbol AAPL --episodes 200
```

This will:
1. Load AAPL historical data (or fetch if missing)
2. Preprocess and add technical indicators
3. Split into train/validation sets (80/20)
4. Train DQN agent for 200 episodes
5. Save model to `models/` directory
6. Generate training plots in `logs/`

### Understanding Training Output

```
Episode 10/200
  Train Return: 0.0523     # 5.23% profit on training data
  Val Return: 0.0341       # 3.41% profit on validation data
  Epsilon: 0.9045          # Exploration rate (decreases over time)
  Avg Loss: 0.002341       # Neural network training loss
  Sharpe: 1.23             # Risk-adjusted return metric
  Trades: 45               # Number of buy/sell actions
```

### Training Tips

1. **Start Small**: Begin with 100-200 episodes to test
2. **Monitor Validation**: Watch val_return to avoid overfitting
3. **Patience**: Training can take 10-30 minutes per 100 episodes
4. **Multiple Stocks**: Train separate models for each stock

### Advanced Training

```bash
# Train with custom episodes
python train.py --symbol GOOGL --episodes 500

# Train with custom model name
python train.py --symbol MSFT --episodes 300 --model-name msft_long_term
```

## Using the Dashboard

### Launch Dashboard

```bash
streamlit run app.py
```

Opens browser at `http://localhost:8501`

### Dashboard Features

#### 1. Stock Selection
- Choose from configured symbols (AAPL, GOOGL, MSFT)
- View real-time price data
- See technical indicators

#### 2. Data Visualization
- **Price Chart**: Candlesticks with moving averages
- **Volume**: Trading volume over time
- **RSI**: Relative Strength Index
- **Data Table**: Raw data view

#### 3. Model Selection
- List of trained models
- Select model for simulation
- Configure initial balance

#### 4. Run Simulation
- Click "Run Simulation" button
- Agent makes trading decisions
- View results in real-time

#### 5. Performance Metrics
- **Total Return**: Overall profit/loss %
- **Sharpe Ratio**: Risk-adjusted performance
- **Max Drawdown**: Worst peak-to-trough decline
- **Number of Trades**: Total actions taken

#### 6. Portfolio Visualization
- Portfolio value over time
- Cash vs holdings breakdown
- Trading action timeline

## Advanced Configuration

### Edit Configuration (`src/config.py`)

```python
# Trading Parameters
INITIAL_BALANCE = 10000        # Starting capital
TRANSACTION_FEE = 0.001        # 0.1% per trade
WINDOW_SIZE = 30               # Days of history to consider

# RL Training Parameters
LEARNING_RATE = 0.0001         # Neural network learning rate
GAMMA = 0.99                   # Discount factor
EPSILON_START = 1.0            # Initial exploration
EPSILON_END = 0.01             # Minimum exploration
EPSILON_DECAY = 0.995          # Decay rate per episode
BATCH_SIZE = 64                # Training batch size
MEMORY_SIZE = 10000            # Replay buffer size
NUM_EPISODES = 1000            # Default training episodes
```

### Environment Variables (`.env`)

```bash
# Alpaca API (optional)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
INITIAL_BALANCE=10000
STOCK_SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA
```

### Custom Technical Indicators

Edit `src/data/preprocessor.py` to add indicators:

```python
def add_technical_indicators(self, df):
    # Add your custom indicators here
    df['Custom_Indicator'] = ...
    return df
```

## Troubleshooting

### Common Issues

#### 1. Module Not Found Errors

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 2. No Data Available

```bash
# Solution: Fetch data
python fetch_data.py --symbols AAPL --period 2y
```

#### 3. CUDA/GPU Errors

```bash
# Solution: Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

#### 4. Streamlit Connection Error

```bash
# Solution: Specify port
streamlit run app.py --server.port 8502
```

#### 5. Training Too Slow

- Reduce `NUM_EPISODES` in config.py
- Use GPU if available
- Reduce `WINDOW_SIZE` for faster training

### Performance Tips

1. **GPU Acceleration**: Install CUDA-enabled PyTorch for faster training
2. **Batch Size**: Increase if you have more RAM
3. **Episodes**: More episodes = better learning but slower
4. **Window Size**: Smaller window = faster but less context

### Getting Help

1. Check the [README.md](README.md) for overview
2. Run `python demo.py` to see configuration
3. Run `python test_setup.py` to verify setup
4. Check logs in `logs/` directory
5. Review training plots for insights

## Next Steps

1. **Experiment**: Try different stocks and parameters
2. **Analyze**: Study the training plots in `logs/`
3. **Compare**: Train multiple models and compare performance
4. **Optimize**: Tune hyperparameters for better results
5. **Extend**: Add new features or indicators

## Best Practices

1. **Data Quality**: Always fetch recent data before training
2. **Validation**: Monitor validation returns during training
3. **Patience**: Good models need 200-500 episodes
4. **Diversification**: Train on multiple stocks
5. **Paper Trading**: Test thoroughly before real money
6. **Risk Management**: Use appropriate position sizing
7. **Regular Updates**: Retrain models with new data

## Important Disclaimer

⚠️ **This is for educational purposes only!**

- Past performance doesn't guarantee future results
- Always paper trade before using real money
- Understand the risks involved in trading
- Consider transaction costs and taxes
- Consult financial advisors for investment advice

---

Ready to start? Run:

```bash
python fetch_data.py
python train.py --symbol AAPL --episodes 100
streamlit run app.py
```

Happy trading! 📈
