# 📈 Stock AI Predictor

An advanced RL-based stock market prediction application using Deep Q-Network (DQN) reinforcement learning. Features real-time data ingestion via Alpaca API, historical financial datasets training, and an interactive Streamlit dashboard for predictions and portfolio simulation.

## 🌟 Features

- **Reinforcement Learning**: DQN agent trained on historical stock data
- **Data Ingestion**: Real-time and historical data via Alpaca API and Yahoo Finance
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands, and more
- **Interactive Dashboard**: Streamlit-based UI for visualization and simulation
- **Portfolio Simulation**: Real-time trading simulation with performance metrics
- **Multiple Stocks**: Support for training and trading multiple stock symbols

## 🏗️ Architecture

```
stock-ai-predictor/
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   ├── data/
│   │   ├── data_fetcher.py    # Data ingestion from APIs
│   │   └── preprocessor.py    # Feature engineering and preprocessing
│   └── rl_agent/
│       ├── environment.py     # Gym trading environment
│       └── dqn_agent.py       # DQN implementation
├── app.py                      # Streamlit dashboard
├── train.py                    # Training script
├── requirements.txt            # Python dependencies
├── models/                     # Saved models
├── data/                       # Stock data cache
└── logs/                       # Training logs and plots

```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/ashikbro/stock-ai-predictor.git
cd stock-ai-predictor

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` with your Alpaca API credentials (optional - uses Yahoo Finance by default):

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

INITIAL_BALANCE=10000
STOCK_SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA
```

### 3. Fetch Data

```bash
python -c "from src.data.data_fetcher import DataFetcher; f = DataFetcher(); d = f.fetch_multiple_stocks(period='2y'); f.save_data(d)"
```

### 4. Train the Model

```bash
# Train on AAPL with 200 episodes
python train.py --symbol AAPL --episodes 200

# Train on a different symbol
python train.py --symbol GOOGL --episodes 300 --model-name googl_model
```

Training progress will be displayed and plots saved to `logs/`.

### 5. Run the Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the interactive dashboard.

## 📊 Dashboard Features

### Stock Data Visualization
- **Price Charts**: Real-time price data with moving averages
- **Volume Analysis**: Trading volume visualization
- **Technical Indicators**: RSI, MACD, Bollinger Bands

### AI Trading Simulation
- **Model Selection**: Choose from trained models
- **Portfolio Tracking**: Real-time portfolio value tracking
- **Performance Metrics**: 
  - Total Return
  - Sharpe Ratio
  - Maximum Drawdown
  - Number of Trades
- **Action History**: View all trading decisions (Buy/Hold/Sell)

## 🤖 RL Agent Details

### Environment
- **State Space**: Historical price window + technical indicators + portfolio state
- **Action Space**: Discrete(3) - {Sell, Hold, Buy}
- **Reward**: Portfolio value change percentage

### DQN Architecture
- **Network**: 3-layer MLP with dropout
- **Experience Replay**: 10,000 transitions
- **Target Network**: Updated every 10 episodes
- **Exploration**: Epsilon-greedy with decay

### Training Parameters
- Learning Rate: 0.0001
- Gamma (Discount): 0.99
- Batch Size: 64
- Episodes: 200-1000 (configurable)

## 📈 Performance Metrics

The system tracks multiple metrics:

1. **Total Return**: Overall profit/loss percentage
2. **Sharpe Ratio**: Risk-adjusted returns
3. **Maximum Drawdown**: Largest peak-to-trough decline
4. **Win Rate**: Percentage of profitable trades
5. **Portfolio Value**: Real-time tracking

## 🛠️ Advanced Usage

### Custom Training Configuration

Edit `src/config.py` to customize:
- Learning parameters (learning rate, gamma, epsilon)
- Environment parameters (window size, transaction fees)
- Technical indicators selection

### Multiple Stock Training

```bash
# Train on multiple stocks sequentially
for symbol in AAPL GOOGL MSFT TSLA; do
    python train.py --symbol $symbol --episodes 200
done
```

### Extending the Agent

The modular architecture allows easy extensions:
- Add new technical indicators in `src/data/preprocessor.py`
- Modify reward function in `src/rl_agent/environment.py`
- Implement different RL algorithms (PPO, A3C) in `src/rl_agent/`

## 📦 Dependencies

Core libraries:
- **PyTorch**: Deep learning framework for RL
- **Streamlit**: Interactive dashboard
- **Alpaca-py**: Market data API
- **Gym**: RL environment interface
- **Pandas/NumPy**: Data processing
- **Plotly**: Interactive visualizations

## 🔬 Technical Indicators

Implemented indicators:
- Simple Moving Average (SMA) - 20, 50 day
- Exponential Moving Average (EMA) - 12, 26 day
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators

## 🎯 Roadmap

- [ ] Add more RL algorithms (PPO, SAC, A3C)
- [ ] Multi-asset portfolio optimization
- [ ] Real-time trading integration
- [ ] Backtesting framework
- [ ] Risk management strategies
- [ ] News sentiment analysis integration
- [ ] Docker containerization

## 📝 License

MIT License - feel free to use for learning and commercial purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Do not use it for actual trading without thorough testing and understanding of the risks involved. Past performance does not guarantee future results.
