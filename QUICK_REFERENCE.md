# Quick Reference Card

## 🚀 Quick Start Commands

### First Time Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data for default stocks
python fetch_data.py

# Train your first model
python train.py --symbol AAPL --episodes 100

# Launch dashboard
streamlit run app.py
```

## 📊 Common Commands

### Data Management
```bash
# Fetch specific stock data
python fetch_data.py --symbols AAPL GOOGL MSFT --period 2y

# Fetch data for single stock
python fetch_data.py --symbols TSLA --period 1y
```

### Training Models
```bash
# Quick training (testing)
python train.py --symbol AAPL --episodes 50

# Standard training
python train.py --symbol AAPL --episodes 200

# Long training (better results)
python train.py --symbol AAPL --episodes 500

# Train with custom name
python train.py --symbol GOOGL --episodes 300 --model-name googl_v1
```

### Running Dashboard
```bash
# Standard launch
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Open in browser automatically
streamlit run app.py --server.headless false
```

### Utilities
```bash
# Run demo
python demo.py

# Verify setup
python test_setup.py

# Run example
python examples/basic_usage.py
```

## 📁 Project Structure Quick Reference

```
stock-ai-predictor/
├── 📄 Core Scripts
│   ├── app.py              # Dashboard (streamlit run app.py)
│   ├── train.py            # Training (python train.py)
│   ├── fetch_data.py       # Data fetcher
│   ├── demo.py             # Demonstration
│   └── test_setup.py       # Verification
│
├── 📚 Source Code
│   └── src/
│       ├── config.py       # Configuration
│       ├── data/           # Data modules
│       │   ├── data_fetcher.py
│       │   └── preprocessor.py
│       └── rl_agent/       # RL modules
│           ├── environment.py
│           └── dqn_agent.py
│
├── 💾 Data & Models
│   ├── data/               # Cached stock data
│   ├── models/             # Saved models
│   └── logs/               # Training logs
│
├── 📖 Documentation
│   ├── README.md           # Overview
│   ├── GETTING_STARTED.md  # Detailed guide
│   ├── CONTRIBUTING.md     # Dev guide
│   └── PROJECT_SUMMARY.md  # Summary
│
└── 🔧 Configuration
    ├── requirements.txt    # Dependencies
    ├── .env.example        # Config template
    └── setup.sh            # Auto setup
```

## 🎯 Key Configuration Options

### Environment Variables (.env)
```bash
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
INITIAL_BALANCE=10000
STOCK_SYMBOLS=AAPL,GOOGL,MSFT,AMZN,TSLA
```

### Training Parameters (src/config.py)
```python
LEARNING_RATE = 0.0001      # Neural network learning rate
GAMMA = 0.99                # Discount factor
EPSILON_START = 1.0         # Initial exploration
EPSILON_DECAY = 0.995       # Exploration decay
BATCH_SIZE = 64             # Training batch size
NUM_EPISODES = 1000         # Training episodes
WINDOW_SIZE = 30            # Historical window (days)
```

## 🎨 Dashboard Features

### Main Tabs
1. **Price Chart**: Stock prices with moving averages
2. **Data Table**: Raw historical data
3. **Statistics**: Current metrics (price, volume, RSI)

### Actions
- Select stock from dropdown
- Fetch new data (checkbox)
- Choose trained model
- Set initial balance
- Run simulation

### Results Display
- Total Return %
- Final Portfolio Value
- Sharpe Ratio
- Number of Trades
- Portfolio value chart
- Trading actions table

## 🔍 Performance Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Total Return | Profit/Loss % | > 5% |
| Sharpe Ratio | Risk-adjusted return | > 1.0 |
| Max Drawdown | Worst decline | > -20% |
| Win Rate | % profitable trades | > 50% |

## 🐛 Common Issues & Solutions

### "Module not found"
```bash
pip install -r requirements.txt
```

### "No data available"
```bash
python fetch_data.py --symbols AAPL
```

### "CUDA error"
```bash
# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Training too slow
- Reduce NUM_EPISODES
- Decrease WINDOW_SIZE
- Use GPU if available

### Dashboard won't start
```bash
streamlit run app.py --server.port 8502
```

## 💡 Best Practices

### Training
1. Start with 100-200 episodes
2. Monitor validation returns
3. Save models every 50 episodes
4. Train separate models per stock
5. Use 2 years of historical data

### Data
1. Update data regularly
2. Verify data quality
3. Check for missing values
4. Ensure sufficient history

### Simulation
1. Test on unseen data
2. Compare to buy-and-hold
3. Consider transaction costs
4. Use appropriate position sizing
5. Paper trade before real trading

## 📈 Typical Workflow

```
1. Fetch Data
   ↓
2. Train Model (2-5 times with different parameters)
   ↓
3. Evaluate on Dashboard
   ↓
4. Compare Models
   ↓
5. Select Best Model
   ↓
6. Run Simulation
   ↓
7. Analyze Results
   ↓
8. Iterate or Deploy
```

## 🔗 Quick Links

- Full Guide: [GETTING_STARTED.md](GETTING_STARTED.md)
- Architecture: [README.md](README.md)
- Development: [CONTRIBUTING.md](CONTRIBUTING.md)
- Summary: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)
- Examples: [examples/](examples/)

## ⚡ Performance Tips

### Speed Up Training
- Use GPU: 3-5x faster
- Reduce episodes for testing
- Smaller window size
- Larger batch size (if RAM allows)

### Improve Results
- More training episodes (500+)
- Tune hyperparameters
- Add more technical indicators
- Ensemble multiple models
- Regular retraining with new data

## 🆘 Getting Help

1. Check GETTING_STARTED.md
2. Run demo.py
3. Run test_setup.py
4. Check logs/ directory
5. Review training plots
6. Open an issue on GitHub

## ⚠️ Important Disclaimer

**For educational purposes only!**
- Not financial advice
- Past performance ≠ future results
- Always paper trade first
- Understand the risks
- Consult professionals

---

**Ready to start?** Run: `python fetch_data.py && python train.py --symbol AAPL --episodes 100 && streamlit run app.py` 🚀
