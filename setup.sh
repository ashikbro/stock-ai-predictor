#!/bin/bash
# Quick start script for Stock AI Predictor

echo "Stock AI Predictor - Quick Start"
echo "================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo "Creating directories..."
mkdir -p models data logs

# Fetch initial data
echo ""
echo "Fetching stock data for AAPL, GOOGL, MSFT..."
python -c "
from src.data.data_fetcher import DataFetcher
fetcher = DataFetcher()
symbols = ['AAPL', 'GOOGL', 'MSFT']
data = fetcher.fetch_multiple_stocks(symbols, period='2y')
fetcher.save_data(data)
print('Data fetched successfully!')
"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train a model: python train.py --symbol AAPL --episodes 100"
echo "2. Launch dashboard: streamlit run app.py"
echo ""
