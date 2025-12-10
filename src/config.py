"""
Configuration module for stock AI predictor.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alpaca API Configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Trading Configuration
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '10000'))
STOCK_SYMBOLS = os.getenv('STOCK_SYMBOLS', 'AAPL,GOOGL,MSFT').split(',')

# RL Training Parameters
LEARNING_RATE = 0.0001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE = 10
NUM_EPISODES = 1000

# Environment Parameters
MAX_STEPS = 252  # Trading days in a year
TRANSACTION_FEE = 0.001
WINDOW_SIZE = 30  # Days of historical data to consider

# Model Paths
MODEL_DIR = 'models'
DATA_DIR = 'data'
LOGS_DIR = 'logs'

# Feature Engineering
TECHNICAL_INDICATORS = ['SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volume_Change']
