"""
Utility script to fetch and update stock data.
"""
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.data.data_fetcher import DataFetcher
from src import config


def fetch_data(symbols=None, period='2y'):
    """
    Fetch and save stock data.
    
    Args:
        symbols: List of stock symbols (default: from config)
        period: Time period to fetch
    """
    fetcher = DataFetcher()
    
    if symbols is None:
        symbols = config.STOCK_SYMBOLS
    
    print(f"Fetching data for: {', '.join(symbols)}")
    print(f"Period: {period}")
    print("-" * 60)
    
    data = fetcher.fetch_multiple_stocks(symbols, period=period)
    fetcher.save_data(data)
    
    print("-" * 60)
    print(f"Successfully fetched data for {len(data)} symbols")
    
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} rows, "
              f"Date range: {df['Date'].min()} to {df['Date'].max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch stock data')
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Stock symbols to fetch (e.g., AAPL GOOGL MSFT)'
    )
    parser.add_argument(
        '--period',
        type=str,
        default='2y',
        choices=['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        help='Time period to fetch'
    )
    
    args = parser.parse_args()
    
    fetch_data(args.symbols, args.period)
