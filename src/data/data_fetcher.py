"""
Data ingestion module using Alpaca API and Yahoo Finance.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


class DataFetcher:
    """Fetches stock data from various sources."""
    
    def __init__(self):
        self.symbols = config.STOCK_SYMBOLS
    
    def fetch_historical_data(
        self, 
        symbol: str, 
        start_date: str = None, 
        end_date: str = None,
        period: str = '2y'
    ) -> pd.DataFrame:
        """
        Fetch historical stock data using yfinance.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Period to fetch if dates not specified (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                df = ticker.history(start=start_date, end=end_date)
            else:
                df = ticker.history(period=period)
            
            df = df.reset_index()
            df['Symbol'] = symbol
            
            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
        period: str = '2y'
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple stocks.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            period: Period to fetch if dates not specified
            
        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        if symbols is None:
            symbols = self.symbols
        
        data = {}
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = self.fetch_historical_data(symbol, start_date, end_date, period)
            if not df.empty:
                data[symbol] = df
        
        return data
    
    def save_data(self, data: Dict[str, pd.DataFrame], directory: str = None):
        """
        Save fetched data to CSV files.
        
        Args:
            data: Dictionary mapping symbols to DataFrames
            directory: Directory to save files (default: config.DATA_DIR)
        """
        if directory is None:
            directory = config.DATA_DIR
        
        os.makedirs(directory, exist_ok=True)
        
        for symbol, df in data.items():
            filepath = os.path.join(directory, f"{symbol}.csv")
            df.to_csv(filepath, index=False)
            print(f"Saved {symbol} data to {filepath}")
    
    def load_data(self, symbol: str, directory: str = None) -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            symbol: Stock ticker symbol
            directory: Directory containing CSV files
            
        Returns:
            DataFrame with stock data
        """
        if directory is None:
            directory = config.DATA_DIR
        
        filepath = os.path.join(directory, f"{symbol}.csv")
        
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            print(f"File not found: {filepath}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    fetcher = DataFetcher()
    
    # Fetch and save historical data
    data = fetcher.fetch_multiple_stocks(period='2y')
    fetcher.save_data(data)
    
    print(f"\nFetched data for {len(data)} symbols")
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows")
