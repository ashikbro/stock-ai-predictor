"""
Data preprocessing and feature engineering module.
"""
import pandas as pd
import numpy as np
from typing import List
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


class DataPreprocessor:
    """Preprocesses stock data and adds technical indicators."""
    
    def __init__(self):
        pass
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        
        # Drop NaN values from calculations
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Series of prices
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize specified columns using min-max scaling.
        
        Args:
            df: DataFrame to normalize
            columns: List of columns to normalize (None for all numeric)
            
        Returns:
            DataFrame with normalized columns
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        window_size: int = None,
        features: List[str] = None
    ) -> np.ndarray:
        """
        Create sequences for time series prediction.
        
        Args:
            df: DataFrame with features
            window_size: Number of time steps to include
            features: List of feature columns to use
            
        Returns:
            3D numpy array of shape (samples, window_size, features)
        """
        if window_size is None:
            window_size = config.WINDOW_SIZE
        
        if features is None:
            features = ['Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Volume_Change']
        
        # Ensure all features exist
        features = [f for f in features if f in df.columns]
        
        data = df[features].values
        sequences = []
        
        for i in range(len(data) - window_size):
            sequences.append(data[i:i + window_size])
        
        return np.array(sequences)
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete preprocessing pipeline for training data.
        
        Args:
            df: Raw DataFrame with OHLCV data
            
        Returns:
            Preprocessed DataFrame ready for training
        """
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Sort by date
        if 'Date' in df.columns:
            df = df.sort_values('Date').reset_index(drop=True)
        
        return df


if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    # Example usage
    fetcher = DataFetcher()
    preprocessor = DataPreprocessor()
    
    # Load data
    symbol = 'AAPL'
    df = fetcher.load_data(symbol)
    
    if not df.empty:
        # Preprocess
        df_processed = preprocessor.prepare_training_data(df)
        print(f"Processed data shape: {df_processed.shape}")
        print(f"Columns: {df_processed.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df_processed.head())
