"""
Stock trading environment using OpenAI Gym interface.
"""
import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config


class StockTradingEnv(gym.Env):
    """
    A stock trading environment for reinforcement learning.
    
    Action Space: Discrete(3)
        0: Sell
        1: Hold
        2: Buy
    
    State Space: Box
        - Current price normalized
        - Portfolio holdings
        - Technical indicators
        - Historical price window
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = None,
        transaction_fee: float = None,
        window_size: int = None
    ):
        """
        Initialize the trading environment.
        
        Args:
            df: DataFrame with preprocessed stock data
            initial_balance: Starting cash balance
            transaction_fee: Transaction fee percentage
            window_size: Number of historical steps to include in state
        """
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance or config.INITIAL_BALANCE
        self.transaction_fee = transaction_fee or config.TRANSACTION_FEE
        self.window_size = window_size or config.WINDOW_SIZE
        
        # Feature columns for state representation
        self.feature_columns = [
            'Close', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 
            'MACD', 'Volume_Change', 'Price_Change'
        ]
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Action space: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)
        
        # State space: window of features + portfolio state
        n_features = len(self.feature_columns)
        state_size = n_features * self.window_size + 3  # +3 for cash, holdings, portfolio_value
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_asset_value = self.initial_balance
        self.portfolio_values = []
        self.trades = []
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_asset_value = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        # Get historical window
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        window_data = self.df[self.feature_columns].iloc[start_idx:end_idx].values
        
        # Normalize window data
        window_flat = window_data.flatten()
        
        # Current portfolio state
        current_price = self.df.loc[self.current_step, 'Close']
        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.holdings * current_price / self.initial_balance,
            self.total_asset_value / self.initial_balance
        ])
        
        # Combine into single state vector
        state = np.concatenate([window_flat, portfolio_state])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0=Sell, 1=Hold, 2=Buy)
            
        Returns:
            observation, reward, done, info
        """
        current_price = self.df.loc[self.current_step, 'Close']
        prev_portfolio_value = self.total_asset_value
        
        # Execute action
        if action == 2:  # Buy
            # Buy as much as possible
            max_shares = int(self.balance / (current_price * (1 + self.transaction_fee)))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_fee)
                self.balance -= cost
                self.holdings += max_shares
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'shares': max_shares,
                    'price': current_price
                })
        
        elif action == 0:  # Sell
            # Sell all holdings
            if self.holdings > 0:
                revenue = self.holdings * current_price * (1 - self.transaction_fee)
                self.balance += revenue
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'shares': self.holdings,
                    'price': current_price
                })
                self.holdings = 0
        
        # Update portfolio value
        self.total_asset_value = self.balance + self.holdings * current_price
        self.portfolio_values.append(self.total_asset_value)
        
        # Calculate reward (change in portfolio value)
        reward = (self.total_asset_value - prev_portfolio_value) / prev_portfolio_value
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Get next observation (or last observation if done)
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.total_asset_value,
            'balance': self.balance,
            'holdings': self.holdings,
            'total_return': (self.total_asset_value - self.initial_balance) / self.initial_balance
        }
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            current_price = self.df.loc[self.current_step, 'Close']
            print(f"Step: {self.current_step}")
            print(f"Price: ${current_price:.2f}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Holdings: {self.holdings} shares")
            print(f"Portfolio Value: ${self.total_asset_value:.2f}")
            print(f"Total Return: {((self.total_asset_value - self.initial_balance) / self.initial_balance * 100):.2f}%")
            print("-" * 50)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        portfolio_returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        total_return = (self.total_asset_value - self.initial_balance) / self.initial_balance
        
        if len(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-9) * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'final_value': self.total_asset_value
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        portfolio_values = np.array(self.portfolio_values)
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax
        return np.min(drawdown)
