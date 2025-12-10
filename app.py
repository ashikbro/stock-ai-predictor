"""
Streamlit dashboard for stock prediction and portfolio simulation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src import config
from src.data.data_fetcher import DataFetcher
from src.data.preprocessor import DataPreprocessor
from src.rl_agent.environment import StockTradingEnv
from src.rl_agent.dqn_agent import DQNAgent


# Page configuration
st.set_page_config(
    page_title="Stock AI Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_model(model_path: str, state_size: int) -> DQNAgent:
    """Load trained DQN model."""
    agent = DQNAgent(state_size, action_size=3)
    if os.path.exists(model_path):
        agent.load(model_path)
        return agent
    return None


def run_simulation(agent: DQNAgent, env: StockTradingEnv):
    """Run trading simulation with trained agent."""
    state = env.reset()
    done = False
    
    portfolio_history = []
    actions_history = []
    
    while not done:
        action = agent.select_action(state, evaluate=True)
        state, reward, done, info = env.step(action)
        
        portfolio_history.append({
            'step': env.current_step,
            'portfolio_value': info['portfolio_value'],
            'balance': info['balance'],
            'holdings': info['holdings']
        })
        
        action_name = ['Sell', 'Hold', 'Buy'][action]
        actions_history.append({
            'step': env.current_step,
            'action': action_name
        })
    
    return portfolio_history, actions_history, env.get_performance_metrics()


def plot_stock_data(df: pd.DataFrame, symbol: str):
    """Plot stock price and technical indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{symbol} Price', 'Volume', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and moving averages
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color='blue')),
        row=1, col=1
    )
    
    if 'SMA_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange', dash='dash')),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], name='SMA 50', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    
    return fig


def plot_portfolio_performance(portfolio_history: list, df: pd.DataFrame):
    """Plot portfolio performance over time."""
    portfolio_df = pd.DataFrame(portfolio_history)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Portfolio Value', 'Cash vs Holdings Value'),
        row_heights=[0.6, 0.4]
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df['step'],
            y=portfolio_df['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='green', width=2)
        ),
        row=1, col=1
    )
    
    # Initial balance line
    initial_balance = portfolio_df['portfolio_value'].iloc[0]
    fig.add_hline(
        y=initial_balance,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Balance",
        row=1, col=1
    )
    
    # Cash vs Holdings
    fig.add_trace(
        go.Scatter(x=portfolio_df['step'], y=portfolio_df['balance'], name='Cash', fill='tonexty'),
        row=2, col=1
    )
    
    holdings_value = portfolio_df['holdings'] * df.iloc[portfolio_df['step']]['Close'].values
    fig.add_trace(
        go.Scatter(x=portfolio_df['step'], y=holdings_value, name='Holdings Value', fill='tonexty'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Trading Step", row=2, col=1)
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig


def main():
    st.title("📈 Stock AI Predictor")
    st.markdown("### RL-based Stock Market Prediction and Portfolio Simulation")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    available_symbols = config.STOCK_SYMBOLS
    selected_symbol = st.sidebar.selectbox("Select Stock", available_symbols)
    
    # Data fetching options
    st.sidebar.subheader("Data Options")
    fetch_new_data = st.sidebar.checkbox("Fetch New Data", value=False)
    
    if fetch_new_data:
        period = st.sidebar.selectbox("Period", ['1mo', '3mo', '6mo', '1y', '2y', '5y'], index=4)
        if st.sidebar.button("Fetch Data"):
            with st.spinner(f"Fetching data for {selected_symbol}..."):
                fetcher = DataFetcher()
                data = fetcher.fetch_historical_data(selected_symbol, period=period)
                fetcher.save_data({selected_symbol: data})
                st.sidebar.success(f"Data fetched and saved!")
    
    # Model selection
    st.sidebar.subheader("Model Options")
    model_files = [f for f in os.listdir(config.MODEL_DIR) if f.endswith('.pth')] if os.path.exists(config.MODEL_DIR) else []
    
    if model_files:
        selected_model = st.sidebar.selectbox("Select Model", model_files)
        use_model = True
    else:
        st.sidebar.warning("No trained models found. Run training first.")
        use_model = False
    
    # Initial balance for simulation
    initial_balance = st.sidebar.number_input(
        "Initial Balance ($)",
        min_value=1000,
        max_value=1000000,
        value=10000,
        step=1000
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(f"Stock: {selected_symbol}")
    
    with col2:
        run_simulation_btn = st.button("🚀 Run Simulation", type="primary", disabled=not use_model)
    
    # Load and display data
    try:
        fetcher = DataFetcher()
        preprocessor = DataPreprocessor()
        
        df = fetcher.load_data(selected_symbol)
        
        if df.empty:
            st.warning(f"No data available for {selected_symbol}. Please fetch data first.")
            return
        
        df = preprocessor.prepare_training_data(df)
        
        # Display stock data
        st.subheader("📊 Stock Data and Technical Indicators")
        
        tab1, tab2, tab3 = st.tabs(["Price Chart", "Data Table", "Statistics"])
        
        with tab1:
            fig = plot_stock_data(df, selected_symbol)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.dataframe(df.tail(100), use_container_width=True)
        
        with tab3:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
            with col2:
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[-2]
                st.metric("Daily Change", f"${price_change:.2f}", f"{price_change/df['Close'].iloc[-2]*100:.2f}%")
            with col3:
                st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
            with col4:
                if 'RSI' in df.columns:
                    st.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")
        
        # Run simulation
        if run_simulation_btn and use_model:
            st.subheader("🤖 AI Trading Simulation")
            
            with st.spinner("Running simulation..."):
                # Create environment
                env = StockTradingEnv(df, initial_balance=initial_balance)
                
                # Load model
                state_size = env.observation_space.shape[0]
                model_path = os.path.join(config.MODEL_DIR, selected_model)
                agent = load_model(model_path, state_size)
                
                if agent is None:
                    st.error(f"Failed to load model: {selected_model}")
                    return
                
                # Run simulation
                portfolio_history, actions_history, metrics = run_simulation(agent, env)
                
                # Display results
                st.success("Simulation completed!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Return",
                        f"{metrics['total_return']*100:.2f}%",
                        f"${metrics['final_value'] - initial_balance:.2f}"
                    )
                with col2:
                    st.metric("Final Portfolio Value", f"${metrics['final_value']:.2f}")
                with col3:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                with col4:
                    st.metric("Number of Trades", metrics['num_trades'])
                
                # Portfolio chart
                st.subheader("Portfolio Performance")
                portfolio_fig = plot_portfolio_performance(portfolio_history, df)
                st.plotly_chart(portfolio_fig, use_container_width=True)
                
                # Actions table
                st.subheader("Trading Actions")
                actions_df = pd.DataFrame(actions_history)
                actions_df = actions_df[actions_df['action'] != 'Hold'].tail(20)
                st.dataframe(actions_df, use_container_width=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **About**
        
        This app uses Deep Q-Network (DQN) reinforcement learning
        to predict optimal trading actions and simulate portfolio performance.
        
        **Actions:**
        - Buy: Purchase stocks
        - Hold: Maintain position
        - Sell: Liquidate holdings
        """
    )


if __name__ == "__main__":
    main()
