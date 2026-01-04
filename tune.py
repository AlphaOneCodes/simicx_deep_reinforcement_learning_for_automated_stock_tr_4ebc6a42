"""
Hyperparameter Tuning and Agent Selection Module for DRL Trading System.

This module provides functionality to:
1. Load training data based on phase (limited/full ticker set)
2. Compute global statistics (mean returns, covariance matrix, turbulence threshold)
3. Train ensemble DRL agents (PPO, A2C, DDPG) on training data
4. Validate agents on held-out data and compute Sharpe ratios
5. Select best agent and save weights/parameters for production use

The training process uses an actor-critic approach with advantage estimation,
following PPO-style updates for all agent types.
"""

import importlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Dynamic module imports to comply with import policy
_signal_gen = importlib.import_module('signal_gen')
_data_loader = importlib.import_module('simicx.data_loader')

# Re-export symbols from signal_gen
PPOAgent = _signal_gen.PPOAgent
A2CAgent = _signal_gen.A2CAgent
DDPGAgent = _signal_gen.DDPGAgent
get_device = _signal_gen.get_device
calc_features = _signal_gen.calc_features
calc_turbulence = _signal_gen.calc_turbulence
construct_state = _signal_gen.construct_state
load_config = _signal_gen.load_config

# Re-export symbols from data_loader
get_training_data = _data_loader.get_training_data


# Constants
STATE_DIM = 181  # 1 + 30*6 (balance + 6 features per ticker)
ACTION_DIM = 30  # One action per ticker
INITIAL_CASH = 1_000_000.0
H_MAX = 100
CONFIG_PATH = 'simicx/alpha_config.json'
VALIDATION_MONTHS = 3
LOOKBACK_DAYS = 252  # Trading days for turbulence calculation
LEARNING_RATE = 1e-4
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE parameter
CLIP_EPSILON = 0.2  # PPO clip parameter


def compute_returns(ohlcv_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Compute daily returns for each ticker from OHLCV data.
    
    Pivots the close prices by ticker and computes percentage changes.
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, open, high, low, close, volume]
        tickers: List of ticker symbols (must match tickers in ohlcv_df)
        
    Returns:
        pd.DataFrame: Returns DataFrame with time index and ticker columns.
                     First row is NaN due to pct_change, so it's dropped.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=5).repeat(2),
        ...     'ticker': ['AAPL', 'MSFT'] * 5,
        ...     'close': [100, 200, 102, 204, 104, 208, 106, 212, 108, 216]
        ... })
        >>> returns = compute_returns(df, ['AAPL', 'MSFT'])
        >>> returns.shape[1]  # 2 tickers
        2
        >>> len(returns)  # 4 rows (5 - 1 for pct_change)
        4
    """
    # Pivot to get close prices per ticker
    pivot_close = ohlcv_df.pivot(index='time', columns='ticker', values='close')
    pivot_close = pivot_close[tickers]  # Ensure correct column order
    
    # Compute daily returns and drop NaN from first row
    returns = pivot_close.pct_change().dropna()
    
    return returns


def compute_global_statistics(
    ohlcv_df: pd.DataFrame, 
    tickers: List[str]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute global mean, covariance of returns, and turbulence threshold.
    
    Calculates statistics over the ENTIRE training period for use in:
    - Turbulence parameter storage (mu, sigma)
    - Risk management threshold (99th percentile)
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, close, ...]
        tickers: List of ticker symbols (must be 30)
        
    Returns:
        Tuple of:
            - mu: Mean returns array of shape (30,) as float64
            - sigma: Covariance matrix of shape (30, 30) as float64
            - threshold: 99th percentile turbulence threshold
            
    Example:
        >>> mu, sigma, threshold = compute_global_statistics(ohlcv_df, tickers)
        >>> mu.shape
        (30,)
        >>> sigma.shape
        (30, 30)
        >>> threshold > 0
        True
    """
    # Compute returns over entire training set
    returns = compute_returns(ohlcv_df, tickers)
    
    # Calculate mean and covariance matrix
    mu = returns.mean().values.astype(np.float64)
    sigma = returns.cov().values.astype(np.float64)
    
    # Calculate turbulence for all training days
    turbulence = calc_turbulence(ohlcv_df, tickers, lookback=LOOKBACK_DAYS)
    
    # Set threshold at 99th percentile
    valid_turbulence = turbulence.dropna()
    threshold = float(np.quantile(valid_turbulence.values, 0.99))
    
    return mu, sigma, threshold


def split_train_validation(
    ohlcv_df: pd.DataFrame, 
    validation_months: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training and validation sets temporally.
    
    Training: start to (end - validation_months)
    Validation: last validation_months
    
    Args:
        ohlcv_df: OHLCV DataFrame with 'time' column
        validation_months: Number of months for validation (default: 3)
        
    Returns:
        Tuple of (train_df, val_df) DataFrames
        
    Example:
        >>> train_df, val_df = split_train_validation(ohlcv_df, validation_months=3)
        >>> train_df['time'].max() < val_df['time'].min()
        True
    """
    unique_dates = sorted(ohlcv_df['time'].unique())
    
    # Get end date and compute split point
    end_date = pd.Timestamp(unique_dates[-1])
    split_date = end_date - pd.DateOffset(months=validation_months)
    
    # Split data
    train_df = ohlcv_df[ohlcv_df['time'] < split_date].copy()
    val_df = ohlcv_df[ohlcv_df['time'] >= split_date].copy()
    
    return train_df, val_df


def get_prices_for_date(
    ohlcv_df: pd.DataFrame, 
    date: pd.Timestamp, 
    tickers: List[str]
) -> np.ndarray:
    """Get close prices for all tickers on a specific date.
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, close, ...]
        date: Target date to get prices for
        tickers: List of ticker symbols (30 tickers)
        
    Returns:
        np.ndarray: Prices array of shape (30,) as float32
    """
    day_data = ohlcv_df[ohlcv_df['time'] == date]
    prices = []
    for ticker in tickers:
        ticker_data = day_data[day_data['ticker'] == ticker]
        if len(ticker_data) > 0:
            prices.append(float(ticker_data['close'].values[0]))
        else:
            prices.append(0.0)
    return np.array(prices, dtype=np.float32)


def get_features_for_date(
    features_df: pd.DataFrame, 
    date: pd.Timestamp, 
    tickers: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract technical indicator features for a specific date.
    
    Args:
        features_df: Features DataFrame from calc_features, indexed by time
        date: Target date to get features for
        tickers: List of ticker symbols (30 tickers)
        
    Returns:
        Tuple of (macd, rsi, cci, adx) arrays, each of shape (30,) as float32
        Returns default values if date not found.
    """
    # Handle date not in index
    if date not in features_df.index:
        return (
            np.zeros(len(tickers), dtype=np.float32),
            np.ones(len(tickers), dtype=np.float32) * 50.0,
            np.zeros(len(tickers), dtype=np.float32),
            np.ones(len(tickers), dtype=np.float32) * 25.0
        )
    
    row = features_df.loc[date]
    
    # Extract features for each ticker
    macd = np.array(
        [float(row.get(f'{t}_macd', 0.0)) for t in tickers], 
        dtype=np.float32
    )
    rsi = np.array(
        [float(row.get(f'{t}_rsi', 50.0)) for t in tickers], 
        dtype=np.float32
    )
    cci = np.array(
        [float(row.get(f'{t}_cci', 0.0)) for t in tickers], 
        dtype=np.float32
    )
    adx = np.array(
        [float(row.get(f'{t}_adx', 25.0)) for t in tickers], 
        dtype=np.float32
    )
    
    return macd, rsi, cci, adx


def compute_portfolio_value(
    balance: float, 
    holdings: np.ndarray, 
    prices: np.ndarray
) -> float:
    """Compute total portfolio value (cash + holdings).
    
    Args:
        balance: Cash balance
        holdings: Holdings array of shape (30,)
        prices: Prices array of shape (30,)
        
    Returns:
        Total portfolio value as float
    """
    return balance + float(np.sum(holdings * prices))


def execute_actions(
    actions: np.ndarray,
    balance: float,
    holdings: np.ndarray,
    prices: np.ndarray,
    h_max: int = H_MAX
) -> Tuple[float, np.ndarray]:
    """Execute trading actions and update portfolio state.
    
    Actions in [-1, 1] are scaled by h_max to get share quantities.
    Positive actions = buy, Negative actions = sell.
    
    Trades are constrained by:
    - Available cash (for buys)
    - Available holdings (for sells)
    
    Args:
        actions: Action array of shape (30,) in range [-1, 1]
        balance: Current cash balance
        holdings: Current holdings array of shape (30,)
        prices: Current prices array of shape (30,)
        h_max: Maximum shares per trade (default: 100)
        
    Returns:
        Tuple of (new_balance, new_holdings)
        
    Example:
        >>> actions = np.array([0.5, -0.3] + [0.0]*28)  # buy, sell, hold rest
        >>> balance = 100000.0
        >>> holdings = np.zeros(30, dtype=np.float32)
        >>> holdings[1] = 100  # Have 100 shares to sell
        >>> prices = np.ones(30, dtype=np.float32) * 100
        >>> new_bal, new_hold = execute_actions(actions, balance, holdings, prices)
        >>> new_hold[0] > 0  # Bought first stock
        True
        >>> new_hold[1] < 100  # Sold second stock
        True
    """
    new_balance = balance
    new_holdings = holdings.copy()
    
    # Scale actions to share quantities
    trade_quantities = actions * h_max
    
    for i, qty in enumerate(trade_quantities):
        price = prices[i]
        if price <= 0:
            continue
            
        if qty > 0:  # Buy
            # Limit by available cash
            max_buyable = int(new_balance / price)
            actual_qty = min(int(qty), max_buyable)
            if actual_qty > 0:
                new_balance -= actual_qty * price
                new_holdings[i] += actual_qty
        elif qty < 0:  # Sell
            # Limit by available holdings
            actual_qty = min(int(abs(qty)), int(new_holdings[i]))
            if actual_qty > 0:
                new_balance += actual_qty * price
                new_holdings[i] -= actual_qty
                
    return new_balance, new_holdings


def train_agent(
    agent: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_df: pd.DataFrame,
    features_df: pd.DataFrame,
    tickers: List[str],
    device: torch.device,
    h_max: int = H_MAX
) -> torch.nn.Module:
    """Train a single agent on the training data using actor-critic updates.
    
    Uses PPO-style training with:
    - Generalized Advantage Estimation (GAE) for advantage computation
    - Clipped surrogate objective for policy updates
    - Value function loss for critic updates
    
    Args:
        agent: Agent to train (PPOAgent, A2CAgent, or DDPGAgent)
        optimizer: PyTorch optimizer for agent parameters
        train_df: Training OHLCV DataFrame
        features_df: Features DataFrame from calc_features
        tickers: List of ticker symbols (30 tickers)
        device: Computation device (cuda/mps/cpu)
        h_max: Maximum shares per trade (default: 100)
        
    Returns:
        Trained agent (modified in-place, also returned for convenience)
        
    Example:
        >>> agent = PPOAgent(device=device)
        >>> optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
        >>> trained = train_agent(agent, optimizer, train_df, features_df, tickers, device)
    """
    agent.train()
    
    # Get unique dates in training set
    unique_dates = sorted(train_df['time'].unique())
    
    if len(unique_dates) < 2:
        return agent
    
    # Initialize portfolio state
    balance = INITIAL_CASH
    holdings = np.zeros(len(tickers), dtype=np.float32)
    
    # Experience storage for batch updates
    states: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    rewards: List[float] = []
    values: List[float] = []
    log_probs: List[float] = []
    
    update_frequency = 20  # Update every N steps
    
    for t in range(len(unique_dates) - 1):
        date = unique_dates[t]
        next_date = unique_dates[t + 1]
        
        # Get current prices and features
        prices = get_prices_for_date(train_df, date, tickers)
        macd, rsi, cci, adx = get_features_for_date(features_df, date, tickers)
        
        # Construct state vector
        state = construct_state(
            balance=balance,
            prices=prices,
            holdings=holdings,
            macd=macd,
            rsi=rsi,
            cci=cci,
            adx=adx
        )
        
        # Get action from agent
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_tensor, value_tensor = agent(state_tensor)
        
        action = action_tensor.squeeze(0).cpu().numpy()
        value = value_tensor.squeeze().item()
        
        # Compute log probability (assuming Gaussian policy with fixed std)
        action_std = 0.5
        action_dist = torch.distributions.Normal(
            action_tensor, 
            torch.ones_like(action_tensor) * action_std
        )
        log_prob = action_dist.log_prob(action_tensor).sum(dim=-1).item()
        
        # Execute actions
        new_balance, new_holdings = execute_actions(
            action, balance, holdings, prices, h_max
        )
        
        # Get next prices for reward computation
        next_prices = get_prices_for_date(train_df, next_date, tickers)
        
        # Compute portfolio values
        current_value = compute_portfolio_value(balance, holdings, prices)
        next_value = compute_portfolio_value(new_balance, new_holdings, next_prices)
        
        # Reward is change in portfolio value (scaled for numerical stability)
        reward = (next_value - current_value) / INITIAL_CASH * 100
        
        # Store experience
        states.append(state)
        actions_list.append(action)
        rewards.append(reward)
        values.append(value)
        log_probs.append(log_prob)
        
        # Update portfolio state
        balance = new_balance
        holdings = new_holdings
        
        # Perform update at intervals or end of episode
        should_update = (t + 1) % update_frequency == 0 or t == len(unique_dates) - 2
        
        if should_update and len(states) > 0:
            # Compute advantages using GAE
            advantages: List[float] = []
            returns_list: List[float] = []
            gae = 0.0
            
            for i in reversed(range(len(rewards))):
                if i == len(rewards) - 1:
                    next_val = 0.0
                else:
                    next_val = values[i + 1]
                
                delta = rewards[i] + GAMMA * next_val - values[i]
                gae = delta + GAMMA * GAE_LAMBDA * gae
                advantages.insert(0, gae)
                returns_list.insert(0, gae + values[i])
            
            # Convert to tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(device)
            actions_tensor = torch.FloatTensor(np.array(actions_list)).to(device)
            advantages_tensor = torch.FloatTensor(advantages).to(device)
            returns_tensor = torch.FloatTensor(returns_list).to(device)
            old_log_probs_tensor = torch.FloatTensor(log_probs).to(device)
            
            # Normalize advantages for stability
            if len(advantages_tensor) > 1:
                adv_mean = advantages_tensor.mean()
                adv_std = advantages_tensor.std() + 1e-8
                advantages_tensor = (advantages_tensor - adv_mean) / adv_std
            
            # Forward pass with gradients
            new_actions, new_values = agent(states_tensor)
            new_dist = torch.distributions.Normal(
                new_actions, 
                torch.ones_like(new_actions) * action_std
            )
            new_log_probs = new_dist.log_prob(actions_tensor).sum(dim=-1)
            
            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(
                ratio, 
                1.0 - CLIP_EPSILON, 
                1.0 + CLIP_EPSILON
            ) * advantages_tensor
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (MSE)
            critic_loss = torch.nn.functional.mse_loss(
                new_values.squeeze(), 
                returns_tensor
            )
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Gradient update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()
            
            # Clear experience buffer
            states = []
            actions_list = []
            rewards = []
            values = []
            log_probs = []
    
    return agent


def validate_agent(
    agent: torch.nn.Module,
    val_df: pd.DataFrame,
    features_df: pd.DataFrame,
    tickers: List[str],
    device: torch.device,
    h_max: int = H_MAX
) -> float:
    """Validate agent on held-out data and compute Sharpe ratio.
    
    Runs the agent in inference mode (no training) through the validation
    period and computes the annualized Sharpe ratio of daily returns.
    
    Args:
        agent: Trained agent to validate
        val_df: Validation OHLCV DataFrame
        features_df: Features DataFrame from calc_features
        tickers: List of ticker symbols (30 tickers)
        device: Computation device
        h_max: Maximum shares per trade (default: 100)
        
    Returns:
        Annualized Sharpe ratio (can be negative)
        
    Example:
        >>> sharpe = validate_agent(agent, val_df, features_df, tickers, device)
        >>> isinstance(sharpe, float)
        True
    """
    agent.eval()
    
    unique_dates = sorted(val_df['time'].unique())
    
    if len(unique_dates) < 2:
        return float('-inf')
    
    # Initialize portfolio
    balance = INITIAL_CASH
    holdings = np.zeros(len(tickers), dtype=np.float32)
    
    daily_returns: List[float] = []
    prev_value = INITIAL_CASH
    
    for t in range(len(unique_dates) - 1):
        date = unique_dates[t]
        next_date = unique_dates[t + 1]
        
        # Get current data
        prices = get_prices_for_date(val_df, date, tickers)
        macd, rsi, cci, adx = get_features_for_date(features_df, date, tickers)
        
        # Construct state
        state = construct_state(
            balance=balance,
            prices=prices,
            holdings=holdings,
            macd=macd,
            rsi=rsi,
            cci=cci,
            adx=adx
        )
        
        # Get action (inference, no gradient)
        action = agent.get_action(state)
        
        # Execute trades
        balance, holdings = execute_actions(action, balance, holdings, prices, h_max)
        
        # Get next day value
        next_prices = get_prices_for_date(val_df, next_date, tickers)
        current_value = compute_portfolio_value(balance, holdings, next_prices)
        
        # Compute daily return
        if prev_value > 0:
            daily_return = (current_value - prev_value) / prev_value
            daily_returns.append(daily_return)
        
        prev_value = current_value
    
    # Compute annualized Sharpe ratio
    if len(daily_returns) < 2:
        return float('-inf')
    
    returns_array = np.array(daily_returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array, ddof=1)
    
    if std_return < 1e-10:
        return 0.0
    
    # Annualized: multiply by sqrt(252) for daily to annual conversion
    sharpe = (mean_return / std_return) * np.sqrt(252)
    
    return float(sharpe)


def tune(phase: str = 'limited') -> Dict:
    """Main tuning function to train ensemble agents and select best one.

    This function:
    1. Loads configuration and training data based on phase
    2. Computes global statistics (mean, covariance, turbulence threshold)
    3. Splits data into training and validation sets
    4. Trains PPO, A2C, and DDPG agents
    5. Validates each agent and selects the one with highest Sharpe ratio
    6. Saves best agent weights to 'best_agent.pth'
    7. Saves parameters to 'best_params.json'

    Args:
        phase: 'limited' or 'full' ticker set (default: 'limited')

    Returns:
        Dict with keys:
            - agent_type: Best agent type ('ppo', 'a2c', or 'ddpg')
            - sharpe: Validation Sharpe ratio of best agent
            - params_path: Path to saved parameters JSON
            - model_path: Path to saved model weights

    Example:
        >>> result = tune(phase='limited')
        >>> result['agent_type'] in ['ppo', 'a2c', 'ddpg']
        True
        >>> os.path.exists(result['model_path'])
        True
    """
    # Load configuration
    config = load_config(CONFIG_PATH)

    # Set tickers based on phase
    if phase == 'limited':
        tickers = config['LIMITED_TICKERS']
    else:
        tickers = config['FULL_TICKERS']

    # Calculate dimensions dynamically based on ticker count
    num_tickers = len(tickers)
    state_dim = 1 + num_tickers * 6  # balance + 6 features per ticker
    action_dim = num_tickers  # One action per ticker

    # Get computation device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Phase: {phase}, Tickers: {num_tickers}, State dim: {state_dim}, Action dim: {action_dim}")

    # Load training data
    print(f"Loading training data for phase: {phase}")
    df = get_training_data(phase=phase)
    print(f"Loaded {len(df)} records spanning {df['time'].nunique()} unique dates")

    # Compute global statistics
    print("Computing global statistics...")
    mu, sigma, turbulence_threshold = compute_global_statistics(df, tickers)
    print(f"Turbulence threshold (99th percentile): {turbulence_threshold:.4f}")

    # Compute technical features for entire dataset
    print("Computing technical features...")
    features_df = calc_features(df, tickers)
    print(f"Features computed for {len(features_df)} dates")

    # Split into train and validation sets
    print(f"Splitting data (validation = last {VALIDATION_MONTHS} months)...")
    train_df, val_df = split_train_validation(df, VALIDATION_MONTHS)
    train_dates = train_df['time'].nunique()
    val_dates = val_df['time'].nunique()
    print(f"Training: {train_dates} dates, Validation: {val_dates} dates")

    # Agent configurations
    agent_classes = {
        'ppo': PPOAgent,
        'a2c': A2CAgent,
        'ddpg': DDPGAgent
    }

    best_agent = None
    best_agent_type: Optional[str] = None
    best_sharpe = float('-inf')

    # Train and evaluate each agent type
    for agent_type, AgentClass in agent_classes.items():
        print(f"\n{'='*50}")
        print(f"Training {agent_type.upper()} Agent")
        print(f"{'='*50}")

        # Instantiate agent on correct device with dynamic dimensions
        agent = AgentClass(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

        # Create optimizer
        optimizer = torch.optim.Adam(agent.parameters(), lr=LEARNING_RATE)

        # Train agent
        agent = train_agent(
            agent=agent,
            optimizer=optimizer,
            train_df=train_df,
            features_df=features_df,
            tickers=tickers,
            device=device,
            h_max=H_MAX
        )

        # Validate agent
        sharpe = validate_agent(
            agent=agent,
            val_df=val_df,
            features_df=features_df,
            tickers=tickers,
            device=device,
            h_max=H_MAX
        )
        print(f"{agent_type.upper()} Validation Sharpe Ratio: {sharpe:.4f}")

        # Track best agent
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_agent = agent
            best_agent_type = agent_type

    if best_agent is None or best_agent_type is None:
        raise RuntimeError("No agent was successfully trained")

    print(f"\n{'='*50}")
    print(f"Best Agent: {best_agent_type.upper()} (Sharpe: {best_sharpe:.4f})")
    print(f"{'='*50}")

    # Save best agent weights
    model_path = 'best_agent.pth'
    torch.save({'state_dict': best_agent.state_dict()}, model_path)
    print(f"Saved model weights to: {model_path}")

    # Prepare parameters dictionary
    best_params = {
        'agent_type': best_agent_type,
        'model_path': model_path,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'num_tickers': num_tickers,
        'tickers': tickers,
        'turbulence_params': {
            'mu': mu.tolist(),
            'sigma': sigma.tolist(),
            'threshold': turbulence_threshold
        },
        'h_max': H_MAX,
        'technical_params': {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'cci_period': 14,
            'adx_period': 14
        }
    }

    # Save parameters to JSON
    params_path = 'best_params.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved parameters to: {params_path}")

    return {
        'agent_type': best_agent_type,
        'sharpe': best_sharpe,
        'params_path': params_path,
        'model_path': model_path
    }


def main():
    """Entry point for tuning script with CLI argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Train ensemble DRL agents and select best for trading'
    )
    parser.add_argument(
        '--phase',
        choices=['limited', 'full'],
        default='limited',
        help='Ticker set to use: limited (10 tickers) or full (30 tickers)'
    )
    
    args = parser.parse_args()
    
    result = tune(phase=args.phase)
    
    print(f"\n{'='*50}")
    print("TUNING COMPLETE")
    print(f"{'='*50}")
    print(f"Best Agent Type: {result['agent_type'].upper()}")
    print(f"Validation Sharpe: {result['sharpe']:.4f}")
    print(f"Model saved to: {result['model_path']}")
    print(f"Params saved to: {result['params_path']}")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def simicx_test_compute_returns():
    """Test return computation from OHLCV data."""
    # Create synthetic test data
    dates = pd.date_range('2024-01-01', periods=5)
    df = pd.DataFrame({
        'time': list(dates) * 2,
        'ticker': ['A'] * 5 + ['B'] * 5,
        'open': [100, 101, 102, 103, 104] + [200, 202, 204, 206, 208],
        'high': [105] * 5 + [210] * 5,
        'low': [95] * 5 + [190] * 5,
        'close': [100, 102, 104, 106, 108] + [200, 204, 208, 212, 216],
        'volume': [1000] * 10
    })
    
    returns = compute_returns(df, ['A', 'B'])
    
    # Should have 4 rows (5 - 1 for pct_change)
    assert len(returns) == 4, f"Expected 4 rows, got {len(returns)}"
    
    # Check columns
    assert 'A' in returns.columns, "Missing column A"
    assert 'B' in returns.columns, "Missing column B"
    
    # Check return values (A: 100->102 is 2% return)
    expected_return_a = 0.02
    actual_return_a = returns['A'].iloc[0]
    assert abs(actual_return_a - expected_return_a) < 0.001, \
        f"Expected ~{expected_return_a}, got {actual_return_a}"
    
    print("simicx_test_compute_returns PASSED")


def simicx_test_portfolio_operations():
    """Test portfolio value computation and action execution."""
    # Test portfolio value
    balance = 10000.0
    holdings = np.array([10, 20], dtype=np.float32)
    prices = np.array([100.0, 50.0], dtype=np.float32)
    
    value = compute_portfolio_value(balance, holdings, prices)
    expected = 10000 + (10 * 100) + (20 * 50)  # 12000
    assert abs(value - expected) < 0.01, f"Expected {expected}, got {value}"
    
    # Test buy action
    # 30 tickers with 2 real, rest zeros
    full_holdings = np.zeros(30, dtype=np.float32)
    full_holdings[0:2] = holdings
    full_prices = np.zeros(30, dtype=np.float32)
    full_prices[0:2] = prices
    
    actions = np.zeros(30, dtype=np.float32)
    actions[0] = 0.5  # Buy 50 shares of first (h_max=100 -> 50 shares)
    
    new_balance, new_holdings = execute_actions(
        actions, balance, full_holdings, full_prices, h_max=100
    )
    
    # Should have bought shares (limited by available cash)
    assert new_balance < balance, "Balance should decrease after buying"
    assert new_holdings[0] > full_holdings[0], "Holdings should increase"
    
    # Test sell action
    actions_sell = np.zeros(30, dtype=np.float32)
    actions_sell[0] = -0.3  # Sell 30 shares
    
    new_balance2, new_holdings2 = execute_actions(
        actions_sell, new_balance, new_holdings, full_prices, h_max=100
    )
    
    assert new_balance2 > new_balance, "Balance should increase after selling"
    assert new_holdings2[0] < new_holdings[0], "Holdings should decrease"
    
    print("simicx_test_portfolio_operations PASSED")


def simicx_test_integration_with_signal_gen():
    """Integration test exercising signal_gen module interfaces."""
    import tempfile
    from pathlib import Path
    
    # Test get_device
    device = get_device()
    assert device is not None, "get_device should return a device"
    assert device.type in ['cuda', 'mps', 'cpu'], f"Unexpected device type: {device.type}"
    
    # Test agent instantiation
    agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)
    
    # Verify agent has required methods
    assert hasattr(agent, 'forward'), "Agent missing forward method"
    assert hasattr(agent, 'get_action'), "Agent missing get_action method"
    assert hasattr(agent, 'load_weights'), "Agent missing load_weights method"
    assert hasattr(agent, 'parameters'), "Agent missing parameters method"
    assert hasattr(agent, 'state_dict'), "Agent missing state_dict method"
    
    # Test construct_state
    state = construct_state(
        balance=1000000.0,
        prices=np.ones(30, dtype=np.float32) * 100,
        holdings=np.zeros(30, dtype=np.float32),
        macd=np.zeros(30, dtype=np.float32),
        rsi=np.ones(30, dtype=np.float32) * 50,
        cci=np.zeros(30, dtype=np.float32),
        adx=np.ones(30, dtype=np.float32) * 25
    )
    
    assert state.shape == (181,), f"State should be (181,), got {state.shape}"
    assert state.dtype == np.float32, f"State dtype should be float32, got {state.dtype}"
    
    # Test agent inference
    action = agent.get_action(state)
    assert action.shape == (30,), f"Action should be (30,), got {action.shape}"
    assert np.all((action >= -1) & (action <= 1)), "Actions should be in [-1, 1]"
    
    # Test forward pass
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    action_tensor, value_tensor = agent(state_tensor)
    assert action_tensor.shape == (1, 30), f"Action tensor shape mismatch: {action_tensor.shape}"
    assert value_tensor.shape == (1, 1), f"Value tensor shape mismatch: {value_tensor.shape}"
    
    # Test load_config
    try:
        config = load_config(CONFIG_PATH)
        assert 'LIMITED_TICKERS' in config, "Config missing LIMITED_TICKERS"
        assert 'FULL_TICKERS' in config, "Config missing FULL_TICKERS"
        assert len(config['FULL_TICKERS']) == 30, "FULL_TICKERS should have 30 tickers"
        print(f"  Config loaded: {len(config['LIMITED_TICKERS'])} limited, "
              f"{len(config['FULL_TICKERS'])} full tickers")
    except FileNotFoundError:
        print("  Config file not found (expected in test environment)")
    
    # Test save/load weights
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        model_path = tmp_path / "test_agent.pth"
        
        # Save weights
        torch.save({'state_dict': agent.state_dict()}, str(model_path))
        assert model_path.exists(), "Model file should exist after save"
        
        # Load into new agent
        new_agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)
        new_agent.load_weights(str(model_path))
        
        # Verify loaded correctly (should produce same actions)
        new_action = new_agent.get_action(state)
        assert np.allclose(action, new_action, atol=1e-5), \
            "Loaded agent should produce identical actions"
        
        # Test JSON params saving
        params = {
            'agent_type': 'ppo',
            'model_path': 'best_agent.pth',
            'turbulence_params': {
                'mu': [0.0] * 30,
                'sigma': [[0.0] * 30 for _ in range(30)],
                'threshold': 140.0
            },
            'h_max': 100,
            'technical_params': {
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'rsi_period': 14,
                'cci_period': 14,
                'adx_period': 14
            }
        }
        params_path = tmp_path / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        # Verify JSON is valid
        with open(params_path, 'r') as f:
            loaded_params = json.load(f)
        assert loaded_params['agent_type'] == 'ppo'
        assert len(loaded_params['turbulence_params']['mu']) == 30
    
    print("simicx_test_integration_with_signal_gen PASSED")


if __name__ == '__main__':
    main()