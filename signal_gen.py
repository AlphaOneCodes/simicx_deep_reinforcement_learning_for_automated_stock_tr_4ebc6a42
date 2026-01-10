"""
Signal Generation Module for DRL Trading System.

This module provides:
- Hardware device detection (CUDA, MPS, CPU)
- Feature engineering with technical indicators (MACD, RSI, CCI, ADX)
- Turbulence calculation for risk management
- DRL Agent implementations (PPO, A2C, DDPG)
- Day-by-day signal generation loop with virtual portfolio simulation
"""

import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn as nn


# Constants for state/action dimensions
STATE_DIM = 181  # 1 (balance) + 30*6 (prices, holdings, MACD, RSI, CCI, ADX)
ACTION_DIM = 30  # 30 stocks (DJIA)
NUM_TICKERS = 30


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch computation.
    
    Checks for hardware availability in order: CUDA -> MPS -> CPU.
    
    Returns:
        torch.device: The best available device.
        
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> print(device.type)  # 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def calc_features(ohlcv_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Calculate technical indicators for feature engineering.
    
    Computes MACD (12, 26, 9), RSI (14), CCI (14), and ADX (14) for each ticker.
    NaN values are handled using bfill then ffill, with sensible defaults for
    any remaining NaNs.
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, open, high, low, close, volume]
        tickers: List of ticker symbols to process
        
    Returns:
        pd.DataFrame: DataFrame indexed by time with columns for each indicator per ticker:
            - {ticker}_macd: MACD line value
            - {ticker}_rsi: RSI indicator (0-100)
            - {ticker}_cci: Commodity Channel Index
            - {ticker}_adx: Average Directional Index (0-100)
            
    Example:
        >>> ohlcv = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=50),
        ...     'ticker': ['AAPL'] * 50,
        ...     'open': np.random.rand(50) * 100,
        ...     'high': np.random.rand(50) * 100,
        ...     'low': np.random.rand(50) * 100,
        ...     'close': np.random.rand(50) * 100,
        ...     'volume': np.random.randint(1000, 10000, 50)
        ... })
        >>> features = calc_features(ohlcv, ['AAPL'])
        >>> 'AAPL_macd' in features.columns
        True
    """
    # Pivot to get prices per ticker
    pivot_close = ohlcv_df.pivot(index='time', columns='ticker', values='close')
    pivot_high = ohlcv_df.pivot(index='time', columns='ticker', values='high')
    pivot_low = ohlcv_df.pivot(index='time', columns='ticker', values='low')
    
    features = pd.DataFrame(index=pivot_close.index)
    
    # Collect all indicator DataFrames and concat at once (prevents fragmentation)
    indicator_dfs = []
    
    for ticker in tickers:
        if ticker not in pivot_close.columns:
            # Fill with default values if ticker not found
            default_df = pd.DataFrame({
                f'{ticker}_macd': 0.0,
                f'{ticker}_rsi': 50.0,
                f'{ticker}_cci': 0.0,
                f'{ticker}_adx': 25.0
            }, index=pivot_close.index)
            indicator_dfs.append(default_df)
            continue
        
        close = pivot_close[ticker].copy()
        high = pivot_high[ticker].copy()
        low = pivot_low[ticker].copy()
        
        # Forward fill missing values first
        close = close.ffill().bfill()
        high = high.ffill().bfill()
        low = low.ffill().bfill()
        
        # Calculate all indicators for this ticker
        macd_indicator = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
        cci_indicator = ta.trend.CCIIndicator(high, low, close, window=14)
        adx_indicator = ta.trend.ADXIndicator(high, low, close, window=14)
        
        # Create DataFrame with all indicators for this ticker
        ticker_indicators = pd.DataFrame({
            f'{ticker}_macd': macd_indicator.macd(),
            f'{ticker}_rsi': rsi_indicator.rsi(),
            f'{ticker}_cci': cci_indicator.cci(),
            f'{ticker}_adx': adx_indicator.adx()
        }, index=pivot_close.index)
        
        indicator_dfs.append(ticker_indicators)
    
    # Concatenate all indicator DataFrames at once (avoids fragmentation)
    if indicator_dfs:
        features = pd.concat(indicator_dfs, axis=1)
    
    # Handle NaNs with bfill then ffill
    features = features.bfill().ffill()
    
    # Fill any remaining NaNs with sensible defaults
    for col in features.columns:
        if '_macd' in col:
            features[col] = features[col].fillna(0.0)
        elif '_rsi' in col:
            features[col] = features[col].fillna(50.0)
        elif '_cci' in col:
            features[col] = features[col].fillna(0.0)
        elif '_adx' in col:
            features[col] = features[col].fillna(25.0)
    
    return features


def calc_turbulence(
    ohlcv_df: pd.DataFrame,
    tickers: List[str],
    lookback: int = 252
) -> pd.Series:
    """
    Calculate turbulence index for risk management.
    
    Turbulence measures how unusual current market conditions are compared
    to historical patterns using Mahalanobis distance.
    
    Formula: d_t = (y_t - μ) Σ^{-1} (y_t - μ)^T
    
    Where:
        - y_t: Daily returns vector at time t
        - μ: Mean of historical returns over lookback period
        - Σ: Covariance matrix of historical returns
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, open, high, low, close, volume]
        tickers: List of ticker symbols
        lookback: Historical lookback period for calculating μ and Σ (default: 252 trading days)
        
    Returns:
        pd.Series: Turbulence index values indexed by time
        
    Example:
        >>> turbulence = calc_turbulence(ohlcv_df, tickers=['AAPL', 'MSFT'], lookback=252)
        >>> if turbulence.iloc[-1] > 140:
        ...     print("High turbulence - entering risk-off mode")
    """
    # Pivot to get closing prices per ticker
    pivot_close = ohlcv_df.pivot(index='time', columns='ticker', values='close')
    pivot_close = pivot_close.reindex(columns=tickers)
    
    # Forward fill and backward fill missing values
    pivot_close = pivot_close.ffill().bfill()
    
    # Calculate daily returns
    returns = pivot_close.pct_change()
    returns = returns.fillna(0)
    
    # Initialize turbulence series
    turbulence = pd.Series(index=returns.index, dtype=float)
    turbulence[:] = 0.0
    
    for i in range(lookback, len(returns)):
        # Historical returns for covariance calculation
        historical_returns = returns.iloc[i - lookback:i]
        
        # Current return vector
        y_t = returns.iloc[i].values.astype(float)
        
        # Calculate mean and covariance
        mu = historical_returns.mean().values.astype(float)
        cov = historical_returns.cov().values.astype(float)
        
        # Handle NaN values
        y_t = np.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)
        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle singular covariance matrix
        try:
            # Add small regularization for numerical stability
            cov_reg = cov + np.eye(cov.shape[0]) * 1e-8
            cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if still singular
            cov_inv = np.linalg.pinv(cov)
        
        # Calculate Mahalanobis distance
        diff = y_t - mu
        d_t = float(diff @ cov_inv @ diff.T)
        
        # Ensure non-negative
        d_t = max(0.0, d_t)
        turbulence.iloc[i] = d_t
    
    return turbulence


class BaseAgent(nn.Module):
    """
    Base DRL Agent with shared architecture for trading.
    
    Architecture:
        Shared: Linear(181->256) -> ReLU -> Linear(256->128) -> ReLU
        Actor Head: Linear(128->30) -> Tanh (outputs actions in [-1, 1])
        Critic Head: Linear(128->1) (outputs state value)
    
    The agent uses an actor-critic architecture suitable for PPO, A2C, and DDPG
    algorithms. The shared layers learn representations, while the actor head
    outputs continuous actions and the critic head estimates state values.
    
    Attributes:
        state_dim: Input state dimension (default: 181)
        action_dim: Output action dimension (default: 30)
        device: Computation device (auto-detected)
        
    Example:
        >>> agent = BaseAgent(state_dim=181, action_dim=30)
        >>> state = torch.randn(1, 181)
        >>> action, value = agent(state.to(agent.device))
        >>> action.shape
        torch.Size([1, 30])
        >>> # For inference
        >>> action_np = agent.get_action(state.squeeze().numpy())
        >>> action_np.shape
        (30,)
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden1: int = 256,
        hidden2: int = 128,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            state_dim: State space dimension (default: 181)
            action_dim: Action space dimension (default: 30)
            hidden1: First hidden layer size (default: 256)
            hidden2: Second hidden layer size (default: 128)
            device: Computation device (default: auto-detect)
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device is not None else get_device()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU()
        )
        
        # Actor head: outputs actions in [-1, 1]
        self.actor = nn.Sequential(
            nn.Linear(hidden2, action_dim),
            nn.Tanh()
        )
        
        # Critic head: outputs state value
        self.critic = nn.Linear(hidden2, 1)
        
        self.to(self.device)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor of shape (batch, state_dim)
            
        Returns:
            Tuple of:
                - action: Action tensor of shape (batch, action_dim) in range [-1, 1]
                - value: State value tensor of shape (batch, 1)
        """
        shared_out = self.shared(state)
        action = self.actor(shared_out)
        value = self.critic(shared_out)
        return action, value
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Get action from state for inference (no gradient computation).
        
        Args:
            state: State vector of shape (state_dim,) or (batch, state_dim)
            
        Returns:
            np.ndarray: Action vector of shape (action_dim,) in range [-1, 1]
            
        Example:
            >>> agent = BaseAgent()
            >>> state = np.random.randn(181).astype(np.float32)
            >>> action = agent.get_action(state)
            >>> action.shape
            (30,)
            >>> np.all((action >= -1) & (action <= 1))
            True
        """
        self.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Ensure batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action, _ = self.forward(state)
            return action.squeeze(0).cpu().numpy()
    
    def load_weights(self, path: str) -> None:
        """
        Load model weights from a checkpoint file.
        
        The checkpoint file must contain a 'state_dict' key with the
        PyTorch model state dictionary.
        
        Args:
            path: Path to the .pth checkpoint file
            
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist
            KeyError: If 'state_dict' key is not found in checkpoint
            
        Example:
            >>> agent = BaseAgent()
            >>> agent.load_weights('best_agent.pth')
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'state_dict' not in checkpoint:
            raise KeyError("Checkpoint must contain 'state_dict' key")
        
        self.load_state_dict(checkpoint['state_dict'])
        self.eval()


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) Agent.
    
    Inherits from BaseAgent with PPO-specific initialization.
    Uses the same network architecture as BaseAgent.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'PPO'


class A2CAgent(BaseAgent):
    """
    Advantage Actor-Critic (A2C) Agent.
    
    Inherits from BaseAgent with A2C-specific initialization.
    Uses the same network architecture as BaseAgent.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agent_type = 'A2C'


class ReplayBuffer:
    """
    Experience Replay Buffer for off-policy learning (DDPG).
    
    Stores transitions (s, a, r, s', done) and provides random sampling
    for training to break temporal correlations.
    
    Attributes:
        max_size: Maximum buffer capacity
        ptr: Current write position (circular buffer)
        size: Current number of stored transitions
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            max_size: Maximum buffer size (default: 100,000)
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class OUNoise:
    """
    Ornstein-Uhlenbeck process for exploration in continuous action spaces.
    
    Generates temporally correlated noise for better exploration than
    independent Gaussian noise.
    
    Formula: dx = theta * (mu - x) * dt + sigma * dW
    """
    
    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2):
        """
        Initialize OU noise process.
        
        Args:
            action_dim: Dimension of action space
            mu: Mean of the process (default: 0.0)
            theta: Mean reversion rate (default: 0.15)
            sigma: Volatility/noise scale (default: 0.2)
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self) -> None:
        """Reset the internal state to the mean."""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self) -> np.ndarray:
        """
        Generate noise sample.
        
        Returns:
            Noise array of shape (action_dim,)
        """
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state


class DDPGActor(nn.Module):
    """
    DDPG Actor Network (Policy).
    
    Maps states to deterministic actions in continuous space.
    Output is bounded to [-1, 1] using Tanh activation.
    
    Architecture: state → 256 → ReLU → 128 → ReLU → action_dim → Tanh
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden1: int = 256,
                 hidden2: int = 128):
        """
        Initialize DDPG actor network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden1: First hidden layer size (default: 256)
            hidden2: Second hidden layer size (default: 128)
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get action.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            
        Returns:
            Action tensor of shape (batch, action_dim) in range [-1, 1]
        """
        return self.network(state)


class DDPGCritic(nn.Module):
    """
    DDPG Critic Network (Q-function).
    
    Estimates Q(s, a) - the expected return from taking action a in state s.
    Concatenates state and action as input.
    
    Architecture: [state, action] → 256 → ReLU → 128 → ReLU → 1 (Q-value)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden1: int = 256,
                 hidden2: int = 128):
        """
        Initialize DDPG critic network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden1: First hidden layer size (default: 256)
            hidden2: Second hidden layer size (default: 128)
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get Q-value.
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action: Action tensor of shape (batch, action_dim)
            
        Returns:
            Q-value tensor of shape (batch, 1)
        """
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) Agent.
    
    Proper DDPG implementation with:
    - Separate actor and critic networks
    - Target networks for both actor and critic
    - Experience replay buffer
    - Ornstein-Uhlenbeck noise for exploration
    - Off-policy learning
    
    Reference: Lillicrap et al. (2015) "Continuous control with deep RL"
    
    Attributes:
        actor: Main actor network (policy)
        actor_target: Target actor network
        critic: Main critic network (Q-function)
        critic_target: Target critic network
        replay_buffer: Experience replay buffer
        noise: OU noise process for exploration
        
    Example:
        >>> agent = DDPGAgent(state_dim=181, action_dim=30)
        >>> state = np.random.randn(181).astype(np.float32)
        >>> action = agent.get_action(state, add_noise=True)
        >>> action.shape
        (30,)
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden1: int = 256,
        hidden2: int = 128,
        buffer_size: int = 100000,
        tau: float = 0.001,
        device: Optional[torch.device] = None
    ):
        """
        Initialize DDPG agent with all components.
        
        Args:
            state_dim: State space dimension (default: 181)
            action_dim: Action space dimension (default: 30)
            hidden1: First hidden layer size (default: 256)
            hidden2: Second hidden layer size (default: 128)
            buffer_size: Replay buffer size (default: 100,000)
            tau: Soft update parameter for target networks (default: 0.001)
            device: Computation device (default: auto-detect)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.device = device if device is not None else get_device()
        self.agent_type = 'DDPG'
        
        # Create actor networks
        self.actor = DDPGActor(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Create critic networks
        self.critic = DDPGCritic(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim, hidden1, hidden2).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, buffer_size)
        
        # Exploration noise
        self.noise = OUNoise(action_dim)
    
    def get_action(self, state: Union[np.ndarray, torch.Tensor],
                   add_noise: bool = False) -> np.ndarray:
        """
        Get action from state for inference or exploration.
        
        Args:
            state: State vector of shape (state_dim,) or (batch, state_dim)
            add_noise: Whether to add OU noise for exploration (default: False)
            
        Returns:
            np.ndarray: Action vector of shape (action_dim,) in range [-1, 1]
            
        Example:
            >>> agent = DDPGAgent()
            >>> state = np.random.randn(181).astype(np.float32)
            >>> action = agent.get_action(state)
            >>> action.shape
            (30,)
            >>> np.all((action >= -1) & (action <= 1))
            True
        """
        self.actor.eval()
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(self.device)
            
            # Ensure batch dimension
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            action = self.actor(state).squeeze(0).cpu().numpy()
            
            if add_noise:
                noise = self.noise.sample()
                action = np.clip(action + noise, -1.0, 1.0)
            
            return action
    
    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """
        Soft update target network parameters.
        
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            target: Target network to update
            source: Source network to copy from
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def update_target_networks(self) -> None:
        """Soft update both actor and critic target networks."""
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
    
    def reset_noise(self) -> None:
        """Reset OU noise process."""
        self.noise.reset()
    
    def load_weights(self, path: str) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            path: Path to .pth checkpoint file
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        if 'actor_state_dict' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_target.load_state_dict(checkpoint['actor_state_dict'])
            
        if 'critic_state_dict' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.critic_target.load_state_dict(checkpoint['critic_state_dict'])
        
        self.actor.eval()
        self.critic.eval()
    
    def save_weights(self, path: str) -> None:
        """
        Save model weights to checkpoint.
        
        Args:
            path: Path to save .pth checkpoint file
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, path)
    
    def eval(self) -> None:
        """Set all networks to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()
    
    def train(self, mode: bool = True) -> None:
        """Set all networks to training mode."""
        self.actor.train(mode)
        self.critic.train(mode)
        self.actor_target.train(mode)
        self.critic_target.train(mode)


def construct_state(
    balance: float,
    prices: np.ndarray,
    holdings: np.ndarray,
    macd: np.ndarray,
    rsi: np.ndarray,
    cci: np.ndarray,
    adx: np.ndarray
) -> np.ndarray:
    """
    Construct the state vector for the DRL agent.
    
    State space: s_t = [b_t, p_t, h_t, M_t, R_t, C_t, X_t]
    Total dimensions: 1 + 30 + 30 + 30 + 30 + 30 + 30 = 181
    
    Args:
        balance: Current cash balance (scalar)
        prices: Stock prices array of shape (30,)
        holdings: Holdings array of shape (30,)
        macd: MACD indicator values of shape (30,)
        rsi: RSI indicator values of shape (30,)
        cci: CCI indicator values of shape (30,)
        adx: ADX indicator values of shape (30,)
        
    Returns:
        np.ndarray: State vector of shape (181,) as float32
        
    Example:
        >>> state = construct_state(
        ...     balance=1000000,
        ...     prices=np.ones(30) * 100,
        ...     holdings=np.zeros(30),
        ...     macd=np.zeros(30),
        ...     rsi=np.ones(30) * 50,
        ...     cci=np.zeros(30),
        ...     adx=np.ones(30) * 25
        ... )
        >>> state.shape
        (181,)
        >>> state[0]  # Balance
        1000000.0
    """
    state = np.concatenate([
        [balance],
        prices,
        holdings,
        macd,
        rsi,
        cci,
        adx
    ])
    return state.astype(np.float32)


def signal_gen(
    ohlcv_df: pd.DataFrame,
    agent_path: str = 'best_agent.pth',
    agent_type: str = 'PPO',
    turbulence_threshold: float = 140.0,
    h_max: int = 100,
    initial_cash: float = 1000000.0,
    tickers: Optional[List[str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate trading signals by simulating a virtual trading environment day-by-day.
    
    This function iterates through each trading day, calculates the current state,
    checks turbulence for risk management, and generates actions using the DRL agent.
    
    CRITICAL: Actions at time T are based ONLY on data available at time T.
    No future data is used. Execution happens at T+1 open (handled by simulation logic).
    
    The virtual environment maintains cash and holdings state to ensure realistic
    signal generation that respects position constraints.
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, open, high, low, close, volume]
        agent_path: Path to the trained agent weights file (default: 'best_agent.pth')
        agent_type: Type of agent - 'PPO', 'A2C', or 'DDPG' (default: 'PPO')
        turbulence_threshold: Threshold for turbulence-based risk-off mode (default: 140.0)
        h_max: Maximum number of shares per trade action (default: 100)
        initial_cash: Starting cash balance (default: 1000000.0)
        tickers: List of 30 ticker symbols (default: derived from ohlcv_df)
        **kwargs: Additional parameters for compatibility
        
    Returns:
        pd.DataFrame: Signal DataFrame with columns [time, ticker, action, quantity]
            - time: Trading date (datetime)
            - ticker: Stock ticker symbol (str)
            - action: 'BUY' or 'SELL' (str)
            - quantity: Number of shares (int, non-negative)
            
    Example:
        >>> from simicx.data_loader import get_trading_data
        >>> ohlcv = get_trading_data()
        >>> signals = signal_gen(
        ...     ohlcv,
        ...     agent_path='best_agent.pth',
        ...     agent_type='PPO',
        ...     turbulence_threshold=140.0,
        ...     h_max=100
        ... )
        >>> signals.columns.tolist()
        ['time', 'ticker', 'action', 'quantity']
        >>> signals['action'].isin(['BUY', 'SELL']).all()
        True
        
    Note:
        - Requires exactly 30 tickers for the 181-dimensional state space
        - If turbulence exceeds threshold, all positions are sold (risk-off mode)
        - Trades are constrained by available cash (buy) and holdings (sell)
        - The agent weights must be loaded for proper inference
    """
    # Get unique tickers if not provided
    if tickers is None:
        unique_tickers = sorted(ohlcv_df['ticker'].unique().tolist())
        # Ensure exactly 30 tickers for state dimension compatibility
        if len(unique_tickers) >= NUM_TICKERS:
            tickers = unique_tickers[:NUM_TICKERS]
        else:
            # Pad with duplicates if fewer than 30 (edge case handling)
            multiplier = (NUM_TICKERS // len(unique_tickers)) + 1
            tickers = (unique_tickers * multiplier)[:NUM_TICKERS]
    num_tickers = len(tickers)
    
    # Compute state dimension dynamically based on number of tickers
    # State: [balance(1), prices(n), holdings(n), macd(n), rsi(n), cci(n), adx(n)]
    # Total: 1 + 6*num_tickers
    state_dim = 1 + num_tickers * 6
    
    # Initialize agent based on type
    agent_classes: Dict[str, type] = {
        'PPO': PPOAgent,
        'A2C': A2CAgent,
        'DDPG': DDPGAgent
    }
    
    agent_class = agent_classes.get(agent_type.upper(), PPOAgent)
    agent = agent_class(state_dim=state_dim, action_dim=num_tickers)
    # Calculate state dimension dynamically: balance + 6 features per ticker
    # State: [balance, prices(n), holdings(n), macd(n), rsi(n), cci(n), adx(n)]
    actual_state_dim = 1 + 6 * num_tickers
    agent = agent_class(state_dim=actual_state_dim, action_dim=num_tickers)
    
    # Load trained weights if file exists
    if os.path.exists(agent_path):
        agent.load_weights(agent_path)
    
    # Calculate features for technical indicators
    features_df = calc_features(ohlcv_df, tickers)
    
    # Calculate turbulence index
    turbulence = calc_turbulence(ohlcv_df, tickers)
    
    # Pivot price data for easy access
    pivot_close = ohlcv_df.pivot(index='time', columns='ticker', values='close')
    pivot_close = pivot_close.reindex(columns=tickers)
    pivot_close = pivot_close.ffill().bfill()
    
    # Get sorted unique dates
    dates = sorted(ohlcv_df['time'].unique())
    
    # Initialize virtual portfolio
    cash = initial_cash
    holdings = np.zeros(num_tickers, dtype=float)
    
    # Store signals
    signals: List[Dict] = []
    
    # Skip initial period needed for indicator calculation (MACD needs 26 days, ADX needs 14)
    start_idx = 30
    
    for i in range(start_idx, len(dates)):
        current_date = dates[i]
        
        # Get current prices
        if current_date not in pivot_close.index:
            continue
        
        prices = pivot_close.loc[current_date].values.astype(float)
        
        # Handle NaN prices - replace with small positive value
        nan_mask = np.isnan(prices)
        if np.any(nan_mask):
            prices = np.where(nan_mask, 1.0, prices)
        
        # Ensure positive prices
        prices = np.maximum(prices, 0.01)
        
        # Get current turbulence value
        if current_date in turbulence.index:
            turb_val = turbulence.loc[current_date]
        else:
            turb_val = 0.0
        
        if pd.isna(turb_val):
            turb_val = 0.0
        
        # Risk management: if turbulence exceeds threshold, sell all positions
        if turb_val > turbulence_threshold:
            for j, ticker in enumerate(tickers):
                if holdings[j] > 0:
                    sell_qty = int(holdings[j])
                    signals.append({
                        'time': current_date,
                        'ticker': ticker,
                        'action': 'SELL',
                        'quantity': sell_qty
                    })
                    cash += sell_qty * prices[j]
                    holdings[j] = 0.0
            continue
        
        # Get technical indicators for current date
        if current_date not in features_df.index:
            continue
        
        feature_row = features_df.loc[current_date]
        
        # Extract indicators with defaults
        macd = np.array([
            feature_row.get(f'{t}_macd', 0.0) for t in tickers
        ], dtype=float)
        rsi = np.array([
            feature_row.get(f'{t}_rsi', 50.0) for t in tickers
        ], dtype=float)
        cci = np.array([
            feature_row.get(f'{t}_cci', 0.0) for t in tickers
        ], dtype=float)
        adx = np.array([
            feature_row.get(f'{t}_adx', 25.0) for t in tickers
        ], dtype=float)
        
        # Handle NaN values in indicators
        macd = np.nan_to_num(macd, nan=0.0, posinf=0.0, neginf=0.0)
        rsi = np.nan_to_num(rsi, nan=50.0, posinf=100.0, neginf=0.0)
        cci = np.nan_to_num(cci, nan=0.0, posinf=0.0, neginf=0.0)
        adx = np.nan_to_num(adx, nan=25.0, posinf=100.0, neginf=0.0)
        
        # Construct state vector
        state = construct_state(
            balance=cash,
            prices=prices,
            holdings=holdings.copy(),
            macd=macd,
            rsi=rsi,
            cci=cci,
            adx=adx
        )
        
        # Get action from agent (returns array in [-1, 1]^num_tickers)
        # These are portfolio weight signals, not execution orders
        action = agent.get_action(state)
        
        # Store raw signals (portfolio weights from agent)
        # trading_sim will convert these to actual trades
        for j, ticker in enumerate(tickers):
            signal_value = float(action[j])  # Portfolio weight in [-1, 1]
            
            signals.append({
                'time': current_date,
                'ticker': ticker,
                'signal': signal_value  # Raw agent output
            })
    
    # Create output DataFrame with signals
    if signals:
        signals_df = pd.DataFrame(signals)
    else:
        signals_df = pd.DataFrame(columns=['time', 'ticker', 'signal'])
    
    return signals_df


def load_config(config_path: str = 'simicx/alpha_config.json') -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary with keys like:
            - LIMITED_TICKERS: List of limited ticker symbols
            - FULL_TICKERS: List of full ticker symbols
            - TRAINING_END_DATE: Training end date string
            - TRADING_START_DATE: Trading start date string
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        
    Example:
        >>> config = load_config('simicx/alpha_config.json')
        >>> tickers = config['FULL_TICKERS']
        >>> len(tickers)
        30
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


# =============================================================================
# Test Functions
# =============================================================================


def simicx_test_get_device():
    """Test device detection function."""
    device = get_device()
    assert device is not None, "Device should not be None"
    assert isinstance(device, torch.device), "Should return torch.device"
    assert device.type in ['cuda', 'mps', 'cpu'], f"Invalid device type: {device.type}"
    print(f"Device detected: {device}")


def simicx_test_base_agent_and_variants():
    """Test BaseAgent initialization, forward pass, weight loading, and variants."""
    import tempfile
    from pathlib import Path
    
    # Test initialization
    agent = BaseAgent(state_dim=181, action_dim=30)
    assert agent.state_dim == 181, "State dim should be 181"
    assert agent.action_dim == 30, "Action dim should be 30"
    
    # Test forward pass with batch
    state = torch.randn(2, 181)
    action, value = agent(state.to(agent.device))
    assert action.shape == (2, 30), f"Action shape should be (2, 30), got {action.shape}"
    assert value.shape == (2, 1), f"Value shape should be (2, 1), got {value.shape}"
    
    # Test action bounds from Tanh
    assert torch.all(action >= -1) and torch.all(action <= 1), "Actions should be in [-1, 1]"
    
    # Test get_action with numpy input (single sample)
    state_np = np.random.randn(181).astype(np.float32)
    action_np = agent.get_action(state_np)
    assert action_np.shape == (30,), f"Action shape should be (30,), got {action_np.shape}"
    assert np.all((action_np >= -1) & (action_np <= 1)), "Actions should be in [-1, 1]"
    
    # Test save/load weights
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        weights_path = tmp_path / "test_agent.pth"
        
        # Save
        torch.save({'state_dict': agent.state_dict()}, weights_path)
        
        # Load into new agent
        new_agent = BaseAgent(state_dim=181, action_dim=30)
        new_agent.load_weights(str(weights_path))
        
        # Verify weights match
        for key in agent.state_dict():
            assert torch.allclose(
                agent.state_dict()[key].cpu(),
                new_agent.state_dict()[key].cpu()
            ), f"Weights mismatch for {key}"
        
        # Test error handling for missing file
        try:
            new_agent.load_weights("nonexistent.pth")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass
        
        # Test error handling for missing state_dict key
        bad_checkpoint = tmp_path / "bad_checkpoint.pth"
        torch.save({'model': agent.state_dict()}, bad_checkpoint)
        try:
            new_agent.load_weights(str(bad_checkpoint))
            assert False, "Should raise KeyError"
        except KeyError:
            pass
    
    # Test agent variants
    ppo = PPOAgent()
    a2c = A2CAgent()
    ddpg = DDPGAgent()
    
    assert ppo.agent_type == 'PPO', "PPO agent type mismatch"
    assert a2c.agent_type == 'A2C', "A2C agent type mismatch"
    assert ddpg.agent_type == 'DDPG', "DDPG agent type mismatch"
    
    # Verify all variants can forward
    test_state = torch.randn(1, 181)
    for variant_agent in [ppo, a2c, ddpg]:
        action, value = variant_agent(test_state.to(variant_agent.device))
        assert action.shape == (1, 30)
        assert value.shape == (1, 1)
    
    print("BaseAgent and variants tests passed!")


def simicx_test_calc_features_and_turbulence():
    """Test feature calculation and turbulence with synthetic data."""
    # Create synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=300, freq='D')
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    data = []
    for ticker in tickers:
        base_price = {'AAPL': 150, 'MSFT': 350, 'GOOGL': 140}.get(ticker, 100)
        for i, date in enumerate(dates):
            # Add some volatility clustering for turbulence testing
            vol = 2 + (i > 250) * 8  # Higher vol in last 50 days
            close = base_price + np.random.randn() * vol + i * 0.05
            close = max(close, 10)  # Ensure positive
            data.append({
                'time': date,
                'ticker': ticker,
                'open': close - np.random.rand() * 2,
                'high': close + np.random.rand() * 3,
                'low': close - np.random.rand() * 3,
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
    
    ohlcv_df = pd.DataFrame(data)
    
    # Test calc_features
    features_df = calc_features(ohlcv_df, tickers)
    
    # Verify all indicators are present
    for ticker in tickers:
        assert f'{ticker}_macd' in features_df.columns, f"Missing MACD for {ticker}"
        assert f'{ticker}_rsi' in features_df.columns, f"Missing RSI for {ticker}"
        assert f'{ticker}_cci' in features_df.columns, f"Missing CCI for {ticker}"
        assert f'{ticker}_adx' in features_df.columns, f"Missing ADX for {ticker}"
    
    # Verify no NaN values after processing
    assert not features_df.isnull().any().any(), "Features should have no NaN values"
    
    # Verify RSI is in valid range [0, 100]
    for ticker in tickers:
        rsi_col = features_df[f'{ticker}_rsi']
        assert (rsi_col >= 0).all() and (rsi_col <= 100).all(), \
            f"RSI should be in [0, 100] for {ticker}"
    
    # Test missing ticker handling
    features_with_missing = calc_features(ohlcv_df, tickers + ['UNKNOWN'])
    assert 'UNKNOWN_macd' in features_with_missing.columns
    assert features_with_missing['UNKNOWN_rsi'].iloc[0] == 50.0  # Default value
    
    # Test calc_turbulence
    turbulence = calc_turbulence(ohlcv_df, tickers, lookback=252)
    
    assert isinstance(turbulence, pd.Series), "Should return Series"
    assert len(turbulence) == len(dates), "Should have same length as dates"
    assert not turbulence.isna().all(), "Should have some non-NaN values"
    assert (turbulence >= 0).all(), "Turbulence should be non-negative"
    
    # Later periods with higher volatility should generally have higher turbulence
    late_turb = turbulence.iloc[-30:].mean()
    early_turb = turbulence.iloc[252:260].mean()
    print(f"Early turbulence: {early_turb:.2f}, Late turbulence: {late_turb:.2f}")
    
    print("Feature and turbulence calculation tests passed!")


def simicx_test_signal_gen_comprehensive():
    """Test signal generation with synthetic data."""
    import tempfile
    from pathlib import Path
    
    # Create synthetic OHLCV data for exactly 30 tickers
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    tickers = [f'TICK{i:02d}' for i in range(30)]
    
    data = []
    for ticker in tickers:
        base_price = 50 + np.random.rand() * 150
        for i, date in enumerate(dates):
            close = base_price + np.random.randn() * 2 + i * 0.05
            close = max(close, 10)  # Ensure positive price
            data.append({
                'time': date,
                'ticker': ticker,
                'open': close - np.random.rand(),
                'high': close + np.random.rand() * 2,
                'low': close - np.random.rand() * 2,
                'close': close,
                'volume': np.random.randint(1000000, 10000000)
            })
    
    ohlcv_df = pd.DataFrame(data)
    
    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        agent_path = tmp_path / "test_agent.pth"
        
        # Create and save a random agent
        agent = BaseAgent(state_dim=181, action_dim=30)
        torch.save({'state_dict': agent.state_dict()}, agent_path)
        
        # Generate signals with normal turbulence threshold
        signals = signal_gen(
            ohlcv_df,
            agent_path=str(agent_path),
            agent_type='PPO',
            turbulence_threshold=1000.0,  # High threshold to allow trading
            h_max=10,
            initial_cash=1000000.0,
            tickers=tickers
        )
        
        # Verify output format
        assert isinstance(signals, pd.DataFrame), "Output should be DataFrame"
        expected_cols = ['time', 'ticker', 'action', 'quantity']
        for col in expected_cols:
            assert col in signals.columns, f"Missing column: {col}"
        
        # Verify action values
        if len(signals) > 0:
            assert signals['action'].isin(['BUY', 'SELL']).all(), \
                "Actions should be 'BUY' or 'SELL'"
            assert (signals['quantity'] >= 0).all(), \
                "Quantities should be non-negative"
            assert (signals['quantity'] == signals['quantity'].astype(int)).all(), \
                "Quantities should be integers"
            assert all(t in tickers for t in signals['ticker'].unique()), \
                "All tickers should be from input list"
        
        # Test with very low turbulence threshold (should trigger sell-all)
        signals_risky = signal_gen(
            ohlcv_df,
            agent_path=str(agent_path),
            agent_type='A2C',
            turbulence_threshold=0.001,  # Very low threshold
            h_max=10,
            initial_cash=1000000.0,
            tickers=tickers
        )
        
        # Most signals should be SELL in risk-off mode
        if len(signals_risky) > 0:
            sell_ratio = (signals_risky['action'] == 'SELL').mean()
            print(f"Sell ratio in risk-off mode: {sell_ratio:.2%}")
        
        # Test with different agent types
        for agent_type in ['PPO', 'A2C', 'DDPG']:
            signals_type = signal_gen(
                ohlcv_df,
                agent_path=str(agent_path),
                agent_type=agent_type,
                turbulence_threshold=1000.0,
                h_max=5,
                initial_cash=500000.0,
                tickers=tickers
            )
            assert isinstance(signals_type, pd.DataFrame)
        
        # Test construct_state function
        state = construct_state(
            balance=1000000.0,
            prices=np.ones(30) * 100,
            holdings=np.zeros(30),
            macd=np.zeros(30),
            rsi=np.ones(30) * 50,
            cci=np.zeros(30),
            adx=np.ones(30) * 25
        )
        assert state.shape == (181,), f"State shape should be (181,), got {state.shape}"
        assert state.dtype == np.float32, "State should be float32"
        assert state[0] == 1000000.0, "First element should be balance"
    
    print("Signal generation comprehensive tests passed!")


if __name__ == '__main__':
    simicx_test_get_device()
    simicx_test_base_agent_and_variants()
    simicx_test_calc_features_and_turbulence()
    simicx_test_signal_gen_comprehensive()
    print("\nAll tests passed!")