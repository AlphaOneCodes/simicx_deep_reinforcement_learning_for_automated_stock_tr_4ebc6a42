"""
SimicX Trading Simulation Module

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Comprehensive backtesting engine for trading strategies with realistic fee modeling,
position tracking, constraint validation, and performance metrics.

Usage:
    from alpha_stream.tools.trading_sim import trading_sim
    
    pnl, pnl_details = trading_sim(
        trading_sheet=trading_df,
        initial_capital=1_000_000.0
    )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import warnings

@dataclass
class Position:
    """Represents a position in a single asset with FIFO (First-In-First-Out) cost basis tracking.
    
    This class manages both long and short positions for a single security ticker,
    tracking individual lots for accurate cost basis and P&L calculations.
    
    Position Types:
        - **Long positions**: positive quantity (bought stock, expecting price increase)
        - **Short positions**: negative quantity (borrowed and sold stock, expecting price decrease)
    
    FIFO Accounting:
        When selling/covering positions, the oldest lots are consumed first. This provides
        accurate cost basis tracking for tax purposes and precise realized P&L calculations.
    
    Attributes:
        ticker: The asset ticker (e.g., 'AAPL', 'DIA')
        lots: List of (quantity, price_per_share) tuples. Each tuple represents a lot:
              - quantity > 0 for long lots (shares owned)
              - quantity < 0 for short lots (shares owed)
              - price is the cost basis per share (including commission for buys)
    
    Example:
        >>> # Create a new position and add lots
        >>> pos = Position(ticker='AAPL')
        >>> pos.add(100, 150.0)  # Buy 100 shares at $150
        >>> pos.add(50, 155.0)   # Buy 50 more at $155
        >>> 
        >>> # Check position state
        >>> pos.quantity  # Total shares: 150
        150.0
        >>> pos.avg_cost  # Weighted average: (100*150 + 50*155) / 150 = 151.67
        151.66666666666666
        >>> pos.market_value  # Total cost basis: 100*150 + 50*155 = 22750
        22750.0
        >>> 
        >>> # Sell using FIFO - removes from oldest lot first
        >>> removed_qty, cost_basis = pos.remove(75)  # Sell 75 shares
        >>> removed_qty  # Actually removed
        75.0
        >>> cost_basis  # Cost basis of sold shares (75 * $150 from first lot)
        11250.0
        >>> pos.lots  # Remaining: 25 shares at $150, 50 shares at $155
        [(25, 150.0), (50, 155.0)]
        
        >>> # Short position example
        >>> short_pos = Position(ticker='TSLA')
        >>> short_pos.add(-100, 200.0)  # Short 100 shares at $200 (net proceeds per share)
        >>> short_pos.quantity  # Negative = owe shares
        -100
    """
    ticker: str
    lots: List[Tuple[float, float]] = field(default_factory=list)  # [(quantity, price), ...]
    # quantity > 0 for long lots, quantity < 0 for short lots
    
    @property
    def quantity(self) -> float:
        """Total quantity held across all lots.
        
        Returns:
            float: Sum of all lot quantities.
                   Positive for net long position (own shares).
                   Negative for net short position (owe shares).
                   Zero if no position or balanced.
        
        Example:
            >>> pos = Position(ticker='AAPL')
            >>> pos.add(100, 150.0)  # Long 100
            >>> pos.add(-30, 155.0)  # Short 30 (partially close)
            >>> pos.quantity
            70.0
        """
        return sum(q for q, _ in self.lots)
    
    @property
    def avg_cost(self) -> float:
        """Weighted average cost basis per share across all lots.
        
        Calculates the volume-weighted average price of all open lots.
        For long positions, this represents the average purchase price.
        For short positions, this represents the average sale price (proceeds received).
        
        Returns:
            float: Average cost per share. Returns 0.0 if no position held.
        
        Formula:
            avg_cost = Σ(quantity_i × price_i) / Σ(quantity_i)
        
        Example:
            >>> pos = Position(ticker='AAPL')
            >>> pos.add(100, 150.0)  # 100 × $150 = $15,000
            >>> pos.add(50, 160.0)   # 50 × $160 = $8,000
            >>> pos.avg_cost  # ($15,000 + $8,000) / 150 = $153.33
            153.33333333333334
            
            >>> # With no position
            >>> empty_pos = Position(ticker='MSFT')
            >>> empty_pos.avg_cost
            0.0
        """
        total_qty = self.quantity
        if total_qty == 0:
            return 0.0
        return sum(q * p for q, p in self.lots) / total_qty
    
    @property
    def market_value(self) -> float:
        """Total cost basis of the position (quantity × cost per share for each lot).
        
        This represents the total capital invested in the position. For unrealized
        P&L calculation, compare this with `quantity × current_market_price`.
        
        Returns:
            float: Sum of (quantity × price) for all lots.
                   Positive for long positions (capital deployed).
                   Negative for short positions (proceeds received).
        
        Note:
            This is NOT the current market value. Use with current prices:
            `unrealized_pnl = (quantity × current_price) - market_value`
        
        Example:
            >>> pos = Position(ticker='AAPL')
            >>> pos.add(100, 150.0)  # Cost: $15,000
            >>> pos.add(50, 160.0)   # Cost: $8,000
            >>> pos.market_value     # Total cost basis: $23,000
            23000.0
            
            >>> # Calculate unrealized P&L if current price is $170
            >>> current_price = 170.0
            >>> unrealized_pnl = (pos.quantity * current_price) - pos.market_value
            >>> unrealized_pnl  # 150 × $170 - $23,000 = $2,500
            2500.0
        """
        return sum(q * p for q, p in self.lots)
    
    def add(self, quantity: float, price_per_share: float) -> None:
        """Add shares to this position as a new lot.
        
        Creates a new lot entry in the position's lot list. Does NOT merge with
        existing lots to maintain accurate FIFO tracking.
        
        Position Direction:
            - quantity > 0: Adding long position (buying shares)
            - quantity < 0: Adding short position (shorting shares)
        
        Args:
            quantity: Number of shares to add. Can be positive (long) or negative (short).
                     Zero quantity is ignored (no lot created).
            price_per_share: Cost basis per share, should INCLUDE commission for accuracy.
                            For buys: (execution_price × quantity + commission) / quantity
                            For shorts: net_proceeds / quantity (after commission)
        
        Returns:
            None. Modifies the position in-place.
        
        Example:
            >>> pos = Position(ticker='AAPL')
            >>> 
            >>> # Add long position (100 shares at $150.15 including commission)
            >>> pos.add(100, 150.15)
            >>> pos.lots
            [(100, 150.15)]
            >>> 
            >>> # Add another lot at different price
            >>> pos.add(50, 155.0)
            >>> pos.lots  # Two separate lots maintained
            [(100, 150.15), (50, 155.0)]
            >>> 
            >>> # Add short position
            >>> short_pos = Position(ticker='TSLA')
            >>> short_pos.add(-100, 200.0)  # Short 100 at $200/share net proceeds
            >>> short_pos.quantity
            -100
            
        Note:
            - Each call creates a new lot, even if price matches existing lots
            - Zero quantity is silently ignored (useful for conditional adds)
            - For accurate P&L, include commission in price_per_share
        """
        if quantity != 0:
            self.lots.append((quantity, price_per_share))
    
    def remove(self, quantity: float, is_short_covering: bool = False) -> Tuple[float, float]:
        """Remove shares from position using FIFO (First-In-First-Out) accounting.
        
        Consumes lots in chronological order (oldest first) until the requested
        quantity is removed. Handles partial lot consumption correctly.
        
        FIFO Logic:
            1. Start with the oldest lot (first in the list)
            2. If lot quantity ≤ remaining to remove: consume entire lot
            3. If lot quantity > remaining: consume partial lot, update lot size
            4. Repeat until requested quantity is removed or no lots remain
        
        Args:
            quantity: Number of shares to remove (always pass a POSITIVE value).
                     The method handles both long and short position removal internally.
            is_short_covering: Optional flag indicating if this remove is for covering
                              a short position. Currently used for documentation purposes
                              but may affect future logic extensions.
        
        Returns:
            Tuple[float, float]: A tuple containing:
                - actual_quantity_removed: How many shares were actually removed.
                  May be less than requested if position is smaller.
                - cost_basis: Total cost basis of removed shares (quantity × price
                  summed across consumed lots). For short positions, this is negative
                  (representing proceeds received).
        
        Example:
            >>> pos = Position(ticker='AAPL')
            >>> pos.add(100, 150.0)  # Lot 1: 100 @ $150
            >>> pos.add(50, 160.0)   # Lot 2: 50 @ $160
            >>> 
            >>> # Remove 75 shares (FIFO takes from oldest lot first)
            >>> removed_qty, cost_basis = pos.remove(75)
            >>> removed_qty
            75.0
            >>> cost_basis  # 75 × $150 from first lot
            11250.0
            >>> pos.lots  # First lot reduced, second untouched
            [(25, 150.0), (50, 160.0)]
            >>> 
            >>> # Remove remaining 25 from first lot + 30 from second
            >>> removed_qty, cost_basis = pos.remove(55)
            >>> cost_basis  # (25 × $150) + (30 × $160) = 3750 + 4800 = 8550
            8550.0
            >>> pos.lots  # 20 shares remaining from second lot
            [(20, 160.0)]
            >>> 
            >>> # Try to remove more than available
            >>> removed_qty, cost_basis = pos.remove(50)
            >>> removed_qty  # Only 20 available
            20.0
            >>> pos.lots  # Position fully closed
            []
            
            >>> # Edge case: remove from empty position
            >>> empty_pos = Position(ticker='MSFT')
            >>> empty_pos.remove(100)
            (0.0, 0.0)
        
        Note:
            - For short positions (negative lot quantities), cost_basis represents
              the proceeds received when the short was opened (negative value).
            - The function gracefully handles partial fills when position is smaller
              than requested quantity.
            - Zero or negative quantity returns (0.0, 0.0) immediately.
        """
        if quantity <= 0:
            return 0.0, 0.0
        
        remaining = quantity
        total_cost_basis = 0.0
        total_removed = 0.0
        
        while remaining > 0 and self.lots:
            lot_qty, lot_price = self.lots[0]
            abs_lot_qty = abs(lot_qty)
            
            if abs_lot_qty <= remaining:
                # Consume entire lot
                total_cost_basis += lot_qty * lot_price
                total_removed += abs_lot_qty
                remaining -= abs_lot_qty
                self.lots.pop(0)
            else:
                # Partial lot consumption
                if lot_qty > 0:  # Long lot
                    total_cost_basis += remaining * lot_price
                    self.lots[0] = (lot_qty - remaining, lot_price)
                else:  # Short lot
                    total_cost_basis += -remaining * lot_price
                    self.lots[0] = (lot_qty + remaining, lot_price)
                total_removed += remaining
                remaining = 0
        
        return total_removed, total_cost_basis


@dataclass 
class Portfolio:
    """Portfolio state tracking with cash and positions.
    
    Manages the overall portfolio state including cash balance and all open positions
    (both long and short). Provides methods to calculate total portfolio value and
    track holdings across multiple assets.
    
    Attributes:
        cash: Current cash balance in the portfolio
        positions: Dictionary mapping ticker -> Position for all active positions
    
    Example:
        >>> portfolio = Portfolio(cash=100_000.0)
        >>> # Buy 100 shares of AAPL at $150
        >>> pos = portfolio.get_position('AAPL')
        >>> pos.add(100, 150.0)
        >>> portfolio.cash -= 100 * 150.0
        >>> 
        >>> # Check portfolio value
        >>> prices = {'AAPL': 155.0}
        >>> total_value = portfolio.get_total_value(prices)
        >>> print(f"Portfolio value: ${total_value:,.2f}")  # Cash + holdings
    """
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    
    def get_position(self, ticker: str) -> Position:
        """Get or create position for a given ticker.
        
        If the ticker doesn't exist in the portfolio, a new empty Position
        is created and added to the portfolio.
        
        Args:
            ticker: Asset ticker (e.g., 'AAPL', 'DIA')
            
        Returns:
            Position object for the specified ticker
        """
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        return self.positions[ticker]
    
    def get_holdings_value(self, prices: Dict[str, float]) -> float:
        """Calculate total market value of all holdings at current prices.
        
        For long positions (quantity > 0): positive value (asset)
        For short positions (quantity < 0): negative value (liability)
        
        Args:
            prices: Dictionary mapping ticker -> current market price
        
        Returns:
            Total market value of all holdings. Positive for net long positions,
            negative for net short positions.
            
        Example:
            >>> portfolio = Portfolio(cash=50_000)
            >>> # Long 100 AAPL, Short 50 TSLA
            >>> portfolio.get_position('AAPL').add(100, 150.0)
            >>> portfolio.get_position('TSLA').add(-50, 200.0)  # Short
            >>> prices = {'AAPL': 155.0, 'TSLA': 195.0}
            >>> holdings = portfolio.get_holdings_value(prices)
            >>> # = (100 * 155) + (-50 * 195) = 15,500 - 9,750 = 5,750
        """
        total = 0.0
        for ticker, pos in self.positions.items():
            if pos.quantity != 0 and ticker in prices:
                total += pos.quantity * prices[ticker]
        return total
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + holdings).
        
        This represents the liquidation value of the portfolio if all positions
        were closed at the given prices.
        
        Args:
            prices: Dictionary mapping ticker -> current market price
            
        Returns:
            Total portfolio value = cash + holdings_value
            
        Note:
            For portfolios with short positions, holdings_value may be negative,
            representing the liability from short positions.
        """
        return self.cash + self.get_holdings_value(prices)


def validate_trading_sheet(trading_sheet: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize trading sheet input.
    
    Performs comprehensive validation and normalization of trading signals:
    - Normalizes column names to lowercase
    - Validates required columns are present
    - Ensures action values are 'buy' or 'sell'
    - Converts time to datetime format
    - Validates numeric columns (quantity, price)
    - Removes invalid rows with warnings
    - Sorts by time chronologically
    
    Args:
        trading_sheet: DataFrame with trade instructions containing columns:
            - time: Trading timestamp (str or datetime)
            - ticker: Asset ticker symbol (str)
            - action: 'buy' or 'sell' (case-insensitive)
            - quantity: Trade quantity (numeric)
            - price: Target execution price (numeric)
        
    Returns:
        Validated DataFrame with normalized column names, sorted by time,
        with only valid rows retained
        
    Raises:
        ValueError: If required columns are missing or action values are invalid
        
    Warnings:
        Issues warnings when removing rows with invalid quantity/price values
        
    Example:
        >>> import pandas as pd
        >>> # Valid input
        >>> trades = pd.DataFrame({
        ...     'TIME': ['2024-01-02 09:30:00', '2024-01-02 10:00:00'],
        ...     'TICKER': ['AAPL', 'MSFT'],
        ...     'Action': ['BUY', 'SELL'],  # Case-insensitive
        ...     'Quantity': [100, 50],
        ...     'Price': [150.50, 380.25]
        ... })
        >>> validated = validate_trading_sheet(trades)
        >>> validated.columns.tolist()
        ['time', 'ticker', 'action', 'quantity', 'price']
        
        >>> # Invalid input - missing column
        >>> bad_trades = pd.DataFrame({'time': ['2024-01-02'], 'ticker': ['AAPL']})
        >>> validate_trading_sheet(bad_trades)  # Raises ValueError
        
        >>> # Invalid input - bad action value
        >>> bad_trades = pd.DataFrame({
        ...     'time': ['2024-01-02'], 'ticker': ['AAPL'],
        ...     'action': ['HOLD'], 'quantity': [100], 'price': [150]
        ... })
        >>> validate_trading_sheet(bad_trades)  # Raises ValueError
        
    Note:
        - Column names are case-insensitive and trimmed of whitespace
        - Empty DataFrames return an empty DataFrame with correct columns
        - Rows with NaN quantity or price are removed with a warning
        - All trades are sorted chronologically for proper execution order
    """
    required_cols = {'time', 'ticker', 'action', 'quantity', 'price'}
    
    if trading_sheet.empty:
        return pd.DataFrame(columns=['time', 'ticker', 'action', 'quantity', 'price'])
    
    # Normalize column names to lowercase
    trading_sheet = trading_sheet.copy()
    trading_sheet.columns = trading_sheet.columns.str.lower().str.strip()
    
    missing_cols = required_cols - set(trading_sheet.columns)
    if missing_cols:
        raise ValueError(f"Trading sheet missing required columns: {missing_cols}")
    
    # Validate action values
    valid_actions = {'buy', 'sell'}
    trading_sheet['action'] = trading_sheet['action'].str.lower().str.strip()
    invalid_actions = set(trading_sheet['action'].unique()) - valid_actions
    if invalid_actions:
        raise ValueError(f"Invalid action values: {invalid_actions}. Must be 'buy' or 'sell'.")
    
    # Ensure time is datetime
    trading_sheet['time'] = pd.to_datetime(trading_sheet['time'])
    
    # Ensure numeric columns
    trading_sheet['quantity'] = pd.to_numeric(trading_sheet['quantity'], errors='coerce')
    trading_sheet['price'] = pd.to_numeric(trading_sheet['price'], errors='coerce')
    
    # Remove invalid rows
    invalid_mask = trading_sheet['quantity'].isna() | trading_sheet['price'].isna()
    if invalid_mask.any():
        warnings.warn(f"Removed {invalid_mask.sum()} rows with invalid quantity/price values")
        trading_sheet = trading_sheet[~invalid_mask]
    
    # Sort by time
    trading_sheet = trading_sheet.sort_values('time').reset_index(drop=True)
    
    return trading_sheet[['time', 'ticker', 'action', 'quantity', 'price']]


def calculate_execution_price(
    target_price: float,
    action: str,
    slippage_rate: float = 0.0005,
    spread_rate: float = 0.0001
) -> float:
    """Calculate realistic execution price with slippage and bid-ask spread.
    
    Models the market impact of executing a trade by adjusting the target price
    for realistic market conditions. Both slippage and spread work against the trader.
    
    Cost Components:
        **Slippage**: The difference between expected and actual execution price due to
        market movement, order size impact, or latency. Always works against you.
        
        **Spread**: The difference between bid and ask prices. Buyers pay the ask (higher),
        sellers receive the bid (lower). The `spread_rate` represents half the spread.
    
    Formulas:
        - Buy:  execution_price = target_price × (1 + slippage_rate + spread_rate)
        - Sell: execution_price = target_price × (1 - slippage_rate - spread_rate)
    
    Args:
        target_price: The target/signal price from your trading strategy.
                     This is typically the close price or a calculated entry price.
        action: Trade direction - 'buy' or 'sell' (case-sensitive).
        slippage_rate: Slippage as a fraction of price. Default 0.0005 (0.05% or 5 bps).
                      Larger orders or less liquid assets typically have higher slippage.
        spread_rate: Half-spread as a fraction of price. Default 0.0001 (0.01% or 1 bp).
                    Represents half of the bid-ask spread. Full spread = 2 × spread_rate.
    
    Returns:
        float: Adjusted execution price after accounting for slippage and spread.
    
    Example:
        >>> # Buying at $100 with default rates
        >>> calculate_execution_price(100.0, 'buy')
        100.06  # = 100 × (1 + 0.0005 + 0.0001)
        
        >>> # Selling at $100 with default rates  
        >>> calculate_execution_price(100.0, 'sell')
        99.94  # = 100 × (1 - 0.0005 - 0.0001)
        
        >>> # High-impact trade with larger slippage
        >>> calculate_execution_price(50.0, 'buy', slippage_rate=0.002, spread_rate=0.001)
        50.15  # = 50 × (1 + 0.002 + 0.001) = 50 × 1.003
        
        >>> # Calculate round-trip cost (buy then sell same price)
        >>> buy_price = calculate_execution_price(100.0, 'buy')
        >>> sell_price = calculate_execution_price(100.0, 'sell')
        >>> round_trip_cost = buy_price - sell_price  # $0.12 per share
        >>> round_trip_cost_pct = (round_trip_cost / 100.0) * 100  # 0.12%
    
    Note:
        - Default values represent typical costs for liquid US equities
        - Crypto, forex, and less liquid assets often have higher rates
        - These costs are IN ADDITION to commissions
        - For limit orders with guaranteed fills, you may set both rates to 0
    """
    # Slippage always works against the trader
    # Spread: buy at ask (higher), sell at bid (lower)
    if action == 'buy':
        return target_price * (1 + slippage_rate + spread_rate)
    else:  # sell
        return target_price * (1 - slippage_rate - spread_rate)


def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02
) -> Dict[str, float]:
    """Calculate comprehensive performance metrics for trading strategy evaluation.
    
    Computes a wide range of industry-standard metrics to assess strategy performance,
    risk characteristics, and return quality. All calculations assume 252 trading days
    per year for annualization.
    
    Metric Categories:
        **Return Metrics**:
        - `total_return`: Cumulative return over the entire period
        - `annualized_return`: Geometric mean annual return
        - `avg_daily_return`: Arithmetic mean of daily returns
        - `volatility`: Annualized standard deviation of returns
        
        **Risk-Adjusted Metrics**:
        - `sharpe_ratio`: Excess return per unit of total risk
        - `sortino_ratio`: Excess return per unit of downside risk
        - `calmar_ratio`: Annualized return / max drawdown
        
        **Drawdown Analysis**:
        - `max_drawdown`: Largest peak-to-trough decline (negative value)
        - `avg_drawdown`: Average drawdown when in drawdown
        - `max_drawdown_duration`: Longest drawdown period in days
        
        **Win/Loss Statistics**:
        - `win_rate`: Percentage of positive return days
        - `profit_factor`: Gross profit / gross loss
        - `payoff_ratio`: Average win / average loss (absolute)
        
        **Distribution Characteristics**:
        - `skewness`: Return distribution asymmetry (positive = right tail)
        - `kurtosis`: Return distribution tail thickness (>3 = fat tails)
        - `var_95`: 5th percentile daily return (Value at Risk)
        - `cvar_95`: Mean of returns below VaR (Conditional VaR / Expected Shortfall)
    
    Args:
        returns: pandas Series of daily returns in decimal format.
                Example: 0.01 represents a +1% daily return, -0.02 represents a -2% return.
                Must be simple returns, not log returns.
        risk_free_rate: Annual risk-free rate in decimal format.
                       Default 0.02 represents 2% annual risk-free rate.
                       Used for Sharpe and Sortino ratio calculations.
    
    Returns:
        Dict[str, float]: Dictionary containing all computed metrics.
        Returns empty dict if returns series is empty or has fewer than 2 values.
    
    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> 
        >>> # Generate sample returns (252 trading days)
        >>> np.random.seed(42)
        >>> daily_returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
        >>> 
        >>> metrics = calculate_performance_metrics(daily_returns)
        >>> 
        >>> # Access individual metrics
        >>> print(f"Total Return: {metrics['total_return']*100:.2f}%")
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        >>> print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        >>> print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        
        >>> # Using with trading_sim results
        >>> pnl, details = trading_sim(trading_sheet=trades, ohlcv_path='data.csv')
        >>> if 'metrics' in details.attrs:
        ...     metrics = details.attrs['metrics']
        ...     print(f"Strategy Sharpe: {metrics['sharpe_ratio']:.2f}")
    
    Formulas:
        - Sharpe Ratio: (E[R] - Rf) × √252 / (σ × √252)
        - Sortino Ratio: (E[R] - Rf) × 252 / (σ_downside × √252)
        - Max Drawdown: min((cumulative - running_max) / running_max)
        - VaR 95%: 5th percentile of return distribution
        - CVaR 95%: E[R | R ≤ VaR_95]
    
    Note:
        - Returns 999.0 for infinite ratios (e.g., Sortino with no downside days)
        - Metrics are computed on non-NaN values only
        - Assumes continuous daily data (gaps may affect drawdown duration)
    """
    if returns.empty or len(returns) < 2:
        return {}
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return {}
    
    daily_rf = risk_free_rate / 252
    n_periods = len(returns_clean)
    
    try:
        # Return metrics
        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1
        avg_daily_return = returns_clean.mean()
        volatility = returns_clean.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        excess_returns = returns_clean - daily_rf
        sharpe_ratio = (excess_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (excess_returns.mean() * 252) / downside_std if downside_std > 0 else np.inf
        
        # Drawdown analysis
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        drawdown_periods = drawdown[drawdown < 0]
        avg_drawdown = drawdown_periods.mean() if len(drawdown_periods) > 0 else 0
        
        # Max drawdown duration
        in_drawdown = drawdown < 0
        max_dd_duration = 0
        current_duration = 0
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                current_duration = 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.inf
        
        # Win/loss statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]
        
        win_rate = len(positive_returns) / len(returns_clean)
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        gross_profit = positive_returns.sum() if len(positive_returns) > 0 else 0
        gross_loss = abs(negative_returns.sum()) if len(negative_returns) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Distribution characteristics
        skewness = float(returns_clean.skew())
        kurtosis = float(returns_clean.kurtosis())
        
        # VaR and CVaR
        var_95 = returns_clean.quantile(0.05)
        tail_returns = returns_clean[returns_clean <= var_95]
        cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'avg_daily_return': float(avg_daily_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio) if not np.isinf(sortino_ratio) else 999.0,
            'calmar_ratio': float(calmar_ratio) if not np.isinf(calmar_ratio) else 999.0,
            'max_drawdown': float(max_drawdown),
            'avg_drawdown': float(avg_drawdown),
            'max_drawdown_duration': int(max_dd_duration),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.0,
            'payoff_ratio': float(payoff_ratio) if not np.isinf(payoff_ratio) else 999.0,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'n_periods': n_periods
        }
        
    except Exception as e:
        warnings.warn(f"Error calculating metrics: {e}")
        return {}





def convert_signals_to_trades(
    signals: pd.DataFrame,
    signal_type: str,
    ohlcv_df: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
    max_position_pct: float = 0.25
) -> pd.DataFrame:
    """Convert raw alpha signals into a trading sheet.
    
    Args:
        signals: DataFrame with [time, ticker, signal]
        signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', or 'BINARY'
        ohlcv_df: Market data for price lookup
        initial_capital: For calculating quantities
        max_position_pct: Max allocation per asset
        
    Returns:
        DataFrame trading_sheet [time, ticker, action, quantity, price]
    """
    signals = signals.copy()
    signals['time'] = pd.to_datetime(signals['time'])
    
    # Standardize signal columns
    if 'signal' not in signals.columns:
        raise ValueError("Signals DataFrame must have 'signal' column")
        
    # Create price lookup
    price_map = {}
    for _, row in ohlcv_df.iterrows():
        price_map[(row['time'], row['ticker'])] = row['close']
        
    trades = []
    
    current_positions = {ticker: 0.0 for ticker in ohlcv_df['ticker'].unique()}
    
    # Sort signals by time
    signals = signals.sort_values('time')
    
    # Re-iterate with state
    for time, day_signals in signals.groupby('time'):
        current_prices = {}
        for ticker in day_signals['ticker']:
             p = price_map.get((time, ticker))
             if p: current_prices[ticker] = p

        # Skip if no prices
        if not current_prices:
            continue
            
        # 1. Calculate Target Weights
        day_signals = day_signals.copy()
        if signal_type == 'TARGET_WEIGHT':
            day_signals['target_weight'] = day_signals['signal']
        elif signal_type == 'ALPHA_SCORE':
            abs_sum = day_signals['signal'].abs().sum()
            scale = 1.0 / abs_sum if abs_sum > 0 else 0
            day_signals['target_weight'] = (day_signals['signal'] * scale).clip(-max_position_pct, max_position_pct)
        elif signal_type == 'BINARY':
            active = (day_signals['signal'] != 0).sum()
            w = min(1.0/active, max_position_pct) if active > 0 else 0
            day_signals['target_weight'] = day_signals['signal'] * w
        else:
            raise ValueError(f"Unknown signal_type: {signal_type}")
            
        # 2. Calculate Target Quantities
        target_qtys = {}
        for _, row in day_signals.iterrows():
            t, w = row['ticker'], row['target_weight']
            if t in current_prices and current_prices[t] > 0:
                val = initial_capital * w
                qty = val / current_prices[t]
                target_qtys[t] = qty
        
        # 3. Generate Rebalancing Orders
        active_tickers = set(target_qtys.keys()) | {t for t, q in current_positions.items() if q != 0}
        
        for ticker in active_tickers:
            tgt_qty = target_qtys.get(ticker, 0.0)
            cur_qty = current_positions.get(ticker, 0.0)
            
            delta = tgt_qty - cur_qty
            
            if abs(delta) > 1e-6: # Epsilon/tolerance
                trades.append({
                    'time': time,
                    'ticker': ticker,
                    'action': 'buy' if delta > 0 else 'sell',
                    'quantity': abs(delta),
                    'price': current_prices.get(ticker) # Execute at market price roughly
                })
                current_positions[ticker] = tgt_qty

    return pd.DataFrame(trades)


def trading_sim(
    trading_sheet: Optional[pd.DataFrame] = None,
    signals: Optional[pd.DataFrame] = None,
    signal_type: Optional[str] = None,
    initial_capital: float = 1_000_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    spread_rate: float = 0.0001,
    min_trade_value: float = 100.0,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_position_pct: float = 0.25,
    risk_free_rate: float = 0.02,
    ohlcv_tickers: Optional[List[str]] = None
) -> Tuple[float, pd.DataFrame]:
    """
    Comprehensive trading simulation/backtesting engine.
    
    Can accept EITHER:
    1. trading_sheet: Explicit buy/sell orders
    2. signals + signal_type: Raw signals to be converted to trades
    
    Args:
        trading_sheet: DataFrame with columns [time, ticker, action, quantity, price(optional)]
        signals: DataFrame with columns [time, ticker, signal]
        signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', 'BINARY'
        initial_capital: Starting cash (default $1,000,000)
        commission_rate: Commission as fraction of trade value (default 0.1%)
        slippage_rate: Slippage as fraction of price (default 0.05%)
        spread_rate: Half-spread as fraction of price (default 0.01%)
        min_trade_value: Minimum trade value threshold (default $100)
        allow_short: Allow short selling (default False)
        allow_leverage: Allow leverage/margin (default False)
        max_position_pct: Max single position as pct of portfolio (default 25%)
        risk_free_rate: Annual risk-free rate for metrics (default 2%)
        ohlcv_tickers: List of tickers to load OHLCV data for. If None, uses LIMITED_TICKERS.
                      Set this to match the tickers used in signal generation to avoid
                      "Ticker not found" errors.
        
    Returns:
        Tuple of:
        - pnl (float): Final portfolio P&L (final_value - initial_capital)
        - pnl_details (pd.DataFrame): Detailed trade-by-trade breakdown
            
    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If OHLCV file not found
    """
    # Load and validate inputs
    try:
        from simicx.data_loader import get_data, LIMITED_TICKERS, FULL_TICKERS
    except ImportError:
        try:
            from src.data_loader import get_data, LIMITED_TICKERS, FULL_TICKERS
        except ImportError:
            from data_loader import get_data, LIMITED_TICKERS, FULL_TICKERS

    # Determine which tickers to load
    if ohlcv_tickers is None:
        tickers_to_load = LIMITED_TICKERS
    else:
        tickers_to_load = ohlcv_tickers
    
    ohlcv_df = get_data(tickers=tickers_to_load)
    
    # --- Input Processing ---
    if signals is not None and not signals.empty:
        if trading_sheet is not None and not trading_sheet.empty:
            warnings.warn("Both 'trading_sheet' and 'signals' provided. Using 'trading_sheet' and ignoring 'signals'.")
        else:
            if not signal_type:
                raise ValueError("Must provide 'signal_type' when inputting raw 'signals'")
            
            trading_sheet = convert_signals_to_trades(
                signals=signals,
                signal_type=signal_type,
                ohlcv_df=ohlcv_df,
                initial_capital=initial_capital,
                max_position_pct=max_position_pct
            )
            
    if trading_sheet is None or trading_sheet.empty:
         return 0.0, pd.DataFrame(columns=[
            'time', 'ticker', 'action', 'quantity', 'target_price', 
            'executed_price', 'commission', 'slippage_cost', 'total_cost',
            'realized_pnl', 'cash_balance', 'holdings_value', 'portfolio_value', 'status', 'notes'
        ])

    trading_sheet = validate_trading_sheet(trading_sheet)
    
    # Initialize portfolio
    portfolio = Portfolio(cash=initial_capital)
    
    # Build price lookups: {(date, ticker): price}
    price_lookup: Dict[Tuple[datetime, str], float] = {}
    high_lookup: Dict[Tuple[datetime, str], float] = {}
    low_lookup: Dict[Tuple[datetime, str], float] = {}
    for _, row in ohlcv_df.iterrows():
        key = (row['time'], row['ticker'])
        price_lookup[key] = row['close']
        high_lookup[key] = row['high']
        low_lookup[key] = row['low']
    
    # Get all unique dates from OHLCV for daily valuation
    all_dates = sorted(ohlcv_df['time'].unique())
    
    # Get all tickers
    all_tickers = set(ohlcv_df['ticker'].unique())
    
    # Trade execution records
    trade_records = []
    
    # Daily portfolio values for returns calculation
    daily_values = []
    prev_value = initial_capital
    date_idx = 0
    
    # Process each trade
    for idx, trade in trading_sheet.iterrows():
        trade_time = trade['time']
        ticker = trade['ticker']
        action = trade['action']
        target_qty = trade['quantity']
        
        # Handle optional target price
        if 'price' in trade and pd.notna(trade['price']):
            target_price = trade['price']
        else:
            # Use market close price if no target provided
            target_price = price_lookup.get((trade_time, ticker), 0.0)
            if target_price == 0:
                # Will be rejected later but need a number
                target_price = 0.0
        
        # Record portfolio values for all dates BEFORE this trade date
        # This captures the portfolio state before any trades on trade_time
        while date_idx < len(all_dates) and all_dates[date_idx] < trade_time:
            date = all_dates[date_idx]
            prices = {t: price_lookup.get((date, t), 0) for t in all_tickers}
            port_value = portfolio.get_total_value(prices)
            if port_value > 0:  # Only record if we have valid prices
                daily_return = (port_value - prev_value) / prev_value if prev_value > 0 else 0
                daily_values.append({
                    'time': date,
                    'portfolio_value': port_value,
                    'return': daily_return
                })
                prev_value = port_value
            date_idx += 1
        
        # Validate ticker exists in data
        if ticker not in all_tickers:
            trade_records.append({
                'time': trade_time,
                'ticker': ticker,
                'action': action,
                'quantity': 0,
                'target_price': target_price,
                'executed_price': 0,
                'commission': 0,
                'slippage_cost': 0,
                'total_cost': 0,
                'realized_pnl': 0,
                'cash_balance': portfolio.cash,
                'holdings_value': portfolio.get_holdings_value(
                    {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                ),
                'portfolio_value': portfolio.get_total_value(
                    {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                ),
                'status': 'REJECTED',
                'notes': 'Ticker not found'
            })
            continue
        
        # Calculate execution price with slippage and spread
        exec_price = calculate_execution_price(
            target_price, action, slippage_rate, spread_rate
        )
        
        # Validate and adjust execution price to stay within day's high/low range
        day_high = high_lookup.get((trade_time, ticker), None)
        day_low = low_lookup.get((trade_time, ticker), None)
        
        original_exec_price = exec_price
        execution_note = ""
        
        if day_high is not None and day_low is not None:
             # If target_price was 0 (missing), we default to Close, which should be valid.
             # If explicit target was given, we check bounds.
             if 'price' in trade and pd.notna(trade['price']) and trade['price'] > 0:
                if action == 'buy':
                    # For buys: willing to pay up to exec_price
                    if exec_price > day_high:
                        # Target is above high - execute at high (better price for buyer)
                        exec_price = day_high
                        execution_note = f" (filled at day high ${day_high:.2f})"
                    elif exec_price < day_low:
                        # Target is below low - cannot execute (market never that cheap)
                        trade_records.append({
                            'time': trade_time,
                            'ticker': ticker,
                            'action': action,
                            'quantity': 0,
                            'target_price': target_price,
                            'executed_price': 0,
                            'commission': 0,
                            'slippage_cost': 0,
                            'total_cost': 0,
                            'realized_pnl': 0,
                            'cash_balance': portfolio.cash,
                            'holdings_value': portfolio.get_holdings_value(
                                {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                            ),
                            'portfolio_value': portfolio.get_total_value(
                                {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                            ),
                            'status': 'REJECTED',
                            'notes': f'Buy price ${exec_price:.2f} below day low ${day_low:.2f}'
                        })
                        continue
                else:  # sell
                    # For sells: willing to sell down to exec_price
                    if exec_price < day_low:
                        # Target is below low - execute at low (better price for seller)
                        exec_price = day_low
                        execution_note = f" (filled at day low ${day_low:.2f})"
                    elif exec_price > day_high:
                        # Target is above high - cannot execute (market never that high)
                        trade_records.append({
                            'time': trade_time,
                            'ticker': ticker,
                            'action': action,
                            'quantity': 0,
                            'target_price': target_price,
                            'executed_price': 0,
                            'commission': 0,
                            'slippage_cost': 0,
                            'total_cost': 0,
                            'realized_pnl': 0,
                            'cash_balance': portfolio.cash,
                            'holdings_value': portfolio.get_holdings_value(
                                {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                            ),
                            'portfolio_value': portfolio.get_total_value(
                                {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
                            ),
                            'status': 'REJECTED',
                            'notes': f'Sell price ${exec_price:.2f} above day high ${day_high:.2f}'
                        })
                        continue
        
        # Calculate trade value and fees
        trade_value = target_qty * exec_price
        commission = trade_value * commission_rate
        slippage_cost = abs(exec_price - target_price) * target_qty
        
        # Get current prices for portfolio valuation
        current_prices = {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
        current_portfolio_value = portfolio.get_total_value(current_prices)
        
        # Position and constraint checking
        position = portfolio.get_position(ticker)
        status = 'EXECUTED'
        notes = execution_note.strip() if execution_note else ''
        realized_pnl = 0.0
        executed_qty = target_qty
        
        if action == 'buy':
            total_cost = trade_value + commission
            
            # Check minimum trade value
            if trade_value < min_trade_value:
                status = 'REJECTED'
                notes = f'Below min trade value (${min_trade_value})'
                executed_qty = 0
                
            # Check cash constraint (no leverage)
            elif not allow_leverage and total_cost > portfolio.cash:
                # Adjust quantity to what we can afford
                # Formula: cash = qty * price * (1 + commission_rate)
                # So: qty = cash / (price * (1 + commission_rate))
                affordable_qty = portfolio.cash / (exec_price * (1 + commission_rate))
                trade_value = affordable_qty * exec_price
                
                if trade_value > min_trade_value:
                    executed_qty = affordable_qty
                    commission = trade_value * commission_rate
                    total_cost = trade_value + commission
                    status = 'EXECUTED'
                    notes = 'Reduced due to cash constraint'
                else:
                    status = 'REJECTED'
                    notes = 'Insufficient cash'
                    executed_qty = 0
                    
            # Check max position constraint using ABSOLUTE VALUES
            if executed_qty > 0:
                # Use current market price for existing position valuation
                current_price = current_prices.get(ticker, exec_price)
                
                # Calculate absolute position values (treats long/short symmetrically)
                current_abs_position_value = abs(position.quantity * current_price)
                new_position_qty = position.quantity + executed_qty
                new_abs_position_value = abs(new_position_qty * exec_price)
                
                # Use absolute portfolio value to handle net-short portfolios correctly
                abs_portfolio_value = abs(current_portfolio_value)
                
                # CRITICAL: Allow covering shorts (risk reduction) even if over limit
                # Only enforce limit when INCREASING risk
                is_covering_short = position.quantity < 0 and new_position_qty <= 0
                is_reducing_risk = new_abs_position_value < current_abs_position_value
                
                # Only check limit when NOT reducing risk
                if not (is_covering_short or is_reducing_risk):
                    if new_abs_position_value > abs_portfolio_value * max_position_pct:
                        # For flipping (short→long), closing short frees up space
                        if position.quantity < 0 and new_position_qty > 0:
                            # Flipping: can use full limit for new long position
                            max_long_value = abs_portfolio_value * max_position_pct
                            max_long_qty = max_long_value / exec_price
                            # Need to cover short first, then add long
                            total_needed = abs(position.quantity) + max_long_qty
                            executed_qty = total_needed
                            trade_value = executed_qty * exec_price
                            commission = trade_value * commission_rate
                            total_cost = trade_value + commission
                            status = 'EXECUTED'
                            notes = 'Reduced due to position limit (flip allowed)'
                        else:
                            # Just adding to existing long position
                            max_addl_value = (abs_portfolio_value * max_position_pct 
                                             - current_abs_position_value)
                            
                            if max_addl_value > min_trade_value:
                                executed_qty = max_addl_value / exec_price
                                trade_value = executed_qty * exec_price
                                commission = trade_value * commission_rate
                                total_cost = trade_value + commission
                                status = 'EXECUTED'
                                notes = 'Reduced due to position limit'
                            else:
                                status = 'REJECTED'
                                notes = 'Position limit reached'
                                executed_qty = 0
            
            if executed_qty > 0:
                # Check if we're covering a short position
                current_position_qty = position.quantity
                
                if current_position_qty < 0:
                    # We have a short position - buying covers it
                    abs_short_qty = abs(current_position_qty)
                    qty_covering_short = min(executed_qty, abs_short_qty)
                    
                    # Remove short position (cover it)
                    covered_qty, short_cost_basis = position.remove(qty_covering_short)
                    
                    # Realized P&L from covering short
                    # short_cost_basis = negative (net proceeds received after short commission)
                    # cost_to_cover = what we pay to buy back (including cover commission)
                    # P&L = proceeds_received - cost_to_cover
                    cover_value = covered_qty * exec_price
                    cover_commission = cover_value * commission_rate
                    cost_to_cover = cover_value + cover_commission
                    realized_pnl = -short_cost_basis - cost_to_cover
                    
                    # Update commission for short cover
                    commission = cover_commission
                    
                    # If buying more than short quantity, add remainder as long position
                    if executed_qty > abs_short_qty:
                        remaining_long_qty = executed_qty - abs_short_qty
                        remaining_cost = remaining_long_qty * exec_price
                        remaining_commission = remaining_cost * commission_rate
                        remaining_total = remaining_cost + remaining_commission
                        cost_per_share_with_commission = remaining_total / remaining_long_qty
                        position.add(remaining_long_qty, cost_per_share_with_commission)
                        commission += remaining_commission  # Total commission for cover + long
                        status = 'EXECUTED'
                        notes = 'Short cover + long'
                    
                    # Recalculate total_cost for the entire buy operation
                    total_cost = executed_qty * exec_price + commission
                else:
                    # Normal long position addition
                    # Include commission in cost basis: total_cost / quantity
                    cost_per_share_with_commission = total_cost / executed_qty
                    position.add(executed_qty, cost_per_share_with_commission)
                
                portfolio.cash -= total_cost
                
        else:  # sell
            # Determine how much we can sell
            current_position_qty = position.quantity
            
            if current_position_qty <= 0 and not allow_short:
                # No long position and shorting disabled
                status = 'REJECTED'
                notes = 'No position to sell (shorting disabled)'
                executed_qty = 0
            elif current_position_qty < target_qty and not allow_short:
                # Partial position and shorting disabled
                executed_qty = current_position_qty
                status = 'EXECUTED'
                notes = 'Reduced to available position'
            elif current_position_qty >= target_qty:
                # We have enough long position to cover the sale
                executed_qty = target_qty
            else:
                # current_position_qty < target_qty and allow_short=True
                # Sell what we have + create short for remainder
                executed_qty = target_qty
            
            # Check max position constraint for SHORT positions (same as BUY logic)
            if executed_qty > 0 and allow_short:
                # Calculate if this sell will create or increase a short position
                potential_short_qty = max(0, executed_qty - max(0, current_position_qty))
                
                if potential_short_qty > 0:
                    # This sell will create/increase short position - check limit
                    current_price = current_prices.get(ticker, exec_price)
                    current_abs_position_value = abs(position.quantity * current_price)
                    
                    # Calculate new position after sell (will be negative for short)
                    new_position_qty = current_position_qty - executed_qty
                    new_abs_position_value = abs(new_position_qty * exec_price)
                    
                    # Use absolute portfolio value to handle net-short portfolios
                    abs_portfolio_value = abs(current_portfolio_value)
                    
                    # CRITICAL: Allow closing longs (risk reduction) even if over limit
                    is_closing_long = current_position_qty > 0 and new_position_qty >= 0
                    is_reducing_risk = new_abs_position_value < current_abs_position_value
                    
                    # Only check limit when NOT reducing risk
                    if not (is_closing_long or is_reducing_risk):
                        if new_abs_position_value > abs_portfolio_value * max_position_pct:
                            # For flipping (long→short), closing long frees up space
                            if current_position_qty > 0 and new_position_qty < 0:
                                # Flipping: can use full limit for new short position
                                max_short_value = abs_portfolio_value * max_position_pct
                                max_short_qty = max_short_value / exec_price
                                # Need to close long first, then add short
                                total_needed = current_position_qty + max_short_qty
                                executed_qty = total_needed
                                status = 'EXECUTED'
                                notes = 'Reduced due to position limit (flip allowed)'
                            else:
                                # Just adding to existing short position
                                max_short_value = abs_portfolio_value * max_position_pct
                                max_addl_short_value = max_short_value - current_abs_position_value
                                
                                if max_addl_short_value > min_trade_value:
                                    # Can create some short, just not full amount
                                    max_short_qty = max_addl_short_value / exec_price
                                    executed_qty = max(0, current_position_qty) + max_short_qty
                                    status = 'EXECUTED'
                                    notes = 'Reduced due to position limit'
                                else:
                                    # Can only close existing long, no new short
                                    if current_position_qty > 0:
                                        executed_qty = current_position_qty
                                        status = 'EXECUTED'
                                        notes = 'Reduced to close long only (short limit reached)'
                                    else:
                                        status = 'REJECTED'
                                        notes = 'Position limit reached'
                                        executed_qty = 0
            
            if executed_qty > 0:
                # Case 1: Selling from long position
                if current_position_qty > 0:
                    qty_from_long = min(executed_qty, current_position_qty)
                    sold_qty, cost_basis = position.remove(qty_from_long)
                    
                    # Calculate proceeds and realized P&L from closing long
                    proceeds = sold_qty * exec_price
                    commission_long = proceeds * commission_rate
                    net_proceeds = proceeds - commission_long
                    realized_pnl = net_proceeds - cost_basis
                    portfolio.cash += net_proceeds
                    commission = commission_long
                    
                    # If selling more than we have, create short for remainder (only if allowed)
                    if executed_qty > current_position_qty:
                        if not allow_short:
                            # This should not happen due to earlier constraints, but guard anyway
                            raise ValueError(
                                f"Internal error: Attempted to create short position when allow_short=False. "
                                f"executed_qty={executed_qty}, current_position_qty={current_position_qty}"
                            )
                        
                        short_qty = executed_qty - current_position_qty
                        
                        # Short sale proceeds (minus commission)
                        short_proceeds = short_qty * exec_price
                        commission_short = short_proceeds * commission_rate
                        net_short_proceeds = short_proceeds - commission_short
                        
                        # Add negative quantity for short position
                        # Store net proceeds per share (includes commission cost)
                        net_proceeds_per_share = net_short_proceeds / short_qty
                        position.add(-short_qty, net_proceeds_per_share)
                        
                        portfolio.cash += net_short_proceeds
                        commission += commission_short
                        status = 'EXECUTED'
                        notes = 'Partial close + short'
                else:
                    # Case 2: Pure short sale (no existing long position)
                    if not allow_short:
                        # This should not happen due to earlier constraints, but guard anyway
                        raise ValueError(
                            f"Internal error: Attempted to create short position when allow_short=False. "
                            f"current_position_qty={current_position_qty}, executed_qty={executed_qty}"
                        )
                    
                    # Short sale proceeds
                    proceeds = executed_qty * exec_price
                    commission = proceeds * commission_rate
                    net_proceeds = proceeds - commission
                    
                    # Add negative quantity for short position
                    # Store net proceeds per share (includes commission cost)
                    net_proceeds_per_share = net_proceeds / executed_qty
                    position.add(-executed_qty, net_proceeds_per_share)
                    
                    portfolio.cash += net_proceeds
                    # No realized P&L yet (will realize when covering)
                    realized_pnl = 0.0
                
                trade_value = executed_qty * exec_price
        
        # Record trade
        current_prices = {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
        
        trade_records.append({
            'time': trade_time,
            'ticker': ticker,
            'action': action,
            'quantity': executed_qty,
            'target_price': target_price,
            'executed_price': exec_price if executed_qty > 0 else 0,
            'commission': commission if executed_qty > 0 else 0,
            'slippage_cost': slippage_cost if executed_qty > 0 else 0,
            'total_cost': trade_value + commission if executed_qty > 0 else 0,
            'realized_pnl': realized_pnl,
            'cash_balance': portfolio.cash,
            'holdings_value': portfolio.get_holdings_value(current_prices),
            'portfolio_value': portfolio.get_total_value(current_prices),
            'status': status,
            'notes': notes
        })
        
        # Check if this is the last trade on this date (or if next trade is on a different date)
        is_last_trade_on_date = (idx == len(trading_sheet) - 1) or \
                                (idx < len(trading_sheet) - 1 and 
                                 trading_sheet.iloc[idx + 1]['time'] != trade_time)
        
        # After processing all trades for this date, record the daily portfolio value
        # This captures the effect of ALL trades executed on trade_time
        if is_last_trade_on_date and date_idx < len(all_dates) and all_dates[date_idx] == trade_time:
            prices = {t: price_lookup.get((trade_time, t), 0) for t in all_tickers}
            port_value = portfolio.get_total_value(prices)
            if port_value > 0:  # Only record if we have valid prices
                daily_return = (port_value - prev_value) / prev_value if prev_value > 0 else 0
                daily_values.append({
                    'time': trade_time,
                    'portfolio_value': port_value,
                    'return': daily_return
                })
                prev_value = port_value
            date_idx += 1
    
    # Record portfolio values for any remaining dates after all trades
    while date_idx < len(all_dates):
        date = all_dates[date_idx]
        prices = {t: price_lookup.get((date, t), 0) for t in all_tickers}
        port_value = portfolio.get_total_value(prices)
        if port_value > 0:  # Only record if we have valid prices
            daily_return = (port_value - prev_value) / prev_value if prev_value > 0 else 0
            daily_values.append({
                'time': date,
                'portfolio_value': port_value,
                'return': daily_return
            })
            prev_value = port_value
        date_idx += 1
    
    # Create output DataFrames
    pnl_details = pd.DataFrame(trade_records)
    
    if pnl_details.empty:
        pnl_details = pd.DataFrame(columns=[
            'time', 'ticker', 'action', 'quantity', 'target_price', 
            'executed_price', 'commission', 'slippage_cost', 'total_cost',
            'realized_pnl', 'cash_balance', 'holdings_value', 'portfolio_value', 'status', 'notes'
        ])
    
    # Calculate final P&L
    if daily_values:
        final_value = daily_values[-1]['portfolio_value']
    else:
        final_value = initial_capital
    
    pnl = final_value - initial_capital
    
    # Calculate and attach performance metrics
    if daily_values:
        returns_series = pd.Series([d['return'] for d in daily_values])
        metrics = calculate_performance_metrics(returns_series, risk_free_rate)
        
        # Add metrics as attributes to the DataFrame
        pnl_details.attrs['metrics'] = metrics
        pnl_details.attrs['initial_capital'] = initial_capital
        pnl_details.attrs['final_value'] = final_value
        pnl_details.attrs['total_pnl'] = pnl
        pnl_details.attrs['total_return_pct'] = (pnl / initial_capital) * 100
    
    return pnl, pnl_details


def generate_performance_report(pnl_details: pd.DataFrame) -> str:
    """Generate a comprehensive, formatted performance report from trading simulation results.
    
    Creates a professional text-based report suitable for logging, display, or export.
    The report includes capital summary, return metrics, risk-adjusted ratios,
    drawdown analysis, win/loss statistics, and trade execution summary.
    
    Report Sections:
        **CAPITAL SUMMARY**: Initial capital, final value, total P&L, and return %
        
        **RETURN METRICS**: Annualized return, volatility, average daily return
        
        **RISK-ADJUSTED METRICS**: Sharpe, Sortino, and Calmar ratios
        
        **DRAWDOWN ANALYSIS**: Maximum drawdown, average drawdown, max duration
        
        **WIN/LOSS STATISTICS**: Win rate, profit factor, payoff ratio
        
        **RISK METRICS**: VaR 95%, CVaR 95%, skewness, kurtosis
        
        **TRADE SUMMARY**: Total trades, executed/rejected counts, commissions, realized P&L
    
    Args:
        pnl_details: DataFrame returned by `trading_sim()` function. Must have `.attrs`
                    dictionary containing 'metrics', 'initial_capital', 'final_value',
                    and 'total_pnl' keys. These are automatically attached by trading_sim.
    
    Returns:
        str: A multi-line formatted string containing the complete performance report.
             Returns "No performance metrics available." if pnl_details lacks metrics.
    
    Example:
        >>> import pandas as pd
        >>> 
        >>> # Create trading signals
        >>> trades = pd.DataFrame({
        ...     'time': ['2024-01-02', '2024-01-15', '2024-02-01'],
        ...     'ticker': ['AAPL', 'AAPL', 'AAPL'],
        ...     'action': ['buy', 'sell', 'buy'],
        ...     'quantity': [100, 100, 50],
        ...     'price': [150.0, 155.0, 152.0]
        ... })
        >>> 
        >>> # Run simulation and generate report
        >>> pnl, details = trading_sim(
        ...     trading_sheet=trades,
        ...     ohlcv_path='market_data/ohlcv.csv',
        ...     initial_capital=100_000
        ... )
        >>> 
        >>> # Generate and print report
        >>> report = generate_performance_report(details)
        >>> print(report)
        ================================================================================
                             TRADING SIMULATION REPORT
        ================================================================================
        
        CAPITAL SUMMARY
        ---------------
          Initial Capital:    $     100,000.00
          Final Value:        $     100,450.00
          ...
        
        >>> # Save report to file
        >>> with open('backtest_report.txt', 'w') as f:
        ...     f.write(report)
    
    Note:
        - Requires pnl_details to have metrics attached via .attrs dictionary
        - All percentages are displayed with proper formatting
        - Currency values use comma separators and 2 decimal places
        - Infinite ratios (e.g., from no losing trades) display as 999.000
    """
    if not hasattr(pnl_details, 'attrs') or 'metrics' not in pnl_details.attrs:
        return "No performance metrics available."
    
    metrics = pnl_details.attrs['metrics']
    initial = pnl_details.attrs.get('initial_capital', 0)
    final = pnl_details.attrs.get('final_value', 0)
    pnl = pnl_details.attrs.get('total_pnl', 0)
    
    report = f"""
================================================================================
                         TRADING SIMULATION REPORT
================================================================================

CAPITAL SUMMARY
---------------
  Initial Capital:    ${initial:>15,.2f}
  Final Value:        ${final:>15,.2f}
  Total P&L:          ${pnl:>15,.2f}
  Total Return:       {(pnl/initial)*100:>14.2f}%

RETURN METRICS
--------------
  Annualized Return:  {metrics.get('annualized_return', 0)*100:>14.2f}%
  Volatility:         {metrics.get('volatility', 0)*100:>14.2f}%
  Avg Daily Return:   {metrics.get('avg_daily_return', 0)*100:>14.4f}%

RISK-ADJUSTED METRICS
---------------------
  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):>14.3f}
  Sortino Ratio:      {metrics.get('sortino_ratio', 0):>14.3f}
  Calmar Ratio:       {metrics.get('calmar_ratio', 0):>14.3f}

DRAWDOWN ANALYSIS
-----------------
  Max Drawdown:       {metrics.get('max_drawdown', 0)*100:>14.2f}%
  Avg Drawdown:       {metrics.get('avg_drawdown', 0)*100:>14.2f}%
  Max DD Duration:    {metrics.get('max_drawdown_duration', 0):>14} days

WIN/LOSS STATISTICS
-------------------
  Win Rate:           {metrics.get('win_rate', 0)*100:>14.2f}%
  Profit Factor:      {metrics.get('profit_factor', 0):>14.3f}
  Payoff Ratio:       {metrics.get('payoff_ratio', 0):>14.3f}

RISK METRICS
------------
  VaR (95%):          {metrics.get('var_95', 0)*100:>14.4f}%
  CVaR (95%):         {metrics.get('cvar_95', 0)*100:>14.4f}%
  Skewness:           {metrics.get('skewness', 0):>14.3f}
  Kurtosis:           {metrics.get('kurtosis', 0):>14.3f}

TRADE SUMMARY
-------------
  Total Trades:       {len(pnl_details):>14}
  Executed Trades:    {len(pnl_details[pnl_details['status'] == 'EXECUTED']):>14}
  Rejected Trades:    {len(pnl_details[pnl_details['status'] == 'REJECTED']):>14}
  Total Commission:   ${pnl_details['commission'].sum():>15,.2f}
  Total Realized PnL: ${pnl_details['realized_pnl'].sum():>15,.2f}

================================================================================
"""
    return report


if __name__ == '__main__':
    # Quick self-test
    print("Trading Simulation Module - Self Test")
    print("=" * 50)
    
    # Check if test data exists
    import os
    test_data_path = 'alpha_stream/shared_files/stock_daily_ohlcv.csv'
    
    if not os.path.exists(test_data_path):
        # Try alternative paths
        test_data_path = 'market_data/stock_daily_top50.csv'
    
    if not os.path.exists(test_data_path):
        test_data_path = 'data/test_ohlcv_data.csv'
    
    if os.path.exists(test_data_path):
        # Monkeypatch data_loader path for testing
        import alpha_stream.shared_files.data_loader as dl
        dl._OHLCV_PATH = test_data_path
        
        # Create a simple trading sheet for testing
        try:
            df = dl.get_data()
            
            # Get first ticker and some dates
            first_ticker = df['ticker'].iloc[0]
            dates = df[df['ticker'] == first_ticker]['time'].iloc[:10].tolist()
            prices = df[df['ticker'] == first_ticker]['close'].iloc[:10].tolist()
        except (ValueError, KeyError) as e:
            print(f"\n⚠ Test data doesn't have required columns: {e}")
            print("  Expected columns: time, ticker, open, high, low, close, volume")
            import sys
            sys.exit(0)
        
        # Create simple buy-then-sell trades
        trading_sheet = pd.DataFrame({
            'time': [dates[0], dates[5]],
            'ticker': [first_ticker, first_ticker],
            'action': ['buy', 'sell'],
            'quantity': [100.0, 100.0],
            'price': [prices[0], prices[5]]
        })
        
        print(f"\nTest ticker: {first_ticker}")
        print(f"Buy price: ${prices[0]:,.2f}")
        print(f"Sell price: ${prices[5]:,.2f}")
        print(f"Expected gross P&L: ${(prices[5] - prices[0]) * 100:,.2f}")
        
        # Run simulation
        pnl, details = trading_sim(
            trading_sheet=trading_sheet,
            initial_capital=1_000_000.0
        )
        
        print(f"\nSimulation Results:")
        print(f"  Final P&L: ${pnl:,.2f}")
        print(f"  Trades executed: {len(details)}")
        print(f"\nTrade Details:")
        print(details.to_string())
        
        if hasattr(details, 'attrs') and 'metrics' in details.attrs:
            print("\n" + generate_performance_report(details))
        
        print("\n✓ Self-test completed successfully!")
    else:
        print(f"\n⚠ Test data not found at {test_data_path}")
        print("  Run from project root or provide correct path.")


def simicx_test_trading_sim():
    import pandas as pd
    
    # Create simple trading sheet using a ticker from LIMITED_TICKERS
    trades = pd.DataFrame({
        'time': ['2024-01-02'],  # Use 2024 date for consistency with training data
        'ticker': ['SPY'],       # SPY is in LIMITED_TICKERS
        'action': ['buy'],
        'quantity': [100],
        'price': [475.0]         # Approximate SPY price in early 2024
    })
    
    # Run sim
    # Note: Uses LIMITED_TICKERS from data_loader for OHLCV data
    # We use a small capital to avoid margin issues
    result_pnl, result_df = trading_sim(trading_sheet=trades, initial_capital=100000.0)
    
    assert isinstance(result_pnl, float)
    assert not result_df.empty
    assert 'realized_pnl' in result_df.columns

