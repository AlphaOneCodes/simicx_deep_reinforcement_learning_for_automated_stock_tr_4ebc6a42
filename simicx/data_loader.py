"""
SimicX Data Loader Module (Database Version)

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Centralized OHLCV data loading from MongoDB with strict temporal controls
for alpha discovery and backtesting.

Key Features:
- Strict train/test split: Training ≤ 2024-12-31, Trading ≥ 2025-01-01
- Date alignment across tickers (consistent coverage)
- 2-phase testing support: LIMITED → FULL tickers
- Multi-asset extensibility (not equity-specific)

Usage:
    from data_loader import get_training_data, get_trading_data
    
    # For tune.py (hyperparameter optimization)
    train_df = get_training_data(LIMITED_TICKERS, years_back=3)
    
    # For main.py (backtesting)
    trade_df = get_trading_data(FULL_TICKERS)
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union
from datetime import datetime, timedelta
from pymongo import MongoClient, ASCENDING
import threading

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load configuration
try:
    CONFIG_PATH = Path(__file__).parent / 'alpha_config.json'
    with open(CONFIG_PATH, 'r') as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"WARNING: Could not load alpha_config.json: {e}. Using fallback defaults.")
    CONFIG = {}

# Ticker configurations for 2-phase testing
LIMITED_TICKERS = CONFIG.get("LIMITED_TICKERS", ["SPY", "NVDA", "QQQ", "AAPL", "MSFT", "AMZN"])

FULL_TICKERS = CONFIG.get("FULL_TICKERS", [
    "SPY", "NVDA", "QQQ", "AAPL", "MSFT", "AMZN", "IWM", "IVV", "GOOGL", "AMD",
    "GOOG", "TLT", "NFLX", "UNH", "JPM", "V", "MU", "HYG", "BA", "WMT",
    "XLF", "XOM", "LQD", "CVX", "DIA", "CSCO", "BAC", "PG", "GLD", "PFE"
])

# Temporal boundaries - CRITICAL for preventing data leaking
# HARDCODED: These dates are immutable to ensure data leakage prevention
# DO NOT load from config - these are fundamental constraints that must never change
TRAINING_END_DATE = "2024-12-31"   # Training/tuning data ends here (inclusive) - IMMUTABLE
TRADING_START_DATE = "2025-01-01"  # Trading simulation starts here (inclusive) - IMMUTABLE

# Validate config dates match hardcoded values (if config exists)
if CONFIG:
    config_training_end = CONFIG.get("TRAINING_END_DATE")
    config_trading_start = CONFIG.get("TRADING_START_DATE")
    if config_training_end and config_training_end != TRAINING_END_DATE:
        print(f"WARNING: Config TRAINING_END_DATE ({config_training_end}) differs from hardcoded value ({TRAINING_END_DATE}). Using hardcoded value.")
    if config_trading_start and config_trading_start != TRADING_START_DATE:
        print(f"WARNING: Config TRADING_START_DATE ({config_trading_start}) differs from hardcoded value ({TRADING_START_DATE}). Using hardcoded value.")

# Training history limits
TRAINING_YEARS_BACK_LIMITED = CONFIG.get("TRAINING_YEARS_BACK_LIMITED", 3)
TRAINING_YEARS_BACK_FULL = CONFIG.get("TRAINING_YEARS_BACK_FULL", None)

# MongoDB connection from environment variables (set by alpha_agent.py)
# CRITICAL: Strict enforcement - no fallback allowed
MONGODB_URI = os.environ.get("SIMICX_MONGODB_URI")
MONGODB_DATABASE = os.environ.get("SIMICX_MONGODB_DATABASE")
OHLCV_COLLECTION = "US_stock_etf_daily_ohlcv"

if not MONGODB_URI:
    # HARD STOP: Cannot proceed without database
    raise ValueError("CRITICAL: SIMICX_MONGODB_URI environment variable not set. Database connection required.")

if not MONGODB_DATABASE:
    raise ValueError("CRITICAL: SIMICX_MONGODB_DATABASE environment variable not set.")

# =============================================================================
# MONGODB CONNECTION (Thread-safe singleton)
# =============================================================================

_mongo_client = None
_mongo_lock = threading.Lock()


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client connection (thread-safe singleton).
    
    Returns:
        MongoClient instance with connection pooling.
    
    Raises:
        RuntimeError: If connection to MongoDB fails (Fail Fast).
    
    Example:
        >>> client = get_mongo_client()
        >>> db = client[MONGODB_DATABASE]
    """
    global _mongo_client
    if _mongo_client is None:
        with _mongo_lock:
            if _mongo_client is None:
                try:
                    client = MongoClient(
                        MONGODB_URI,
                        serverSelectionTimeoutMS=5000,  # 5s timeout (Fail Fast)
                        connectTimeoutMS=5000,
                        socketTimeoutMS=30000,
                        maxPoolSize=50,
                        minPoolSize=5,
                        retryWrites=True,
                        retryReads=True
                    )
                    # Test connection immediately
                    client.admin.command('ping')
                    _mongo_client = client
                except Exception as e:
                    # HARD STOP: Connection failed
                    raise RuntimeError(f"CRITICAL: Failed to connect to MongoDB at {MONGODB_URI}. Error: {e}")
                
    return _mongo_client


def get_collection():
    """Get OHLCV collection instance.
    
    Returns:
        MongoDB collection for OHLCV data.
    """
    client = get_mongo_client()
    db = client[MONGODB_DATABASE]
    return db[OHLCV_COLLECTION]


# =============================================================================
# CORE DATA FUNCTIONS
# =============================================================================

def get_tickers() -> List[str]:
    """Get list of unique ticker symbols available in the database.
    
    Returns:
        List[str]: Sorted list of available ticker symbols.
    
    Example:
        >>> tickers = get_tickers()
        >>> print(f"Found {len(tickers)} tickers")
        >>> 'SPY' in tickers
        True
    """
    collection = get_collection()
    tickers = collection.distinct("symbol")
    return sorted(tickers)


def get_date_range(ticker: str) -> Tuple[datetime, datetime]:
    """Get the date range (start and end dates) for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL').
    
    Returns:
        Tuple[datetime, datetime]: (start_date, end_date) for the ticker.
    
    Raises:
        ValueError: If the ticker is not found in the database.
    
    Example:
        >>> start, end = get_date_range('SPY')
        >>> print(f"SPY data: {start.date()} to {end.date()}")
    """
    collection = get_collection()
    
    # Get min date
    first = collection.find_one(
        {"symbol": ticker},
        sort=[("time", ASCENDING)]
    )
    if not first:
        raise ValueError(f"Ticker '{ticker}' not found in database")
    
    # Get max date
    last = collection.find_one(
        {"symbol": ticker},
        sort=[("time", -1)]
    )
    
    return first['time'], last['time']


def get_data(
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    phase: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    align_dates: bool = True
) -> pd.DataFrame:
    """Get OHLCV data with optional filtering by ticker(s), phase, and date range.
    
    Args:
        ticker: Single ticker symbol (e.g., 'SPY'). Mutually exclusive with tickers/phase.
        tickers: List of ticker symbols. Mutually exclusive with ticker/phase.
        phase: 'limited' (LIMITED_TICKERS) or 'full' (FULL_TICKERS). Overrides ticker/tickers.
        start_date: Start date (inclusive) in 'YYYY-MM-DD' format.
        end_date: End date (inclusive) in 'YYYY-MM-DD' format.
        align_dates: If True, only return dates where ALL tickers have data.
    
    Returns:
        pd.DataFrame: OHLCV data with columns: time, ticker, open, high, low, close, volume
    
    Raises:
        ValueError: If neither phase, ticker nor tickers is provided.
    """
    # Handle phase argument first config-driven
    if phase:
        if phase.lower() == 'limited':
            ticker_list = LIMITED_TICKERS
        elif phase.lower() == 'full':
            ticker_list = FULL_TICKERS
        else:
            raise ValueError(f"Invalid phase '{phase}'. Must be 'limited' or 'full'.")
    
    # Handle manual ticker/tickers parameter
    elif ticker is not None:
        ticker_list = [ticker]
    elif tickers is not None:
        ticker_list = tickers
    else:
        raise ValueError("Either 'phase', 'ticker' or 'tickers' must be provided")
    
    collection = get_collection()
    
    # Build query
    query = {"symbol": {"$in": ticker_list}}
    
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        query["time"] = query.get("time", {})
        query["time"]["$gte"] = start_dt.to_pydatetime()
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        query["time"] = query.get("time", {})
        query["time"]["$lte"] = end_dt.to_pydatetime()
    
    # Fetch data
    cursor = collection.find(
        query,
        {
            "time": 1, "symbol": 1, 
            "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
            "_id": 0
        }
    ).sort([("time", ASCENDING), ("symbol", ASCENDING)])
    
    # Convert to DataFrame
    data = list(cursor)
    if not data:
        return pd.DataFrame(columns=['time', 'ticker', 'open', 'high', 'low', 'close', 'volume'])
    
    df = pd.DataFrame(data)
    
    # Rename 'symbol' to 'ticker' for consistency with existing code
    df = df.rename(columns={'symbol': 'ticker'})
    
    # Ensure proper types
    df['time'] = pd.to_datetime(df['time'])
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
    
    # Date alignment: only keep dates where ALL tickers have data
    if align_dates and len(ticker_list) > 1:
        date_counts = df.groupby('time')['ticker'].nunique()
        valid_dates = date_counts[date_counts == len(ticker_list)].index
        df = df[df['time'].isin(valid_dates)]
    
    # Sort and reset index
    df = df.sort_values(['time', 'ticker']).reset_index(drop=True)
    
    return df[['time', 'ticker', 'open', 'high', 'low', 'close', 'volume']]


def get_training_data(
    tickers: Optional[List[str]] = None,
    phase: Optional[str] = None,
    years_back: Optional[int] = None,
    align_dates: bool = True
) -> pd.DataFrame:
    """Get training/tuning data (all data up to and including 2024-12-31).
    
    CRITICAL: This function ensures NO data after 2024-12-31 is included.
    
    Args:
        tickers: List of ticker symbols. Defaults to FULL_TICKERS if phase not set.
        phase: 'limited' or 'full'. Sets tickers and default years_back from config.
        years_back: Override default years_back.
        align_dates: If True, only return dates where ALL tickers have data.
    
    Returns:
        pd.DataFrame: Training OHLCV data.
    """
    if phase:
        if phase.lower() == 'limited':
            cols = LIMITED_TICKERS
            if years_back is None:
                years_back = TRAINING_YEARS_BACK_LIMITED
        elif phase.lower() == 'full':
            cols = FULL_TICKERS
            if years_back is None:
                years_back = TRAINING_YEARS_BACK_FULL
        else:
            raise ValueError(f"Invalid phase '{phase}'")
        tickers = cols
    elif tickers is None:
        tickers = FULL_TICKERS.copy()
    
    # Calculate start date if years_back is specified
    if years_back is not None:
        # Training ends at 2024-12-31, so start N years before 2025-01-01
        start_date = f"{2025 - years_back}-01-01"
    else:
        start_date = None  # Use all available data
    
    return get_data(
        tickers=tickers,
        start_date=start_date,
        end_date=TRAINING_END_DATE,
        align_dates=align_dates
    )


def get_trading_data(
    tickers: Optional[List[str]] = None,
    align_dates: bool = True
) -> pd.DataFrame:
    """Get trading simulation data (all data from start of 2025 onwards).
    
    CRITICAL: This function ensures ONLY data from 2025-Jan-01 onwards is returned,
    which should be used for backtesting and performance reporting.
    
    Args:
        tickers: List of ticker symbols. Defaults to FULL_TICKERS.
        align_dates: If True, only return dates where ALL tickers have data.
    
    Returns:
        pd.DataFrame: Trading OHLCV data (2025 onwards).
    
    Example:
        >>> # Trading data for backtesting
        >>> trade_df = get_trading_data(FULL_TICKERS)
        >>> trade_df['time'].min()  # Should be >= 2025-Jan-01
    """
    if tickers is None:
        tickers = FULL_TICKERS.copy()
    
    return get_data(
        tickers=tickers,
        start_date=TRADING_START_DATE,
        end_date=None,  # Up to latest available data
        align_dates=align_dates
    )


# =============================================================================
# CONVENIENCE ALIASES (for backward compatibility)
# =============================================================================

def get_ohlcv(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """Convenience alias for get_data() - get OHLCV data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol.
        start_date: Optional start date.
        end_date: Optional end date.
    
    Returns:
        pd.DataFrame: OHLCV data for the specified ticker.
    """
    return get_data(ticker=ticker, start_date=start_date, end_date=end_date)


# =============================================================================
# TESTING
# =============================================================================

def simicx_test_data_loader():
    """Test function for data_loader module.
    
    Verifies:
    1. Database connectivity
    2. Ticker availability
    3. Date range queries
    4. Temporal split integrity (training ≤ 2024, trading ≥ 2025)
    5. Date alignment across tickers
    """
    print("Testing data_loader module...")
    
    # Test 1: Get tickers
    tickers = get_tickers()
    assert len(tickers) > 0, "No tickers found in database"
    assert 'SPY' in tickers, "SPY should be available"
    print(f"✓ get_tickers: Found {len(tickers)} tickers")
    
    # Test 2: Get date range
    start, end = get_date_range('SPY')
    assert start < end, "Start date should be before end date"
    print(f"✓ get_date_range: SPY data from {start.date()} to {end.date()}")
    
    # Test 3: Get training data
    train_df = get_training_data(LIMITED_TICKERS, years_back=2)
    assert not train_df.empty, "Training data should not be empty"
    assert train_df['time'].max() <= pd.Timestamp(TRAINING_END_DATE), \
        f"Training data leak! Max date: {train_df['time'].max()}"
    print(f"✓ get_training_data: {len(train_df)} rows, max date: {train_df['time'].max().date()}")
    
    # Test 4: Get trading data
    trade_df = get_trading_data(LIMITED_TICKERS)
    if not trade_df.empty:
        assert trade_df['time'].min() >= pd.Timestamp(TRADING_START_DATE), \
            f"Trading data contaminated! Min date: {trade_df['time'].min()}"
        print(f"✓ get_trading_data: {len(trade_df)} rows, min date: {trade_df['time'].min().date()}")
    else:
        print("⚠ get_trading_data: No data yet (2025+ data not available)")
    
    # Test 5: Date alignment
    df = get_data(tickers=['SPY', 'AAPL'], start_date='2024-01-01', end_date='2024-06-30')
    if not df.empty:
        dates_spy = set(df[df['ticker'] == 'SPY']['time'])
        dates_aapl = set(df[df['ticker'] == 'AAPL']['time'])
        assert dates_spy == dates_aapl, "Date alignment failed!"
        print(f"✓ Date alignment: {len(dates_spy)} common dates for SPY and AAPL")
    
    # Test 6: Phase-based loading
    print("\nTesting phase based loading...")
    limited_train = get_training_data(phase='limited')
    # Default year back for limited is 3, so start year should be 2022
    expected_start = pd.Timestamp("2022-01-01")
    actual_start = limited_train['time'].min()
    print(f"Limited Train Start: {actual_start.date()} (Expected ~{expected_start.date()})")
    assert actual_start >= expected_start, "Limited training data goes back too far!"
    
    # Check tickers match LIMITED_TICKERS
    unique_tickers = set(limited_train['ticker'].unique())
    assert unique_tickers.issubset(set(LIMITED_TICKERS)), "Found tickers outside LIMITED_TICKERS in limited phase"
    print("✓ Phase=limited uses correct tickers and date range")

    print("\n🎉 All data_loader tests passed!")


if __name__ == '__main__':
    simicx_test_data_loader()
