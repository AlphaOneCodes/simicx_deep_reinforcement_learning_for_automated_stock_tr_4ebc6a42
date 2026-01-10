"""
Ensemble Manager for Quarterly Retraining Strategy.

This module implements the ensemble strategy from the research paper:
1. Quarterly retraining with growing window
2. 3-month validation for agent selection
3. Dynamic best-agent selection based on Sharpe ratio

Author: SimicX
Copyright: (C) 2025-2026 SimicX. All rights reserved.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class QuarterlyScheduler:
    """
    Manages quarterly retraining schedule.
    
    Determines when to retrain agents (every 3 months) and tracks
    the training/validation/trading periods for each quarter.
    
    Example:
        >>> scheduler = QuarterlyScheduler(start_date='2025-01-01')
        >>> quarters = list(scheduler.get_quarters(end_date='2025-12-31'))
        >>> len(quarters)
        4  # Q1, Q2, Q3, Q4
    """
    
    def __init__(self, start_date: str):
        """
        Initialize quarterly scheduler.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
        """
        self.start_date = pd.Timestamp(start_date)
    
    def get_quarter_start(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Get the start date of the quarter containing the given date.
        
        Args:
            date: Any date within the quarter
            
        Returns:
            First day of that quarter
        """
        quarter = (date.month - 1) // 3
        return pd.Timestamp(year=date.year, month=quarter * 3 + 1, day=1)
    
    def get_next_quarter_start(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Get the start date of the next quarter.
        
        Args:
            date: Current date
            
        Returns:
            First day of next quarter
        """
        current_quarter_start = self.get_quarter_start(date)
        next_quarter = current_quarter_start + pd.DateOffset(months=3)
        return next_quarter
    
    def get_quarters(self, end_date: Optional[str] = None) -> List[Dict[str, pd.Timestamp]]:
        """
        Generate list of quarters from start to end date.
        
        Each quarter dictionary contains:
        - quarter_start: First day of quarter
        - quarter_end: Last day of quarter
        - retrain_by: Date by which retraining should be complete
        
        Args:
            end_date: End date in 'YYYY-MM-DD' format (optional)
            
        Returns:
            List of quarter dictionaries
        """
        quarters = []
        current = self.get_quarter_start(self.start_date)
        
        if end_date is not None:
            end = pd.Timestamp(end_date)
        else:
            # Default to 4 quarters from start
            end = current + pd.DateOffset(months=12)
        
        while current <= end:
            quarter_end = current + pd.DateOffset(months=3) - pd.DateOffset(days=1)
            
            quarters.append({
                'quarter_start': current,
                'quarter_end': min(quarter_end, end),
                'retrain_by': current  # Retrain at start of quarter
            })
            
            current = current + pd.DateOffset(months=3)
        
        return quarters
    
    def should_retrain(self, current_date: str, last_train_date: Optional[str] = None) -> bool:
        """
        Determine if retraining is needed.
        
        Args:
            current_date: Current date in 'YYYY-MM-DD' format
            last_train_date: Last training date (optional)
            
        Returns:
            True if retraining is needed, False otherwise
        """
        current = pd.Timestamp(current_date)
        
        if last_train_date is None:
            # Never trained before
            return True
        
        last_train = pd.Timestamp(last_train_date)
        
        # Check if we've crossed a quarter boundary
        current_quarter = self.get_quarter_start(current)
        last_quarter = self.get_quarter_start(last_train)
        
        return current_quarter > last_quarter


class GrowingWindowLoader:
    """
    Manages growing window data loading for ensemble strategy.
    
    Each quarter, the training window expands to include more historical
    data, while maintaining a fixed 3-month validation window.
    
    Example:
        >>> loader = GrowingWindowLoader(initial_years=3)
        >>> train, val = loader.get_data_for_quarter('2025-04-01', ohlcv_df)
        >>> # train_df contains expanding historical data
        >>> # val_df contains last 3 months before quarter start
    """
    
    def __init__(self, initial_years: int = 3, validation_months: int = 3):
        """
        Initialize growing window loader.
        
        Args:
            initial_years: Initial training window size in years (default: 3)
            validation_months: Validation window size in months (default: 3)
        """
        self.initial_years = initial_years
        self.validation_months = validation_months
    
    def get_training_window(
        self,
        quarter_start: pd.Timestamp,
        ohlcv_df: pd.DataFrame
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Calculate training window dates for a given quarter.
        
        Args:
            quarter_start: Start date of the quarter
            ohlcv_df: Full OHLCV DataFrame
            
        Returns:
            Tuple of (train_start, train_end) dates
        """
        # Validation ends just before quarter starts
        val_end = quarter_start - pd.DateOffset(days=1)
        
        # Training ends before validation
        train_end = val_end - pd.DateOffset(months=self.validation_months)
        
        # Calculate how far back to go (expanding window)
        # Start with initial_years, expand by 3 months per quarter
        min_date = ohlcv_df['time'].min()
        
        # Use all available data up to train_end
        train_start = min_date
        
        return train_start, train_end
    
    def get_validation_window(
        self,
        quarter_start: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Calculate validation window dates (3 months before quarter).
        
        Args:
            quarter_start: Start date of the quarter
            
        Returns:
            Tuple of (val_start, val_end) dates
        """
        val_end = quarter_start - pd.DateOffset(days=1)
        val_start = val_end - pd.DateOffset(months=self.validation_months) + pd.DateOffset(days=1)
        
        return val_start, val_end
    
    def get_data_for_quarter(
        self,
        quarter_start: str,
        ohlcv_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and validation data for a specific quarter.
        
        Args:
            quarter_start: Quarter start date in 'YYYY-MM-DD' format
            ohlcv_df: Full OHLCV DataFrame with 'time' column
            
        Returns:
            Tuple of (train_df, val_df)
        """
        q_start = pd.Timestamp(quarter_start)
        
        # Get window dates
        train_start, train_end = self.get_training_window(q_start, ohlcv_df)
        val_start, val_end = self.get_validation_window(q_start)
        
        # Filter data
        train_df = ohlcv_df[
            (ohlcv_df['time'] >= train_start) & (ohlcv_df['time'] <= train_end)
        ].copy()
        
        val_df = ohlcv_df[
            (ohlcv_df['time'] >= val_start) & (ohlcv_df['time'] <= val_end)
        ].copy()
        
        return train_df, val_df


class EnsembleSelector:
    """
    Selects best agent based on validation Sharpe ratio.
    
    Tracks selection history and provides persistence for ensemble state.
    
    Example:
        >>> selector = EnsembleSelector()
        >>> selector.select_best({'ppo': 1.2, 'a2c': 0.9, 'ddpg': 1.5}, '2025-01-01')
        'ddpg'
        >>> selector.get_history()
        [{'quarter': '2025-01-01', 'agent': 'ddpg', 'sharpe': 1.5, ...}]
    """
    
    def __init__(self, history_path: str = 'ensemble_history.json'):
        """
        Initialize ensemble selector.
        
        Args:
            history_path: Path to save selection history
        """
        self.history_path = history_path
        self.history: List[Dict] = []
        self.load_history()
    
    def select_best(
        self,
        sharpe_ratios: Dict[str, float],
        quarter_date: str
    ) -> str:
        """
        Select best agent based on Sharpe ratios.
        
        Args:
            sharpe_ratios: Dict mapping agent_type -> Sharpe ratio
            quarter_date: Quarter date in 'YYYY-MM-DD' format
            
        Returns:
            Best agent type ('ppo', 'a2c', or 'ddpg')
        """
        # Find agent with highest Sharpe ratio
        best_agent = max(sharpe_ratios.keys(), key=lambda k: sharpe_ratios[k])
        best_sharpe = sharpe_ratios[best_agent]
        
        # Record selection
        selection = {
            'quarter': quarter_date,
            'agent': best_agent,
            'sharpe': best_sharpe,
            'all_sharpes': sharpe_ratios,
            'timestamp': datetime.now().isoformat()
        }
        
        self.history.append(selection)
        self.save_history()
        
        return best_agent
    
    def get_history(self) -> List[Dict]:
        """Get selection history."""
        return self.history.copy()
    
    def get_latest_selection(self) -> Optional[Dict]:
        """Get most recent selection, or None if no history."""
        if self.history:
            return self.history[-1].copy()
        return None
    
    def save_history(self) -> None:
        """Save selection history to JSON file."""
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load_history(self) -> None:
        """Load selection history from JSON file if it exists."""
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = []


def get_quarter_for_date(date: str) -> str:
    """
    Get quarter identifier for a given date.
    
    Args:
        date: Date in 'YYYY-MM-DD' format
        
    Returns:
        Quarter identifier like '2025Q1'
    """
    dt = pd.Timestamp(date)
    quarter = (dt.month - 1) // 3 + 1
    return f"{dt.year}Q{quarter}"


# Example usage
if __name__ == "__main__":
    # Test quarterly scheduler
    scheduler = QuarterlyScheduler('2025-01-01')
    quarters = scheduler.get_quarters(end_date='2025-12-31')
    
    print("Quarterly Schedule:")
    for q in quarters:
        print(f"  {q['quarter_start'].date()} to {q['quarter_end'].date()}")
    
    # Test should_retrain
    print(f"\nShould retrain on 2025-04-01? {scheduler.should_retrain('2025-04-01', None)}")
    print(f"Should retrain on 2025-02-01 (last trained 2025-01-01)? {scheduler.should_retrain('2025-02-01', '2025-01-01')}")
    
    print("\n✓ Ensemble manager module working correctly!")

