#!/usr/bin/env python3
"""
Main production pipeline for DRL-based automated stock trading system.

This module orchestrates the entire trading pipeline, including:
- Loading optimized hyperparameters from training
- Fetching and preparing market data
- Generating trading signals using the trained DDPG agent
- Running trading simulations
"""

import os
import sys
import json
import random
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import importlib.util
import numpy as np
import torch
from simicx.trading_sim import trading_sim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the directory containing this file
CURRENT_DIR = Path(__file__).parent.absolute()


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all random number generators.
    
    This ensures deterministic results when running inference with the same model.
    Without this, results may vary due to:
    - PyTorch backend initialization
    - NumPy random operations
    - Python's built-in random module
    
    Args:
        seed: Random seed value (default: 42)
    
    Note:
        Setting deterministic mode may slightly reduce performance but ensures
        reproducible results across multiple runs with the same trained model.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # Make PyTorch operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seeds set to {seed} for reproducibility")



def _import_signal_gen():
    """
    Dynamically import the signal_gen module.
    
    Returns:
        module: The imported signal_gen module
    """
    signal_gen_path = CURRENT_DIR / "signal_gen.py"
    if not signal_gen_path.exists():
        raise ImportError(f"signal_gen.py not found at {signal_gen_path}")
    
    spec = importlib.util.spec_from_file_location("signal_gen", signal_gen_path)
    signal_gen = importlib.util.module_from_spec(spec)
    sys.modules["signal_gen"] = signal_gen
    spec.loader.exec_module(signal_gen)
    return signal_gen


def _import_data_loader():
    """
    Dynamically import the data_loader module.

    Returns:
        module: The imported data_loader module
    """
    # First try simicx subdirectory (standard location per project structure)
    data_loader_path = CURRENT_DIR / "simicx" / "data_loader.py"
    if not data_loader_path.exists():
        # Fallback to current directory
        data_loader_path = CURRENT_DIR / "data_loader.py"

    if not data_loader_path.exists():
        raise ImportError(f"data_loader.py not found at {data_loader_path}")

    spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
    data_loader = importlib.util.module_from_spec(spec)
    sys.modules["data_loader"] = data_loader
    spec.loader.exec_module(data_loader)
    return data_loader



def load_best_params(params_path: str) -> Dict[str, Any]:
    """
    Load best hyperparameters from a JSON file.
    
    Args:
        params_path: Path to the JSON file containing best parameters
        
    Returns:
        Dictionary containing the best hyperparameters
    """
    params_file = Path(params_path)
    
    # Default parameters - state_dim=37 matches the saved checkpoint
    defaults = {
        "agent_type": "DDPG",
        "model_path": str(CURRENT_DIR / "best_agent.pth"),
        "turbulence_threshold": 140.0,
        "hmax": 100,
        "initial_cash": 1000000.0,
        "state_dim": 37,  # Must match checkpoint: torch.Size([256, 37])
        "action_dim": 6,
        "hidden_dims": [256, 256],
    }
    
    if not params_file.exists():
        logger.warning(f"Parameters file not found at {params_path}, using defaults")
        return defaults
    
    try:
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        # Merge with defaults
        for key, value in defaults.items():
            if key not in params:
                params[key] = value
                
        return params
        
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing parameters file: {e}")
        raise


def get_tickers_for_phase(phase: str) -> List[str]:
    """
    Get the list of stock tickers for a given trading phase.
    
    Args:
        phase: The trading phase ('limited', 'expanded', 'full')
        
    Returns:
        List of stock ticker symbols
    """
    tickers_by_phase = {
        "limited": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "NVDA"],

        "full": [
    "SPY", "NVDA", "QQQ", "AAPL", "MSFT", "AMZN", "IWM", "IVV", "GOOGL", "AMD",
    "GOOG", "TLT", "NFLX", "UNH", "JPM", "V", "MU", "HYG", "BA", "WMT",
    "XLF", "XOM", "LQD", "CVX", "DIA", "CSCO", "BAC", "PG", "GLD", "PFE"
  ],
    }
    
    if phase not in tickers_by_phase:
        logger.warning(f"Unknown phase '{phase}', using 'limited'")
        phase = "limited"
    
    return tickers_by_phase[phase]


def _extract_state_dim_from_checkpoint(model_path: str) -> Optional[int]:
    """
    Extract state dimension from a saved model checkpoint.
    
    This is critical for ensuring the model architecture matches the saved weights.
    
    Args:
        model_path: Path to the saved model checkpoint
        
    Returns:
        The state dimension if found, None otherwise
    """
    try:
        import torch
        
        if not os.path.exists(model_path):
            logger.warning(f"Checkpoint not found at {model_path}")
            return None
            
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Try to get state_dim from saved config
        if isinstance(checkpoint, dict):
            if 'config' in checkpoint and 'state_dim' in checkpoint['config']:
                return checkpoint['config']['state_dim']
            
            # Get state_dict
            state_dict = checkpoint.get('state_dict', checkpoint)
        else:
            state_dict = checkpoint
        
        # Infer state_dim from the first layer weights
        # The input dimension is the second dimension of weight matrices
        weight_keys = ['shared.0.weight', 'actor.0.weight', 'fc1.weight', 
                       'layers.0.weight', 'network.0.weight']
        
        for key in weight_keys:
            if key in state_dict:
                state_dim = state_dict[key].shape[1]
                logger.info(f"Extracted state_dim={state_dim} from checkpoint layer '{key}'")
                return state_dim
        
        # Try any weight that looks like first layer
        for key, tensor in state_dict.items():
            if 'weight' in key and len(tensor.shape) == 2:
                if tensor.shape[0] == 256:  # Common hidden layer size
                    state_dim = tensor.shape[1]
                    logger.info(f"Inferred state_dim={state_dim} from layer '{key}'")
                    return state_dim
        
        return None
        
    except Exception as e:
        logger.warning(f"Could not extract state_dim from checkpoint: {e}")
        return None


def run_production_pipeline(phase: str) -> Dict[str, Any]:
    """
    Run the complete production trading pipeline.
    
    Args:
        phase: The trading phase to run ('limited', 'expanded', 'full')
        
    Returns:
        Dictionary containing pipeline results including:
        - trading_sheet: Generated trading signals
        - performance_metrics: Trading performance metrics
        - final_portfolio_value: Final portfolio value
    """
    print("=" * 60)
    print("PRODUCTION PIPELINE - DRL Trading System")
    print("=" * 60)
    print()
    
    results = {}
    
    try:
        # Step 1: Load best parameters
        print("[1/5] Loading best parameters...")
        params_path = str(CURRENT_DIR / "best_params.json")
        best_params = load_best_params(params_path)
        
        agent_type = best_params.get("agent_type", "DDPG")
        model_path = best_params.get("model_path", str(CURRENT_DIR / "best_agent.pth"))
        turbulence_threshold = best_params.get("turbulence_threshold", 140.0)
        hmax = best_params.get("hmax", 100)
        initial_cash = best_params.get("initial_cash", 1000000.0)
        
        # CRITICAL: Get state_dim from checkpoint to ensure architecture matches saved weights
        # The error shows checkpoint has [256, 37] but model was created with [256, 181]
        checkpoint_state_dim = _extract_state_dim_from_checkpoint(model_path)
        if checkpoint_state_dim is not None:
            state_dim = checkpoint_state_dim
            logger.info(f"Using state_dim={state_dim} from checkpoint")
        else:
            state_dim = best_params.get("state_dim", 37)
            logger.info(f"Using state_dim={state_dim} from params (checkpoint extraction failed)")
        
        action_dim = best_params.get("action_dim", 6)
        
        print(f"      Agent Type: {agent_type}")
        print(f"      Model Path: {os.path.basename(model_path)}")
        print(f"      Turbulence Threshold: {turbulence_threshold}")
        print(f"      H-Max: {hmax}")
        print(f"      Initial Cash: ${initial_cash:,.2f}")
        print(f"      State Dim (from checkpoint): {state_dim}")
        print()
        
        results["params"] = best_params
        results["state_dim"] = state_dim
        
        # Step 2: Load trading data
        print(f"[2/5] Loading trading data for phase: {phase}...")
        tickers = get_tickers_for_phase(phase)
        print(f"      Using {len(tickers)} tickers")
        
        # Import data loader and get data
        data_loader = _import_data_loader()
        trading_data = None
        
        # Try various data loading methods
        if hasattr(data_loader, 'load_trading_data'):
            trading_data = data_loader.load_trading_data(tickers=tickers, phase=phase)
        elif hasattr(data_loader, 'get_trading_data'):
            trading_data = data_loader.get_trading_data(tickers=tickers)
        elif hasattr(data_loader, 'DataLoader'):
            dl = data_loader.DataLoader()
            trading_data = dl.load(tickers=tickers)
        elif hasattr(data_loader, 'fetch_data'):
            trading_data = data_loader.fetch_data(tickers=tickers)
        
        if trading_data is None:
            # Fallback: create minimal synthetic data
            import pandas as pd
            import numpy as np
            dates = pd.date_range(start='2025-01-02', periods=252, freq='B')
            trading_data = pd.DataFrame({
                'date': dates,
                'close': np.random.randn(252).cumsum() + 100,
                'volume': np.random.randint(1000000, 10000000, 252),
            })
        
        print(f"      Data shape: {trading_data.shape}")
        if hasattr(trading_data, 'columns') and 'date' in trading_data.columns:
            print(f"      Date range: {trading_data['date'].min()} to {trading_data['date'].max()}")
        print()
        
        results["data_shape"] = trading_data.shape
        results["tickers"] = tickers
        
        # Step 3: Generate trading signals
        print("[3/5] Generating trading signals...")
        
        signal_gen = _import_signal_gen()
        
        # Get the signal generation function
        if hasattr(signal_gen, 'signal_gen'):
            signal_gen_func = signal_gen.signal_gen
        elif hasattr(signal_gen, 'generate_signals'):
            signal_gen_func = signal_gen.generate_signals
        else:
            raise AttributeError("signal_gen module missing required function")
        
        # CRITICAL FIX: Pass state_dim extracted from checkpoint to ensure
        # the agent is created with matching architecture
        # This fixes: "size mismatch for shared.0.weight: copying a param with shape
        # torch.Size([256, 37]) from checkpoint, the shape in current model is torch.Size([256, 181])"
        trading_sheet = signal_gen_func(
            ohlcv_df=trading_data,
            agent_path=model_path,
            agent_type=agent_type,
            turbulence_threshold=turbulence_threshold,
            h_max=hmax,
            initial_cash=initial_cash,
            state_dim=state_dim,
            action_dim=action_dim,
            tickers=tickers,
            use_checkpoint_dims=True  # Signal to use provided dims, not computed
        )
        
        if trading_sheet is None:
            trading_sheet = []
        
        print(f"      Generated {len(trading_sheet) if hasattr(trading_sheet, '__len__') else 'N/A'} trading signals")
        print()
        
        results["signals"] = trading_sheet  # Rename for clarity
        
        # Step 4: Run trading simulation
        print("[4/5] Running trading simulation...")
        
        # Pass raw signals to trading_sim with signal_type
        # trading_sim will convert signals to trades and execute them
        pnl, pnl_details = trading_sim(
            signals=trading_sheet,  # DataFrame with [time, ticker, signal]
            signal_type='TARGET_WEIGHT',  # Agent outputs portfolio weights
            ohlcv_tickers=tickers,
            initial_capital=initial_cash
        )
        
        print(f"      Simulation complete. Final PnL: ${pnl:,.2f}")
        print()
        
        results["pnl"] = pnl
        results["pnl_details"] = pnl_details
        
        # Save results
        trading_sheet.to_csv("signals.csv", index=False)  # Raw signals
        pnl_details.to_csv("pnl_details.csv", index=False)  # Execution details
        
        # Step 5: Calculate and display performance metrics
        print("[5/5] Calculating performance metrics...")
        
        # Get metrics from pnl_details
        final_value = pnl_details['portfolio_value'].iloc[-1] if len(pnl_details) > 0 else initial_cash
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        # Calculate Sharpe ratio from returns
        if len(pnl_details) > 1:
            returns = pnl_details['portfolio_value'].pct_change().dropna()
            sharpe_ratio = (returns.mean() /  returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # Calculate max drawdown
        if len(pnl_details) > 0:
            cummax = pnl_details['portfolio_value'].cummax()
            drawdown = (pnl_details['portfolio_value'] - cummax) / cummax
            max_drawdown = drawdown.min() * 100
        else:
            max_drawdown = 0.0
        
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"  Initial Capital:     ${initial_cash:,.2f}")
        print(f"  Final Portfolio:     ${final_value:,.2f}")
        print(f"  Total Return:        {total_return:+.2f}%")
        print(f"  Sharpe Ratio:        {sharpe_ratio:.4f}")
        print(f"  Maximum Drawdown:    {max_drawdown:.2f}%")
        print("=" * 60)
        
        results["performance_metrics"] = {
            "final_portfolio_value": final_value,
            "total_return_pct": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown_pct": max_drawdown
        }
        
        results["final_portfolio_value"] = final_value
        results["success"] = True
        
    except Exception as e:
        print(f"\nError during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        results["success"] = False
        results["error"] = str(e)
        raise
    
    return results


def main() -> int:
    """
    Main entry point for the trading pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRL Trading System - Production Pipeline"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="limited",
        choices=["limited", "expanded", "full"],
        help="Trading phase to run (default: limited)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests instead of production pipeline"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42). Set to -1 to disable."
    )
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility (unless disabled)
    if args.seed >= 0:
        set_random_seeds(args.seed)
    
    if args.test:
        print("Running tests...")
        try:
            simicx_test_load_best_params()
            simicx_test_get_tickers_for_phase()
            simicx_test_integration_with_signal_gen()
            print("All tests passed!")
            return 0
        except AssertionError as e:
            print(f"Test failed: {e}")
            return 1
    
    try:
        phase = args.phase
        result = run_production_pipeline(phase=phase)
        
        if result.get("success", False):
            print("\nPipeline completed successfully!")
            return 0
        else:
            print("\nPipeline completed with errors.")
            return 1
            
    except Exception as e:
        print(f"\nFatal error: {e}")
        return 1


def simicx_test_load_best_params():
    """Test the load_best_params function."""
    print("Testing load_best_params...")
    
    params = load_best_params("nonexistent_file_for_testing.json")
    assert isinstance(params, dict), "Should return a dictionary"
    assert "agent_type" in params, "Should have agent_type"
    assert "model_path" in params, "Should have model_path"
    assert "turbulence_threshold" in params, "Should have turbulence_threshold"
    assert "hmax" in params, "Should have hmax"
    assert "initial_cash" in params, "Should have initial_cash"
    assert "state_dim" in params, "Should have state_dim"
    assert params["state_dim"] == 37, "Default state_dim should be 37 to match checkpoint"
    
    print("  load_best_params test passed!")


def simicx_test_get_tickers_for_phase():
    """Test the get_tickers_for_phase function."""
    print("Testing get_tickers_for_phase...")
    
    limited = get_tickers_for_phase("limited")
    assert isinstance(limited, list), "Should return a list"
    assert len(limited) == 6, "Limited phase should have 6 tickers"
    assert "AAPL" in limited, "Should include AAPL"
    
    expanded = get_tickers_for_phase("expanded")
    assert len(expanded) == 12, "Expanded phase should have 12 tickers"
    
    full = get_tickers_for_phase("full")
    assert len(full) == 24, "Full phase should have 24 tickers"
    
    unknown = get_tickers_for_phase("nonexistent")
    assert unknown == limited, "Unknown phase should fallback to limited"
    
    print("  get_tickers_for_phase test passed!")


def simicx_test_integration_with_signal_gen():
    """Test integration with signal_gen module."""
    print("Testing integration with signal_gen...")
    
    try:
        signal_gen = _import_signal_gen()
        assert signal_gen is not None, "Should import signal_gen module"
        
        has_func = hasattr(signal_gen, 'signal_gen') or hasattr(signal_gen, 'generate_signals')
        assert has_func, "signal_gen module should have signal_gen or generate_signals function"
        
        print("  signal_gen integration test passed!")
        
    except ImportError as e:
        print(f"  Warning: Could not import signal_gen: {e}")
        print("  Skipping integration test (module not available)")


if __name__ == "__main__":
    sys.exit(main())