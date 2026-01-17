"""
Data Loading Module
===================

Provides functions to load processed data for analysis.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import PROCESSED_DATA_DIR, MANUAL_DATA_DIR


def load_master_data() -> pd.DataFrame:
    """
    Load the master analysis dataset.
    
    Returns:
        DataFrame with all variables aligned by date
    """
    filepath = PROCESSED_DATA_DIR / "master_data.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Master data not found at {filepath}. "
            "Run 'python -m src.data.process' first."
        )
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    return df


def load_portfolio_returns() -> pd.DataFrame:
    """
    Load portfolio returns (factors and mispricing portfolios).
    
    Returns:
        DataFrame with portfolio returns
    """
    filepath = PROCESSED_DATA_DIR / "portfolio_returns.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Portfolio returns not found at {filepath}. "
            "Run 'python -m src.data.process' first."
        )
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    return df


def load_industry_returns() -> pd.DataFrame:
    """
    Load 30 industry portfolio returns.
    
    Returns:
        DataFrame with industry returns
    """
    filepath = PROCESSED_DATA_DIR / "industry_returns.csv"
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Industry returns not found at {filepath}. "
            "Run 'python -m src.data.process' first."
        )
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    return df


def load_all_data() -> dict:
    """
    Load all processed data.
    
    Returns:
        Dictionary with keys: 'master', 'portfolios', 'industry'
    """
    return {
        'master': load_master_data(),
        'portfolios': load_portfolio_returns(),
        'industry': load_industry_returns()
    }


def check_data_availability() -> dict:
    """
    Check which data files are available.
    
    Returns:
        Dictionary with file availability status
    """
    files = {
        'master_data': PROCESSED_DATA_DIR / "master_data.csv",
        'portfolio_returns': PROCESSED_DATA_DIR / "portfolio_returns.csv",
        'industry_returns': PROCESSED_DATA_DIR / "industry_returns.csv",
        'monetary_shocks_manual': MANUAL_DATA_DIR / "monetary_shocks.csv",
        'fiscal_shocks_manual': MANUAL_DATA_DIR / "fiscal_shocks.csv",
    }
    
    status = {}
    for name, path in files.items():
        status[name] = path.exists()
    
    return status


def get_sample_info() -> dict:
    """
    Get information about the loaded sample.
    
    Returns:
        Dictionary with sample information
    """
    try:
        master = load_master_data()
        
        info = {
            'start_date': master.index[0],
            'end_date': master.index[-1],
            'n_obs': len(master),
            'n_variables': len(master.columns),
            'has_mp_shock': 'MP_SHOCK' in master.columns,
            'has_fiscal_shock': 'FISCAL_SHOCK' in master.columns,
            'has_constraints': 'CONSTRAINT_COMPOSITE' in master.columns,
        }
        
        # Count shock events
        if 'MP_SHOCK_IND' in master.columns:
            info['n_mp_shocks'] = master['MP_SHOCK_IND'].sum()
        
        if 'HIGH_CONSTRAINT' in master.columns:
            info['n_high_constraint'] = master['HIGH_CONSTRAINT'].sum()
        
        return info
        
    except FileNotFoundError:
        return {'error': 'Data not yet processed'}


# Module-level init
__all__ = [
    'load_master_data',
    'load_portfolio_returns', 
    'load_industry_returns',
    'load_all_data',
    'check_data_availability',
    'get_sample_info'
]
