"""
Data Processing Module
======================

Cleans, merges, and processes all raw data into analysis-ready format.

Outputs:
- processed/master_data.csv: Main analysis dataset
- processed/portfolios.csv: Portfolio returns
- processed/industry_returns.csv: 30 industry portfolio returns
"""

import sys
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MANUAL_DATA_DIR,
    SAMPLE_START, SAMPLE_END, EXTENDED_START,
    SHOCK_THRESHOLD_PERCENTILE, CONSTRAINT_THRESHOLD_PERCENTILE,
    CAPITAL_RATIO_THRESHOLD_PERCENTILE
)

warnings.filterwarnings('ignore')


def load_raw_csv(filename: str, directory: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load a CSV file from raw data directory."""
    filepath = directory / filename
    if filepath.exists():
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    else:
        print(f"  ✗ File not found: {filepath}")
        return pd.DataFrame()


def process_ff5_factors() -> pd.DataFrame:
    """Process Fama-French 5 factors."""
    print("  Processing FF5 factors...")
    
    df = load_raw_csv("ff5_factors.csv")
    if df.empty:
        return df
    
    # Standardize column names
    df.columns = [c.strip().upper().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Rename columns
    rename_map = {
        'MKT_RF': 'MKT-RF',
        'MKTRF': 'MKT-RF',
    }
    df = df.rename(columns=rename_map)
    
    # Ensure we have the expected columns
    expected = ['MKT-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    available = [c for c in expected if c in df.columns]
    
    print(f"    ✓ FF5 factors: {len(df)} obs, columns: {available}")
    
    return df[available]


def process_momentum() -> pd.DataFrame:
    """Process momentum factor."""
    print("  Processing Momentum factor...")
    
    df = load_raw_csv("momentum.csv")
    if df.empty:
        return df
    
    # Standardize column names
    df.columns = [c.strip().upper() for c in df.columns]
    
    # Rename to MOM
    if len(df.columns) == 1:
        df.columns = ['MOM']
    elif 'MOM' not in df.columns:
        # Take first column
        df = df[[df.columns[0]]]
        df.columns = ['MOM']
    
    print(f"    ✓ Momentum: {len(df)} obs")
    
    return df[['MOM']]


def process_reversal_factors() -> pd.DataFrame:
    """Process short-term and long-term reversal factors."""
    print("  Processing Reversal factors...")
    
    st_rev = load_raw_csv("st_reversal.csv")
    lt_rev = load_raw_csv("lt_reversal.csv")
    
    factors = []
    
    if not st_rev.empty:
        st_rev.columns = [c.strip().upper() for c in st_rev.columns]
        if len(st_rev.columns) == 1:
            st_rev.columns = ['ST_REV']
        else:
            st_rev = st_rev[[st_rev.columns[0]]]
            st_rev.columns = ['ST_REV']
        factors.append(st_rev)
        print(f"    ✓ ST Reversal: {len(st_rev)} obs")
    
    if not lt_rev.empty:
        lt_rev.columns = [c.strip().upper() for c in lt_rev.columns]
        if len(lt_rev.columns) == 1:
            lt_rev.columns = ['LT_REV']
        else:
            lt_rev = lt_rev[[lt_rev.columns[0]]]
            lt_rev.columns = ['LT_REV']
        factors.append(lt_rev)
        print(f"    ✓ LT Reversal: {len(lt_rev)} obs")
    
    if factors:
        return pd.concat(factors, axis=1)
    return pd.DataFrame()


def process_bab() -> pd.DataFrame:
    """Process BAB factor."""
    print("  Processing BAB factor...")
    
    df = load_raw_csv("bab_factor.csv")
    if df.empty:
        return df
    
    df.columns = ['BAB']
    
    print(f"    ✓ BAB: {len(df)} obs")
    
    return df


def process_industry_portfolios() -> pd.DataFrame:
    """Process 30 industry portfolios."""
    print("  Processing 30 Industry portfolios...")
    
    df = load_raw_csv("industry_30.csv")
    if df.empty:
        return df
    
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    
    print(f"    ✓ Industry portfolios: {len(df)} obs, {len(df.columns)} industries")
    
    return df


def process_fred_data() -> pd.DataFrame:
    """Process FRED data (VIX, TED spread)."""
    print("  Processing FRED data...")
    
    df = load_raw_csv("fred_data.csv")
    if df.empty:
        return df
    
    print(f"    ✓ FRED data: {len(df)} obs, columns: {list(df.columns)}")
    
    return df


def process_intermediary_capital() -> pd.DataFrame:
    """Process intermediary capital ratio."""
    print("  Processing Intermediary Capital...")
    
    # Try raw first, then manual
    df = load_raw_csv("intermediary_capital.csv")
    
    if df.empty:
        df = load_raw_csv("intermediary_capital.csv", MANUAL_DATA_DIR)
    
    if df.empty:
        print("    ✗ Intermediary capital not found")
        return df
    
    # Standardize column name
    if 'CAPITAL_RATIO' not in df.columns:
        # Take first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df = df[[num_cols[0]]]
            df.columns = ['CAPITAL_RATIO']
    
    print(f"    ✓ Intermediary Capital: {len(df)} obs")
    
    return df[['CAPITAL_RATIO']]


def process_monetary_shocks() -> pd.DataFrame:
    """
    Process monetary policy shocks.
    
    ⚠️ REQUIRES MANUAL DATA FILE: data/manual/monetary_shocks.csv
    """
    print("  Processing Monetary Policy Shocks...")
    
    filepath = MANUAL_DATA_DIR / "monetary_shocks.csv"
    
    if not filepath.exists():
        print("    ✗ MANUAL DATA REQUIRED: data/manual/monetary_shocks.csv")
        print("      See README.md for format specification")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Standardize column name
    if 'shock' in df.columns:
        df = df.rename(columns={'shock': 'MP_SHOCK'})
    elif 'MP_SHOCK' not in df.columns:
        df.columns = ['MP_SHOCK']
    
    print(f"    ✓ Monetary Shocks: {len(df)} obs, {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    
    return df[['MP_SHOCK']]


def process_fiscal_shocks() -> pd.DataFrame:
    """
    Process fiscal policy shocks.
    
    ⚠️ OPTIONAL MANUAL DATA FILE: data/manual/fiscal_shocks.csv
    """
    print("  Processing Fiscal Policy Shocks...")
    
    filepath = MANUAL_DATA_DIR / "fiscal_shocks.csv"
    
    if not filepath.exists():
        print("    ⚠ Optional file not found: data/manual/fiscal_shocks.csv")
        print("      Fiscal shock analysis will be skipped")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Standardize column name
    if 'shock' in df.columns:
        df = df.rename(columns={'shock': 'FISCAL_SHOCK'})
    elif 'FISCAL_SHOCK' not in df.columns:
        df.columns = ['FISCAL_SHOCK']
    
    # Resample quarterly to monthly (forward fill within quarter)
    if df.index.freq is None or 'Q' in str(df.index.freq):
        df = df.resample('M').ffill()
    
    print(f"    ✓ Fiscal Shocks: {len(df)} obs")
    
    return df[['FISCAL_SHOCK']]


def create_shock_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary shock indicators based on percentile thresholds."""
    
    result = df.copy()
    
    # Monetary shock indicator (contractionary = negative shock)
    if 'MP_SHOCK' in df.columns:
        threshold = np.nanpercentile(df['MP_SHOCK'], SHOCK_THRESHOLD_PERCENTILE)
        result['MP_SHOCK_IND'] = (df['MP_SHOCK'] < threshold).astype(int)
        print(f"    Monetary shock threshold (p{SHOCK_THRESHOLD_PERCENTILE}): {threshold:.4f}")
        print(f"    Contractionary shock months: {result['MP_SHOCK_IND'].sum()}")
    
    # Fiscal shock indicator
    if 'FISCAL_SHOCK' in df.columns:
        threshold = np.nanpercentile(df['FISCAL_SHOCK'], SHOCK_THRESHOLD_PERCENTILE)
        result['FISCAL_SHOCK_IND'] = (df['FISCAL_SHOCK'] < threshold).astype(int)
        print(f"    Fiscal shock threshold (p{SHOCK_THRESHOLD_PERCENTILE}): {threshold:.4f}")
    
    return result


def create_constraint_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Create constraint indicators based on percentile thresholds."""
    
    result = df.copy()
    
    # Individual constraint indicators
    constraints = []
    
    # Capital ratio (inverted: low capital = constrained)
    if 'CAPITAL_RATIO' in df.columns:
        threshold = np.nanpercentile(df['CAPITAL_RATIO'], CAPITAL_RATIO_THRESHOLD_PERCENTILE)
        result['CAPITAL_CONSTRAINED'] = (df['CAPITAL_RATIO'] < threshold).astype(int)
        constraints.append('CAPITAL_CONSTRAINED')
        print(f"    Capital ratio threshold (p{CAPITAL_RATIO_THRESHOLD_PERCENTILE}): {threshold:.4f}")
    
    # TED spread (high = constrained)
    if 'TED_SPREAD' in df.columns:
        threshold = np.nanpercentile(df['TED_SPREAD'], CONSTRAINT_THRESHOLD_PERCENTILE)
        result['TED_CONSTRAINED'] = (df['TED_SPREAD'] > threshold).astype(int)
        constraints.append('TED_CONSTRAINED')
        print(f"    TED spread threshold (p{CONSTRAINT_THRESHOLD_PERCENTILE}): {threshold:.4f}")
    
    # VIX (high = constrained)
    if 'VIX' in df.columns:
        threshold = np.nanpercentile(df['VIX'], CONSTRAINT_THRESHOLD_PERCENTILE)
        result['VIX_CONSTRAINED'] = (df['VIX'] > threshold).astype(int)
        constraints.append('VIX_CONSTRAINED')
        print(f"    VIX threshold (p{CONSTRAINT_THRESHOLD_PERCENTILE}): {threshold:.4f}")
    
    # Composite constraint indicator (average of available)
    if constraints:
        result['CONSTRAINT_COMPOSITE'] = result[constraints].mean(axis=1)
        result['HIGH_CONSTRAINT'] = (result['CONSTRAINT_COMPOSITE'] >= 0.5).astype(int)
        print(f"    High constraint months: {result['HIGH_CONSTRAINT'].sum()}")
    
    return result


def create_time_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based variables for clustering and fixed effects."""
    
    result = df.copy()
    
    # Year and month
    result['year'] = result.index.year
    result['month'] = result.index.month
    
    # Quarter (for clustering)
    result['quarter'] = result.index.to_period('Q').astype(str)
    
    # Year-month string
    result['ym'] = result.index.to_period('M').astype(str)
    
    return result


def merge_all_data() -> tuple:
    """
    Merge all processed data into master dataset.
    
    Returns:
        master_data: DataFrame with all variables
        portfolio_returns: DataFrame with factor and portfolio returns
        industry_returns: DataFrame with 30 industry portfolio returns
    """
    print("=" * 60)
    print("PROCESSING DATA")
    print("=" * 60)
    
    # =========================================================================
    # PROCESS INDIVIDUAL DATASETS
    # =========================================================================
    print("\n[1/4] Processing individual datasets...")
    
    ff5 = process_ff5_factors()
    mom = process_momentum()
    rev = process_reversal_factors()
    bab = process_bab()
    ind30 = process_industry_portfolios()
    fred = process_fred_data()
    int_cap = process_intermediary_capital()
    mp_shock = process_monetary_shocks()
    fiscal_shock = process_fiscal_shocks()
    
    # =========================================================================
    # MERGE PORTFOLIO RETURNS
    # =========================================================================
    print("\n[2/4] Merging portfolio returns...")
    
    portfolio_dfs = [df for df in [ff5, mom, rev, bab] if not df.empty]
    
    if portfolio_dfs:
        portfolio_returns = pd.concat(portfolio_dfs, axis=1)
        portfolio_returns = portfolio_returns.loc[~portfolio_returns.index.duplicated(keep='first')]
        print(f"    Combined portfolios: {len(portfolio_returns)} obs, {len(portfolio_returns.columns)} series")
    else:
        print("    ✗ No portfolio data available")
        portfolio_returns = pd.DataFrame()
    
    # =========================================================================
    # MERGE CONSTRAINT VARIABLES
    # =========================================================================
    print("\n[3/4] Merging constraint and shock data...")
    
    constraint_dfs = []
    
    if not fred.empty:
        constraint_dfs.append(fred)
    
    if not int_cap.empty:
        # Interpolate quarterly to monthly
        int_cap_monthly = int_cap.resample('M').last().interpolate(method='linear')
        constraint_dfs.append(int_cap_monthly)
    
    if constraint_dfs:
        constraints = pd.concat(constraint_dfs, axis=1)
        constraints = constraints.loc[~constraints.index.duplicated(keep='first')]
    else:
        constraints = pd.DataFrame()
    
    # Add shocks
    shock_dfs = [df for df in [mp_shock, fiscal_shock] if not df.empty]
    if shock_dfs:
        shocks = pd.concat(shock_dfs, axis=1)
        shocks = shocks.loc[~shocks.index.duplicated(keep='first')]
    else:
        shocks = pd.DataFrame()
    
    # =========================================================================
    # CREATE MASTER DATASET
    # =========================================================================
    print("\n[4/4] Creating master dataset...")
    
    # Combine all
    all_dfs = [df for df in [portfolio_returns, constraints, shocks] if not df.empty]
    
    if all_dfs:
        master = pd.concat(all_dfs, axis=1)
        master = master.loc[~master.index.duplicated(keep='first')]
        
        # Filter to sample period
        sample_start = pd.to_datetime(SAMPLE_START)
        sample_end = pd.to_datetime(SAMPLE_END)
        master = master[(master.index >= sample_start) & (master.index <= sample_end)]
        
        # Create indicators
        print("\n  Creating indicators...")
        master = create_shock_indicators(master)
        master = create_constraint_indicators(master)
        master = create_time_variables(master)
        
        print(f"\n  Master dataset: {len(master)} obs, {len(master.columns)} variables")
        print(f"  Sample period: {master.index[0].strftime('%Y-%m')} to {master.index[-1].strftime('%Y-%m')}")
    else:
        print("    ✗ No data to merge")
        master = pd.DataFrame()
    
    # Filter industry returns to sample period
    if not ind30.empty:
        ind30 = ind30[(ind30.index >= sample_start) & (ind30.index <= sample_end)]
    
    return master, portfolio_returns, ind30


def save_processed_data(master: pd.DataFrame, portfolios: pd.DataFrame, 
                        industry: pd.DataFrame) -> None:
    """Save processed data to CSV files."""
    
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    if not master.empty:
        master.to_csv(PROCESSED_DATA_DIR / "master_data.csv")
        print(f"  ✓ Saved: master_data.csv ({len(master)} obs)")
    
    if not portfolios.empty:
        portfolios.to_csv(PROCESSED_DATA_DIR / "portfolio_returns.csv")
        print(f"  ✓ Saved: portfolio_returns.csv ({len(portfolios)} obs)")
    
    if not industry.empty:
        industry.to_csv(PROCESSED_DATA_DIR / "industry_returns.csv")
        print(f"  ✓ Saved: industry_returns.csv ({len(industry)} obs)")
    
    # Save data dictionary
    if not master.empty:
        data_dict = pd.DataFrame({
            'variable': master.columns,
            'dtype': master.dtypes.astype(str),
            'non_null': master.notna().sum(),
            'mean': master.select_dtypes(include=[np.number]).mean(),
            'std': master.select_dtypes(include=[np.number]).std()
        })
        data_dict.to_csv(PROCESSED_DATA_DIR / "data_dictionary.csv", index=False)
        print(f"  ✓ Saved: data_dictionary.csv")


def process_all():
    """Main processing function."""
    
    master, portfolios, industry = merge_all_data()
    save_processed_data(master, portfolios, industry)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    
    # Check for required data
    if 'MP_SHOCK' not in master.columns:
        print("\n⚠️  WARNING: Monetary shocks not found!")
        print("   Analysis cannot proceed without data/manual/monetary_shocks.csv")
    
    return master, portfolios, industry


if __name__ == "__main__":
    process_all()
