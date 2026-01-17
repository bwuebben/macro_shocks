"""
Data Download Module
====================

Downloads all publicly available data for the replication.

Data that CANNOT be downloaded automatically:
- Monetary policy shocks (Nakamura-Steinsson)
- Fiscal policy shocks (Ramey-Zubairy) 

These must be placed in data/manual/ - see README.md
"""

import os
import sys
import io
import zipfile
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    RAW_DATA_DIR, URLS, SAMPLE_START, SAMPLE_END, EXTENDED_START
)

# Suppress warnings
warnings.filterwarnings('ignore')


def download_file(url: str, desc: str = "Downloading") -> bytes:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    content = b""
    with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            content += chunk
            pbar.update(len(chunk))
    
    return content


def download_french_data(url: str, name: str) -> pd.DataFrame:
    """
    Download and parse data from Kenneth French's website.
    
    French's CSV files have a specific format with multiple sections.
    We extract the first section (monthly data).
    """
    print(f"  Downloading {name}...")
    
    try:
        content = download_file(url, desc=f"  {name}")
        
        # Unzip
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            # Get the CSV file name (usually only one file)
            csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
            csv_content = z.read(csv_name).decode('utf-8')
        
        # Parse the CSV
        # French's files have header rows and multiple sections
        lines = csv_content.strip().split('\n')
        
        # Find the start of data (first line that starts with a date-like number)
        data_start = 0
        for i, line in enumerate(lines):
            first_col = line.split(',')[0].strip()
            if first_col.isdigit() and len(first_col) == 6:  # YYYYMM format
                data_start = i
                break
        
        # Find the end of the first section (blank line or non-numeric start)
        data_end = len(lines)
        for i in range(data_start + 1, len(lines)):
            first_col = lines[i].split(',')[0].strip()
            if not first_col or (not first_col.replace('-', '').replace('.', '').isdigit()):
                data_end = i
                break
        
        # Get header (line before data)
        header_line = lines[data_start - 1] if data_start > 0 else lines[0]
        headers = [h.strip() for h in header_line.split(',')]
        
        # Parse data section
        data_lines = lines[data_start:data_end]
        data_str = '\n'.join(data_lines)
        
        df = pd.read_csv(io.StringIO(data_str), header=None, names=headers)
        
        # Parse date column (first column, format YYYYMM)
        date_col = df.columns[0]
        df[date_col] = df[date_col].astype(str).str.strip()
        df = df[df[date_col].str.match(r'^\d{6}$', na=False)]
        df['date'] = pd.to_datetime(df[date_col], format='%Y%m') + pd.offsets.MonthEnd(0)
        df = df.drop(columns=[date_col])
        
        # Convert to numeric
        for col in df.columns:
            if col != 'date':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Set date as index
        df = df.set_index('date').sort_index()
        
        # French data is in percent, convert to decimal
        df = df / 100
        
        print(f"    ✓ {name}: {len(df)} observations, {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
        
        return df
        
    except Exception as e:
        print(f"    ✗ Failed to download {name}: {e}")
        return None


def download_fred_data() -> pd.DataFrame:
    """
    Download VIX and TED spread from FRED.
    
    Uses pandas_datareader or direct FRED API.
    """
    print("  Downloading FRED data...")
    
    try:
        import pandas_datareader.data as web
        
        start = pd.to_datetime(EXTENDED_START)
        end = pd.to_datetime(SAMPLE_END)
        
        # Download VIX
        print("    Downloading VIX...")
        try:
            vix = web.DataReader(URLS["VIX"], 'fred', start, end)
            vix.columns = ['VIX']
            print(f"    ✓ VIX: {len(vix)} observations")
        except Exception as e:
            print(f"    ✗ VIX download failed: {e}")
            vix = pd.DataFrame()
        
        # Download TED spread
        print("    Downloading TED spread...")
        try:
            ted = web.DataReader(URLS["TED_SPREAD"], 'fred', start, end)
            ted.columns = ['TED_SPREAD']
            print(f"    ✓ TED spread: {len(ted)} observations")
        except Exception as e:
            print(f"    ✗ TED spread download failed: {e}")
            ted = pd.DataFrame()
        
        # Combine and resample to monthly
        fred_data = pd.concat([vix, ted], axis=1)
        
        # Resample to end-of-month
        fred_monthly = fred_data.resample('M').last()
        
        return fred_monthly
        
    except ImportError:
        print("    ✗ pandas_datareader not installed. Install with: pip install pandas-datareader")
        return pd.DataFrame()


def download_intermediary_capital() -> pd.DataFrame:
    """
    Download He-Kelly-Manela intermediary capital ratio.
    
    This data is hosted on Google Drive. If download fails,
    user must provide manually.
    """
    print("  Downloading Intermediary Capital data...")
    
    try:
        # Try to download from the URL
        url = URLS["INTERMEDIARY_CAPITAL"]
        
        # This is a Google Drive link, need special handling
        response = requests.get(url)
        
        if response.status_code == 200:
            df = pd.read_csv(io.StringIO(response.text))
            
            # Parse date
            if 'yyyyq' in df.columns:
                # Format: YYYYQ (e.g., 19901 = 1990 Q1)
                df['year'] = df['yyyyq'] // 10
                df['quarter'] = df['yyyyq'] % 10
                df['date'] = pd.to_datetime(df['year'].astype(str) + 'Q' + df['quarter'].astype(str)) + pd.offsets.QuarterEnd(0)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            # Get capital ratio column
            cap_col = [c for c in df.columns if 'capital' in c.lower() or 'intermediary' in c.lower()]
            if cap_col:
                df = df[['date', cap_col[0]]].copy()
                df.columns = ['date', 'CAPITAL_RATIO']
            
            df = df.set_index('date').sort_index()
            
            print(f"    ✓ Intermediary Capital: {len(df)} observations")
            return df
        else:
            raise Exception(f"HTTP {response.status_code}")
            
    except Exception as e:
        print(f"    ✗ Intermediary Capital download failed: {e}")
        print("      → Please provide data/manual/intermediary_capital.csv")
        return pd.DataFrame()


def download_bab_factor() -> pd.DataFrame:
    """
    Download Betting Against Beta factor from AQR.
    
    AQR provides this as an Excel file.
    """
    print("  Downloading BAB factor from AQR...")
    
    try:
        url = URLS["BAB"]
        
        # Download Excel file
        response = requests.get(url)
        response.raise_for_status()
        
        # Read Excel
        df = pd.read_excel(io.BytesIO(response.content), sheet_name='BAB Factors', skiprows=18)
        
        # Parse date
        df = df.rename(columns={df.columns[0]: 'date'})
        df['date'] = pd.to_datetime(df['date'])
        
        # Get US BAB factor
        bab_col = [c for c in df.columns if 'USA' in str(c).upper() or 'US' in str(c).upper()]
        if bab_col:
            df = df[['date', bab_col[0]]].copy()
            df.columns = ['date', 'BAB']
        else:
            # Just take the second column
            df = df[['date', df.columns[1]]].copy()
            df.columns = ['date', 'BAB']
        
        df = df.set_index('date').sort_index()
        df = df[df.index.notna()]
        
        # Convert to decimal if in percent
        if df['BAB'].abs().mean() > 1:
            df['BAB'] = df['BAB'] / 100
        
        print(f"    ✓ BAB factor: {len(df)} observations")
        return df
        
    except Exception as e:
        print(f"    ✗ BAB download failed: {e}")
        print("      → Download manually from AQR website")
        return pd.DataFrame()


def download_all():
    """
    Download all publicly available data.
    
    Returns dict of DataFrames.
    """
    print("=" * 60)
    print("DOWNLOADING DATA")
    print("=" * 60)
    
    data = {}
    
    # =========================================================================
    # KENNETH FRENCH DATA
    # =========================================================================
    print("\n[1/5] Kenneth French Data Library")
    
    # Fama-French 5 factors
    ff5 = download_french_data(URLS["FF5_FACTORS"], "FF5 Factors")
    if ff5 is not None:
        data['ff5_factors'] = ff5
        ff5.to_csv(RAW_DATA_DIR / "ff5_factors.csv")
    
    # Momentum factor
    mom = download_french_data(URLS["MOMENTUM"], "Momentum Factor")
    if mom is not None:
        data['momentum'] = mom
        mom.to_csv(RAW_DATA_DIR / "momentum.csv")
    
    # 30 Industry portfolios
    ind30 = download_french_data(URLS["30_INDUSTRY"], "30 Industry Portfolios")
    if ind30 is not None:
        # Keep only value-weighted returns (first section is VW)
        data['industry_30'] = ind30
        ind30.to_csv(RAW_DATA_DIR / "industry_30.csv")
    
    # Short-term reversal
    st_rev = download_french_data(URLS["ST_REVERSAL"], "ST Reversal Factor")
    if st_rev is not None:
        data['st_reversal'] = st_rev
        st_rev.to_csv(RAW_DATA_DIR / "st_reversal.csv")
    
    # Long-term reversal
    lt_rev = download_french_data(URLS["LT_REVERSAL"], "LT Reversal Factor")
    if lt_rev is not None:
        data['lt_reversal'] = lt_rev
        lt_rev.to_csv(RAW_DATA_DIR / "lt_reversal.csv")
    
    # =========================================================================
    # FRED DATA
    # =========================================================================
    print("\n[2/5] FRED Data (VIX, TED Spread)")
    
    fred = download_fred_data()
    if not fred.empty:
        data['fred'] = fred
        fred.to_csv(RAW_DATA_DIR / "fred_data.csv")
    
    # =========================================================================
    # INTERMEDIARY CAPITAL
    # =========================================================================
    print("\n[3/5] Intermediary Capital (He-Kelly-Manela)")
    
    int_cap = download_intermediary_capital()
    if not int_cap.empty:
        data['intermediary_capital'] = int_cap
        int_cap.to_csv(RAW_DATA_DIR / "intermediary_capital.csv")
    
    # =========================================================================
    # AQR BAB FACTOR
    # =========================================================================
    print("\n[4/5] Betting Against Beta (AQR)")
    
    bab = download_bab_factor()
    if not bab.empty:
        data['bab'] = bab
        bab.to_csv(RAW_DATA_DIR / "bab_factor.csv")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n[5/5] Download Summary")
    print("-" * 60)
    
    downloaded = [k for k, v in data.items() if v is not None and not (isinstance(v, pd.DataFrame) and v.empty)]
    print(f"Successfully downloaded: {len(downloaded)} datasets")
    for name in downloaded:
        print(f"  ✓ {name}")
    
    print("\n⚠️  MANUAL DATA REQUIRED:")
    print("  - data/manual/monetary_shocks.csv (REQUIRED)")
    print("  - data/manual/fiscal_shocks.csv (optional)")
    if 'intermediary_capital' not in downloaded:
        print("  - data/manual/intermediary_capital.csv")
    
    print("=" * 60)
    
    return data


if __name__ == "__main__":
    download_all()
