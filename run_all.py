#!/usr/bin/env python
"""
Master Replication Script
=========================

Run this script to execute the full replication:
    python run_all.py

This will:
1. Download publicly available data
2. Process and merge all data
3. Run all analyses
4. Generate all tables and figures

BEFORE RUNNING: Place required manual data files in data/manual/
See README.md for details.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    PROJECT_ROOT, MANUAL_DATA_DIR, PROCESSED_DATA_DIR,
    TABLES_DIR, FIGURES_DIR
)


def print_header():
    """Print script header."""
    print("\n")
    print("=" * 70)
    print("  REPLICATION: Macroeconomic Shocks and Effective Dimension")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Project root: {PROJECT_ROOT}")
    print("=" * 70)


def check_manual_data():
    """Check if required manual data files exist."""
    print("\n" + "=" * 70)
    print("  CHECKING MANUAL DATA REQUIREMENTS")
    print("=" * 70)
    
    required = {
        'monetary_shocks.csv': 'REQUIRED - Nakamura-Steinsson monetary policy shocks'
    }
    
    optional = {
        'fiscal_shocks.csv': 'OPTIONAL - Ramey-Zubairy fiscal policy shocks',
        'intermediary_capital.csv': 'OPTIONAL - He-Kelly-Manela (attempted auto-download)'
    }
    
    all_ok = True
    
    print("\n  Required files:")
    for filename, description in required.items():
        filepath = MANUAL_DATA_DIR / filename
        if filepath.exists():
            print(f"    ✓ {filename} - Found")
        else:
            print(f"    ✗ {filename} - MISSING")
            print(f"      {description}")
            all_ok = False
    
    print("\n  Optional files:")
    for filename, description in optional.items():
        filepath = MANUAL_DATA_DIR / filename
        if filepath.exists():
            print(f"    ✓ {filename} - Found")
        else:
            print(f"    ○ {filename} - Not found")
            print(f"      {description}")
    
    if not all_ok:
        print("\n" + "!" * 70)
        print("  ERROR: Required manual data files are missing!")
        print("  Please provide the following files in data/manual/:")
        for filename, description in required.items():
            filepath = MANUAL_DATA_DIR / filename
            if not filepath.exists():
                print(f"    - {filename}")
        print("\n  See README.md for file format specifications.")
        print("!" * 70)
        return False
    
    return True


def run_data_pipeline():
    """Run data download and processing."""
    from src.data.download import download_all
    from src.data.process import process_all
    
    print("\n")
    download_all()
    
    print("\n")
    master, portfolios, industry = process_all()
    
    return master, portfolios, industry


def run_analysis_pipeline(master, portfolios, industry):
    """Run all analyses."""
    from src.analysis.effective_dimension import run_dimension_analysis
    from src.analysis.shock_response import run_shock_analysis
    from src.analysis.constraint_interaction import run_constraint_analysis
    from src.analysis.portfolio_loadings import run_portfolio_analysis
    from src.analysis.robustness import run_robustness_analysis
    
    results = {}
    
    # 1. Effective Dimension Analysis
    print("\n")
    dim_results = run_dimension_analysis(industry, master)
    results['dimension'] = dim_results
    
    # Update master with dimension
    if 'd_eff' not in master.columns and 'dimension_series' in dim_results:
        master = master.join(dim_results['dimension_series'][['d_eff']], how='left')
    
    # 2. Shock Response Analysis (H1)
    print("\n")
    shock_results = run_shock_analysis(master)
    results['shock'] = shock_results
    
    # 3. Constraint Mechanism Analysis (H2)
    print("\n")
    constraint_results = run_constraint_analysis(master)
    results['constraint'] = constraint_results
    
    # 4. Portfolio Loadings Analysis (H3)
    print("\n")
    if 'd_eff' in master.columns:
        portfolio_results = run_portfolio_analysis(portfolios, industry, master['d_eff'])
        results['portfolio'] = portfolio_results
    else:
        print("Skipping portfolio analysis - dimension not computed")
    
    # 5. Robustness Checks
    print("\n")
    robust_results = run_robustness_analysis(industry, master)
    results['robustness'] = robust_results
    
    return results


def print_summary():
    """Print summary of outputs."""
    print("\n" + "=" * 70)
    print("  OUTPUT SUMMARY")
    print("=" * 70)
    
    # Tables
    print("\n  Tables generated:")
    tables = list(TABLES_DIR.glob("*.tex"))
    for t in sorted(tables):
        print(f"    - {t.name}")
    
    # Figures
    print("\n  Figures generated:")
    figures = list(FIGURES_DIR.glob("*.pdf")) + list(FIGURES_DIR.glob("*.png"))
    for f in sorted(figures):
        print(f"    - {f.name}")
    
    # Data
    print("\n  Processed data:")
    data_files = list(PROCESSED_DATA_DIR.glob("*.csv"))
    for d in sorted(data_files):
        print(f"    - {d.name}")


def main():
    """Main entry point."""
    start_time = time.time()
    
    print_header()
    
    # Check manual data
    if not check_manual_data():
        print("\n  Exiting due to missing required data.")
        sys.exit(1)
    
    # Run data pipeline
    print("\n")
    print("=" * 70)
    print("  STAGE 1: DATA DOWNLOAD AND PROCESSING")
    print("=" * 70)
    
    master, portfolios, industry = run_data_pipeline()
    
    if master.empty:
        print("\n  ERROR: Data processing failed. Cannot continue.")
        sys.exit(1)
    
    # Run analysis pipeline
    print("\n")
    print("=" * 70)
    print("  STAGE 2: ANALYSIS")
    print("=" * 70)
    
    results = run_analysis_pipeline(master, portfolios, industry)
    
    # Summary
    print_summary()
    
    # Timing
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"  REPLICATION COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
