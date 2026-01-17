#!/usr/bin/env python
"""
Analysis-Only Script
====================

Run this script after data has been downloaded and processed:
    python run_analysis.py

This assumes data is already in data/processed/
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.load import load_master_data, load_portfolio_returns, load_industry_returns
from src.analysis.effective_dimension import run_dimension_analysis
from src.analysis.shock_response import run_shock_analysis
from src.analysis.constraint_interaction import run_constraint_analysis
from src.analysis.portfolio_loadings import run_portfolio_analysis
from src.analysis.robustness import run_robustness_analysis


def main():
    """Run all analyses."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    try:
        master = load_master_data()
        portfolios = load_portfolio_returns()
        industry = load_industry_returns()
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run data download and processing first:")
        print("  python -m src.data.download")
        print("  python -m src.data.process")
        sys.exit(1)
    
    print(f"\n  Master data: {len(master)} observations")
    print(f"  Portfolios: {len(portfolios)} observations, {len(portfolios.columns)} series")
    print(f"  Industries: {len(industry)} observations, {len(industry.columns)} series")
    
    results = {}
    
    # 1. Effective Dimension
    print("\n")
    dim_results = run_dimension_analysis(industry, master)
    results['dimension'] = dim_results
    
    if 'dimension_series' in dim_results:
        master = master.join(dim_results['dimension_series'][['d_eff']], how='left')
    
    # 2. Shock Response
    print("\n")
    results['shock'] = run_shock_analysis(master)
    
    # 3. Constraint Mechanism
    print("\n")
    results['constraint'] = run_constraint_analysis(master)
    
    # 4. Portfolio Loadings
    print("\n")
    if 'd_eff' in master.columns:
        results['portfolio'] = run_portfolio_analysis(portfolios, industry, master['d_eff'])
    
    # 5. Robustness
    print("\n")
    results['robustness'] = run_robustness_analysis(industry, master)
    
    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
