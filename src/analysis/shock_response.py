"""
Shock Response Analysis
=======================

Tests Hypothesis 1: Contractionary macroeconomic shocks increase 
the effective dimension of asset returns.

This module estimates:
    d_eff(t+1) = α + β * Shock_t + controls + ε_t
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    TABLES_DIR, FIGURES_DIR, FIGURE_SIZE, FIGURE_DPI,
    COLOR_PRIMARY, COLOR_SECONDARY, DECIMALS_COEFFICIENTS
)
from src.utils.helpers import ols_regression, create_latex_table, format_coefficient, format_se


def test_shock_dimension_response(data: pd.DataFrame,
                                  shock_var: str = 'MP_SHOCK_IND',
                                  dimension_var: str = 'd_eff',
                                  controls: List[str] = None,
                                  cluster_var: str = 'quarter') -> Dict:
    """
    Test whether shocks predict higher effective dimension.
    
    Parameters:
        data: DataFrame with shock indicators and dimension
        shock_var: Name of shock indicator variable
        dimension_var: Name of dimension variable
        controls: List of control variables
        cluster_var: Variable to cluster standard errors on
    
    Returns:
        Dictionary with regression results
    """
    # Check required columns
    required = [shock_var, dimension_var]
    if controls:
        required.extend(controls)
    
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Create lead of dimension (shock at t predicts dimension at t+1)
    data = data.copy()
    data['d_eff_lead'] = data[dimension_var].shift(-1)
    
    # Drop missing
    analysis_vars = ['d_eff_lead', shock_var] + (controls or [])
    if cluster_var in data.columns:
        analysis_vars.append(cluster_var)
    
    clean_data = data[analysis_vars].dropna()
    
    # Run regression
    y = clean_data['d_eff_lead']
    X = clean_data[[shock_var] + (controls or [])]
    cluster = clean_data[cluster_var] if cluster_var in clean_data.columns else None
    
    results = ols_regression(y, X, cluster_var=cluster)
    
    # Add interpretation
    shock_coef = results['coef'][shock_var]
    shock_pval = results['p_value'][shock_var]
    
    results['interpretation'] = {
        'shock_effect': shock_coef,
        'shock_pvalue': shock_pval,
        'significant_5pct': shock_pval < 0.05,
        'significant_10pct': shock_pval < 0.10,
        'mean_dimension': y.mean(),
        'effect_pct': (shock_coef / y.mean()) * 100
    }
    
    return results


def test_conditional_shock_response(data: pd.DataFrame,
                                    shock_var: str = 'MP_SHOCK_IND',
                                    constraint_var: str = 'HIGH_CONSTRAINT',
                                    dimension_var: str = 'd_eff',
                                    cluster_var: str = 'quarter') -> Dict:
    """
    Test shock response conditional on constraint states (H2 test).
    
    Estimates:
        d_eff(t+1) = α + β1*Shock + β2*Constraint + β3*Shock×Constraint + ε
    
    Parameters:
        data: DataFrame with variables
        shock_var: Shock indicator
        constraint_var: Constraint indicator
        dimension_var: Dimension variable
        cluster_var: Clustering variable
    
    Returns:
        Dictionary with results including interaction effects
    """
    data = data.copy()
    
    # Create lead dimension
    data['d_eff_lead'] = data[dimension_var].shift(-1)
    
    # Create interaction
    data['shock_x_constraint'] = data[shock_var] * data[constraint_var]
    
    # Clean data
    vars_needed = ['d_eff_lead', shock_var, constraint_var, 'shock_x_constraint']
    if cluster_var in data.columns:
        vars_needed.append(cluster_var)
    
    clean_data = data[vars_needed].dropna()
    
    # Full model with interaction
    y = clean_data['d_eff_lead']
    X_full = clean_data[[shock_var, constraint_var, 'shock_x_constraint']]
    cluster = clean_data[cluster_var] if cluster_var in clean_data.columns else None
    
    results_full = ols_regression(y, X_full, cluster_var=cluster)
    
    # Separate regressions for high/low constraint
    high_mask = clean_data[constraint_var] == 1
    low_mask = clean_data[constraint_var] == 0
    
    results_high = ols_regression(
        clean_data.loc[high_mask, 'd_eff_lead'],
        clean_data.loc[high_mask, [shock_var]],
        cluster_var=clean_data.loc[high_mask, cluster_var] if cluster is not None else None
    )

    results_low = ols_regression(
        clean_data.loc[low_mask, 'd_eff_lead'],
        clean_data.loc[low_mask, [shock_var]],
        cluster_var=clean_data.loc[low_mask, cluster_var] if cluster is not None else None
    )
    
    # Test if difference is significant
    coef_diff = results_high['coef'][shock_var] - results_low['coef'][shock_var]
    se_diff = np.sqrt(results_high['se'][shock_var]**2 + results_low['se'][shock_var]**2)
    t_diff = coef_diff / se_diff
    p_diff = 2 * (1 - stats.t.cdf(abs(t_diff), df=len(clean_data) - 4))
    
    return {
        'full_model': results_full,
        'high_constraint': results_high,
        'low_constraint': results_low,
        'difference_test': {
            'coef_diff': coef_diff,
            'se_diff': se_diff,
            't_stat': t_diff,
            'p_value': p_diff
        }
    }


def create_table2_dimension_shocks(data: pd.DataFrame,
                                   save_path: Path = None) -> pd.DataFrame:
    """
    Create Table 2: Effective Dimension Response to Macroeconomic Shocks.
    
    Panel A: Unconditional response
    Panel B: Conditional on constraints
    """
    results = {}
    
    # Panel A: Unconditional
    # Column 1: Monetary shock only
    if 'MP_SHOCK_IND' in data.columns:
        results['mp_only'] = test_shock_dimension_response(
            data, shock_var='MP_SHOCK_IND'
        )
    
    # Column 2: Fiscal shock only
    if 'FISCAL_SHOCK_IND' in data.columns:
        results['fiscal_only'] = test_shock_dimension_response(
            data, shock_var='FISCAL_SHOCK_IND'
        )
    
    # Column 3: Both shocks
    if 'MP_SHOCK_IND' in data.columns and 'FISCAL_SHOCK_IND' in data.columns:
        results['both'] = test_shock_dimension_response(
            data, shock_var='MP_SHOCK_IND', controls=['FISCAL_SHOCK_IND']
        )
    
    # Panel B: Conditional
    if 'MP_SHOCK_IND' in data.columns and 'HIGH_CONSTRAINT' in data.columns:
        results['conditional_mp'] = test_conditional_shock_response(
            data, shock_var='MP_SHOCK_IND'
        )
    
    if 'FISCAL_SHOCK_IND' in data.columns and 'HIGH_CONSTRAINT' in data.columns:
        results['conditional_fiscal'] = test_conditional_shock_response(
            data, shock_var='FISCAL_SHOCK_IND'
        )
    
    # Format as table
    rows = []
    
    # Panel A header
    rows.append(['\\multicolumn{4}{l}{\\textit{Panel A: Unconditional}}', '', '', ''])
    rows.append(['', '', '', ''])
    
    # Monetary shock row
    if 'mp_only' in results:
        mp = results['mp_only']
        row = ['Monetary shock ($\\varepsilon_t^m < 0$)']
        row.append(format_coefficient(mp['coef']['MP_SHOCK_IND'], mp['se']['MP_SHOCK_IND'], mp['p_value']['MP_SHOCK_IND']))
        row.append('')
        if 'both' in results:
            both = results['both']
            row.append(format_coefficient(both['coef']['MP_SHOCK_IND'], both['se']['MP_SHOCK_IND'], both['p_value']['MP_SHOCK_IND']))
        else:
            row.append('')
        rows.append(row)
        
        # SE row
        row_se = ['']
        row_se.append(format_se(mp['se']['MP_SHOCK_IND']))
        row_se.append('')
        if 'both' in results:
            row_se.append(format_se(both['se']['MP_SHOCK_IND']))
        else:
            row_se.append('')
        rows.append(row_se)
    
    # Fiscal shock row
    if 'fiscal_only' in results:
        fiscal = results['fiscal_only']
        row = ['Fiscal shock ($\\varepsilon_t^f < 0$)']
        row.append('')
        row.append(format_coefficient(fiscal['coef']['FISCAL_SHOCK_IND'], fiscal['se']['FISCAL_SHOCK_IND'], fiscal['p_value']['FISCAL_SHOCK_IND']))
        if 'both' in results:
            both = results['both']
            row.append(format_coefficient(both['coef']['FISCAL_SHOCK_IND'], both['se']['FISCAL_SHOCK_IND'], both['p_value']['FISCAL_SHOCK_IND']))
        else:
            row.append('')
        rows.append(row)
        
        # SE row
        row_se = ['']
        row_se.append('')
        row_se.append(format_se(fiscal['se']['FISCAL_SHOCK_IND']))
        if 'both' in results:
            row_se.append(format_se(both['se']['FISCAL_SHOCK_IND']))
        else:
            row_se.append('')
        rows.append(row_se)
    
    # R-squared row
    row_r2 = ['$R^2$']
    for key in ['mp_only', 'fiscal_only', 'both']:
        if key in results:
            row_r2.append(f"{results[key]['r2']:.2f}")
        else:
            row_r2.append('')
    rows.append(row_r2)
    
    # Panel B
    rows.append(['\\midrule', '', '', ''])
    rows.append(['\\multicolumn{4}{l}{\\textit{Panel B: Conditional on Constraints}}', '', '', ''])
    rows.append(['', '', '', ''])
    
    if 'conditional_mp' in results:
        cond = results['conditional_mp']
        
        # High constraint
        high = cond['high_constraint']
        row = ['Shock $\\times$ High constraints']
        row.append(format_coefficient(high['coef']['MP_SHOCK_IND'], high['se']['MP_SHOCK_IND'], high['p_value']['MP_SHOCK_IND']))
        if 'conditional_fiscal' in results:
            fiscal_high = results['conditional_fiscal']['high_constraint']
            row.append(format_coefficient(fiscal_high['coef']['FISCAL_SHOCK_IND'], fiscal_high['se']['FISCAL_SHOCK_IND'], fiscal_high['p_value']['FISCAL_SHOCK_IND']))
        else:
            row.append('')
        row.append('')
        rows.append(row)
        
        row_se = ['']
        row_se.append(format_se(high['se']['MP_SHOCK_IND']))
        if 'conditional_fiscal' in results:
            row_se.append(format_se(results['conditional_fiscal']['high_constraint']['se']['FISCAL_SHOCK_IND']))
        else:
            row_se.append('')
        row_se.append('')
        rows.append(row_se)
        
        # Low constraint
        low = cond['low_constraint']
        row = ['Shock $\\times$ Low constraints']
        row.append(format_coefficient(low['coef']['MP_SHOCK_IND'], low['se']['MP_SHOCK_IND'], low['p_value']['MP_SHOCK_IND']))
        if 'conditional_fiscal' in results:
            fiscal_low = results['conditional_fiscal']['low_constraint']
            row.append(format_coefficient(fiscal_low['coef']['FISCAL_SHOCK_IND'], fiscal_low['se']['FISCAL_SHOCK_IND'], fiscal_low['p_value']['FISCAL_SHOCK_IND']))
        else:
            row.append('')
        row.append('')
        rows.append(row)
        
        row_se = ['']
        row_se.append(format_se(low['se']['MP_SHOCK_IND']))
        if 'conditional_fiscal' in results:
            row_se.append(format_se(results['conditional_fiscal']['low_constraint']['se']['FISCAL_SHOCK_IND']))
        else:
            row_se.append('')
        row_se.append('')
        rows.append(row_se)
        
        # p-value for difference
        diff = cond['difference_test']
        row = ['$p$-value: High = Low']
        row.append(f"{diff['p_value']:.3f}")
        if 'conditional_fiscal' in results:
            diff_fiscal = results['conditional_fiscal']['difference_test']
            row.append(f"{diff_fiscal['p_value']:.3f}")
        else:
            row.append('')
        row.append('')
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=['', '(1)', '(2)', '(3)'])
    
    if save_path:
        # Create LaTeX
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Effective Dimension Response to Macroeconomic Shocks}',
            '\\label{tab:dimension_shocks}',
            '\\begin{tabular}{lccc}',
            '\\toprule',
            ' & (1) & (2) & (3) \\\\',
            ' & $\\hat{d}_{\\text{eff}}(t+1)$ & $\\hat{d}_{\\text{eff}}(t+1)$ & $\\hat{d}_{\\text{eff}}(t+1)$ \\\\',
            '\\midrule'
        ]
        
        for _, row in df.iterrows():
            line = ' & '.join(str(v) for v in row.values) + ' \\\\'
            latex_lines.append(line)
        
        latex_lines.extend([
            '\\bottomrule',
            '\\end{tabular}',
            '',
            '\\vspace{0.5em}',
            '\\footnotesize',
            '\\textit{Notes:} The dependent variable is effective dimension in month $t+1$. Shock indicators equal one for negative shocks below the 10th percentile. High constraints indicates periods when the composite constraint measure exceeds 0.5. Standard errors in parentheses, clustered by quarter. ***, **, * denote significance at 1\\%, 5\\%, 10\\%.',
            '\\end{table}'
        ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: {save_path}")
    
    return df, results


def run_shock_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete shock response analysis.
    
    Parameters:
        data: Master dataset with dimension and shocks
    
    Returns:
        Dictionary with all results
    """
    print("=" * 60)
    print("SHOCK RESPONSE ANALYSIS (H1)")
    print("=" * 60)
    
    results = {}
    
    # Check data availability
    has_mp = 'MP_SHOCK_IND' in data.columns
    has_fiscal = 'FISCAL_SHOCK_IND' in data.columns
    has_dim = 'd_eff' in data.columns
    has_constraint = 'HIGH_CONSTRAINT' in data.columns
    
    print(f"\n  Data availability:")
    print(f"    Monetary shocks: {'✓' if has_mp else '✗'}")
    print(f"    Fiscal shocks: {'✓' if has_fiscal else '✗'}")
    print(f"    Effective dimension: {'✓' if has_dim else '✗'}")
    print(f"    Constraint indicator: {'✓' if has_constraint else '✗'}")
    
    if not has_dim:
        print("\n  ✗ Cannot run analysis without effective dimension")
        return results
    
    if not has_mp and not has_fiscal:
        print("\n  ✗ Cannot run analysis without shock data")
        return results
    
    # Create Table 2
    print("\n  Creating Table 2: Dimension response to shocks...")
    table2, table2_results = create_table2_dimension_shocks(
        data,
        save_path=TABLES_DIR / "table2_dimension_shocks.tex"
    )
    results['table2'] = table2
    results['table2_results'] = table2_results
    
    # Print key results
    if 'mp_only' in table2_results:
        mp = table2_results['mp_only']
        print(f"\n  Key result (H1):")
        print(f"    Monetary shock → dimension: {mp['coef']['MP_SHOCK_IND']:.2f}")
        print(f"    t-statistic: {mp['t_stat']['MP_SHOCK_IND']:.2f}")
        print(f"    p-value: {mp['p_value']['MP_SHOCK_IND']:.3f}")
    
    if 'conditional_mp' in table2_results:
        cond = table2_results['conditional_mp']
        print(f"\n  Key result (H2):")
        print(f"    High constraint effect: {cond['high_constraint']['coef']['MP_SHOCK_IND']:.2f}")
        print(f"    Low constraint effect: {cond['low_constraint']['coef']['MP_SHOCK_IND']:.2f}")
        print(f"    Difference p-value: {cond['difference_test']['p_value']:.3f}")
    
    print("\n" + "=" * 60)
    print("SHOCK ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from src.data.load import load_master_data
    
    try:
        data = load_master_data()
        results = run_shock_analysis(data)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
