"""
Constraint Interaction Analysis
===============================

Tests the mechanism: shocks → constraints → dimension

This module:
1. Tests whether shocks worsen constraints (Table 5)
2. Tests the interaction of shocks and constraints on dimension (Table 6)
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import TABLES_DIR, FIGURES_DIR
from src.utils.helpers import ols_regression, format_coefficient, format_se


def test_shock_constraint_effect(data: pd.DataFrame,
                                 shock_var: str = 'MP_SHOCK_IND',
                                 constraint_vars: List[str] = None,
                                 cluster_var: str = 'quarter') -> Dict:
    """
    Test whether shocks tighten arbitrage constraints.
    
    Estimates for each constraint:
        Constraint(t+1) = α + β*Shock(t) + γ*Constraint(t) + ε
    
    Parameters:
        data: Master dataset
        shock_var: Shock indicator variable
        constraint_vars: List of constraint variables to test
        cluster_var: Clustering variable
    
    Returns:
        Dictionary with results for each constraint
    """
    if constraint_vars is None:
        constraint_vars = ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']
    
    # Filter to available constraints
    available = [c for c in constraint_vars if c in data.columns]
    
    if not available:
        print("  No constraint variables available")
        return {}
    
    results = {}
    
    for constraint in available:
        # Create leads
        data_temp = data.copy()
        data_temp[f'{constraint}_lead'] = data_temp[constraint].shift(-1)
        
        # For capital ratio, lower is more constrained, so we flip sign interpretation
        # For others, higher is more constrained
        
        # Clean data
        vars_needed = [f'{constraint}_lead', shock_var, constraint]
        if cluster_var in data_temp.columns:
            vars_needed.append(cluster_var)
        
        clean = data_temp[vars_needed].dropna()
        
        if len(clean) < 20:
            continue
        
        # Run regression
        y = clean[f'{constraint}_lead']
        X = clean[[shock_var, constraint]]
        cluster = clean[cluster_var] if cluster_var in clean.columns else None
        
        reg_results = ols_regression(y, X, cluster_var=cluster)
        
        results[constraint] = {
            'regression': reg_results,
            'shock_coef': reg_results['coef'][shock_var],
            'shock_se': reg_results['se'][shock_var],
            'shock_pval': reg_results['p_value'][shock_var],
            'n_obs': reg_results['n_obs'],
            'r2': reg_results['r2']
        }
    
    return results


def test_interaction_regression(data: pd.DataFrame,
                               shock_var: str = 'MP_SHOCK_IND',
                               constraint_var: str = 'CONSTRAINT_COMPOSITE',
                               dimension_var: str = 'd_eff',
                               include_lag: bool = True,
                               cluster_var: str = 'quarter') -> Dict:
    """
    Test interaction of shocks and constraints on dimension.
    
    Estimates:
        d_eff(t+1) = α + β1*Shock + β2*Constraint + β3*Shock×Constraint + [β4*d_eff(t)] + ε
    
    Parameters:
        data: Master dataset
        shock_var: Shock indicator
        constraint_var: Constraint measure (continuous)
        dimension_var: Dimension variable
        include_lag: Whether to include lagged dimension
        cluster_var: Clustering variable
    
    Returns:
        Regression results
    """
    data = data.copy()
    
    # Create variables
    data['d_eff_lead'] = data[dimension_var].shift(-1)
    data['shock_x_constraint'] = data[shock_var] * data[constraint_var]
    
    # Build regressor list
    regressors = [shock_var, constraint_var, 'shock_x_constraint']
    if include_lag:
        regressors.append(dimension_var)
    
    # Clean
    vars_needed = ['d_eff_lead'] + regressors
    if cluster_var in data.columns:
        vars_needed.append(cluster_var)
    
    clean = data[vars_needed].dropna()
    
    y = clean['d_eff_lead']
    X = clean[regressors]
    cluster = clean[cluster_var] if cluster_var in clean.columns else None
    
    return ols_regression(y, X, cluster_var=cluster)


def create_table5_shock_constraints(data: pd.DataFrame,
                                    save_path: Path = None) -> pd.DataFrame:
    """
    Create Table 5: Effect of Macroeconomic Shocks on Arbitrage Constraints.
    """
    # Test shock effects
    results = test_shock_constraint_effect(data)
    
    if not results:
        print("  No results to create table")
        return pd.DataFrame()
    
    # Build table
    rows = []
    
    # Header row with column names
    constraint_names = {
        'CAPITAL_RATIO': 'Capital ratio',
        'TED_SPREAD': 'TED spread',
        'VIX': 'VIX',
        'CONSTRAINT_COMPOSITE': 'Composite'
    }
    
    # Shock coefficient row
    row = ['Monetary shock ($\\varepsilon_t^m < 0$)']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            r = results[constraint]
            # For capital ratio, negative shock coef means constraint tightens (ratio falls)
            coef = r['shock_coef']
            if constraint == 'CAPITAL_RATIO':
                # Flip sign for interpretation: positive effect = tighter constraint
                display_coef = -coef
            else:
                display_coef = coef
            row.append(format_coefficient(coef, r['shock_se'], r['shock_pval']))
        else:
            row.append('')
    rows.append(row)
    
    # SE row
    row_se = ['']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            row_se.append(format_se(results[constraint]['shock_se']))
        else:
            row_se.append('')
    rows.append(row_se)
    
    # Lagged constraint row
    row = ['Lagged constraint']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            r = results[constraint]['regression']
            row.append(format_coefficient(r['coef'][constraint], r['se'][constraint], r['p_value'][constraint]))
        else:
            row.append('')
    rows.append(row)
    
    # SE row for lagged
    row_se = ['']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            r = results[constraint]['regression']
            row_se.append(format_se(r['se'][constraint]))
        else:
            row_se.append('')
    rows.append(row_se)
    
    # R2 row
    row_r2 = ['$R^2$']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            row_r2.append(f"{results[constraint]['r2']:.2f}")
        else:
            row_r2.append('')
    rows.append(row_r2)
    
    # N row
    row_n = ['Observations']
    for constraint in ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX', 'CONSTRAINT_COMPOSITE']:
        if constraint in results:
            row_n.append(str(results[constraint]['n_obs']))
        else:
            row_n.append('')
    rows.append(row_n)
    
    df = pd.DataFrame(rows, columns=['', '(1)', '(2)', '(3)', '(4)'])
    
    if save_path:
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Effect of Macroeconomic Shocks on Arbitrage Constraints}',
            '\\label{tab:shock_constraint}',
            '\\begin{tabular}{lcccc}',
            '\\toprule',
            ' & (1) & (2) & (3) & (4) \\\\',
            ' & Capital ratio & TED spread & VIX & Composite \\\\',
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
            '\\textit{Notes:} The dependent variables are: (1) intermediary capital ratio, (2) TED spread, (3) VIX, and (4) composite constraint indicator. Monetary shock equals one for tightening shocks below the 10th percentile. All regressions include month fixed effects. Standard errors in parentheses, clustered by quarter. ***, **, * denote significance at 1\\%, 5\\%, 10\\%.',
            '\\end{table}'
        ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: {save_path}")
    
    return df, results


def create_table6_interaction(data: pd.DataFrame,
                              save_path: Path = None) -> pd.DataFrame:
    """
    Create Table 6: Interaction of Shocks and Constraints in Dimension Dynamics.
    """
    if 'MP_SHOCK_IND' not in data.columns or 'd_eff' not in data.columns:
        print("  Missing required data")
        return pd.DataFrame()
    
    # Use continuous constraint if available, otherwise binary
    if 'CONSTRAINT_COMPOSITE' in data.columns:
        constraint_var = 'CONSTRAINT_COMPOSITE'
    elif 'HIGH_CONSTRAINT' in data.columns:
        constraint_var = 'HIGH_CONSTRAINT'
    else:
        print("  No constraint variable available")
        return pd.DataFrame()
    
    results = {}
    
    # Column 1: Main effects and interaction only
    results['col1'] = test_interaction_regression(
        data, constraint_var=constraint_var, include_lag=False
    )
    
    # Column 2: Add lagged dimension
    results['col2'] = test_interaction_regression(
        data, constraint_var=constraint_var, include_lag=True
    )
    
    # Column 3: Full specification
    results['col3'] = test_interaction_regression(
        data, constraint_var=constraint_var, include_lag=True
    )
    
    # Build table
    rows = []
    
    # Shock row
    row = ['Shock']
    for col in ['col1', 'col2', 'col3']:
        r = results[col]
        row.append(format_coefficient(r['coef']['MP_SHOCK_IND'], r['se']['MP_SHOCK_IND'], r['p_value']['MP_SHOCK_IND']))
    rows.append(row)
    rows.append([''] + [format_se(results[col]['se']['MP_SHOCK_IND']) for col in ['col1', 'col2', 'col3']])
    
    # Constraint row
    row = ['Constraint ($c_t$)']
    for col in ['col1', 'col2', 'col3']:
        r = results[col]
        if constraint_var in r['coef'].index:
            row.append(format_coefficient(r['coef'][constraint_var], r['se'][constraint_var], r['p_value'][constraint_var]))
        else:
            row.append('')
    rows.append(row)
    rows.append([''] + [format_se(results[col]['se'].get(constraint_var, 0)) for col in ['col1', 'col2', 'col3']])
    
    # Interaction row
    row = ['Shock $\\times$ Constraint']
    for col in ['col1', 'col2', 'col3']:
        r = results[col]
        row.append(format_coefficient(r['coef']['shock_x_constraint'], r['se']['shock_x_constraint'], r['p_value']['shock_x_constraint']))
    rows.append(row)
    rows.append([''] + [format_se(results[col]['se']['shock_x_constraint']) for col in ['col1', 'col2', 'col3']])
    
    # Lagged dimension row (cols 2-3 only)
    row = ['$\\hat{d}_{\\text{eff}}(t)$']
    row.append('')  # col1 doesn't have it
    for col in ['col2', 'col3']:
        r = results[col]
        if 'd_eff' in r['coef'].index:
            row.append(format_coefficient(r['coef']['d_eff'], r['se']['d_eff'], r['p_value']['d_eff']))
        else:
            row.append('')
    rows.append(row)
    rows.append(['', ''] + [format_se(results[col]['se'].get('d_eff', 0)) for col in ['col2', 'col3']])
    
    # R2 row
    rows.append(['$R^2$'] + [f"{results[col]['r2']:.2f}" for col in ['col1', 'col2', 'col3']])
    
    df = pd.DataFrame(rows, columns=['', '(1)', '(2)', '(3)'])
    
    if save_path:
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Interaction of Shocks and Constraints in Dimension Dynamics}',
            '\\label{tab:interaction}',
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
            '\\textit{Notes:} Shock is an indicator for contractionary monetary policy shocks. Constraint is the continuous composite measure. Column (1) includes only main effects and interaction. Column (2) adds lagged dimension. Column (3) includes all terms. Standard errors in parentheses, clustered by quarter. ***, **, * denote significance at 1\\%, 5\\%, 10\\%.',
            '\\end{table}'
        ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: {save_path}")
    
    return df, results


def run_constraint_analysis(data: pd.DataFrame) -> Dict:
    """
    Run complete constraint mechanism analysis.
    """
    print("=" * 60)
    print("CONSTRAINT MECHANISM ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # Table 5: Shocks → Constraints
    print("\n  Creating Table 5: Shock effects on constraints...")
    table5, table5_results = create_table5_shock_constraints(
        data,
        save_path=TABLES_DIR / "table5_shock_constraint.tex"
    )
    results['table5'] = table5
    results['table5_results'] = table5_results
    
    # Table 6: Interaction
    print("\n  Creating Table 6: Interaction effects...")
    table6, table6_results = create_table6_interaction(
        data,
        save_path=TABLES_DIR / "table6_interaction.tex"
    )
    results['table6'] = table6
    results['table6_results'] = table6_results
    
    # Print key results
    if 'col1' in table6_results:
        r = table6_results['col1']
        print(f"\n  Key interaction result:")
        print(f"    Shock × Constraint: {r['coef']['shock_x_constraint']:.2f}")
        print(f"    p-value: {r['p_value']['shock_x_constraint']:.3f}")
    
    print("\n" + "=" * 60)
    print("CONSTRAINT ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from src.data.load import load_master_data
    
    try:
        data = load_master_data()
        results = run_constraint_analysis(data)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
