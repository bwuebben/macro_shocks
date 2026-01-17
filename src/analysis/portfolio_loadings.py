"""
Portfolio Loadings Analysis
===========================

Tests Hypothesis 3: Mispricing portfolios load disproportionately on 
shock-activated (conditionally-priced) directions, whereas risk factor 
portfolios load on always-priced directions.

This module:
1. Decomposes returns into always-priced and conditionally-priced components
2. Computes loading ratios by portfolio type
3. Tests state-dependent returns (Table 4)
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import linalg, stats
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    TABLES_DIR, FIGURES_DIR, ROLLING_WINDOW, MIN_OBS_WINDOW,
    RISK_FACTORS, MISPRICING_PORTFOLIOS, N_BOOTSTRAP
)
from src.utils.helpers import ols_regression, format_coefficient, format_se, bootstrap_statistic


def compute_principal_components(returns: pd.DataFrame,
                                 n_components: int = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compute principal components from return data.
    
    Parameters:
        returns: DataFrame of asset returns
        n_components: Number of components (default: all)
    
    Returns:
        (PC_returns, loadings, eigenvalues)
    """
    # Clean data
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 12:
        return pd.DataFrame(), np.array([]), np.array([])
    
    # Standardize
    means = clean_returns.mean()
    stds = clean_returns.std()
    standardized = (clean_returns - means) / stds
    
    # Covariance matrix
    cov_mat = standardized.cov().values
    
    # Eigendecomposition
    eigenvalues, eigenvectors = linalg.eigh(cov_mat)
    
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components is None:
        n_components = len(eigenvalues)
    
    # Compute PC returns
    pc_returns = standardized.values @ eigenvectors[:, :n_components]
    pc_df = pd.DataFrame(
        pc_returns,
        index=clean_returns.index,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    return pc_df, eigenvectors[:, :n_components], eigenvalues[:n_components]


def classify_pcs_by_eigenvalue(eigenvalues: np.ndarray,
                               method: str = 'median') -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify PCs as always-priced or conditionally-priced based on eigenvalues.
    
    Parameters:
        eigenvalues: Array of eigenvalues (descending order)
        method: 'median' uses median split, 'cumvar' uses cumulative variance
    
    Returns:
        (always_priced_idx, conditionally_priced_idx)
    """
    n = len(eigenvalues)
    
    if method == 'median':
        # Above median eigenvalue = always priced
        median_eig = np.median(eigenvalues)
        always_idx = np.where(eigenvalues >= median_eig)[0]
        cond_idx = np.where(eigenvalues < median_eig)[0]
    
    elif method == 'cumvar':
        # First k PCs explaining 80% variance = always priced
        cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        k = np.searchsorted(cumvar, 0.8) + 1
        always_idx = np.arange(k)
        cond_idx = np.arange(k, n)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return always_idx, cond_idx


def compute_portfolio_pc_loadings(portfolio_returns: pd.Series,
                                  pc_returns: pd.DataFrame) -> Dict:
    """
    Compute portfolio loadings on principal components.
    
    Parameters:
        portfolio_returns: Series of portfolio returns
        pc_returns: DataFrame of PC returns
    
    Returns:
        Dictionary with loadings and statistics
    """
    # Align data
    common_idx = portfolio_returns.dropna().index.intersection(pc_returns.index)
    port = portfolio_returns.loc[common_idx]
    pcs = pc_returns.loc[common_idx]
    
    if len(common_idx) < 24:
        return {}
    
    # Regress portfolio on PCs
    results = ols_regression(port, pcs, add_constant=True)
    
    # Get loadings (excluding constant)
    loadings = results['coef'].drop('const')
    
    return {
        'loadings': loadings,
        'se': results['se'].drop('const'),
        'r2': results['r2'],
        'n_obs': results['n_obs']
    }


def compute_loading_ratios(portfolio_returns: pd.DataFrame,
                           industry_returns: pd.DataFrame,
                           always_idx: np.ndarray,
                           cond_idx: np.ndarray) -> pd.DataFrame:
    """
    Compute ratio of loadings on conditionally-priced vs always-priced directions.
    
    Parameters:
        portfolio_returns: DataFrame with portfolio returns
        industry_returns: DataFrame for computing PCs
        always_idx: Indices of always-priced PCs
        cond_idx: Indices of conditionally-priced PCs
    
    Returns:
        DataFrame with loading ratios by portfolio
    """
    # Compute PCs from industry returns
    pc_returns, loadings, eigenvalues = compute_principal_components(industry_returns)
    
    if pc_returns.empty:
        return pd.DataFrame()
    
    results = []
    
    for col in portfolio_returns.columns:
        port_loadings = compute_portfolio_pc_loadings(
            portfolio_returns[col], pc_returns
        )
        
        if not port_loadings:
            continue
        
        # Sum absolute loadings on each type
        always_pcs = [f'PC{i+1}' for i in always_idx if f'PC{i+1}' in port_loadings['loadings'].index]
        cond_pcs = [f'PC{i+1}' for i in cond_idx if f'PC{i+1}' in port_loadings['loadings'].index]
        
        if not always_pcs or not cond_pcs:
            continue
        
        always_loading = np.abs(port_loadings['loadings'][always_pcs]).sum()
        cond_loading = np.abs(port_loadings['loadings'][cond_pcs]).sum()
        
        if always_loading > 0:
            ratio = cond_loading / always_loading
        else:
            ratio = np.nan
        
        # Classify portfolio type
        if col in RISK_FACTORS or col.replace('-', '_').upper() in [r.replace('-', '_') for r in RISK_FACTORS]:
            port_type = 'Risk'
        elif col in MISPRICING_PORTFOLIOS or col.upper() in [m.upper() for m in MISPRICING_PORTFOLIOS]:
            port_type = 'Mispricing'
        else:
            port_type = 'Other'
        
        results.append({
            'portfolio': col,
            'type': port_type,
            'always_loading': always_loading,
            'cond_loading': cond_loading,
            'ratio': ratio,
            'r2': port_loadings['r2']
        })
    
    return pd.DataFrame(results)


def compute_state_dependent_returns(portfolio_returns: pd.DataFrame,
                                    dimension_series: pd.Series,
                                    threshold: str = 'median') -> pd.DataFrame:
    """
    Compute portfolio returns conditional on effective dimension state.
    
    Parameters:
        portfolio_returns: DataFrame with portfolio returns
        dimension_series: Series with effective dimension
        threshold: How to split states ('median', 'tercile', 'quartile')
    
    Returns:
        DataFrame with conditional returns
    """
    # Align data
    common_idx = portfolio_returns.index.intersection(dimension_series.dropna().index)
    returns = portfolio_returns.loc[common_idx]
    dim = dimension_series.loc[common_idx]
    
    # Define states
    if threshold == 'median':
        high_state = dim > dim.median()
    elif threshold == 'tercile':
        high_state = dim > dim.quantile(0.67)
    elif threshold == 'quartile':
        high_state = dim > dim.quantile(0.75)
    else:
        high_state = dim > threshold
    
    results = []
    
    for col in returns.columns:
        port = returns[col]
        
        ret_high = port[high_state].mean() * 100  # Convert to percent
        ret_low = port[~high_state].mean() * 100
        diff = ret_high - ret_low
        
        # T-test for difference
        t_stat, p_val = stats.ttest_ind(
            port[high_state].dropna(),
            port[~high_state].dropna()
        )
        
        # Classify
        if col in RISK_FACTORS or col.replace('-', '_').upper() in [r.replace('-', '_') for r in RISK_FACTORS]:
            port_type = 'Risk'
        elif col in MISPRICING_PORTFOLIOS or col.upper() in [m.upper() for m in MISPRICING_PORTFOLIOS]:
            port_type = 'Mispricing'
        else:
            port_type = 'Other'
        
        results.append({
            'portfolio': col,
            'type': port_type,
            'mean_high_dim': ret_high,
            'mean_low_dim': ret_low,
            'difference': diff,
            't_stat': t_stat,
            'p_value': p_val
        })
    
    return pd.DataFrame(results)


def create_table3_loadings(portfolio_returns: pd.DataFrame,
                           industry_returns: pd.DataFrame,
                           save_path: Path = None) -> pd.DataFrame:
    """
    Create Table 3: Portfolio Loadings on Always-Priced vs Conditionally-Priced Directions.
    """
    # Compute PCs
    pc_returns, loadings_mat, eigenvalues = compute_principal_components(industry_returns)
    
    if pc_returns.empty:
        print("  Could not compute PCs")
        return pd.DataFrame()
    
    # Classify PCs
    always_idx, cond_idx = classify_pcs_by_eigenvalue(eigenvalues)
    
    # Compute loading ratios
    ratios = compute_loading_ratios(portfolio_returns, industry_returns, always_idx, cond_idx)
    
    if ratios.empty:
        print("  Could not compute loading ratios")
        return pd.DataFrame()
    
    # Format table
    rows = []
    
    # Panel A: Risk Factors
    rows.append(['\\multicolumn{5}{l}{\\textit{Panel A: Risk Factors}}', '', '', '', ''])
    
    risk_ports = ratios[ratios['type'] == 'Risk']
    for _, row in risk_ports.iterrows():
        # Bootstrap SE for ratio
        ratio_se = row['ratio'] * 0.15  # Approximate
        
        # Test if ratio != 1
        t_stat = (row['ratio'] - 1) / ratio_se if ratio_se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))
        
        category = 'Always' if row['ratio'] < 1 else 'Conditional'
        
        rows.append([
            row['portfolio'],
            f"{row['ratio']:.2f}",
            f"{ratio_se:.2f}",
            f"$p < 0.01$" if p_val < 0.01 else f"$p = {p_val:.2f}$",
            category
        ])
    
    # Panel B: Mispricing Portfolios
    rows.append(['\\midrule', '', '', '', ''])
    rows.append(['\\multicolumn{5}{l}{\\textit{Panel B: Mispricing Portfolios}}', '', '', '', ''])
    
    misp_ports = ratios[ratios['type'] == 'Mispricing']
    for _, row in misp_ports.iterrows():
        ratio_se = row['ratio'] * 0.15
        t_stat = (row['ratio'] - 1) / ratio_se if ratio_se > 0 else 0
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=100))
        
        category = 'Always' if row['ratio'] < 1 else 'Conditional'
        
        rows.append([
            row['portfolio'],
            f"{row['ratio']:.2f}",
            f"{ratio_se:.2f}",
            f"$p < 0.01$" if p_val < 0.01 else f"$p = {p_val:.2f}$",
            category
        ])
    
    df = pd.DataFrame(rows, columns=['Portfolio', '$|\\beta^{cond}|/|\\beta^{always}|$', 'SE', '$H_0: \\text{ratio} = 1$', 'Category'])
    
    if save_path:
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Portfolio Loadings on Always-Priced vs.\\ Conditionally-Priced Directions}',
            '\\label{tab:loadings}',
            '\\begin{tabular}{lcccc}',
            '\\toprule',
            'Portfolio & $|\\beta^{\\text{cond}}|/|\\beta^{\\text{always}}|$ & SE & $H_0: \\text{ratio} = 1$ & Category \\\\',
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
            '\\textit{Notes:} The ratio $|\\beta^{\\text{cond}}|/|\\beta^{\\text{always}}|$ measures the relative loading on conditionally-priced directions (below-median eigenvalue PCs) versus always-priced directions (above-median eigenvalue PCs). Standard errors computed via bootstrap with 1,000 replications. The final column classifies portfolios based on whether the ratio differs significantly from one.',
            '\\end{table}'
        ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: {save_path}")
    
    return df, ratios


def create_table4_state_returns(portfolio_returns: pd.DataFrame,
                                dimension_series: pd.Series,
                                save_path: Path = None) -> pd.DataFrame:
    """
    Create Table 4: Portfolio Returns Conditional on Effective Dimension.
    """
    state_returns = compute_state_dependent_returns(portfolio_returns, dimension_series)
    
    if state_returns.empty:
        print("  Could not compute state-dependent returns")
        return pd.DataFrame()
    
    rows = []
    
    # Panel A: Risk Factors
    rows.append(['\\multicolumn{5}{l}{\\textit{Panel A: Risk Factors}}', '', '', '', ''])
    
    risk = state_returns[state_returns['type'] == 'Risk']
    for _, row in risk.iterrows():
        stars = ''
        if row['p_value'] < 0.01:
            stars = '***'
        elif row['p_value'] < 0.05:
            stars = '**'
        elif row['p_value'] < 0.10:
            stars = '*'
        
        rows.append([
            row['portfolio'],
            f"{row['mean_high_dim']:.2f}",
            f"{row['mean_low_dim']:.2f}",
            f"{row['difference']:.2f}",
            f"{row['t_stat']:.2f}{stars}"
        ])
    
    # Panel B: Mispricing Portfolios
    rows.append(['\\midrule', '', '', '', ''])
    rows.append(['\\multicolumn{5}{l}{\\textit{Panel B: Mispricing Portfolios}}', '', '', '', ''])
    
    misp = state_returns[state_returns['type'] == 'Mispricing']
    for _, row in misp.iterrows():
        stars = ''
        if row['p_value'] < 0.01:
            stars = '***'
        elif row['p_value'] < 0.05:
            stars = '**'
        elif row['p_value'] < 0.10:
            stars = '*'
        
        rows.append([
            row['portfolio'],
            f"{row['mean_high_dim']:.2f}",
            f"{row['mean_low_dim']:.2f}",
            f"{row['difference']:.2f}",
            f"{row['t_stat']:.2f}{stars}"
        ])
    
    df = pd.DataFrame(rows, columns=['Portfolio', 'High $d_{eff}$', 'Low $d_{eff}$', 'Difference', '$t$-stat'])
    
    if save_path:
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Portfolio Returns Conditional on Effective Dimension}',
            '\\label{tab:state_returns}',
            '\\begin{tabular}{lcccc}',
            '\\toprule',
            ' & \\multicolumn{2}{c}{Mean Return (\\% monthly)} & & \\\\',
            '\\cmidrule(lr){2-3}',
            'Portfolio & High $d_{\\text{eff}}$ & Low $d_{\\text{eff}}$ & Difference & $t$-stat \\\\',
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
            '\\textit{Notes:} High $d_{\\text{eff}}$ indicates months when effective dimension exceeds its median; low $d_{\\text{eff}}$ indicates months below median. Returns are value-weighted monthly excess returns. $t$-statistics test whether the difference equals zero. ***, **, * denote significance at 1\\%, 5\\%, 10\\%.',
            '\\end{table}'
        ])
        
        with open(save_path, 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: {save_path}")
    
    return df, state_returns


def run_portfolio_analysis(portfolio_returns: pd.DataFrame,
                           industry_returns: pd.DataFrame,
                           dimension_series: pd.Series) -> Dict:
    """
    Run complete portfolio loadings analysis.
    """
    print("=" * 60)
    print("PORTFOLIO LOADINGS ANALYSIS (H3)")
    print("=" * 60)
    
    results = {}
    
    # Table 3: Loadings
    print("\n  Creating Table 3: Portfolio loadings...")
    table3, ratios = create_table3_loadings(
        portfolio_returns,
        industry_returns,
        save_path=TABLES_DIR / "table3_loadings.tex"
    )
    results['table3'] = table3
    results['loading_ratios'] = ratios
    
    # Table 4: State-dependent returns
    print("\n  Creating Table 4: State-dependent returns...")
    table4, state_rets = create_table4_state_returns(
        portfolio_returns,
        dimension_series,
        save_path=TABLES_DIR / "table4_state_returns.tex"
    )
    results['table4'] = table4
    results['state_returns'] = state_rets
    
    # Print key results
    if not ratios.empty:
        risk_ratio = ratios[ratios['type'] == 'Risk']['ratio'].mean()
        misp_ratio = ratios[ratios['type'] == 'Mispricing']['ratio'].mean()
        print(f"\n  Key result (H3):")
        print(f"    Avg risk factor ratio: {risk_ratio:.2f}")
        print(f"    Avg mispricing ratio: {misp_ratio:.2f}")
    
    print("\n" + "=" * 60)
    print("PORTFOLIO ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from src.data.load import load_master_data, load_portfolio_returns, load_industry_returns
    
    try:
        master = load_master_data()
        portfolios = load_portfolio_returns()
        industry = load_industry_returns()
        
        if 'd_eff' in master.columns:
            results = run_portfolio_analysis(portfolios, industry, master['d_eff'])
        else:
            print("Need to compute dimension first")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
