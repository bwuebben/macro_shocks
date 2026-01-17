"""
Robustness Checks
=================

This module runs all robustness tests:
1. Alternative rolling window lengths
2. Alternative constraint specifications
3. Placebo tests (expansionary shocks, pseudo-shocks)
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    TABLES_DIR, N_PERMUTATIONS, ROLLING_WINDOW,
    EXPANSIONARY_THRESHOLD_PERCENTILE
)
from src.utils.helpers import ols_regression, format_coefficient, format_se
from src.analysis.effective_dimension import compute_effective_dimension_series
from src.analysis.shock_response import test_conditional_shock_response


def robustness_window_length(industry_returns: pd.DataFrame,
                             master_data: pd.DataFrame,
                             windows: List[int] = [36, 60, 84]) -> Dict:
    """
    Test robustness to different rolling window lengths.
    """
    results = {}
    
    for window in windows:
        print(f"    Window = {window} months...")
        
        # Compute dimension with this window
        dim_series = compute_effective_dimension_series(
            industry_returns, window=window, show_progress=False
        )
        
        # Merge with master
        data = master_data.copy()
        data['d_eff'] = dim_series['d_eff']
        
        # Run conditional shock test
        if 'MP_SHOCK_IND' in data.columns and 'HIGH_CONSTRAINT' in data.columns:
            cond_results = test_conditional_shock_response(data)
            
            results[window] = {
                'dimension_mean': dim_series['d_eff'].mean(),
                'dimension_std': dim_series['d_eff'].std(),
                'high_coef': cond_results['high_constraint']['coef']['MP_SHOCK_IND'],
                'high_se': cond_results['high_constraint']['se']['MP_SHOCK_IND'],
                'high_pval': cond_results['high_constraint']['p_value']['MP_SHOCK_IND'],
                'low_coef': cond_results['low_constraint']['coef']['MP_SHOCK_IND'],
                'low_se': cond_results['low_constraint']['se']['MP_SHOCK_IND'],
                'low_pval': cond_results['low_constraint']['p_value']['MP_SHOCK_IND'],
            }
        else:
            results[window] = {
                'dimension_mean': dim_series['d_eff'].mean(),
                'dimension_std': dim_series['d_eff'].std(),
            }
    
    return results


def robustness_constraint_specification(data: pd.DataFrame) -> Dict:
    """
    Test robustness to different constraint specifications.
    """
    results = {}
    
    if 'MP_SHOCK_IND' not in data.columns or 'd_eff' not in data.columns:
        return results
    
    data = data.copy()
    data['d_eff_lead'] = data['d_eff'].shift(-1)
    
    # Specification 1: Baseline
    if 'CONSTRAINT_COMPOSITE' in data.columns:
        data['shock_x_constraint_base'] = data['MP_SHOCK_IND'] * data['CONSTRAINT_COMPOSITE']
        
        clean = data[['d_eff_lead', 'MP_SHOCK_IND', 'CONSTRAINT_COMPOSITE', 'shock_x_constraint_base', 'quarter']].dropna()
        
        if len(clean) > 20:
            results['baseline'] = ols_regression(
                clean['d_eff_lead'],
                clean[['MP_SHOCK_IND', 'CONSTRAINT_COMPOSITE', 'shock_x_constraint_base']],
                cluster_var=clean['quarter']
            )
    
    # Specification 2: Continuous
    constraint_vars = ['CAPITAL_RATIO', 'TED_SPREAD', 'VIX']
    available_constraints = [c for c in constraint_vars if c in data.columns]
    
    if available_constraints:
        for c in available_constraints:
            data[f'{c}_std'] = (data[c] - data[c].mean()) / data[c].std()
        
        std_cols = [f'{c}_std' for c in available_constraints]
        if 'CAPITAL_RATIO_std' in data.columns:
            data['CAPITAL_RATIO_std'] = -data['CAPITAL_RATIO_std']
        
        data['CONSTRAINT_CONTINUOUS'] = data[std_cols].mean(axis=1)
        data['shock_x_constraint_cont'] = data['MP_SHOCK_IND'] * data['CONSTRAINT_CONTINUOUS']
        
        clean = data[['d_eff_lead', 'MP_SHOCK_IND', 'CONSTRAINT_CONTINUOUS', 'shock_x_constraint_cont', 'quarter']].dropna()
        
        if len(clean) > 20:
            results['continuous'] = ols_regression(
                clean['d_eff_lead'],
                clean[['MP_SHOCK_IND', 'CONSTRAINT_CONTINUOUS', 'shock_x_constraint_cont']],
                cluster_var=clean['quarter']
            )
    
    return results


def placebo_expansionary_shocks(data: pd.DataFrame) -> Dict:
    """
    Placebo test using expansionary (positive) shocks.
    """
    results = {}
    
    if 'MP_SHOCK' not in data.columns or 'd_eff' not in data.columns:
        return results
    
    data = data.copy()
    
    threshold = np.nanpercentile(data['MP_SHOCK'], EXPANSIONARY_THRESHOLD_PERCENTILE)
    data['MP_SHOCK_EXPANSIONARY'] = (data['MP_SHOCK'] > threshold).astype(int)
    
    print(f"    Expansionary shock threshold (p{EXPANSIONARY_THRESHOLD_PERCENTILE}): {threshold:.4f}")
    print(f"    Expansionary shock months: {data['MP_SHOCK_EXPANSIONARY'].sum()}")
    
    data['d_eff_lead'] = data['d_eff'].shift(-1)
    
    if 'HIGH_CONSTRAINT' in data.columns:
        data['exp_x_constraint'] = data['MP_SHOCK_EXPANSIONARY'] * data['HIGH_CONSTRAINT']
        
        clean = data[['d_eff_lead', 'MP_SHOCK_EXPANSIONARY', 'HIGH_CONSTRAINT', 'exp_x_constraint', 'quarter']].dropna()
        
        if len(clean) > 20:
            results['expansionary_interaction'] = ols_regression(
                clean['d_eff_lead'],
                clean[['MP_SHOCK_EXPANSIONARY', 'HIGH_CONSTRAINT', 'exp_x_constraint']],
                cluster_var=clean['quarter']
            )
    
    return results


def placebo_permutation_test(data: pd.DataFrame,
                             n_permutations: int = 500) -> Dict:
    """
    Permutation test: randomly shuffle shock series.
    """
    if 'MP_SHOCK_IND' not in data.columns or 'd_eff' not in data.columns:
        return {}
    
    if 'HIGH_CONSTRAINT' not in data.columns:
        return {}
    
    data = data.copy()
    data['d_eff_lead'] = data['d_eff'].shift(-1)
    data['shock_x_constraint'] = data['MP_SHOCK_IND'] * data['HIGH_CONSTRAINT']
    clean = data[['d_eff_lead', 'MP_SHOCK_IND', 'HIGH_CONSTRAINT', 'shock_x_constraint', 'quarter']].dropna()
    
    if len(clean) < 30:
        return {}
    
    true_result = ols_regression(
        clean['d_eff_lead'],
        clean[['MP_SHOCK_IND', 'HIGH_CONSTRAINT', 'shock_x_constraint']],
        cluster_var=clean['quarter']
    )
    true_coef = true_result['coef']['shock_x_constraint']
    
    perm_coefs = []
    
    for _ in tqdm(range(n_permutations), desc="    Permutation test"):
        perm_shock = clean['MP_SHOCK_IND'].sample(frac=1, replace=False).values
        clean_perm = clean.copy()
        clean_perm['MP_SHOCK_IND'] = perm_shock
        clean_perm['shock_x_constraint'] = clean_perm['MP_SHOCK_IND'] * clean_perm['HIGH_CONSTRAINT']
        
        try:
            perm_result = ols_regression(
                clean_perm['d_eff_lead'],
                clean_perm[['MP_SHOCK_IND', 'HIGH_CONSTRAINT', 'shock_x_constraint']],
                add_constant=True
            )
            perm_coefs.append(perm_result['coef']['shock_x_constraint'])
        except:
            pass
    
    perm_coefs = np.array(perm_coefs)
    p_value = np.mean(np.abs(perm_coefs) >= np.abs(true_coef))
    
    return {
        'true_coef': true_coef,
        'perm_median': np.median(perm_coefs),
        'perm_ci_low': np.percentile(perm_coefs, 2.5),
        'perm_ci_high': np.percentile(perm_coefs, 97.5),
        'p_value': p_value
    }


def create_robustness_tables(window_results: Dict,
                             constraint_results: Dict,
                             placebo_results: Dict,
                             save_dir: Path = TABLES_DIR) -> None:
    """Create LaTeX tables for robustness results."""
    
    # Table A1: Window length
    if window_results:
        rows = []
        for window in sorted(window_results.keys()):
            r = window_results[window]
            if 'high_coef' in r:
                rows.append([
                    f'{window} months',
                    format_coefficient(r['high_coef'], r['high_se'], r['high_pval']),
                    format_se(r['high_se']),
                    format_coefficient(r['low_coef'], r['low_se'], r['low_pval']),
                    format_se(r['low_se']),
                ])
        
        if rows:
            latex_lines = [
                '\\begin{table}[htbp]',
                '\\centering',
                '\\caption{Robustness to Rolling Window Length}',
                '\\label{tab:robustness_window}',
                '\\begin{tabular}{lcccc}',
                '\\toprule',
                'Window & High Coef & SE & Low Coef & SE \\\\',
                '\\midrule'
            ]
            for row in rows:
                latex_lines.append(' & '.join(row) + ' \\\\')
            latex_lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
            
            with open(save_dir / 'tableA1_robustness_window.tex', 'w') as f:
                f.write('\n'.join(latex_lines))
            print(f"  Saved: tableA1_robustness_window.tex")
    
    # Table A2: Constraint specification
    if constraint_results:
        rows = []
        for spec, name in [('baseline', 'Baseline'), ('continuous', 'Continuous')]:
            if spec in constraint_results:
                r = constraint_results[spec]
                int_vars = [v for v in r['coef'].index if 'shock_x' in v]
                if int_vars:
                    iv = int_vars[0]
                    rows.append([name, format_coefficient(r['coef'][iv], r['se'][iv], r['p_value'][iv]), format_se(r['se'][iv])])
        
        if rows:
            latex_lines = [
                '\\begin{table}[htbp]',
                '\\centering',
                '\\caption{Robustness to Constraint Specification}',
                '\\label{tab:robustness_constraint}',
                '\\begin{tabular}{lcc}',
                '\\toprule',
                'Specification & Shock $\\times$ Constraint & SE \\\\',
                '\\midrule'
            ]
            for row in rows:
                latex_lines.append(' & '.join(row) + ' \\\\')
            latex_lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
            
            with open(save_dir / 'tableA2_robustness_constraint.tex', 'w') as f:
                f.write('\n'.join(latex_lines))
            print(f"  Saved: tableA2_robustness_constraint.tex")
    
    # Table 7: Placebo
    if placebo_results:
        latex_lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            '\\caption{Placebo Tests}',
            '\\label{tab:placebo}',
            '\\begin{tabular}{lcc}',
            '\\toprule',
            ' & Coefficient & $p$-value \\\\',
            '\\midrule'
        ]
        
        if 'expansionary' in placebo_results and placebo_results['expansionary']:
            r = placebo_results['expansionary'].get('expansionary_interaction', {})
            if r:
                iv = [v for v in r['coef'].index if 'exp_x' in v]
                if iv:
                    latex_lines.append(f"Expansionary $\\times$ High & {r['coef'][iv[0]]:.2f} & {r['p_value'][iv[0]]:.2f} \\\\")
        
        if 'permutation' in placebo_results and placebo_results['permutation']:
            p = placebo_results['permutation']
            latex_lines.append('\\midrule')
            latex_lines.append(f"Permutation $p$-value & & {p['p_value']:.3f} \\\\")
        
        latex_lines.extend(['\\bottomrule', '\\end{tabular}', '\\end{table}'])
        
        with open(save_dir / 'table7_placebo.tex', 'w') as f:
            f.write('\n'.join(latex_lines))
        print(f"  Saved: table7_placebo.tex")


def run_robustness_analysis(industry_returns: pd.DataFrame,
                            master_data: pd.DataFrame) -> Dict:
    """Run all robustness checks."""
    print("=" * 60)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    print("\n[1/4] Testing window length robustness...")
    results['window'] = robustness_window_length(industry_returns, master_data)
    
    print("\n[2/4] Testing constraint specification robustness...")
    results['constraint'] = robustness_constraint_specification(master_data)
    
    print("\n[3/4] Running expansionary shocks placebo...")
    results['expansionary'] = placebo_expansionary_shocks(master_data)
    
    print("\n[4/4] Running permutation test...")
    results['permutation'] = placebo_permutation_test(master_data, n_permutations=min(N_PERMUTATIONS, 200))
    
    print("\n  Creating robustness tables...")
    create_robustness_tables(
        results['window'],
        results['constraint'],
        {'expansionary': results['expansionary'], 'permutation': results['permutation']}
    )
    
    print("\n" + "=" * 60)
    print("ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    from src.data.load import load_master_data, load_industry_returns
    
    try:
        master = load_master_data()
        industry = load_industry_returns()
        results = run_robustness_analysis(industry, master)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
