"""
Utility Functions
=================

Statistical helpers, table formatting, and other utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, Tuple, List, Union
import warnings


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def newey_west_se(y: np.ndarray, X: np.ndarray, lags: int = None) -> np.ndarray:
    """
    Compute Newey-West heteroskedasticity and autocorrelation consistent standard errors.
    
    Parameters:
        y: Dependent variable (n,)
        X: Independent variables (n, k)
        lags: Number of lags for HAC (default: floor(4*(n/100)^(2/9)))
    
    Returns:
        Standard errors (k,)
    """
    n, k = X.shape
    
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    
    # OLS estimates
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    
    # Meat of sandwich
    S = np.zeros((k, k))
    
    for l in range(lags + 1):
        weight = 1 - l / (lags + 1) if l > 0 else 1
        
        for t in range(l, n):
            S += weight * np.outer(X[t] * resid[t], X[t - l] * resid[t - l])
            if l > 0:
                S += weight * np.outer(X[t - l] * resid[t - l], X[t] * resid[t])
    
    # Bread of sandwich
    XtX_inv = np.linalg.inv(X.T @ X)
    
    # Sandwich
    V = n * XtX_inv @ S @ XtX_inv
    
    return np.sqrt(np.diag(V))


def clustered_se(y: np.ndarray, X: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """
    Compute cluster-robust standard errors.
    
    Parameters:
        y: Dependent variable (n,)
        X: Independent variables (n, k)
        clusters: Cluster identifiers (n,)
    
    Returns:
        Standard errors (k,)
    """
    n, k = X.shape
    
    # OLS estimates
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    
    # Get unique clusters
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    
    # Meat of sandwich
    B = np.zeros((k, k))
    
    for g in unique_clusters:
        mask = clusters == g
        Xg = X[mask]
        eg = resid[mask]
        score_g = Xg.T @ eg
        B += np.outer(score_g, score_g)
    
    # Bread - use pseudo-inverse for numerical stability
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)

    # Sandwich with small-sample adjustment
    adjustment = (G / (G - 1)) * ((n - 1) / (n - k))
    V = adjustment * XtX_inv @ B @ XtX_inv

    # Ensure non-negative variances
    variances = np.diag(V)
    variances = np.maximum(variances, 0)

    return np.sqrt(variances)


def ols_regression(y: pd.Series, X: pd.DataFrame, 
                   cluster_var: Optional[pd.Series] = None,
                   add_constant: bool = True) -> dict:
    """
    Run OLS regression with optional clustered standard errors.
    
    Parameters:
        y: Dependent variable
        X: Independent variables
        cluster_var: Variable to cluster on (optional)
        add_constant: Whether to add a constant
    
    Returns:
        Dictionary with coefficients, standard errors, t-stats, p-values
    """
    # Align indices
    common_idx = y.dropna().index.intersection(X.dropna().index)
    y_clean = y.loc[common_idx].values
    X_clean = X.loc[common_idx].values
    
    if add_constant:
        X_clean = np.column_stack([np.ones(len(y_clean)), X_clean])
        var_names = ['const'] + list(X.columns)
    else:
        var_names = list(X.columns)
    
    n, k = X_clean.shape
    
    # OLS coefficients
    beta = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
    
    # Residuals and R-squared
    resid = y_clean - X_clean @ beta
    tss = np.sum((y_clean - np.mean(y_clean)) ** 2)
    rss = np.sum(resid ** 2)
    r2 = 1 - rss / tss
    
    # Standard errors
    if cluster_var is not None:
        clusters = cluster_var.loc[common_idx].values
        se = clustered_se(y_clean, X_clean, clusters)
    else:
        se = newey_west_se(y_clean, X_clean)
    
    # t-stats and p-values
    t_stats = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
    
    return {
        'coef': pd.Series(beta, index=var_names),
        'se': pd.Series(se, index=var_names),
        't_stat': pd.Series(t_stats, index=var_names),
        'p_value': pd.Series(p_values, index=var_names),
        'r2': r2,
        'n_obs': n,
        'resid': resid
    }


def bootstrap_statistic(data: np.ndarray, statistic_func: callable,
                        n_bootstrap: int = 1000, 
                        confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a statistic.
    
    Parameters:
        data: Data array
        statistic_func: Function that computes statistic from data
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    n = len(data)
    point_estimate = statistic_func(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        boot_sample = data[np.random.choice(n, n, replace=True)]
        bootstrap_stats.append(statistic_func(boot_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def permutation_test(data1: np.ndarray, data2: np.ndarray,
                     test_stat_func: callable,
                     n_permutations: int = 1000) -> Tuple[float, float]:
    """
    Permutation test for difference between two groups.
    
    Parameters:
        data1, data2: Two data arrays
        test_stat_func: Function computing test statistic from two arrays
        n_permutations: Number of permutations
    
    Returns:
        (observed_stat, p_value)
    """
    observed = test_stat_func(data1, data2)
    
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    
    count_extreme = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined)
        perm_stat = test_stat_func(combined[:n1], combined[n1:])
        if np.abs(perm_stat) >= np.abs(observed):
            count_extreme += 1
    
    p_value = count_extreme / n_permutations
    
    return observed, p_value


# =============================================================================
# TABLE FORMATTING
# =============================================================================

def format_coefficient(coef: float, se: float, p_value: float,
                       decimals: int = 2) -> str:
    """Format coefficient with standard error and significance stars."""
    stars = ''
    if p_value < 0.01:
        stars = '***'
    elif p_value < 0.05:
        stars = '**'
    elif p_value < 0.10:
        stars = '*'
    
    return f"{coef:.{decimals}f}{stars}"


def format_se(se: float, decimals: int = 2) -> str:
    """Format standard error in parentheses."""
    return f"({se:.{decimals}f})"


def results_to_latex_column(results: dict, var_order: List[str] = None,
                            decimals: int = 2) -> pd.DataFrame:
    """
    Convert regression results to LaTeX table column format.
    
    Parameters:
        results: Dictionary from ols_regression
        var_order: Order of variables to display
        decimals: Decimal places
    
    Returns:
        DataFrame with coefficient and SE rows
    """
    if var_order is None:
        var_order = list(results['coef'].index)
    
    rows = []
    for var in var_order:
        if var in results['coef'].index:
            coef_str = format_coefficient(
                results['coef'][var], 
                results['se'][var],
                results['p_value'][var],
                decimals
            )
            se_str = format_se(results['se'][var], decimals)
        else:
            coef_str = ''
            se_str = ''
        
        rows.append(coef_str)
        rows.append(se_str)
    
    # Add R2 and N
    rows.append(f"{results['r2']:.3f}")
    rows.append(str(results['n_obs']))
    
    index = []
    for var in var_order:
        index.append(var)
        index.append('')
    index.extend(['$R^2$', 'Observations'])
    
    return pd.DataFrame({'results': rows}, index=index)


def create_latex_table(df: pd.DataFrame, caption: str, label: str,
                       note: str = None, column_format: str = None) -> str:
    """
    Create a LaTeX table from a DataFrame.
    
    Parameters:
        df: DataFrame to convert
        caption: Table caption
        label: Table label for references
        note: Optional table note
        column_format: LaTeX column format (e.g., 'lccc')
    
    Returns:
        LaTeX table string
    """
    n_cols = len(df.columns) + 1  # +1 for index
    
    if column_format is None:
        column_format = 'l' + 'c' * len(df.columns)
    
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(f'\\caption{{{caption}}}')
    latex.append(f'\\label{{{label}}}')
    latex.append(f'\\begin{{tabular}}{{{column_format}}}')
    latex.append(r'\toprule')
    
    # Header
    header = ' & '.join([''] + list(df.columns)) + r' \\'
    latex.append(header)
    latex.append(r'\midrule')
    
    # Body
    for idx, row in df.iterrows():
        row_str = str(idx) + ' & ' + ' & '.join(str(v) for v in row.values) + r' \\'
        latex.append(row_str)
    
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    
    if note:
        latex.append('')
        latex.append(r'\vspace{0.5em}')
        latex.append(r'\footnotesize')
        latex.append(f'\\textit{{Notes:}} {note}')
    
    latex.append(r'\end{table}')
    
    return '\n'.join(latex)


# =============================================================================
# EIGENVALUE ANALYSIS
# =============================================================================

def effective_dimension(eigenvalues: np.ndarray) -> float:
    """
    Compute effective dimension from eigenvalues.
    
    Formula: (sum(λ))² / sum(λ²)
    
    Parameters:
        eigenvalues: Array of eigenvalues
    
    Returns:
        Effective dimension
    """
    eigenvalues = np.array(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zeros/negatives
    
    if len(eigenvalues) == 0:
        return np.nan
    
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def cumulative_variance_share(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Compute cumulative variance share for each eigenvalue.
    
    Parameters:
        eigenvalues: Array of eigenvalues (in descending order)
    
    Returns:
        Cumulative variance share array
    """
    eigenvalues = np.array(eigenvalues)
    total = np.sum(eigenvalues)
    
    if total == 0:
        return np.zeros_like(eigenvalues)
    
    return np.cumsum(eigenvalues) / total


# =============================================================================
# TIME SERIES UTILITIES
# =============================================================================

def rolling_apply(df: pd.DataFrame, window: int, func: callable,
                  min_obs: int = None) -> pd.Series:
    """
    Apply a function over rolling windows.
    
    Parameters:
        df: Input DataFrame
        window: Window size
        func: Function to apply (takes DataFrame, returns scalar)
        min_obs: Minimum observations required
    
    Returns:
        Series of results
    """
    if min_obs is None:
        min_obs = window // 2
    
    results = []
    dates = []
    
    for i in range(len(df)):
        if i < window - 1:
            results.append(np.nan)
        else:
            window_data = df.iloc[i - window + 1:i + 1]
            if window_data.notna().all().all() and len(window_data) >= min_obs:
                results.append(func(window_data))
            else:
                results.append(np.nan)
        dates.append(df.index[i])
    
    return pd.Series(results, index=dates)


def create_lags(df: pd.DataFrame, lags: List[int], 
                columns: List[str] = None) -> pd.DataFrame:
    """
    Create lagged variables.
    
    Parameters:
        df: Input DataFrame
        lags: List of lag lengths
        columns: Columns to lag (default: all)
    
    Returns:
        DataFrame with lagged columns appended
    """
    if columns is None:
        columns = df.columns
    
    result = df.copy()
    
    for col in columns:
        for lag in lags:
            if lag > 0:
                result[f'{col}_L{lag}'] = df[col].shift(lag)
            elif lag < 0:
                result[f'{col}_F{-lag}'] = df[col].shift(lag)
    
    return result


# Module init
__all__ = [
    'newey_west_se',
    'clustered_se',
    'ols_regression',
    'bootstrap_statistic',
    'permutation_test',
    'format_coefficient',
    'format_se',
    'results_to_latex_column',
    'create_latex_table',
    'effective_dimension',
    'cumulative_variance_share',
    'rolling_apply',
    'create_lags'
]
