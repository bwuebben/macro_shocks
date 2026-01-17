"""
Effective Dimension Analysis
============================

Core module for computing time-varying effective dimension of asset returns.

This implements the KNS-style effective dimension measure:
    d_eff(t) = (Σλᵢ)² / Σλᵢ²

where λᵢ are eigenvalues of the return covariance matrix.
"""

import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import (
    ROLLING_WINDOW, MIN_OBS_WINDOW, FIGURES_DIR, TABLES_DIR,
    FIGURE_SIZE, FIGURE_DPI, COLOR_PRIMARY, COLOR_CRISIS
)
from src.utils.helpers import effective_dimension, cumulative_variance_share


def compute_rolling_covariance(returns: pd.DataFrame, 
                               window: int = ROLLING_WINDOW,
                               min_obs: int = MIN_OBS_WINDOW) -> dict:
    """
    Compute rolling covariance matrices.
    
    Parameters:
        returns: DataFrame of asset returns (T x N)
        window: Rolling window length in months
        min_obs: Minimum observations required
    
    Returns:
        Dictionary with dates as keys, covariance matrices as values
    """
    cov_matrices = {}
    
    for i in range(window - 1, len(returns)):
        date = returns.index[i]
        window_data = returns.iloc[i - window + 1:i + 1]
        
        # Check for sufficient non-missing data
        valid_data = window_data.dropna(axis=1, how='any')
        
        if len(valid_data) >= min_obs and valid_data.shape[1] > 1:
            cov_mat = valid_data.cov().values
            cov_matrices[date] = {
                'cov': cov_mat,
                'columns': valid_data.columns.tolist(),
                'n_obs': len(valid_data)
            }
    
    return cov_matrices


def compute_effective_dimension_series(returns: pd.DataFrame,
                                       window: int = ROLLING_WINDOW,
                                       min_obs: int = MIN_OBS_WINDOW,
                                       show_progress: bool = True) -> pd.DataFrame:
    """
    Compute time series of effective dimension.
    
    Parameters:
        returns: DataFrame of asset returns (T x N)
        window: Rolling window length
        min_obs: Minimum observations required
        show_progress: Whether to show progress bar
    
    Returns:
        DataFrame with columns: d_eff, n_assets, n_obs, eigenvalues
    """
    results = []
    
    dates = returns.index[window - 1:]
    iterator = tqdm(range(len(dates)), desc="Computing effective dimension") if show_progress else range(len(dates))
    
    for i in iterator:
        idx = i + window - 1
        date = returns.index[idx]
        window_data = returns.iloc[idx - window + 1:idx + 1]
        
        # Drop columns with any missing values in window
        valid_data = window_data.dropna(axis=1, how='any')
        
        if len(valid_data) >= min_obs and valid_data.shape[1] > 1:
            # Compute covariance matrix
            cov_mat = valid_data.cov().values
            
            # Compute eigenvalues
            eigenvalues = linalg.eigvalsh(cov_mat)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros
            
            # Compute effective dimension
            d_eff = effective_dimension(eigenvalues)
            
            results.append({
                'date': date,
                'd_eff': d_eff,
                'n_assets': valid_data.shape[1],
                'n_obs': len(valid_data),
                'eigenvalue_1': eigenvalues[0] if len(eigenvalues) > 0 else np.nan,
                'eigenvalue_2': eigenvalues[1] if len(eigenvalues) > 1 else np.nan,
                'eigenvalue_3': eigenvalues[2] if len(eigenvalues) > 2 else np.nan,
                'total_variance': np.sum(eigenvalues),
                'variance_pc1': eigenvalues[0] / np.sum(eigenvalues) if len(eigenvalues) > 0 else np.nan
            })
        else:
            results.append({
                'date': date,
                'd_eff': np.nan,
                'n_assets': np.nan,
                'n_obs': np.nan,
                'eigenvalue_1': np.nan,
                'eigenvalue_2': np.nan,
                'eigenvalue_3': np.nan,
                'total_variance': np.nan,
                'variance_pc1': np.nan
            })
    
    df = pd.DataFrame(results)
    df = df.set_index('date')
    
    return df


def compute_conditional_eigenvalues(returns: pd.DataFrame,
                                    condition: pd.Series,
                                    window: int = ROLLING_WINDOW) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute average eigenvalue structure in high vs low condition states.
    
    Parameters:
        returns: DataFrame of asset returns
        condition: Binary series (1 = high state, 0 = low state)
        window: Rolling window length
    
    Returns:
        (eigenvalues_high, eigenvalues_low) - average eigenvalues in each state
    """
    # Align data
    common_idx = returns.index.intersection(condition.index)
    returns = returns.loc[common_idx]
    condition = condition.loc[common_idx]
    
    eigenvalues_high = []
    eigenvalues_low = []
    
    for i in range(window - 1, len(returns)):
        date = returns.index[i]
        window_data = returns.iloc[i - window + 1:i + 1].dropna(axis=1, how='any')
        
        if len(window_data) >= MIN_OBS_WINDOW and window_data.shape[1] > 1:
            cov_mat = window_data.cov().values
            eigs = linalg.eigvalsh(cov_mat)
            eigs = np.sort(eigs)[::-1]
            
            # Normalize by total variance for comparability
            eigs_normalized = eigs / np.sum(eigs)
            
            if condition.loc[date] == 1:
                eigenvalues_high.append(eigs_normalized)
            else:
                eigenvalues_low.append(eigs_normalized)
    
    # Pad to same length and average
    def pad_and_average(eig_list, target_len=30):
        padded = []
        for eigs in eig_list:
            if len(eigs) < target_len:
                eigs = np.concatenate([eigs, np.zeros(target_len - len(eigs))])
            padded.append(eigs[:target_len])
        return np.mean(padded, axis=0) if padded else np.zeros(target_len)
    
    avg_high = pad_and_average(eigenvalues_high)
    avg_low = pad_and_average(eigenvalues_low)
    
    return avg_high, avg_low


def plot_dimension_timeseries(dimension_series: pd.DataFrame,
                              crisis_periods: list = None,
                              save_path: Path = None) -> plt.Figure:
    """
    Plot effective dimension time series.
    
    Parameters:
        dimension_series: DataFrame with 'd_eff' column
        crisis_periods: List of (start, end, label) tuples for shading
        save_path: Path to save figure
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    # Plot dimension
    ax.plot(dimension_series.index, dimension_series['d_eff'], 
            color=COLOR_PRIMARY, linewidth=1.5, label='Effective Dimension')
    
    # Add mean line
    mean_d = dimension_series['d_eff'].mean()
    ax.axhline(mean_d, color='gray', linestyle='--', linewidth=1, 
               label=f'Mean = {mean_d:.1f}')
    
    # Shade crisis periods
    if crisis_periods is None:
        crisis_periods = [
            ('1998-07-01', '1998-12-31', 'LTCM'),
            ('2007-12-01', '2009-06-30', 'GFC'),
            ('2020-02-01', '2020-06-30', 'COVID'),
        ]
    
    for start, end, label in crisis_periods:
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        
        if start_dt >= dimension_series.index[0] and end_dt <= dimension_series.index[-1]:
            ax.axvspan(start_dt, end_dt, alpha=0.2, color=COLOR_CRISIS)
            # Add label at peak
            mask = (dimension_series.index >= start_dt) & (dimension_series.index <= end_dt)
            if mask.any():
                peak_idx = dimension_series.loc[mask, 'd_eff'].idxmax()
                peak_val = dimension_series.loc[peak_idx, 'd_eff']
                ax.annotate(label, xy=(peak_idx, peak_val), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Effective Dimension', fontsize=11)
    ax.set_title('Time-Varying Effective Dimension of Asset Returns', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_scree_comparison(eigenvalues_high: np.ndarray,
                          eigenvalues_low: np.ndarray,
                          n_components: int = 10,
                          save_path: Path = None) -> plt.Figure:
    """
    Plot scree plots comparing high vs low dimension states.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.arange(1, n_components + 1)
    
    # Left panel: Low dimension (normal times)
    ax = axes[0]
    ax.bar(x, eigenvalues_low[:n_components], color=COLOR_PRIMARY, alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Share')
    ax.set_title('Normal Times (Low $d_{eff}$)')
    ax.set_xticks(x)
    
    d_eff_low = effective_dimension(eigenvalues_low[:n_components])
    ax.annotate(f'$d_{{eff}}$ ≈ {d_eff_low:.1f}', xy=(0.95, 0.95), 
                xycoords='axes fraction', ha='right', va='top',
                fontsize=11, fontweight='bold')
    
    # Right panel: High dimension (crisis times)
    ax = axes[1]
    ax.bar(x, eigenvalues_high[:n_components], color=COLOR_CRISIS, alpha=0.7)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Share')
    ax.set_title('Crisis Times (High $d_{eff}$)')
    ax.set_xticks(x)
    
    d_eff_high = effective_dimension(eigenvalues_high[:n_components])
    ax.annotate(f'$d_{{eff}}$ ≈ {d_eff_high:.1f}', xy=(0.95, 0.95), 
                xycoords='axes fraction', ha='right', va='top',
                fontsize=11, fontweight='bold')
    
    # Match y-axis scales
    max_y = max(eigenvalues_low[:n_components].max(), eigenvalues_high[:n_components].max())
    for ax in axes:
        ax.set_ylim(0, max_y * 1.1)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def plot_cumulative_variance(eigenvalues_high: np.ndarray,
                             eigenvalues_low: np.ndarray,
                             n_components: int = 10,
                             save_path: Path = None) -> plt.Figure:
    """
    Plot cumulative variance share curves.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    x = np.arange(1, n_components + 1)
    
    cumvar_low = cumulative_variance_share(eigenvalues_low[:n_components])
    cumvar_high = cumulative_variance_share(eigenvalues_high[:n_components])
    
    ax.plot(x, cumvar_low, 'o-', color=COLOR_PRIMARY, linewidth=2, 
            markersize=8, label='Low $d_{eff}$ (Normal)')
    ax.plot(x, cumvar_high, 's--', color=COLOR_CRISIS, linewidth=2, 
            markersize=8, label='High $d_{eff}$ (Crisis)')
    
    # Reference line at 90%
    ax.axhline(0.9, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(n_components + 0.2, 0.9, '90%', va='center', fontsize=9, color='gray')
    
    ax.set_xlabel('Number of Principal Components', fontsize=11)
    ax.set_ylabel('Cumulative Variance Share', fontsize=11)
    ax.set_title('Cumulative Variance Explained by Principal Components', fontsize=12)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    return fig


def create_dimension_summary_table(dimension_series: pd.DataFrame,
                                   condition: pd.Series = None,
                                   save_path: Path = None) -> pd.DataFrame:
    """
    Create summary statistics table for effective dimension.
    """
    d_eff = dimension_series['d_eff'].dropna()
    
    summary = {
        'Statistic': ['Mean', 'Std Dev', 'Min', 'P25', 'Median', 'P75', 'Max', 'N'],
        'Full Sample': [
            f'{d_eff.mean():.2f}',
            f'{d_eff.std():.2f}',
            f'{d_eff.min():.2f}',
            f'{d_eff.quantile(0.25):.2f}',
            f'{d_eff.median():.2f}',
            f'{d_eff.quantile(0.75):.2f}',
            f'{d_eff.max():.2f}',
            f'{len(d_eff)}'
        ]
    }
    
    if condition is not None:
        # Align
        common_idx = d_eff.index.intersection(condition.index)
        d_eff_aligned = d_eff.loc[common_idx]
        cond_aligned = condition.loc[common_idx]
        
        d_high = d_eff_aligned[cond_aligned == 1]
        d_low = d_eff_aligned[cond_aligned == 0]
        
        summary['High Constraint'] = [
            f'{d_high.mean():.2f}',
            f'{d_high.std():.2f}',
            f'{d_high.min():.2f}',
            f'{d_high.quantile(0.25):.2f}',
            f'{d_high.median():.2f}',
            f'{d_high.quantile(0.75):.2f}',
            f'{d_high.max():.2f}',
            f'{len(d_high)}'
        ]
        
        summary['Low Constraint'] = [
            f'{d_low.mean():.2f}',
            f'{d_low.std():.2f}',
            f'{d_low.min():.2f}',
            f'{d_low.quantile(0.25):.2f}',
            f'{d_low.median():.2f}',
            f'{d_low.quantile(0.75):.2f}',
            f'{d_low.max():.2f}',
            f'{len(d_low)}'
        ]
    
    df = pd.DataFrame(summary)
    
    if save_path:
        # Save as LaTeX
        latex = df.to_latex(index=False, escape=False)
        with open(save_path, 'w') as f:
            f.write(latex)
        print(f"  Saved: {save_path}")
    
    return df


def run_dimension_analysis(industry_returns: pd.DataFrame,
                           master_data: pd.DataFrame) -> dict:
    """
    Run complete effective dimension analysis.
    
    Parameters:
        industry_returns: 30 industry portfolio returns
        master_data: Master dataset with constraint indicators
    
    Returns:
        Dictionary with dimension series and figures
    """
    print("=" * 60)
    print("EFFECTIVE DIMENSION ANALYSIS")
    print("=" * 60)
    
    results = {}
    
    # =========================================================================
    # COMPUTE EFFECTIVE DIMENSION
    # =========================================================================
    print("\n[1/4] Computing effective dimension series...")
    
    dimension_series = compute_effective_dimension_series(industry_returns)
    results['dimension_series'] = dimension_series
    
    # Merge with master data
    master_with_dim = master_data.join(dimension_series[['d_eff']], how='left')
    results['master_with_dim'] = master_with_dim
    
    print(f"  Computed dimension for {dimension_series['d_eff'].notna().sum()} months")
    print(f"  Mean dimension: {dimension_series['d_eff'].mean():.2f}")
    print(f"  Std dimension: {dimension_series['d_eff'].std():.2f}")
    
    # =========================================================================
    # CONDITIONAL EIGENVALUES
    # =========================================================================
    print("\n[2/4] Computing conditional eigenvalue structure...")
    
    if 'HIGH_CONSTRAINT' in master_data.columns:
        eig_high, eig_low = compute_conditional_eigenvalues(
            industry_returns, 
            master_data['HIGH_CONSTRAINT']
        )
        results['eigenvalues_high'] = eig_high
        results['eigenvalues_low'] = eig_low
        
        d_eff_high = effective_dimension(eig_high[:10])
        d_eff_low = effective_dimension(eig_low[:10])
        print(f"  Avg dimension (high constraint): {d_eff_high:.2f}")
        print(f"  Avg dimension (low constraint): {d_eff_low:.2f}")
    else:
        print("  ⚠ No constraint indicator available, skipping conditional analysis")
        eig_high = eig_low = None
    
    # =========================================================================
    # FIGURES
    # =========================================================================
    print("\n[3/4] Creating figures...")
    
    # Figure 2: Time series
    fig_ts = plot_dimension_timeseries(
        dimension_series,
        save_path=FIGURES_DIR / "figure2_dimension_timeseries.pdf"
    )
    results['fig_timeseries'] = fig_ts
    plt.close(fig_ts)
    
    # Figure 3: Cumulative variance
    if eig_high is not None:
        fig_cumvar = plot_cumulative_variance(
            eig_high, eig_low,
            save_path=FIGURES_DIR / "figure3_eigenvalue_structure.pdf"
        )
        results['fig_cumvar'] = fig_cumvar
        plt.close(fig_cumvar)
        
        # Figure 4: Scree comparison
        fig_scree = plot_scree_comparison(
            eig_high, eig_low,
            save_path=FIGURES_DIR / "figure4_scree_comparison.pdf"
        )
        results['fig_scree'] = fig_scree
        plt.close(fig_scree)
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    print("\n[4/4] Creating summary table...")
    
    condition = master_data.get('HIGH_CONSTRAINT')
    summary_table = create_dimension_summary_table(
        dimension_series,
        condition=condition,
        save_path=TABLES_DIR / "dimension_summary.tex"
    )
    results['summary_table'] = summary_table
    
    print("\n" + "=" * 60)
    print("DIMENSION ANALYSIS COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test with sample data
    from src.data.load import load_industry_returns, load_master_data
    
    try:
        industry = load_industry_returns()
        master = load_master_data()
        results = run_dimension_analysis(industry, master)
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        print("Run data download and processing first.")
