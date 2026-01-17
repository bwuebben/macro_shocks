"""
Global Configuration Settings
=============================

All parameters that control the analysis are defined here.
Modify these to change sample periods, thresholds, etc.
"""

from pathlib import Path
from datetime import date

# =============================================================================
# PATHS
# =============================================================================

# Project root (parent of config directory)
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANUAL_DATA_DIR = DATA_DIR / "manual"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MANUAL_DATA_DIR, 
                 TABLES_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SAMPLE PERIOD
# =============================================================================

# Full sample period (constrained by monetary shock availability)
SAMPLE_START = "1990-01-01"
SAMPLE_END = "2024-12-31"

# Extended sample for portfolios (used where shocks not needed)
EXTENDED_START = "1963-07-01"


# =============================================================================
# EFFECTIVE DIMENSION ESTIMATION
# =============================================================================

# Rolling window length in months for covariance estimation
ROLLING_WINDOW = 60

# Minimum observations required in window
MIN_OBS_WINDOW = 48


# =============================================================================
# SHOCK IDENTIFICATION
# =============================================================================

# Percentile threshold for contractionary shock indicator
# Shock_t = 1 if shock < percentile(shock, SHOCK_THRESHOLD_PERCENTILE)
SHOCK_THRESHOLD_PERCENTILE = 10

# Percentile for expansionary shocks (for placebo tests)
EXPANSIONARY_THRESHOLD_PERCENTILE = 90


# =============================================================================
# CONSTRAINT MEASURES
# =============================================================================

# Percentile threshold for high constraint indicator
# High_constraint_t = 1 if constraint > percentile(constraint, threshold)
CONSTRAINT_THRESHOLD_PERCENTILE = 75

# For capital ratio (inverted: low capital = high constraint)
CAPITAL_RATIO_THRESHOLD_PERCENTILE = 25

# Rolling window for rolling threshold robustness check
CONSTRAINT_ROLLING_WINDOW = 60


# =============================================================================
# PORTFOLIO CLASSIFICATION
# =============================================================================

# Risk factors (always-priced, from Fama-French)
RISK_FACTORS = ["MKT-RF", "SMB", "HML", "RMW", "CMA"]

# Mispricing portfolios (conditionally-priced)
MISPRICING_PORTFOLIOS = ["MOM", "ST_REV", "LT_REV", "BAB", "MGMT", "PERF"]

# Test assets for dimension estimation
TEST_ASSETS = "30_INDUSTRY"  # Use 30 industry portfolios


# =============================================================================
# EIGENVALUE ANALYSIS
# =============================================================================

# Threshold for classifying always-priced vs conditionally-priced directions
# Directions with eigenvalue rank <= EIGENVALUE_MEDIAN_SPLIT are "always-priced"
EIGENVALUE_MEDIAN_SPLIT = "median"  # Use median eigenvalue as cutoff


# =============================================================================
# STATISTICAL INFERENCE
# =============================================================================

# Clustering for standard errors
CLUSTER_VARIABLE = "quarter"  # Cluster by calendar quarter

# Number of bootstrap replications
N_BOOTSTRAP = 1000

# Number of permutations for placebo tests
N_PERMUTATIONS = 1000

# Significance levels
SIGNIFICANCE_LEVELS = [0.01, 0.05, 0.10]


# =============================================================================
# FIGURE SETTINGS
# =============================================================================

# Figure size (width, height) in inches
FIGURE_SIZE = (10, 6)
FIGURE_SIZE_SMALL = (8, 5)
FIGURE_SIZE_WIDE = (12, 5)

# DPI for saved figures
FIGURE_DPI = 300

# Style
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# Colors
COLOR_PRIMARY = "#1f77b4"      # Blue
COLOR_SECONDARY = "#ff7f0e"    # Orange
COLOR_CRISIS = "#d62728"       # Red
COLOR_NORMAL = "#2ca02c"       # Green


# =============================================================================
# TABLE SETTINGS
# =============================================================================

# Number of decimal places
DECIMALS_COEFFICIENTS = 2
DECIMALS_SE = 2
DECIMALS_PVALUE = 3
DECIMALS_RETURNS = 2


# =============================================================================
# DATA SOURCE URLS
# =============================================================================

URLS = {
    # Kenneth French Data Library
    "FF5_FACTORS": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "MOMENTUM": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
    "30_INDUSTRY": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_CSV.zip",
    "ST_REVERSAL": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_CSV.zip",
    "LT_REVERSAL": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_LT_Reversal_Factor_CSV.zip",
    
    # FRED series
    "VIX": "VIXCLS",
    "TED_SPREAD": "TEDRATE",
    "FF_RATE": "FEDFUNDS",
    
    # He-Kelly-Manela intermediary capital
    "INTERMEDIARY_CAPITAL": "https://drive.google.com/uc?export=download&id=1ahJpH_bPwVaVCrqF8Kx1RZ8Y4YcIbNWS",
    
    # AQR Betting Against Beta
    "BAB": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
}


# =============================================================================
# FRED API KEY
# =============================================================================

# Set your FRED API key here, or set environment variable FRED_API_KEY
# Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = None  # Will try environment variable if None
